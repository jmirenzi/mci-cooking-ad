"""Pure JAX inference kernels for the joint HSMM (docs/model.md).

The belief is b_t(r, i, j) = log P(recipe r, subtask i, elapsed j+1 | o_0..o_t), shape (K_R, K, D).
`predict` and `update` are one tick of the filter; `filter_scan` is the same two kernels scanned
over a trial, so a live run and a whole-trial run agree by construction.
"""
import functools

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp


# ---------------------------------------------------------------------------------------------
# Emissions


def sequence_loglik(verbs, nouns, log_emit_v, log_emit_n, mask):
    """(T, K) log P(v_t | i) + log P(n_t | i); padded ticks are 0 (docs/model.md §1)."""
    loglik = log_emit_v[:, verbs].T + log_emit_n[:, nouns].T
    return jnp.where(mask[:, None], loglik, 0.0)


# ---------------------------------------------------------------------------------------------
# The filter (docs/model.md §3)


def initial_belief(log_b0, d_max):
    """b before o_0: log P(r) + log P(z_0 = i | r) in column j = 0, -inf elsewhere."""
    k_r, k = log_b0.shape
    rest = jnp.full((k_r, k, d_max - 1), -jnp.inf, dtype=log_b0.dtype)
    return jnp.concatenate([log_b0[..., None], rest], axis=-1)


def predict(log_b, log_A, log_h, log_stay):
    """b_{t-1} -> (b_pred, log_end, log_m), docs/model.md §3.1.

    log_end[r,k,j]: subtask k ended at t-1 after j+1 ticks; log_m = logsumexp_j log_end.
    Ended mass moves through A into column 0; the rest ages one column."""
    log_end = log_b + log_h
    log_m = logsumexp(log_end, axis=-1)                               # (K_R, K)
    new = logsumexp(log_m[:, :, None] + log_A, axis=1)                # (K_R, K): into column 0
    cont = log_b[..., :-1] + log_stay[..., :-1]
    return jnp.concatenate([new[..., None], cont], axis=-1), log_end, log_m


def update(b_pred, e):
    """Multiply by P(o_t | i) and normalise -> (b_t, log P(o_t | o_<t)), docs/model.md §3.2."""
    joint = b_pred + e[None, :, None]
    log_norm = logsumexp(joint)
    return joint - log_norm, log_norm


def occupancy(b_pred):
    """pi_tilde_t(i) = P(z_t = i | o_<t), split into 'stayed' (j >= 1) and 'switched' (j = 0)."""
    return (logsumexp(b_pred, axis=(0, 2)), logsumexp(b_pred[..., 1:], axis=(0, 2)),
            logsumexp(b_pred[..., 0], axis=0))


@jax.jit
def filter_scan(log_b0, log_A, log_h, log_stay, loglik, mask):
    """predict + update over a padded trial -> (log_b (T,K_R,K,D), log_norm (T,), pi_tilde,
    log_stay, log_switch (T,K)). Padded ticks repeat the last belief and add log_norm 0."""
    b0 = initial_belief(log_b0, log_h.shape[-1])

    def tick(carry, inp):
        log_b_prev, is_first = carry
        e, real = inp
        b_pred, _, _ = predict(log_b_prev, log_A, log_h, log_stay)
        b_pred = jnp.where(is_first, b0, b_pred)
        log_b, log_norm = update(b_pred, e)
        pi_tilde, stay, switch = occupancy(b_pred)
        log_b = jnp.where(real, log_b, log_b_prev)
        log_norm = jnp.where(real, log_norm, 0.0)
        return (log_b, jnp.where(real, False, is_first)), (log_b, log_norm, pi_tilde, stay, switch)

    _, out = jax.lax.scan(tick, (b0, jnp.array(True)), (loglik, mask))
    return out


# ---------------------------------------------------------------------------------------------
# Backward messages (docs/model.md §5)


def backward_step(beta_next, e_next, log_A, log_h, log_stay):
    """beta_t(r,i,j) = log P(o_{t+1..} | r, z_t = i, e_t = j+1) from beta_{t+1} and o_{t+1}."""
    stay = log_stay[..., :-1] + e_next[None, :, None] + beta_next[..., 1:]
    stay = jnp.concatenate([stay, jnp.full(stay.shape[:-1] + (1,), -jnp.inf, stay.dtype)], axis=-1)
    into = logsumexp(log_A + (e_next[None, :] + beta_next[..., 0])[:, None, :], axis=-1)
    leave = log_h + into[..., None]
    return jnp.logaddexp(stay, leave)


@jax.jit
def full_beta(log_A, log_h, log_stay, loglik, mask):
    """L = inf: beta_t = log P(o_{t+1 .. T-1} | r, z_t, e_t) for every t, as one reverse scan."""
    shape = tuple(log_h.shape)
    nxt_real = jnp.concatenate([mask[1:], jnp.zeros((1,), bool)])
    e_next = jnp.concatenate([loglik[1:], jnp.zeros_like(loglik[:1])], axis=0)

    def tick(beta_next, inp):
        e, real = inp
        beta = jnp.where(real, backward_step(beta_next, e, log_A, log_h, log_stay),
                         jnp.zeros(shape, loglik.dtype))
        return beta, beta

    _, betas = jax.lax.scan(tick, jnp.zeros(shape, loglik.dtype), (e_next, nxt_real), reverse=True)
    return betas


@functools.partial(jax.jit, static_argnames=("lag",))
def fixed_lag_beta(log_A, log_h, log_stay, loglik, t_true, lag):
    """beta_t = log P(o_{t+1 .. min(t+L, T-1)} | r, z_t, e_t) for every t at once; L = 0 is zeros."""
    t_max = loglik.shape[0]
    beta = jnp.zeros((t_max,) + tuple(log_h.shape), dtype=loglik.dtype)
    if lag == 0:
        return beta
    has_next = (jnp.arange(t_max) + 1) < t_true
    e_next = jnp.concatenate([loglik[1:], jnp.zeros_like(loglik[:1])], axis=0)
    step = jax.vmap(backward_step, in_axes=(0, 0, None, None, None))
    for _ in range(lag):
        beta_next = jnp.concatenate([beta[1:], jnp.zeros_like(beta[:1])], axis=0)
        beta = jnp.where(has_next[:, None, None, None],
                         step(beta_next, e_next, log_A, log_h, log_stay), 0.0)
    return beta


# ---------------------------------------------------------------------------------------------
# One-recipe forward pass in segment form (docs/model.md §6). Used for P(o | r) and Viterbi.


def padded_cumsum(loglik, d_max):
    """(cum_padded, offset): cum_padded[offset + i] = sum_{t<i} loglik[t], edge-padded by d_max."""
    zeros = jnp.zeros((1, loglik.shape[1]), dtype=loglik.dtype)
    cum = jnp.concatenate([zeros, jnp.cumsum(loglik, axis=0)], axis=0)  # (T+1, K)
    pad_left = jnp.tile(cum[0:1], (d_max, 1))
    pad_right = jnp.tile(cum[-1:], (d_max, 1))
    return jnp.concatenate([pad_left, cum, pad_right], axis=0), d_max


def _segment_loglik(cum_padded, offset, t, d_max, shift=1):
    """(D, K) log P(o_{t-r .. t+shift-1} | i) for lookback r = 0..D-1; -inf before tick 0."""
    end_val = jax.lax.dynamic_slice_in_dim(cum_padded, offset + t + shift, 1, axis=0)[0]
    window = jax.lax.dynamic_slice_in_dim(cum_padded, offset + t - d_max + 1, d_max, axis=0)
    valid = jnp.arange(d_max) <= t
    return jnp.where(valid[:, None], end_val[None, :] - jnp.flip(window, axis=0), -jnp.inf)


def _start_window(log_init, d_max, dtype):
    k = log_init.shape[0]
    return jnp.concatenate([log_init[None, :], jnp.full((d_max - 1, k), -jnp.inf, dtype=dtype)],
                           axis=0)


def forward_pass(loglik, log_init, log_A, log_pmf, log_surv, mask, d_max):
    """One recipe's explicit-duration forward pass -> (log_norm (T,), astar (T,K)).

    sum(log_norm) = log P(o | r). The window holds log P(a segment starts at t - r)."""
    pmf_t, surv_t = log_pmf.T, log_surv.T                                  # (D, K)
    cum_padded, offset = padded_cumsum(loglik, d_max)

    def tick(carry, t):
        window, logc_prev = carry
        seg = _segment_loglik(cum_padded, offset, t, d_max)
        astar = logsumexp(window + pmf_t + seg, axis=0)                    # segment ends at t
        logc = logsumexp(logsumexp(window + surv_t + seg, axis=0))         # still running at t
        new = logsumexp(astar[:, None] + log_A, axis=0)
        window_next = jnp.concatenate([new[None, :], window[:-1, :]], axis=0)
        real = mask[t]
        return ((jnp.where(real, window_next, window), jnp.where(real, logc, logc_prev)),
                (jnp.where(real, logc - logc_prev, 0.0), astar))

    init = (_start_window(log_init, d_max, loglik.dtype), jnp.array(0.0, dtype=loglik.dtype))
    _, (log_norm, astar) = jax.lax.scan(tick, init, jnp.arange(loglik.shape[0]))
    return log_norm, astar


def predictive_occupancy(loglik, log_init, log_A, log_pmf, log_surv, mask, d_max):
    """(T, K) log pi_tilde_t(i) = log P(z_t = i | o_<t) under one recipe, by forward_pass's recursion."""
    pmf_t, surv_t = log_pmf.T, log_surv.T
    cum_padded, offset = padded_cumsum(loglik, d_max)

    def tick(window, t):
        seg = _segment_loglik(cum_padded, offset, t, d_max)
        seg_pred = _segment_loglik(cum_padded, offset, t, d_max, shift=0)   # excludes o_t
        pi_unnorm = logsumexp(window + surv_t + seg_pred, axis=0)
        astar = logsumexp(window + pmf_t + seg, axis=0)
        new = logsumexp(astar[:, None] + log_A, axis=0)
        window_next = jnp.concatenate([new[None, :], window[:-1, :]], axis=0)
        return jnp.where(mask[t], window_next, window), pi_unnorm - logsumexp(pi_unnorm)

    _, pi_tilde = jax.lax.scan(tick, _start_window(log_init, d_max, loglik.dtype),
                               jnp.arange(loglik.shape[0]))
    return pi_tilde


# ---------------------------------------------------------------------------------------------
# Viterbi (docs/model.md §7): forward_pass with max for logsumexp, plus backpointers.


def viterbi_decode(loglik, log_init, log_A, log_pmf, log_surv, mask, d_max):
    """-> (astar, dur_bp, asurv, dur_bp_surv, prev_bp), each (T, K); read them with `traceback`."""
    pmf_t, surv_t = log_pmf.T, log_surv.T
    cum_padded, offset = padded_cumsum(loglik, d_max)

    def tick(window, t):
        seg = _segment_loglik(cum_padded, offset, t, d_max)
        ended = window + pmf_t + seg
        running = window + surv_t + seg
        into = jnp.max(ended, axis=0)[:, None] + log_A          # rows: from, cols: to
        window_next = jnp.concatenate([jnp.max(into, axis=0)[None, :], window[:-1, :]], axis=0)
        out = (jnp.max(ended, axis=0), jnp.argmax(ended, axis=0), jnp.max(running, axis=0),
               jnp.argmax(running, axis=0), jnp.argmax(into, axis=0))
        return jnp.where(mask[t], window_next, window), out

    _, out = jax.lax.scan(tick, _start_window(log_init, d_max, loglik.dtype),
                          jnp.arange(loglik.shape[0]))
    return out


def traceback(t_true, k_star, dur_bp, dur_bp_surv, prev_bp):
    """MAP segments [(subtask, duration)] oldest first; the last one is right-censored."""
    segments, e, k, first = [], t_true - 1, k_star, True
    while e >= 0:
        r = int(dur_bp_surv[e, k] if first else dur_bp[e, k])
        segments.append((int(k), r + 1))
        first = False
        if e - r == 0:
            break
        k = int(prev_bp[e - r - 1, k])
        e = e - r - 1
    return segments[::-1]
