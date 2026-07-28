import functools

import jax
import jax.numpy as jnp
import numpy as np

from cook_ad.hsmm import emissions, params
from cook_ad.hsmm.messages import _padded_cumsum


def viterbi_decode(loglik, log_init, log_trans, log_dur_pmf, log_dur_survival, mask, d_max):
    """Max-product analogue of hsmm.messages.forward_pass -- same (D_max,K) window/cumsum
    machinery, logsumexp replaced by max, with backpointers recorded for traceback.

    Returns four (T,K) arrays (T,K int for the two *_bp):
      astar_all:      best (pmf-weighted) log-score of a segment ending at t in state k.
      dur_bp_all:      argmax lookback r (duration d=r+1) achieving astar_all[t,k].
      asurv_all:       best (survival-weighted) log-score of a segment ending at t in state
                       k -- used only where t is a sequence's true final tick, to select the
                       right-censored terminal segment (mirrors aocc_terms in forward_pass).
      dur_bp_surv_all: argmax lookback r achieving asurv_all[t,k].
      prev_bp_all:     prev_bp_all[t,k'] = argmax predecessor state k transitioning into k'
                       at boundary time t+1 (i.e. F(t+1,:)'s backpointer; shift by one when
                       reading it for boundary time u, same off-by-one _boundary_from_astar
                       handles in messages.py).

    Traceback (see `traceback`) walks these from a sequence's last real tick backward.
    """
    T, K = loglik.shape
    cum_padded, offset = _padded_cumsum(loglik, d_max)

    r_range = jnp.arange(d_max)
    log_dur_pmf_t = log_dur_pmf.T             # (D,K)
    log_dur_survival_t = log_dur_survival.T   # (D,K)

    window_init = jnp.concatenate(
        [log_init[None, :], jnp.full((d_max - 1, K), -jnp.inf, dtype=loglik.dtype)], axis=0
    )

    def step(window, t):
        end_val = jax.lax.dynamic_slice_in_dim(cum_padded, offset + t + 1, 1, axis=0)[0]
        cum_window = jax.lax.dynamic_slice_in_dim(cum_padded, offset + t - d_max + 1, d_max, axis=0)
        cum_window = jnp.flip(cum_window, axis=0)
        logL = end_val[None, :] - cum_window  # (D,K): logL(t-r, t, k)

        valid_r = r_range <= t
        logL = jnp.where(valid_r[:, None], logL, -jnp.inf)

        astar_terms = window + log_dur_pmf_t + logL       # (D,K)
        asurv_terms = window + log_dur_survival_t + logL  # (D,K)

        astar = jnp.max(astar_terms, axis=0)          # (K,)
        dur_bp = jnp.argmax(astar_terms, axis=0)       # (K,)
        asurv = jnp.max(asurv_terms, axis=0)          # (K,)
        dur_bp_surv = jnp.argmax(asurv_terms, axis=0)  # (K,)

        trans_terms = astar[:, None] + log_trans  # (K,K): rows=prev state, cols=next state
        new_boundary = jnp.max(trans_terms, axis=0)     # (K,): F(t+1,:)
        prev_bp = jnp.argmax(trans_terms, axis=0)       # (K,)

        window_next = jnp.concatenate([new_boundary[None, :], window[:-1, :]], axis=0)
        is_real = mask[t]
        window_next = jnp.where(is_real, window_next, window)

        return window_next, (astar, dur_bp, asurv, dur_bp_surv, prev_bp)

    _, outputs = jax.lax.scan(step, window_init, jnp.arange(T))
    astar_all, dur_bp_all, asurv_all, dur_bp_surv_all, prev_bp_all = outputs
    return astar_all, dur_bp_all, asurv_all, dur_bp_surv_all, prev_bp_all


@functools.partial(jax.jit, static_argnames=("d_max",))
def _viterbi_batch(log_init, log_trans, log_dur_pmf, log_dur_survival, loglik, mask, d_max):
    return jax.vmap(viterbi_decode, in_axes=(0, None, None, None, None, 0, None))(
        loglik, log_init, log_trans, log_dur_pmf, log_dur_survival, mask, d_max
    )


@functools.partial(jax.jit, static_argnames=("d_max",))
def _viterbi_batch_conditioned(log_init, log_trans, log_dur_pmf, log_dur_survival, loglik, mask, d_max):
    """Same as _viterbi_batch, but the four dynamics tables also vary per trial (axis 0) --
    used when each trial has its own MAP-recipe tables rather than one shared set."""
    return jax.vmap(viterbi_decode, in_axes=(0, 0, 0, 0, 0, 0, None))(
        loglik, log_init, log_trans, log_dur_pmf, log_dur_survival, mask, d_max
    )


def traceback(t_true, k_star, dur_bp, dur_bp_surv, prev_bp):
    """Numpy reconstruction of the MAP segmentation for one sequence, from the arrays
    `viterbi_decode` returns (each already sliced to this sequence, (T,K) numpy).

    Starts at the true final tick (t_true-1) in the terminal state k_star (found by the
    caller as argmax_k asurv_all[t_true-1,:] -- the one place the right-censored/survival
    weighting applies). The first segment popped uses `dur_bp_surv` (it IS that terminal,
    possibly-censored segment); every earlier segment uses the ordinary pmf-weighted
    `dur_bp`. Stops when a segment's start reaches tick 0 (log_init, no predecessor).

    Returns segments oldest-to-newest: list of (subtask_id, duration).
    """
    segments = []
    e = t_true - 1
    k = k_star
    first = True
    while e >= 0:
        r = int(dur_bp_surv[e, k] if first else dur_bp[e, k])
        d = r + 1
        u = e - r
        segments.append((int(k), d))
        first = False
        if u == 0:
            break
        k = int(prev_bp[u - 1, k])
        e = u - 1
    segments.reverse()
    return segments


def segments_to_per_tick(segments, t_true):
    subtask_per_tick = np.zeros(t_true, dtype=np.int64)
    pos = 0
    for subtask_id, d in segments:
        subtask_per_tick[pos : pos + d] = subtask_id
        pos += d
    return subtask_per_tick


def _assemble_segment_results(dur_bp_all, dur_bp_surv_all, prev_bp_all, asurv_all, mask):
    """Shared numpy traceback loop: batched Viterbi backpointer arrays -> per-sequence
    {"segments", "subtask_per_tick"} dicts. Factored out so both the shared-table path
    (segment_all_from_log_probs) and the per-trial-conditioned path (segment_all_conditioned)
    reuse the exact same assembly logic."""
    dur_bp_np = np.asarray(dur_bp_all)
    dur_bp_surv_np = np.asarray(dur_bp_surv_all)
    prev_bp_np = np.asarray(prev_bp_all)
    asurv_np = np.asarray(asurv_all)
    mask_np = np.asarray(mask)

    results = []
    n = mask_np.shape[0]
    for i in range(n):
        t_true = int(mask_np[i].sum())
        k_star = int(np.argmax(asurv_np[i, t_true - 1, :]))
        segments = traceback(t_true, k_star, dur_bp_np[i], dur_bp_surv_np[i], prev_bp_np[i])
        results.append({
            "segments": segments,
            "subtask_per_tick": segments_to_per_tick(segments, t_true),
        })
    return results


def segment_all_from_log_probs(log_probs, verb_ids, noun_ids, mask, d_max):
    """Batched driver: emissions -> Viterbi decode (jit+vmap) -> per-sequence numpy
    traceback, given already-normalized HSMMLogProbs (one shared set of tables for the whole
    batch). Returns a list of {"segments": [(subtask_id,duration),...],
    "subtask_per_tick": (T_true,) int64 array} aligned with the input batch order.
    """
    loglik = jax.vmap(emissions.sequence_loglik, in_axes=(0, 0, None, None, 0))(
        verb_ids, noun_ids, log_probs.log_emit_v, log_probs.log_emit_n, mask
    )
    _, dur_bp_all, asurv_all, dur_bp_surv_all, prev_bp_all = _viterbi_batch(
        log_probs.log_init, log_probs.log_trans, log_probs.log_dur_pmf, log_probs.log_dur_survival,
        loglik, mask, d_max,
    )
    return _assemble_segment_results(dur_bp_all, dur_bp_surv_all, prev_bp_all, asurv_all, mask)


def segment_all(hsmm_params, verb_ids, noun_ids, mask, d_max):
    """Batched driver: HSMMParams -> normalize -> segment_all_from_log_probs. Unchanged
    interface/behavior for existing callers."""
    log_probs = params.to_log_probs(hsmm_params, d_max)
    return segment_all_from_log_probs(log_probs, verb_ids, noun_ids, mask, d_max)


def segment_all_conditioned(joint_log_probs, r_hat, verb_ids, noun_ids, mask, d_max):
    """Recipe-conditioned batched Viterbi: each trial i is decoded under its own MAP recipe's
    init/trans/duration tables (gathered from the JointHSMMLogProbs's K_R axis via r_hat),
    while emissions stay shared across recipes. Mirrors segment_all_from_log_probs but with
    per-trial dynamics tables instead of one shared set.
    """
    r_hat = jnp.asarray(r_hat)
    loglik = jax.vmap(emissions.sequence_loglik, in_axes=(0, 0, None, None, 0))(
        verb_ids, noun_ids, joint_log_probs.log_emit_v, joint_log_probs.log_emit_n, mask
    )

    log_init_i = joint_log_probs.log_init[r_hat]                  # (N,K)
    log_trans_i = joint_log_probs.log_trans[r_hat]                # (N,K,K)
    log_dur_pmf_i = joint_log_probs.log_dur_pmf[r_hat]             # (N,K,D)
    log_dur_survival_i = joint_log_probs.log_dur_survival[r_hat]   # (N,K,D)

    _, dur_bp_all, asurv_all, dur_bp_surv_all, prev_bp_all = _viterbi_batch_conditioned(
        log_init_i, log_trans_i, log_dur_pmf_i, log_dur_survival_i, loglik, mask, d_max,
    )
    return _assemble_segment_results(dur_bp_all, dur_bp_surv_all, prev_bp_all, asurv_all, mask)
