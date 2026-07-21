import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp


def _padded_cumsum(loglik, d_max):
    """loglik: (T,K). Returns (cum_padded, offset) where cum_padded[offset+i,:] = Cum[i,:]
    for i=0..T, Cum[0,:]=0, Cum[i,:]=sum_{t<i} loglik[t,:]. Edge-replicated by d_max on both
    sides so any window read during the scan stays in-bounds and finite; validity is
    enforced separately via explicit masks, never by relying on the padding value itself.
    """
    zeros = jnp.zeros((1, loglik.shape[1]), dtype=loglik.dtype)
    cum = jnp.concatenate([zeros, jnp.cumsum(loglik, axis=0)], axis=0)  # (T+1, K)
    pad_left = jnp.tile(cum[0:1], (d_max, 1))
    pad_right = jnp.tile(cum[-1:], (d_max, 1))
    cum_padded = jnp.concatenate([pad_left, cum, pad_right], axis=0)
    return cum_padded, d_max


def forward_pass(loglik, log_init, log_trans, log_dur_pmf, log_dur_survival, mask, d_max):
    """Explicit-duration forward pass (Yu 2010), one jax.lax.scan over ticks.

    loglik: (T,K) from emissions.sequence_loglik. mask: (T,) bool, True for real ticks.
    Returns (log_norm: (T,), astar_all: (T,K)).

    Carry is `window`, shape (D_max,K): window[r,:] = F(t-r,:), the log-probability that a
    NEW segment starts at boundary tick t-r, for lookback r=0..D_max-1 (duration d=r+1).
    Per step: Astar (pmf-weighted, feeds the next boundary/transition) and Aocc
    (survival-weighted, feeds the per-tick normalizer -- this is what makes right-censoring
    at any true sequence length work automatically, with no per-sequence special-casing).
    """
    T, K = loglik.shape
    cum_padded, offset = _padded_cumsum(loglik, d_max)

    r_range = jnp.arange(d_max)
    log_dur_pmf_t = log_dur_pmf.T             # (D,K)
    log_dur_survival_t = log_dur_survival.T   # (D,K)

    window_init = jnp.concatenate(
        [log_init[None, :], jnp.full((d_max - 1, K), -jnp.inf, dtype=loglik.dtype)], axis=0
    )

    def step(carry, t):
        window, logc_prev = carry

        end_val = jax.lax.dynamic_slice_in_dim(cum_padded, offset + t + 1, 1, axis=0)[0]  # (K,)
        cum_window = jax.lax.dynamic_slice_in_dim(cum_padded, offset + t - d_max + 1, d_max, axis=0)
        cum_window = jnp.flip(cum_window, axis=0)  # (D,K): Cum[t-r,:]
        logL = end_val[None, :] - cum_window        # (D,K): logL(t-r, t, k)

        valid_r = r_range <= t
        logL = jnp.where(valid_r[:, None], logL, -jnp.inf)

        astar_terms = window + log_dur_pmf_t + logL
        aocc_terms = window + log_dur_survival_t + logL

        astar = logsumexp(astar_terms, axis=0)  # (K,)
        aocc = logsumexp(aocc_terms, axis=0)    # (K,)

        logc_t = logsumexp(aocc)
        log_norm_t = logc_t - logc_prev

        new_boundary = logsumexp(astar[:, None] + log_trans, axis=0)  # (K,): F(t+1,:)
        window_next = jnp.concatenate([new_boundary[None, :], window[:-1, :]], axis=0)

        is_real = mask[t]
        window_next = jnp.where(is_real, window_next, window)
        logc_next = jnp.where(is_real, logc_t, logc_prev)
        log_norm_t = jnp.where(is_real, log_norm_t, 0.0)

        return (window_next, logc_next), (log_norm_t, astar)

    init_carry = (window_init, jnp.array(0.0, dtype=loglik.dtype))
    _, (log_norm, astar_all) = jax.lax.scan(step, init_carry, jnp.arange(T))
    return log_norm, astar_all


def backward_pass(loglik, log_trans, log_dur_pmf, log_dur_survival, mask, d_max):
    """Mirrored scan, run in reverse (t = T-1 downto 0) via jax.lax.scan(..., reverse=True).

    Carry `bwin`, shape (D_max,K): bwin[r,:] = Bstar(t+1+r,:) -- the backward-variable
    analogue of forward's window, but holding "future" (already-visited, since we scan
    backward) Bstar values instead of "past" F values.

    Base case is *not* a fixed array index: it's wherever the true sequence actually ends,
    `T_true = sum(mask)` (mask is a prefix of Trues by construction). At `t == T_true-1`,
    Bstar is force-set to 0 (log 1, no future observations) instead of computed from the
    transition formula. Sharp correctness point (this is what a first draft got wrong,
    corrected before trusting anything downstream): the weight used inside the recursion
    must be the *survival* function exactly for the one candidate duration whose end lands
    on `T_true-1`, and the *pmf* for every other candidate -- using pmf unconditionally
    double-counts the boundary segment as both possibly-exact and implicitly censored.
    Returns (bstar_all: (T,K), g_all: (T,K)) where g_all[t] = G(t+1,:), the "segment
    starts at t+1" backward variable needed (unmarginalized over next-state) for xi_trans.
    """
    T, K = loglik.shape
    cum_padded, offset = _padded_cumsum(loglik, d_max)
    t_true = jnp.sum(mask)

    r_range = jnp.arange(d_max)
    log_dur_pmf_t = log_dur_pmf.T
    log_dur_survival_t = log_dur_survival.T

    bwin_init = jnp.full((d_max, K), -jnp.inf, dtype=loglik.dtype)

    def step(bwin, t):
        end_cum_window = jax.lax.dynamic_slice_in_dim(cum_padded, offset + t + 2, d_max, axis=0)
        start_val = jax.lax.dynamic_slice_in_dim(cum_padded, offset + t + 1, 1, axis=0)[0]
        logL = end_cum_window - start_val[None, :]  # (D,K): logL(t+1, t+1+r, k)

        end_positions = t + 1 + r_range
        is_last = end_positions == (t_true - 1)
        valid = end_positions <= (t_true - 1)

        weight = jnp.where(is_last[:, None], log_dur_survival_t, log_dur_pmf_t)
        logL = jnp.where(valid[:, None], logL, -jnp.inf)

        combined = bwin + weight + logL  # (D,K)
        g_tplus1 = logsumexp(combined, axis=0)  # (K,): G(t+1,:)

        bstar_t = logsumexp(log_trans + g_tplus1[None, :], axis=1)  # (K,)
        bstar_t = jnp.where(t == t_true - 1, 0.0, bstar_t)

        bwin_next = jnp.concatenate([bstar_t[None, :], bwin[:-1, :]], axis=0)
        return bwin_next, (bstar_t, g_tplus1)

    _, (bstar_all, g_all) = jax.lax.scan(step, bwin_init, jnp.arange(T), reverse=True)
    return bstar_all, g_all


def _boundary_from_astar(astar_all, log_init, log_trans):
    """F(u,k): log-prob a new segment starts at boundary u. F(0,:)=log_init; F(u,:) for
    u>=1 is the transition of astar_all[u-1,:] (the segment-end variable) through log_trans.
    A single vectorized op over the already-materialized astar_all, not part of any scan.
    """
    f_rest = logsumexp(astar_all[:-1, :, None] + log_trans[None, :, :], axis=1)  # (T-1,K)
    return jnp.concatenate([log_init[None, :], f_rest], axis=0)


def combine_sufficient_stats(loglik, mask, log_init, log_trans, log_dur_pmf, log_dur_survival, d_max):
    """Runs forward + backward, then the E-step combination pass.

    Returns (xi_trans, xi_dur, cens, gamma, log_z):
      xi_trans (K,K): expected transition counts, i != j.
      xi_dur, cens (K,D_max): expected exact / right-censored duration histograms.
      gamma (T,K): state-occupancy posterior per tick.
      log_z: total sequence log-likelihood (scalar) -- same quantity as sum(log_norm).

    xi_trans is a direct vectorized (T,K,K) op (cheap enough not to need scanning). xi_dur/
    cens/gamma are organized as a scan over the duration axis d=1..D_max: for a fixed d,
    every quantity needed (a fixed-duration segment's log-lik, which single start point u
    is the true final/censored segment for this d, gamma's range-sum) reduces to a
    shift-and-add over already-materialized (T,K) arrays via the same cumsum trick used in
    forward/backward, keeping memory at O(T*K) per step rather than O(T*D_max*K) overall.
    """
    T, K = loglik.shape
    log_norm, astar_all = forward_pass(loglik, log_init, log_trans, log_dur_pmf, log_dur_survival, mask, d_max)
    bstar_all, g_all = backward_pass(loglik, log_trans, log_dur_pmf, log_dur_survival, mask, d_max)
    f_all = _boundary_from_astar(astar_all, log_init, log_trans)

    log_z = jnp.sum(log_norm)
    t_true = jnp.sum(mask)
    t_idx = jnp.arange(T)

    # Scanned over t rather than a dense (T,K,K) vectorized op: at K=64, T_max=650, batched
    # over 503 sequences via vmap, the dense version allocates >10GB per array (measured:
    # triggers an actual OOM) -- a scan keeps peak memory at O(N*K*K) instead of O(N*T*K*K).
    def trans_step(xi_trans_acc, t):
        term = astar_all[t][:, None] + log_trans + g_all[t][None, :] - log_z  # (K,K)
        term = jnp.where(t <= (t_true - 2), term, -jnp.inf)
        return xi_trans_acc + jnp.exp(term), None

    xi_trans, _ = jax.lax.scan(trans_step, jnp.zeros((K, K), dtype=loglik.dtype), t_idx)

    cum_padded, offset = _padded_cumsum(loglik, d_max)
    log_dur_pmf_t = log_dur_pmf.T
    log_dur_survival_t = log_dur_survival.T
    # bstar_all[u+d-1,:] is needed (the segment's END position), not bstar_all[u,:] -- pad
    # on the right so a dynamic_slice starting at (d-1) can always safely read T entries.
    bstar_padded = jnp.concatenate([bstar_all, jnp.zeros((d_max, K), dtype=bstar_all.dtype)], axis=0)

    def dur_step(gamma_acc, idx):
        d = idx + 1
        cum_shifted = jax.lax.dynamic_slice_in_dim(cum_padded, offset + d, T, axis=0)
        cum_base = jax.lax.dynamic_slice_in_dim(cum_padded, offset, T, axis=0)
        logL_d = cum_shifted - cum_base  # (T,K): logL(u, u+d-1, k) for u=0..T-1
        bstar_end = jax.lax.dynamic_slice_in_dim(bstar_padded, idx, T, axis=0)  # (T,K): Bstar(u+d-1,k)

        end_positions = t_idx + d - 1
        is_final_u = end_positions == (t_true - 1)
        valid_u = (end_positions <= (t_true - 1)) & (t_idx <= t_true - 1)

        weight_d = jnp.where(
            is_final_u[:, None], log_dur_survival_t[idx][None, :], log_dur_pmf_t[idx][None, :]
        )

        term_internal = f_all + weight_d + logL_d + bstar_end
        term_final = f_all + weight_d + logL_d  # Bstar(T_true-1,:) == 0, dropped
        term = jnp.where(is_final_u[:, None], term_final, term_internal)
        term = jnp.where(valid_u[:, None], term, -jnp.inf)

        prob = jnp.exp(term - log_z)  # (T,K)

        xi_dur_d = jnp.sum(jnp.where(is_final_u[:, None], 0.0, prob), axis=0)
        cens_d = jnp.sum(jnp.where(is_final_u[:, None], prob, 0.0), axis=0)

        cumprob = jnp.concatenate([jnp.zeros((1, K), dtype=prob.dtype), jnp.cumsum(prob, axis=0)], axis=0)
        lower = jnp.maximum(0, t_idx - d + 1)
        gamma_from_d = cumprob[t_idx + 1] - cumprob[lower]

        return gamma_acc + gamma_from_d, (xi_dur_d, cens_d)

    gamma_init = jnp.zeros((T, K), dtype=loglik.dtype)
    gamma, (xi_dur_all, cens_all) = jax.lax.scan(dur_step, gamma_init, jnp.arange(d_max))

    return xi_trans, xi_dur_all.T, cens_all.T, gamma, log_z
