import functools

import jax
import jax.numpy as jnp
from jax.scipy.special import betainc, digamma, gammaln, polygamma, xlog1py

FLOOR = 1e-12
EPS = 1e-8

# Duration support convention: d = 1, 2, 3, ... (a segment occupies at least one tick).
# Internally this is NB(r, p) on d' = d - 1 >= 0.


def nb_log_pmf(d, r, p):
    return (
        gammaln(d - 1.0 + r) - gammaln(r) - gammaln(d)
        + r * jnp.log(p) + xlog1py(d - 1.0, -p)
    )


def nb_log_survival(d, r, p):
    """log P(D >= d). Uses the regularized-incomplete-beta identity P(D'<=k) = I_p(r, k+1),
    so P(D>=d) = P(D'>=d-1) = 1 - I_p(r, d-1) for d>=2 (verified against scipy.stats.nbinom
    to ~1e-6). d=1 is P(D>=1)=1 trivially, handled by explicit `where`, not by relying on
    betainc's d-1=0 edge case (betainc's second shape arg must stay > 0).
    """
    d_safe = jnp.maximum(d, 2.0)
    cdf_part = betainc(r, d_safe - 1.0, p)
    return jnp.where(d <= 1.0, 0.0, jnp.log1p(-cdf_part))


def nb_log_cdf(d, r, p):
    """log P(D <= d), the left tail. Same regularized-incomplete-beta identity as
    nb_log_survival: P(D<=d) = P(D'<=d-1) = I_p(r, d) = betainc(r, d, p) for d>=1 (verified
    against scipy.stats.nbinom.cdf(d-1, r, p) to machine precision; the second beta shape arg
    is d itself, always >= 1 on the support, so no d<=1 special-case is needed the way survival
    needs one). Used for the retrospective 'left too early' temporal channel.
    """
    return jnp.log(betainc(r, d, p))


def nb_log_hazard(d, r, p):
    """h(d) = P(D=d)/P(D>=d). Guarded: once the fitted tail has genuinely vanished
    (log_survival == -inf), pmf - survival would be -inf - -inf = NaN; return -inf instead.
    """
    log_pmf = nb_log_pmf(d, r, p)
    log_surv = nb_log_survival(d, r, p)
    return jnp.where(jnp.isfinite(log_surv), log_pmf - log_surv, -jnp.inf)


def duration_tables(dur_r, dur_p, d_max):
    """dur_r, dur_p: (K,). Returns (log_pmf, log_survival), each (K, D_max).

    Vectorized over d=1..d_max and K in one call. The last column of log_pmf is
    overwritten with log_survival's last column: the D_max'th table entry means
    P(D>=D_max), absorbing the infinite NB tail rather than P(D=D_max) -- a segment
    genuinely longer than D_max ticks is bookkept as ending at D_max.
    """
    r = jnp.clip(dur_r, EPS, None)[:, None]
    p = jnp.clip(dur_p, EPS, 1.0 - EPS)[:, None]
    d = jnp.arange(1, d_max + 1, dtype=r.dtype)[None, :]

    log_pmf = nb_log_pmf(d, r, p)
    log_survival = nb_log_survival(d, r, p)
    log_pmf = log_pmf.at[:, -1].set(log_survival[:, -1])
    return log_pmf, log_survival


def impute_censored_histogram(xi_dur, cens, dur_r_old, dur_p_old, d_max):
    """xi_dur, cens: (K, D_max) expected-count histograms from the E-step -- xi_dur for
    exactly-observed segment durations, cens for the one right-censored final segment per
    sequence (known only as "at least d_c" ticks). Redistributes each cens[k,d_c] across
    d=d_c..D_max proportional to pmf_old(k,d)/survival_old(k,d_c) = P(D=d | D>=d_c) under
    the *current* fit -- standard ECM imputation, re-run every M-step as (r,p) improves.
    Mass-conserving: weights for a fixed d_c sum to 1 over d>=d_c by construction (pmf and
    survival come from the same table), so total count is unchanged, only redistributed.
    """
    log_pmf_old, log_survival_old = duration_tables(dur_r_old, dur_p_old, d_max)
    pmf_old = jnp.exp(log_pmf_old)
    survival_old = jnp.exp(log_survival_old)

    d_idx = jnp.arange(d_max)
    valid = d_idx[None, :] >= d_idx[:, None]  # (d_c, d): only redistribute to d >= d_c
    survival_safe = jnp.maximum(survival_old, FLOOR)
    weight = pmf_old[:, None, :] / survival_safe[:, :, None]  # (K, d_c, d)
    weight = jnp.where(valid[None, :, :], weight, 0.0)

    imputed = jnp.einsum("kc,kcd->kd", cens, weight)
    return xi_dur + imputed


def duration_stats_from_histogram(n_hat, d_max):
    """n_hat: (K, D_max) expected-count histogram (already censoring-imputed).
    Returns (N_hat, S_hat): total expected segment count and expected sum of (d-1) per state.
    """
    d = jnp.arange(1, d_max + 1, dtype=n_hat.dtype)
    N_hat = jnp.sum(n_hat, axis=-1)
    S_hat = jnp.sum(n_hat * (d[None, :] - 1.0), axis=-1)
    return N_hat, S_hat


def update_p_given_r(N_hat, S_hat, r):
    N_hat_safe = jnp.maximum(N_hat, EPS)
    p = r * N_hat_safe / (r * N_hat_safe + S_hat + EPS)
    return jnp.clip(p, EPS, 1.0 - EPS)


def method_of_moments_r(n_hat, N_hat, S_hat, fallback=5.0, r_max=1e4):
    """Method-of-moments estimate of r from a duration histogram: r_mom = mean'^2/(var'-mean').

    This exists because plain Newton on the digamma score equation is only locally
    convergent, and the score(r) for this problem is *not* globally well-behaved: it
    decreases from the true root, reaches a minimum, then flattens back toward 0 (never
    re-crossing) as r->inf (the NB->Poisson degeneracy). Starting Newton from an arbitrary
    r (e.g. carried over from a previous EM iteration, or a wide uniform init draw) can
    land on the far side of that minimum, where Newton has no way to find its way back to
    the true root and instead diverges toward r->inf chasing the flattening tail --
    verified directly: starting Newton at r=10 against data truly generated by r=4 diverges
    to r~38000 within 30 iterations. Seeding from the method-of-moments estimate instead
    lands in the correct basin in every case tested (multiple true (r,p) pairs, including
    adversarial starting points), so it is used as the actual Newton starting point,
    regardless of `r_old` -- `r_old` is kept only as the starved-state fallback.

    Falls back to a fixed constant when the histogram is under-dispersed (variance <=
    mean, which NB cannot represent -- the moment formula would be negative/undefined).
    """
    d_max = n_hat.shape[-1]
    d = jnp.arange(1, d_max + 1, dtype=n_hat.dtype)
    N_hat_safe = jnp.maximum(N_hat, EPS)
    mean = S_hat / N_hat_safe
    second_moment = jnp.sum(n_hat * (d[None, :] - 1.0) ** 2, axis=-1) / N_hat_safe
    variance = second_moment - mean**2
    overdispersed = variance > mean + EPS
    r_mom = jnp.where(overdispersed, mean**2 / jnp.maximum(variance - mean, EPS), fallback)
    return jnp.clip(r_mom, EPS, r_max)


def newton_update_r(n_hat, N_hat, S_hat, r_old, n_iters=5, r_max=1e4):
    """Newton's method on the digamma-based score equation for r (no closed form, unlike
    p given r), started from the method-of-moments estimate (see `method_of_moments_r`) --
    not from `r_old` -- for global-convergence safety. Fixed iteration count (not a
    while-loop, keeps static shape). Guards a starved state (N_hat~=0, otherwise a 0/0 in
    p_hat(r)): denominators are floored before every division, and the final update leaves
    a starved state's r completely unmoved rather than letting a numerically-live-but-
    meaningless step drift it. An absolute clip on every step is a final safety net in
    case any real (non-synthetic) histogram shape still pushes Newton somewhere extreme.
    """
    d = jnp.arange(1, n_hat.shape[-1] + 1, dtype=r_old.dtype)
    N_hat_safe = jnp.maximum(N_hat, EPS)
    r_start = method_of_moments_r(n_hat, N_hat, S_hat)

    def step(r, _):
        p_r = r * N_hat_safe / (r * N_hat_safe + S_hat + EPS)
        score = (
            jnp.sum(n_hat * (digamma(d[None, :] - 1.0 + r[:, None]) - digamma(r[:, None])), axis=-1)
            + N_hat_safe * jnp.log(jnp.maximum(p_r, EPS))
        )
        score_prime = (
            jnp.sum(
                n_hat * (polygamma(1, d[None, :] - 1.0 + r[:, None]) - polygamma(1, r[:, None])),
                axis=-1,
            )
            + N_hat_safe * (1.0 / r - N_hat_safe / (r * N_hat_safe + S_hat + EPS))
        )
        safe_prime = jnp.where(jnp.abs(score_prime) > EPS, score_prime, EPS)
        r_new = r - score / safe_prime
        r_new = jnp.where(r_new > 0, r_new, (r + 1e-3) / 2.0)
        r_new = jnp.clip(r_new, EPS, r_max)
        return r_new, None

    r_final, _ = jax.lax.scan(step, r_start, xs=None, length=n_iters)
    return jnp.where(N_hat > EPS, r_final, r_old)


def fit_durations_shrunk(xi_dur_acc, cens_acc, dur_r_old, dur_p_old, d_max, kappa,
                          prev_global_r=None, prev_global_p=None, global_damping=0.0):
    """Per-(recipe,state) duration M-step with shrinkage toward the global per-state NegBin.

    xi_dur_acc, cens_acc: (K_R, K, D_max) responsibility-weighted duration / censoring
    histograms. dur_r_old, dur_p_old: (K_R, K) previous-iteration NB params, used only to
    impute each cell's own right-censored mass. kappa: scalar pseudocount budget (expected
    segments) of the global shape injected into every cell.

    Order is load-bearing: impute per cell with that cell's own old (r,p) -> pool over recipes
    for one global-per-state fit -> shrink each cell toward the global pmf -> re-fit per cell.
    Shrinking before imputing would double-count censored mass against the wrong (r,p).

    Splits sparse (recipe,state) cells' duration histograms into the K_R-fold-finer grid this
    model introduces; many cells have too few expected segments for method-of-moments seeding
    to land in a stable Newton basin (an under-dispersed sparse histogram is exactly where
    method_of_moments_r's fallback fires or Newton starts far from any real root). Adding
    kappa * pmf_global raises a starved cell's effective histogram into the well-behaved
    regime while leaving well-populated cells (N_hat >> kappa) essentially unperturbed.

    Global-fit damping (`prev_global_r`/`prev_global_p`/`global_damping`): the pooled global
    fit (r_global, p_global below) is itself refit FROM SCRATCH every call via method-of-
    moments + Newton -- and for a near-empty state (most cells, at real K_R/K scale), the
    pooled histogram it's fit against is thin and noisy, so r_global can swing by an order of
    magnitude between calls (confirmed directly on a real full-scale checkpoint: r=471->38 in
    one M-step for a state where every recipe's own cell had ~0 occupancy). Because kappa *
    pmf_global is injected into EVERY recipe's copy of that state, one noisy global estimate
    perturbs K_R cells at once -- exactly the shared-instability mechanism, not per-cell noise
    that would average out. With `global_damping` in (0,1) and both `prev_global_*` given, the
    USED global fit is `damping * prev + (1-damping) * fresh` (an EMA across M-step calls, not
    a single fresh fit) -- default 0.0 / prev=None reproduces the undamped behavior exactly
    (verified: test_fit_durations_shrunk_kappa_zero_matches_plain_fit needs no change). The EMA
    state lives in the caller's loop (run_joint_em), NOT in JointHSMMParams, so it resets to
    undamped on a checkpoint resume rather than requiring a schema change to persisted params --
    a deliberate choice to keep existing checkpoints loadable unchanged; the cost is only that
    damping takes a few iterations to "warm up" again after each resume.

    Returns dur_r, dur_p, global_r, global_p: (K_R, K), (K_R, K), (K,), (K,). The last two are
    the USED (possibly damped) global fit -- pass them back in as `prev_global_r`/`prev_global_p`
    on the next call to continue the EMA.
    """
    impute_over_r = jax.vmap(impute_censored_histogram, in_axes=(0, 0, 0, 0, None))
    n_hat = impute_over_r(xi_dur_acc, cens_acc, dur_r_old, dur_p_old, d_max)  # (K_R,K,D)

    n_hat_global = jnp.sum(n_hat, axis=0)  # (K,D): pooled over recipes
    n_hat_global_total, s_hat_global = duration_stats_from_histogram(n_hat_global, d_max)
    r_fallback_global = jnp.mean(dur_r_old, axis=0)  # (K,): starved-state fallback for the global fit
    r_global_fresh = newton_update_r(n_hat_global, n_hat_global_total, s_hat_global, r_fallback_global, n_iters=5)
    p_global_fresh = update_p_given_r(n_hat_global_total, s_hat_global, r_global_fresh)

    if prev_global_r is not None and global_damping > 0.0:
        r_global = global_damping * prev_global_r + (1.0 - global_damping) * r_global_fresh
        p_global = global_damping * prev_global_p + (1.0 - global_damping) * p_global_fresh
    else:
        r_global, p_global = r_global_fresh, p_global_fresh

    log_pmf_global, _ = duration_tables(r_global, p_global, d_max)
    pmf_global = jnp.exp(log_pmf_global)  # (K,D)

    n_hat_shrunk = n_hat + kappa * pmf_global[None, :, :]  # (K_R,K,D)

    stats_over_r = jax.vmap(duration_stats_from_histogram, in_axes=(0, None))
    n_hat_total, s_hat = stats_over_r(n_hat_shrunk, d_max)  # (K_R,K) each

    newton_over_r = jax.vmap(functools.partial(newton_update_r, n_iters=5), in_axes=(0, 0, 0, 0))
    dur_r = newton_over_r(n_hat_shrunk, n_hat_total, s_hat, dur_r_old)

    p_over_r = jax.vmap(update_p_given_r, in_axes=(0, 0, 0))
    dur_p = p_over_r(n_hat_total, s_hat, dur_r)

    return dur_r, dur_p, r_global, p_global


# ---------------------------------------------------------------------------------------------
# numpy/scipy duplicates of the three NB tail functions above.
#
# The jax versions are the ones EM needs (they run inside jitted, vmapped M-steps). The anomaly
# channels do NOT: anomaly/temporal.py is a pure-numpy module scoring a handful of segments per
# trial, and there `jax.scipy.special.betainc` costs ~0.1s PER CALL on a float64 GPU regardless
# of array size -- it is an iterative kernel with a fixed trip count, so a 7-element array pays
# the same as a 7-million-element one. Measured on the full-scale train split that single
# primitive was ~90% of the wall time of every evaluation sweep in the repo (0.3s per trial x
# 378 trials x 6 source groups).
#
# scipy.special.betainc is the same regularized incomplete beta I_x(a,b), on the same
# convention, and is the reference these functions' docstrings already say they were verified
# against -- so this is the accurate side of the pair, not an approximation of the jax one.
# Agreement is asserted in tests/test_temporal.py.
# ---------------------------------------------------------------------------------------------


def nb_log_survival_np(d, r, p):
    """numpy `nb_log_survival`. See the block comment above for why this duplicate exists."""
    import numpy as _np
    from scipy.special import betainc as _betainc

    d = _np.asarray(d, dtype=_np.float64)
    r = _np.asarray(r, dtype=_np.float64)
    p = _np.asarray(p, dtype=_np.float64)
    d_safe = _np.maximum(d, 2.0)
    cdf_part = _betainc(r, d_safe - 1.0, p)
    with _np.errstate(divide="ignore"):
        out = _np.log1p(-cdf_part)
    return _np.where(d <= 1.0, 0.0, out)


def nb_log_cdf_np(d, r, p):
    """numpy `nb_log_cdf`. See the block comment above for why this duplicate exists."""
    import numpy as _np
    from scipy.special import betainc as _betainc

    d = _np.asarray(d, dtype=_np.float64)
    r = _np.asarray(r, dtype=_np.float64)
    p = _np.asarray(p, dtype=_np.float64)
    with _np.errstate(divide="ignore"):
        return _np.log(_betainc(r, d, p))


def nb_log_pmf_np(d, r, p):
    """numpy `nb_log_pmf`. See the block comment above for why this duplicate exists."""
    import numpy as _np
    from scipy.special import gammaln as _gammaln, xlog1py as _xlog1py

    d = _np.asarray(d, dtype=_np.float64)
    r = _np.asarray(r, dtype=_np.float64)
    p = _np.asarray(p, dtype=_np.float64)
    with _np.errstate(divide="ignore"):
        return (
            _gammaln(d - 1.0 + r) - _gammaln(r) - _gammaln(d)
            + r * _np.log(p) + _xlog1py(d - 1.0, -p)
        )
