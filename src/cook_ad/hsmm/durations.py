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
