"""Negative-binomial segment durations on d = 1, 2, ...: pmf, survival, cdf and hazard tables
(docs/model.md §1). Fitting lives in hsmm/train/durations_fit.py."""
import jax.numpy as jnp
from jax.scipy.special import betainc, gammaln, xlog1py

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


# ---------------------------------------------------------------------------------------------


def nb_log_survival_np(d, r, p):
    """numpy `nb_log_survival`."""
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
    """numpy `nb_log_cdf`."""
    import numpy as _np
    from scipy.special import betainc as _betainc

    d = _np.asarray(d, dtype=_np.float64)
    r = _np.asarray(r, dtype=_np.float64)
    p = _np.asarray(p, dtype=_np.float64)
    with _np.errstate(divide="ignore"):
        return _np.log(_betainc(r, d, p))


def nb_log_pmf_np(d, r, p):
    """numpy `nb_log_pmf`."""
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
