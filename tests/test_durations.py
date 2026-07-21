import jax
import jax.numpy as jnp
import pytest

from cook_ad.hsmm import durations

jax.config.update("jax_enable_x64", True)

D_MAX = 150


def test_pmf_sums_to_one_and_survival_shape():
    r = jnp.array([3.0, 5.0, 10.0, 2.0])
    p = jnp.array([0.3, 0.2, 0.1, 0.5])
    log_pmf, log_survival = durations.duration_tables(r, p, D_MAX)

    assert jnp.allclose(jnp.exp(log_pmf).sum(axis=-1), 1.0, atol=1e-6)
    assert jnp.allclose(jnp.exp(log_survival[:, 0]), 1.0, atol=1e-8)
    # Direct comparison, not jnp.diff: once the tail genuinely underflows to -inf, two
    # consecutive -inf entries subtract to NaN, but -inf <= -inf compares fine.
    assert jnp.all(log_survival[:, 1:] <= log_survival[:, :-1] + 1e-9)


def test_tail_mass_small_when_mean_well_under_d_max():
    r = jnp.array([4.0])
    p = jnp.array([0.3])  # mean(d') = r(1-p)/p ~= 9.3, well under D_max=150
    _, log_survival = durations.duration_tables(r, p, D_MAX)
    assert log_survival[0, -1] < -20.0


def test_survival_matches_reverse_cumsum_of_pmf():
    """Two numerically-unrelated code paths (betainc-based survival vs. a plain
    reverse-cumsum of the pmf table) should agree to floating precision."""
    r = jnp.array([2.0, 6.0, 15.0])
    p = jnp.array([0.4, 0.15, 0.6])
    log_pmf, log_survival = durations.duration_tables(r, p, D_MAX)
    pmf = jnp.exp(log_pmf)
    reverse_cumsum_survival = jnp.flip(jnp.cumsum(jnp.flip(pmf, axis=-1), axis=-1), axis=-1)
    assert jnp.max(jnp.abs(reverse_cumsum_survival - jnp.exp(log_survival))) < 1e-6


@pytest.mark.parametrize(
    "r_true, p_true, bad_start",
    [
        (4.0, 0.3, 999.0),
        (1.0, 0.1, 50.0),
        (20.0, 0.5, 1.0),
        (8.0, 0.4, 200.0),
    ],
)
def test_newton_recovers_known_r_p_from_bad_start(r_true, p_true, bad_start):
    """Population-limit check: a histogram built from the exact true pmf (scaled to a large
    fake sample size) should let Newton recover (r,p) exactly, regardless of starting point.
    Plain Newton (started from r_old) diverges here for some of these starts -- verified
    directly, starting at r=10 against r_true=4 data runs away to r~38000 within 30
    iterations -- which is why the M-step seeds from a method-of-moments estimate instead.
    """
    d = jnp.arange(1, D_MAX + 1, dtype=jnp.float64)
    pmf = jnp.exp(durations.nb_log_pmf(d, r_true, p_true))
    n_hat = (pmf * 1000.0)[None, :]
    n_hat_total, s_hat = durations.duration_stats_from_histogram(n_hat, D_MAX)

    r_fit = durations.newton_update_r(n_hat, n_hat_total, s_hat, jnp.array([bad_start]), n_iters=5)
    p_fit = durations.update_p_given_r(n_hat_total, s_hat, r_fit)

    assert jnp.isclose(r_fit[0], r_true, rtol=1e-3)
    assert jnp.isclose(p_fit[0], p_true, rtol=1e-3)


def test_starved_state_duration_unmoved():
    n_hat = jnp.zeros((1, D_MAX))
    n_hat_total, s_hat = durations.duration_stats_from_histogram(n_hat, D_MAX)
    r_old = jnp.array([7.0])
    r_new = durations.newton_update_r(n_hat, n_hat_total, s_hat, r_old, n_iters=5)
    assert jnp.array_equal(r_new, r_old)


def test_impute_censored_histogram_conserves_mass():
    r = jnp.array([4.0, 8.0])
    p = jnp.array([0.3, 0.4])
    xi_dur = jnp.zeros((2, D_MAX))
    cens = jnp.zeros((2, D_MAX)).at[:, 10].set(5.0).at[:, 30].set(3.0)

    n_hat = durations.impute_censored_histogram(xi_dur, cens, r, p, D_MAX)
    assert jnp.allclose(n_hat.sum(axis=-1), cens.sum(axis=-1), atol=1e-6)


def test_hazard_finite_and_guarded():
    r, p = jnp.array([[4.0]]), jnp.array([[0.3]])
    d = jnp.arange(1, D_MAX + 1, dtype=jnp.float64)[None, :]
    hazard = durations.nb_log_hazard(d, r, p)
    assert jnp.all(jnp.isfinite(hazard[:, :50]))
