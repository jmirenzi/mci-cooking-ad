"""hsmm/train/np_hazard.py and the table-duration path of the HSMM detectors."""
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from cook_ad.detectors.hsmm_hindsight import table_tails  # noqa: E402
from cook_ad.hsmm import durations  # noqa: E402
from cook_ad.hsmm.params import to_log_tables  # noqa: E402
from cook_ad.hsmm.train.np_hazard import at_risk, fit_np_hazard, tables_from_hazard  # noqa: E402

D = 12


def test_tables_from_hazard_is_a_distribution():
    h = np.full((1, 2, D), 0.2)
    lp, ls = tables_from_hazard(h)
    np.testing.assert_allclose(np.exp(lp).sum(-1), 1.0)
    np.testing.assert_allclose(np.exp(ls[..., 0]), 1.0)


def test_kaplan_meier_with_censoring():
    xi = np.zeros((1, 1, D)); cens = np.zeros((1, 1, D))
    xi[0, 0, 4] = 6; cens[0, 0, 2] = 2           # 6 segments of 5 ticks, 2 cut at 3 ticks
    n = at_risk(xi, cens)
    assert n[0, 0, 0] == 8 and n[0, 0, 3] == 6 and n[0, 0, 5] == 0


def test_np_hazard_matches_the_data_and_has_a_thin_left_tail():
    rng = np.random.default_rng(0)
    traj = [{"recipe_id": 0, "segments": [(0, int(d)), (1, 3)]} for d in rng.integers(6, 9, 300)]
    flat = np.log(np.full((1, 2, D), 0.15))     # a broad "NB" fallback
    lp, ls = fit_np_hazard(traj, 1, 2, D, flat, kappa=2.0, kappa0=2.0)
    pmf = np.exp(lp[0, 0])
    assert pmf[5:8].sum() > 0.95 and pmf[:3].sum() < 0.01     # lengths 6-8, almost nothing short


def test_table_tails_equal_the_nb_formulas():
    r, p = jnp.asarray([[3.0, 5.0]]), jnp.asarray([[0.4, 0.6]])
    lp, ls = jax.vmap(durations.duration_tables, in_axes=(0, 0, None))(r, p, D)
    t = to_log_tables.__globals__["LogTables"](*([None] * 5), lp, ls, None, None, None)
    d = np.array([1, 3, 7])
    a, b = table_tails(t, 0, np.array([0, 1, 0]), d)
    rr, pp = np.array([3.0, 5.0, 3.0]), np.array([0.4, 0.6, 0.4])
    np.testing.assert_allclose(a, durations.nb_log_survival_np(d.astype(float), rr, pp), atol=1e-9)
    np.testing.assert_allclose(b, durations.nb_log_cdf_np(d.astype(float), rr, pp), atol=1e-9)
