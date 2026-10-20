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
    xi = np.zeros((1, 1, D))
    cens = np.zeros((1, 1, D))
    xi[0, 0, 4] = 6                              # 6 segments of 5 ticks,
    cens[0, 0, 2] = 2                            # 2 cut at 3 ticks
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


def test_thin_cells_keep_the_nb():
    rng = np.random.default_rng(0)
    traj = [{"recipe_id": 0, "segments": [(0, int(d)), (1, 3)]} for d in rng.integers(6, 9, 40)]
    traj += [{"recipe_id": 0, "segments": [(2, 5), (1, 3)]}] * 3          # state 2: 3 segments
    log_h = np.log(np.full((1, 3, D), 0.15))
    lp, _ = fit_np_hazard(traj, 1, 3, D, log_h, kappa=2.0, min_segments=10)
    nb_lp, _ = tables_from_hazard(np.exp(log_h))
    np.testing.assert_allclose(lp[0, 2], nb_lp[0, 2])                     # thin: the NB
    assert np.exp(lp[0, 0, 5:8]).sum() > 0.9                              # rich: the data


def test_sidecar_is_picked_up_by_hsmm_load(tmp_path):
    from cook_ad.hsmm import params as hparams
    from cook_ad.hsmm.model import HSMM, durations_path
    from cook_ad.hsmm.params import JointHSMMParams
    from cook_ad.hsmm.train.np_hazard import save_tables
    k = 2
    p = JointHSMMParams(jnp.ones((1, k)) * 2, jnp.asarray([[[0.0, 3.0], [3.0, 0.0]]]),
                        jnp.ones((k, 3)) * 2, jnp.ones((k, 3)) * 2, jnp.full((1, k), 3.0),
                        jnp.full((1, k), 0.5), jnp.ones(1) * 2)
    path = tmp_path / "m.npz"
    hparams.save(p, str(path))
    assert not HSMM.load(str(path), D).table_durations
    lp, ls = tables_from_hazard(np.full((1, k, D), 0.3))
    save_tables(durations_path(str(path)), lp, ls)
    h = HSMM.load(str(path), D)
    assert h.table_durations and np.allclose(np.asarray(h.tables.log_pmf), lp)
    assert not HSMM.load(str(path), D, durations=None).table_durations


def test_a_fully_gated_model_is_exactly_the_nb():
    """min_segments above every cell: the tables, bins and duration surprises equal the NB path."""
    from cook_ad.detectors.hsmm_hindsight import HindsightHSMMDetector
    from cook_ad.detectors.thresholds import BadSets
    from cook_ad.hsmm.model import HSMM
    from cook_ad.hsmm.params import JointHSMMParams
    k = 3
    p = JointHSMMParams(jnp.ones((1, k)) * 2, jnp.asarray([[[0.0, 3.0, 2.0], [3.0, 0.0, 2.0], [2.0, 3.0, 0.0]]]),
                        jnp.eye(k) * 50 + 1, jnp.eye(k) * 50 + 1, jnp.asarray([[2.0, 40.0, 5.0]]),
                        jnp.asarray([[0.5, 0.9, 0.6]]), jnp.ones(1) * 2)
    Dm = 60
    nb = HSMM(p, Dm)
    traj = [{"recipe_id": 0, "segments": [(0, 4), (1, 9), (2, 5)]}] * 5
    lp, ls = fit_np_hazard(traj, 1, k, Dm, np.asarray(nb.tables.log_h), min_segments=10**6,
                           nb_tables=(nb.tables.log_pmf, nb.tables.log_surv))
    st = HSMM(p, Dm, duration_tables=(lp, ls))
    for f in ("log_pmf", "log_surv", "log_h", "log_stay"):
        np.testing.assert_array_equal(np.asarray(getattr(nb.tables, f)), np.asarray(getattr(st.tables, f)))
    alphas = [0.1, 0.01, 1e-3]
    np.testing.assert_array_equal(np.asarray(BadSets(nb, alphas).early_bin), np.asarray(BadSets(st, alphas).early_bin))
    segs = [(0, 1), (1, 2), (2, 30), (0, 3)]
    a = HindsightHSMMDetector(nb, BadSets(nb, alphas), None).duration_surprise(segs, 0)[0]
    b = HindsightHSMMDetector(st, BadSets(st, alphas), None).duration_surprise(segs, 0)[0]
    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)
