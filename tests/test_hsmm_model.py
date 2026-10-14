"""class HSMM (hsmm/model.py) and BadSets (detectors/thresholds.py): the live step loop and the
whole-trial scan agree, and the whole-trial readouts match the batched paths the eval uses."""
import math
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from cook_ad.detectors import thresholds as T
from cook_ad.detectors.thresholds import BadSets
from cook_ad.hsmm import batch, kernels, params
from cook_ad.hsmm.model import HSMM

K_R, K, V, N, D_MAX = 3, 4, 3, 5, 6
ALPHAS = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 1e-3]
SHIPPED = [
    "dataset/processed/breakfast/joint_params.npz",
    "dataset/processed/breakfast/joint_params_train_lexhard.npz",
    "dataset/processed/breakfast/joint_params_train_dish.npz",
    "dataset/processed/epic_tent/joint_params_clean.npz",
    "dataset/processed/epic_tent/joint_params_clean_prego.npz",
]


def _model(seed=0, k_r=K_R):
    rng = np.random.default_rng(seed)
    trans = rng.gamma(1.0, 3.0, size=(k_r, K, K)) + 1.0
    for r in range(k_r):
        np.fill_diagonal(trans[r], 0.0)
    return params.JointHSMMParams(
        init_counts=jnp.asarray(rng.gamma(1.0, 3.0, size=(k_r, K)) + 1.0),
        trans_counts=jnp.asarray(trans),
        verb_counts=jnp.asarray(rng.gamma(0.5, 4.0, size=(K, V)) + 1.0),
        noun_counts=jnp.asarray(rng.gamma(0.5, 4.0, size=(K, N)) + 1.0),
        dur_r=jnp.asarray(rng.uniform(1.0, 6.0, size=(k_r, K))),
        dur_p=jnp.asarray(rng.uniform(0.3, 0.8, size=(k_r, K))),
        pi_counts=jnp.asarray(rng.gamma(2.0, 2.0, size=(k_r,)) + 1.0),
    )


def _trials(seed=1, n=5, t_range=(8, 15)):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        t = int(rng.integers(*t_range))
        out.append({"verb_ids": rng.integers(0, V, t), "noun_ids": rng.integers(0, N, t)})
    return out


@pytest.mark.parametrize("recipe", [None, 1, "prior"])
def test_step_loop_equals_filter(recipe):
    """predict + update one tick at a time == the scanned filter, belief and log P(o_t | o_<t)."""
    hsmm = HSMM(_model(), D_MAX)
    if recipe == "prior":
        recipe = np.log(np.array([0.7, 0.2, 0.1]))
    for s in _trials():
        fr = hsmm.filter(s["verb_ids"], s["noun_ids"], recipe=recipe)
        hsmm.reset(recipe)
        for t, (v, n) in enumerate(zip(s["verb_ids"], s["noun_ids"])):
            pi_tilde = hsmm.predict().pi_tilde
            ln = hsmm.update(int(v), int(n))
            np.testing.assert_allclose(pi_tilde, fr.pi_tilde[t], rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(ln, fr.log_norm[t], rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(hsmm.belief, fr.log_b[t], rtol=1e-12, atol=1e-12)


def test_filter_loglik_and_r_hat_match_infer_recipe():
    """sum_t log P(o_t | o_<t) == log P(o); HSMM.infer_recipe == the batched one."""
    jp = _model()
    hsmm = HSMM(jp, D_MAX)
    trials = _trials()
    verb, noun, mask = batch.pad([s["verb_ids"] for s in trials], [s["noun_ids"] for s in trials])
    r_hat, rho, trial_ll = batch.infer_recipe(hsmm.tables, verb, noun, mask, D_MAX)
    for i, s in enumerate(trials):
        r, log_post, log_p = hsmm.infer_recipe(s["verb_ids"], s["noun_ids"])
        assert r == int(r_hat[i])
        assert log_p == pytest.approx(float(trial_ll[i]), rel=1e-9)
        np.testing.assert_allclose(np.exp(log_post), rho[i], atol=1e-9)
        assert hsmm.filter(s["verb_ids"], s["noun_ids"]).total_loglik == pytest.approx(log_p, rel=1e-9)
        fr = hsmm.filter(s["verb_ids"], s["noun_ids"])
        hsmm.reset()
        for v, n in zip(s["verb_ids"], s["noun_ids"]):
            hsmm.update(int(v), int(n))
        np.testing.assert_allclose(np.exp(hsmm.recipe_belief), np.exp(log_post), atol=1e-9)
        assert fr.n == len(s["verb_ids"])


def test_viterbi_matches_batched_segmentation():
    jp = _model()
    hsmm = HSMM(jp, D_MAX)
    trials = _trials(n=6)
    verb, noun, mask = batch.pad([s["verb_ids"] for s in trials], [s["noun_ids"] for s in trials])
    r = np.arange(len(trials)) % K_R
    batched = batch.viterbi(hsmm.tables, jnp.asarray(r), verb, noun, mask, D_MAX)
    for i, s in enumerate(trials):
        assert hsmm.viterbi(s["verb_ids"], s["noun_ids"], int(r[i])) == batched[i]


def test_smooth_full_equals_long_fixed_lag():
    hsmm = HSMM(_model(), D_MAX)
    s = _trials(n=1)[0]
    fr = hsmm.filter(s["verb_ids"], s["noun_ids"])
    full = np.asarray(hsmm.smooth(fr, math.inf))[: fr.n]
    lagged = np.asarray(hsmm.smooth(fr, fr.n))[: fr.n]
    np.testing.assert_allclose(full, lagged, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("k_r", [1, K_R])
def test_bad_sets_hindsight_are_the_tail_thresholds(k_r):
    hsmm = HSMM(_model(k_r=k_r), D_MAX)
    bad = BadSets(hsmm, ALPHAS)
    t = hsmm.tables
    for a in (0.05, 1e-3):
        for r in range(k_r):
            got = bad.hindsight(r, a)
            np.testing.assert_array_equal(got.emit, T.joint_quantile_threshold(t.log_emit_v, t.log_emit_n, a))
            np.testing.assert_array_equal(got.verb, T.categorical_quantile_threshold(t.log_emit_v, a))
            np.testing.assert_array_equal(got.transition, T.transition_quantile_threshold(t.log_A[r], a))
            np.testing.assert_array_equal(got.recipe, T.recipe_excess_thresholds(
                t.log_A[r], np.asarray(t.log_A_marginal), a, k_r))
    if k_r == 1:
        assert np.all(np.asarray(bad.recipe_bin) == len(ALPHAS))


def test_initial_belief_is_the_prior():
    hsmm = HSMM(_model(), D_MAX)
    b0 = kernels.initial_belief(hsmm.view().log_b0, D_MAX)
    assert float(jax.scipy.special.logsumexp(b0)) == pytest.approx(0.0, abs=1e-12)
    assert np.all(np.isneginf(np.asarray(b0)[..., 1:]))


@pytest.mark.parametrize("path", SHIPPED)
def test_shipped_params_load_unchanged(path, tmp_path):
    if not os.path.exists(path):
        pytest.skip(f"{path} not present")
    p = params.load(path)
    with np.load(path) as raw:
        assert set(raw.files) == set(params.JointHSMMParams._fields)
        for name in raw.files:
            np.testing.assert_array_equal(np.asarray(getattr(p, name)), raw[name])
    params.save(p, tmp_path / "x.npz")
    q = params.load(tmp_path / "x.npz")
    for a, b in zip(p, q):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
