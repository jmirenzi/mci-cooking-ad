"""HindsightHSMMDetector (detectors/hsmm_hindsight.py): each channel, its threshold, and the flag
rules -- dilution-corrected emission thresholds with a float tolerance, boundary-only transition
channels, the censored final segment, and the one-recipe rule."""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_messages import _brute_force_segmentations, _brute_force_stats, _random_log_probs

jax.config.update("jax_enable_x64", True)

from _fixtures import one_recipe_tables  # noqa: E402
from cook_ad.core import OperatingPoint, Trial  # noqa: E402
from cook_ad.detectors.hsmm_hindsight import (  # noqa: E402
    ATTRIBUTION_MARGIN,
    CHANNELS,
    HindsightHSMMDetector,
    HindsightTrace,
)
from cook_ad.detectors.thresholds import BadSets, categorical_quantile_threshold  # noqa: E402
from cook_ad.hsmm import durations, kernels  # noqa: E402
from cook_ad.hsmm.model import HSMM  # noqa: E402
from cook_ad.hsmm.params import JointHSMMParams, RecipeView, load  # noqa: E402
from cook_ad.synthetic import generate  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
SCALE = 1e6   # counts = 1 + SCALE * p, so the Dirichlet MAP is p (and FLOOR where p = 0)


def model_from_probs(init, trans, verb, noun, dur_r, dur_p, pi=None, d_max=30):
    """An HSMM whose tables are (to rounding) the given probabilities; recipe axis first."""
    init, trans = np.atleast_2d(init), np.asarray(trans)
    trans = trans[None] if trans.ndim == 2 else trans
    k_r, k = init.shape
    pi = np.ones(k_r) / k_r if pi is None else np.asarray(pi)
    p = JointHSMMParams(
        jnp.asarray(1 + SCALE * init), jnp.asarray((1 + SCALE * trans) * (1 - np.eye(k))[None]),
        jnp.asarray(1 + SCALE * np.asarray(verb)), jnp.asarray(1 + SCALE * np.asarray(noun)),
        jnp.asarray(np.broadcast_to(dur_r, (k_r, k)).astype(float)),
        jnp.asarray(np.broadcast_to(dur_p, (k_r, k)).astype(float)), jnp.asarray(1 + SCALE * pi))
    return HSMM(p, d_max)


def detector(hsmm, alpha=0.05):
    return HindsightHSMMDetector(hsmm, BadSets(hsmm, [alpha]), OperatingPoint(alpha))


def trace(z, pi_at_zstar, from_state, r=0, **surprise):
    t = len(z)
    s = {c: np.zeros(t) for c in CHANNELS}
    s.update({k: np.asarray(v, dtype=float) for k, v in surprise.items()})
    segs = [(int(z[0]), 1)]
    return HindsightTrace(r, segs, np.asarray(z), np.zeros(t, int), np.zeros(t, int), s,
                          np.asarray(pi_at_zstar, dtype=float), np.asarray(from_state),
                          np.zeros(t, int), np.zeros(t, int), np.full(t, -1),
                          np.full(t, "none", dtype=object), np.full(t, "none", dtype=object))


# ---- emission channels ---------------------------------------------------------------------


def test_emission_thresholds_add_the_dilution_offset():
    """threshold = z*'s own alpha-quantile + -log P(z* | o_<t): exactly the offset that cancels
    the forecast mixture's dilution, and nothing when the belief is a point mass."""
    hsmm = model_from_probs([[0.5, 0.5]], [[0, 1], [1, 0]], [[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]],
                            [[0.6, 0.4], [0.3, 0.7]], 4.0, 0.3)
    det = detector(hsmm)
    tr = trace(z=[0, 1, 0], pi_at_zstar=[0.5, 1.0, 0.1], from_state=[-1, -1, -1])
    th = det.thresholds(tr)
    table = det.bad.hindsight(0, 0.05)
    offset = -np.log(tr.pi_at_zstar)
    np.testing.assert_allclose(th["s_emit"], table.emit[tr.z_star] + offset)
    np.testing.assert_allclose(th["s_verb"], table.verb[tr.z_star] + offset)
    np.testing.assert_allclose(th["s_noun"], table.noun[tr.z_star] + offset)
    assert th["s_verb"][1] == table.verb[1]


def test_the_dominant_token_is_never_over_its_threshold_under_dilution():
    """For ANY P(z*) in (0, 1], z*'s dominant token stays inside the corrected threshold."""
    rng = np.random.default_rng(0)
    p_star, p_other = rng.dirichlet(np.ones(6)), rng.dirichlet(np.ones(6))
    dom = int(np.argmax(p_star))
    threshold = categorical_quantile_threshold(np.log(p_star)[None, :], alpha=0.05)[0]
    for w in (1.0, 0.99, 0.9, 0.5, 0.1, 0.01, 1e-4):
        s = -np.log(w * p_star[dom] + (1.0 - w) * p_other[dom])
        assert s <= threshold - np.log(w) + 1e-9


def test_a_near_deterministic_states_own_mode_is_never_flagged():
    """The emission surprise and its threshold are computed by two numerical paths that land
    ~1 ULP apart when z* is near-deterministic and observed at its mode; EMISSION_FLAG_ATOL
    keeps that noise from flagging (it once flagged ~99% of healthy trials at every alpha)."""
    v, dom = 36, 2
    p_star, p_other = np.zeros(v), np.zeros(v)
    p_star[dom], p_other[5] = 1.0, 1.0
    hsmm = model_from_probs([[0.5, 0.5]], [[0, 1], [1, 0]], [p_star, p_other], [p_star, p_other],
                            4.0, 0.3)
    det = detector(hsmm)
    for w in (0.999, 0.95, 0.8, 0.5, 0.1, 0.016, 0.011, 0.006, 0.001):
        pi = np.log(np.array([[w, 1.0 - w]]))
        s_emit, s_verb, s_noun = det.emission_surprise(pi, np.array([dom]), np.array([dom]))
        f = det.flags(trace([0], [w], [-1], s_emit=s_emit, s_verb=s_verb, s_noun=s_noun))
        assert not (f["s_emit"][0] or f["s_verb"][0] or f["s_noun"][0]), w


def test_emission_surprise_attributes_a_wrong_object_to_the_item():
    hsmm = model_from_probs([[0.5, 0.5]], [[0, 1], [1, 0]],
                            [[1 / 3, 1 / 3, 1 / 3], [0.9, 0.05, 0.05]],
                            [[1 / 3, 1 / 3, 1 / 3], [0.9, 0.05, 0.05]], 4.0, 0.3)
    det = detector(hsmm)
    pi = np.log(np.array([[0.01, 0.99]]))
    tr = det.assemble(0, [(1, 1)], pi, np.array([0]), np.array([2]))
    assert tr.surprise["s_noun"][0] > tr.surprise["s_verb"][0] + ATTRIBUTION_MARGIN
    assert tr.attribution[0] == "item"


@pytest.mark.parametrize("t, k, d_max", [(6, 3, 3), (8, 3, 4), (7, 4, 4)])
def test_predictive_occupancy_matches_bruteforce(t, k, d_max):
    """pi_tilde_t = P(z_t | o_<t): the posterior at the last tick of the prefix 0..t with o_t's
    evidence zeroed and the final segment censored -- the brute-force enumerator's gamma."""
    rng = np.random.default_rng(1)
    log_init, log_A, log_pmf, log_surv = _random_log_probs(rng, k, d_max)
    loglik = rng.standard_normal((t, k)) * 0.4
    pi = np.exp(np.asarray(kernels.predictive_occupancy(
        jnp.array(loglik), jnp.array(log_init), jnp.array(log_A), jnp.array(log_pmf),
        jnp.array(log_surv), jnp.ones(t, bool), d_max)))
    np.testing.assert_allclose(pi.sum(axis=-1), 1.0, atol=1e-8)
    for tick in range(t):
        ll = loglik[: tick + 1].copy()
        ll[tick] = 0.0
        segs = _brute_force_segmentations(ll, log_init, log_A, log_pmf, log_surv, tick + 1, d_max, k)
        np.testing.assert_allclose(pi[tick], _brute_force_stats(segs, tick + 1, d_max, k)[4][tick],
                                   atol=1e-8)


def test_predictive_occupancy_keeps_a_starved_state_finite_and_inert():
    k, nv, nn, d_max, t, starved = 6, 15, 36, 20, 40, 0
    ki, kt = jax.random.split(jax.random.PRNGKey(3))
    init = jax.random.uniform(ki, (k,), minval=5.0, maxval=50.0).at[starved].set(0.0)
    trans = jax.random.uniform(kt, (k, k), minval=5.0, maxval=50.0) * (1.0 - jnp.eye(k))
    trans = trans.at[starved, :].set(0.0).at[:, starved].set(0.0)
    verb = jnp.full((k, nv), 0.5).at[jnp.arange(1, k), jnp.arange(1, k) % nv].set(200.0).at[starved].set(0.0)
    noun = jnp.full((k, nn), 0.5).at[jnp.arange(1, k), jnp.arange(1, k) % nn].set(200.0).at[starved].set(0.0)
    tb = one_recipe_tables(RecipeView(init, trans, verb, noun, jnp.full((k,), 5.0),
                                      jnp.full((k,), 0.3)), d_max)
    states = jax.random.randint(jax.random.PRNGKey(11), (t,), 1, k)
    mask = jnp.ones((t,), bool)
    e = kernels.sequence_loglik(states % nv, states % nn, tb.log_emit_v, tb.log_emit_n, mask)
    pi = kernels.predictive_occupancy(e, tb.log_init, tb.log_A, tb.log_pmf, tb.log_surv, mask, d_max)
    assert jnp.all(jnp.isfinite(pi))
    assert jnp.max(jnp.exp(pi[:, starved])) < 1e-6


# ---- transition channels ---------------------------------------------------------------------


def test_transition_surprise_sits_on_segment_starts():
    trans = np.array([[0.0, 0.9, 0.1], [0.85, 0.0, 0.15], [0.3, 0.3, 0.4]])
    det = detector(model_from_probs([[1 / 3] * 3], trans, np.eye(3) + 0.1, np.eye(3) + 0.1, 4.0, 0.3))
    with np.errstate(divide="ignore"):
        s, expected = det.transition_surprise([(0, 3), (1, 2), (2, 4)], np.log(trans))
    assert expected[0] == -1 and s[0] == 0.0
    assert expected[3] == 1           # state 0 normally goes to 1, which is what happened
    assert s[5] > s[3]                # 1 -> 2 (0.15) is more surprising than 0 -> 1 (0.9)
    assert expected[5] == 0


def test_transition_threshold_is_the_from_states_own_row():
    """State 0's row is peaked (0.99 / 0.01), state 2's flat (0.5 / 0.5). A transition carrying
    -log 0.5 of surprise is in state 0's alpha-tail but ordinary for state 2."""
    trans = np.array([[0.0, 0.99, 0.01], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
    det = detector(model_from_probs([[1 / 3] * 3], trans, np.eye(3) + 0.1, np.eye(3) + 0.1, 4.0, 0.3))
    s = -np.log(0.5)
    from_peaked = det.flags(trace([2, 2], [1, 1], from_state=[-1, 0], s_transition=[0, s]))
    from_flat = det.flags(trace([1, 1], [1, 1], from_state=[-1, 2], s_transition=[0, s]))
    assert from_peaked["s_transition"][1] and not from_flat["s_transition"][1]


def test_transition_channels_fire_only_at_segment_starts_even_under_a_negative_threshold():
    """The recipe-excess threshold can be negative (a recipe row flatter than the marginal), so
    `0 > threshold` must not flag a tick that is not a segment start."""
    trans_r = np.array([[[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.4, 0.4, 0.2]],
                        [[0.0, 0.98, 0.02], [0.98, 0.0, 0.02], [0.02, 0.02, 0.96]]])
    hsmm = model_from_probs([[1 / 3] * 3] * 2, trans_r, np.eye(3) + 0.1, np.eye(3) + 0.1, 4.0, 0.3)
    det = detector(hsmm, alpha=0.3)
    th = det.bad.hindsight(0, 0.3).recipe
    assert th.min() < 0.0
    f = det.flags(trace([2, 2, 2], [1, 1, 1], from_state=[-1, -1, -1]))
    assert not f["s_recipe_transition"].any() and not f["s_transition"].any()


# ---- duration channels -----------------------------------------------------------------------


def _dur_detector(d_max=150):
    """Two states (a one-state model has no transitions), both with NB(r=4, p=0.3) durations."""
    return detector(model_from_probs([[0.5, 0.5]], [[0, 1], [1, 0]], np.eye(2) + 0.1,
                                     np.eye(2) + 0.1, 4.0, 0.3, d_max=d_max))


def test_stall_surprise_is_the_survival_surprise():
    """-log P(D >= elapsed): monotone within a segment and 0 on its first tick, so -log(alpha)
    is a per-state tail quantile."""
    det = _dur_detector()
    s = det.stall_surprise([(0, 30)], 0)
    log_surv = durations.duration_tables(jnp.array([4.0]), jnp.array([0.3]), 150)[1]
    assert np.all(np.diff(s) >= -1e-9) and s[0] == pytest.approx(0.0, abs=1e-9)
    np.testing.assert_allclose(s, -np.asarray(log_surv)[0, :30], atol=1e-9)


def test_stall_surprise_holds_its_last_value_past_d_max():
    s = _dur_detector(d_max=5).stall_surprise([(0, 10)], 0)
    assert len(s) == 10 and np.all(np.isfinite(s))
    np.testing.assert_allclose(s[4:], s[4])


def test_duration_surprise_catches_both_tails_and_skips_the_censored_last_segment():
    """A ~9-tick subtask (r=4, p=0.3): closing after 1 tick is 'left_early', after 40 'stuck',
    after 9 unremarkable. The trial's last segment has not closed, so it is never scored."""
    det = _dur_detector()
    segs = [(0, 1), (0, 9), (0, 40), (0, 1)]
    s, attr = det.duration_surprise(segs, 0)
    ends = np.cumsum([d for _, d in segs]) - 1
    thresh = -np.log(0.05)
    assert (attr[ends[0]], attr[ends[2]], attr[ends[3]]) == ("left_early", "stuck", "none")
    assert s[ends[0]] > thresh and s[ends[2]] > thresh and s[ends[1]] < thresh
    assert s[ends[3]] == 0.0


def test_numpy_nb_tails_match_scipy():
    from scipy.stats import nbinom
    rng = np.random.default_rng(1)
    r, p = rng.uniform(0.1, 500.0, 2000), rng.uniform(1e-3, 1 - 1e-3, 2000)
    d = rng.integers(1, 200, 2000).astype(float)
    ref, got = nbinom.logcdf(d - 1, r, p), durations.nb_log_cdf_np(d, r, p)
    ok = np.isfinite(ref) & np.isfinite(got)
    assert np.max(np.abs(got[ok] - ref[ok])) < 1e-9
    assert np.array_equal(np.isfinite(got), np.isfinite(ref))
    ref, got = nbinom.logpmf(d - 1, r, p), durations.nb_log_pmf_np(d, r, p)
    ok = np.isfinite(ref) & np.isfinite(got)
    assert np.max(np.abs(got[ok] - ref[ok])) < 1e-9
    assert np.all(durations.nb_log_survival_np(np.ones(5), r[:5], p[:5]) == 0.0)


# ---- whole trials ----------------------------------------------------------------------------


def test_detect_many_end_to_end_on_sampled_trials():
    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    from _fixtures import stack_recipes, weak_limit_params
    jp = stack_recipes([weak_limit_params(k, 5, 6, 7, 15) for k in keys])
    hsmm = HSMM(jp, 15)
    trials = [Trial(str(i), t["verb_ids"], t["noun_ids"]) for i, t in enumerate(
        generate.generate_healthy_joint(jp, 4, np.random.default_rng(0), 60, 15))]
    det = detector(hsmm)
    for d in det.detect_many(trials):
        assert set(det.flags(d.trace)) == set(CHANNELS)
        assert all(a.known_at == len(d.trace.z_star) - 1 for a in d.alarms)


@pytest.mark.parametrize("fit", ["joint_params.npz", "joint_params_clean.npz"])
def test_the_recipe_channel_is_silent_at_one_recipe(fit):
    """At k_recipe = 1 the recipe excess is undefined: the statistic must be exactly zero AND
    its thresholds +inf (marginal_log_trans floors rare transitions, so thresholds built from
    that artifact can land below zero). A one-fit test once hid the second half for a week."""
    root = REPO_ROOT / "dataset" / "processed" / "epic_tent"
    if not (root / fit).exists() or not (root / "sequences.json").exists():
        pytest.skip("needs the EPIC-Tent artifacts")
    hsmm = HSMM(load(root / fit), 200)
    assert hsmm.k_recipe == 1
    seqs = json.loads((root / "sequences.json").read_text())
    alphas = {c: 0.005 for c in CHANNELS}
    alphas["s_temporal"] = 1e-5
    det = HindsightHSMMDetector(hsmm, BadSets(hsmm, sorted(set(alphas.values()))),
                                OperatingPoint(alphas))
    for tr in det.traces([Trial(s["trial_id"], np.asarray(s["verb_ids"]), np.asarray(s["noun_ids"]))
                          for s in seqs]):
        assert not np.any(tr.surprise["s_recipe_transition"])
        assert not det.flags(tr)["s_recipe_transition"].any()
        assert np.all(np.isinf(det.thresholds(tr)["s_recipe_transition"]))
