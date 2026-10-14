"""LiveHSMMDetector / LagHSMMDetector (detectors/hsmm_live.py, hsmm_lag.py): exactness of the
filter and smoother, and rule R2 -- with a point-mass belief on the hindsight answer, every live
flag reduces to the hindsight one."""
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import logsumexp

jax.config.update("jax_enable_x64", True)

from cook_ad.core import OperatingPoint, Trial  # noqa: E402
from cook_ad.detectors import hsmm_live  # noqa: E402
from cook_ad.detectors.hsmm_hindsight import EMISSION_FLAG_ATOL, HindsightHSMMDetector  # noqa: E402
from cook_ad.detectors.hsmm_lag import LagHSMMDetector  # noqa: E402
from cook_ad.detectors.hsmm_live import EVENTS, LiveHSMMDetector  # noqa: E402
from cook_ad.detectors.thresholds import (  # noqa: E402
    BadSets,
    categorical_quantile_threshold,
    joint_quantile_threshold,
)
from cook_ad.hsmm import kernels  # noqa: E402
from cook_ad.hsmm.model import HSMM  # noqa: E402
from cook_ad.hsmm.params import JointHSMMParams  # noqa: E402
from cook_ad.hsmm.train import messages  # noqa: E402

K_R, K, V, N, D_MAX = 2, 4, 3, 5, 6
ALPHAS = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 1e-3]
OP = OperatingPoint(0.05, 0.5)


def _model(seed=0, k_r=K_R):
    rng = np.random.default_rng(seed)
    trans = rng.gamma(1.0, 3.0, size=(k_r, K, K)) + 1.0
    for r in range(k_r):
        np.fill_diagonal(trans[r], 0.0)
    return HSMM(JointHSMMParams(
        init_counts=jnp.asarray(rng.gamma(1.0, 3.0, size=(k_r, K)) + 1.0),
        trans_counts=jnp.asarray(trans),
        verb_counts=jnp.asarray(rng.gamma(0.5, 4.0, size=(K, V)) + 1.0),
        noun_counts=jnp.asarray(rng.gamma(0.5, 4.0, size=(K, N)) + 1.0),
        dur_r=jnp.asarray(rng.uniform(1.0, 6.0, size=(k_r, K))),
        dur_p=jnp.asarray(rng.uniform(0.3, 0.8, size=(k_r, K))),
        pi_counts=jnp.asarray(rng.gamma(2.0, 2.0, size=(k_r,)) + 1.0),
    ), D_MAX)


def _trials(seed=1, n=5, t_range=(8, 15)):
    rng = np.random.default_rng(seed)
    return [Trial(str(i), rng.integers(0, V, t), rng.integers(0, N, t))
            for i, t in enumerate(rng.integers(*t_range, size=n))]


def _one_recipe(hsmm, r=0):
    t = hsmm.tables
    return t.log_init[r], t.log_A[r], t.log_pmf[r], t.log_surv[r]


def test_predictive_belief_matches_the_one_recipe_occupancy():
    """At K_R = 1 the filter's pi_tilde is exactly kernels.predictive_occupancy."""
    hsmm = _model(k_r=1)
    for tr in _trials():
        fr = hsmm.filter(tr.verbs, tr.nouns)
        n = fr.n
        ref = kernels.predictive_occupancy(fr.loglik[:n], *_one_recipe(hsmm)[:2],
                                           *_one_recipe(hsmm)[2:], jnp.ones(n, bool), D_MAX)
        np.testing.assert_allclose(np.exp(fr.occupancy), np.exp(np.asarray(ref)), atol=1e-10)


def test_full_lag_reproduces_the_smoother():
    """With L covering the trial the posterior's subtask marginal is the smoothed gamma, and the
    total switch probability sums to the expected transition count."""
    hsmm = _model(k_r=1)
    for tr in _trials():
        fr = hsmm.filter(tr.verbs, tr.nouns)
        n = fr.n
        beta = hsmm.smooth(fr, math.inf)
        post = (fr.log_b + beta)[:n]
        post = post - logsumexp(post, axis=(1, 2, 3), keepdims=True)
        gamma_live = np.exp(logsumexp(post, axis=(1, 3)))
        li, la, lp, ls = _one_recipe(hsmm)
        xi_trans, _, _, gamma, _ = messages.combine_sufficient_stats(
            fr.loglik[:n], jnp.ones(n, bool), li, la, lp, ls, D_MAX)
        np.testing.assert_allclose(gamma_live, np.asarray(gamma), atol=1e-9)

        det = LiveHSMMDetector(hsmm, BadSets(hsmm, [0.5]), OperatingPoint(0.5, 0.5))
        det.reset()
        every_cell = hsmm_live.Bins(*(jnp.zeros_like(b) for b in det.bins))
        _, _, trans, _ = hsmm_live.event_table(det._dynamics(), every_cell, jnp.asarray([0.5]),
                                               fr.log_b, beta, fr.loglik, fr.mask)
        assert float(np.sum(np.asarray(trans)[:n, 0])) == pytest.approx(
            float(np.asarray(xi_trans).sum()), abs=1e-9)


def test_event_probabilities_are_probabilities_and_nested_in_alpha():
    hsmm = _model()
    bad = BadSets(hsmm, ALPHAS)
    for tr in _trials():
        for det in (LiveHSMMDetector(hsmm, bad, OP), LagHSMMDetector(hsmm, bad, OP, 3)):
            trace = det.trace(tr)
            for c in EVENTS:
                x = trace.events[c]
                assert np.all(x >= -1e-12) and np.all(x <= 1 + 1e-9)
                assert np.all(np.diff(x, axis=1) >= -1e-12)   # alphas ascending -> mass grows
            for p in trace.p.values():
                assert np.all(p > 0) and np.all(p <= 1 + 1e-9)


def test_emission_pvalue_collapses_to_the_hindsight_flag():
    """R2 for the emission channels: under a point-mass belief on z the p-value flag (p <= alpha)
    equals the hindsight rule s > tau_z for every observation and alpha (the dilution term is 0
    at a point mass)."""
    t = _model().tables
    ev, en = t.log_emit_v, t.log_emit_n
    vv, nn = np.meshgrid(np.arange(V), np.arange(N), indexing="ij")
    vv, nn = jnp.asarray(vv.ravel()), jnp.asarray(nn.ravel())
    exceeds = lambda s, tau: np.asarray(s) > tau + EMISSION_FLAG_ATOL  # noqa: E731
    for z in range(K):
        pt = jnp.tile(jnp.where(jnp.arange(K) == z, 0.0, -jnp.inf), (vv.shape[0], 1))
        p_emit, p_verb, p_noun = hsmm_live.forecast_pvalues(pt, ev, en, vv, nn)
        for a in ALPHAS:
            np.testing.assert_array_equal(
                np.asarray(p_emit) <= a,
                exceeds(-(np.asarray(ev)[z, vv] + np.asarray(en)[z, nn]),
                        joint_quantile_threshold(ev, en, a)[z]))
            np.testing.assert_array_equal(np.asarray(p_verb) <= a, exceeds(
                -np.asarray(ev)[z, vv], categorical_quantile_threshold(ev, a)[z]))
            np.testing.assert_array_equal(np.asarray(p_noun) <= a, exceeds(
                -np.asarray(en)[z, nn], categorical_quantile_threshold(en, a)[z]))


def test_event_bad_sets_collapse_to_the_hindsight_flags():
    """R2 for the hidden-event channels: a point-mass belief at the hindsight answer (r_hat, z*,
    elapsed) puts all its event mass on one cell, so the live flag at alpha_g is exactly "that
    cell's bin <= g". Checked against the Hindsight detector's own flags."""
    hsmm = _model(seed=3)
    bad = BadSets(hsmm, ALPHAS)
    hind = HindsightHSMMDetector(hsmm, bad, OperatingPoint(0.05), chunk_size=4)
    stall, early = np.asarray(bad.stall_bin), np.asarray(bad.early_bin)
    trans, recipe = np.asarray(bad.trans_bin), np.asarray(bad.recipe_bin)
    for tr in hind.traces(_trials(seed=4, n=8, t_range=(12, 25))):
        r, z = tr.r_hat, tr.z_star
        starts = np.flatnonzero(np.r_[True, z[1:] != z[:-1]])
        ends = np.r_[starts[1:] - 1, len(z) - 1]
        elapsed = np.concatenate([np.arange(1, b - a + 2) for a, b in zip(starts, ends)])
        for g, a in enumerate(bad.alphas):
            f = hind.flags(tr, OperatingPoint(a))
            np.testing.assert_array_equal(stall[r, z, elapsed - 1] <= g, f["s_temporal"])
            for m in range(1, len(starts)):
                t0, k, i = starts[m], z[starts[m] - 1], z[starts[m]]
                assert (trans[r, k, i] <= g) == bool(f["s_transition"][t0])
                assert (recipe[r, k, i] <= g) == bool(f["s_recipe_transition"][t0])
                # hindsight's early flag sits on the ended run's LAST tick; live's on the next
                # run's first tick (when it becomes knowable)
                d_prev = ends[m - 1] - starts[m - 1] + 1
                short = bool(f["s_dur_two"][ends[m - 1]]) and \
                    tr.temporal_attribution[ends[m - 1]] == "left_early"
                assert (early[r, k, d_prev - 1] <= g) == short


def test_the_split_forecast_sums_to_pi_tilde_and_matches_the_mixture_at_a_point_mass():
    """stay + switch = pi_tilde; at t = 0 all mass is "switch"; and when one hypothesis is
    absent the split p-value equals the plain mixture one -- so R2 holds unchanged."""
    hsmm = _model()
    t = hsmm.tables
    det = LiveHSMMDetector(hsmm, BadSets(hsmm, ALPHAS), OP)
    for tr in _trials():
        fr = hsmm.filter(tr.verbs, tr.nouns)
        np.testing.assert_allclose(np.exp(np.logaddexp(fr.log_stay, fr.log_switch)),
                                   np.exp(fr.pi_tilde), atol=1e-12)
        mix = hsmm_live.forecast_pvalues(fr.pi_tilde[:1], t.log_emit_v, t.log_emit_n,
                                         jnp.asarray(tr.verbs[:1]), jnp.asarray(tr.nouns[:1]))
        assert det.trace(tr).p["s_emit"][0] == pytest.approx(float(mix[0][0]))
    vv, nn = jnp.arange(V).repeat(N), jnp.tile(jnp.arange(N), V)
    pt = jnp.tile(jnp.where(jnp.arange(K) == 1, 0.0, -jnp.inf), (vv.shape[0], 1))
    gone = jnp.full_like(pt, -jnp.inf)
    mixed = hsmm_live.forecast_pvalues(pt, t.log_emit_v, t.log_emit_n, vv, nn)
    for split in (hsmm_live.split_pvalues(pt, gone, t.log_emit_v, t.log_emit_n, vv, nn),
                  hsmm_live.split_pvalues(gone, pt, t.log_emit_v, t.log_emit_n, vv, nn)):
        for a, b in zip(split, mixed):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-12)


def test_full_beta_equals_a_lag_covering_the_trial():
    hsmm = _model()
    for tr in _trials():
        fr = hsmm.filter(tr.verbs, tr.nouns)
        np.testing.assert_allclose(np.asarray(hsmm.smooth(fr, math.inf))[: fr.n],
                                   np.asarray(hsmm.smooth(fr, fr.n))[: fr.n], atol=1e-10)


def test_a_known_recipe_equals_a_one_hot_prior_and_its_own_likelihood():
    """Pinning recipe r and a one-hot prior on r give the same channels, and the filter's
    likelihood is r's own forward pass."""
    hsmm = _model()
    bad = BadSets(hsmm, ALPHAS)
    for tr in _trials(n=3):
        for r in range(K_R):
            prior = np.where(np.arange(K_R) == r, 0.0, -np.inf)
            for det in (LiveHSMMDetector(hsmm, bad, OP), LagHSMMDetector(hsmm, bad, OP, 2)):
                pinned = det.trace(Trial("p", tr.verbs, tr.nouns, r))
                soft = det.trace(Trial("s", tr.verbs, tr.nouns, prior))
                np.testing.assert_allclose(pinned.p["s_emit"], soft.p["s_emit"], atol=1e-12)
                for c in EVENTS:
                    np.testing.assert_allclose(pinned.events[c], soft.events[c], atol=1e-12)
            fr = hsmm.filter(tr.verbs, tr.nouns)
            ln, _ = kernels.forward_pass(fr.loglik[: fr.n], *_one_recipe(hsmm, r)[:2],
                                         *_one_recipe(hsmm, r)[2:], jnp.ones(fr.n, bool), D_MAX)
            assert pinned.log_norm.sum() == pytest.approx(float(jnp.sum(ln)), abs=1e-9)
