"""The AnomalyDetector contract (docs/restructure.md §9.4): fast paths equal the step loop,
alarms are never known before they happen, batching changes nothing, and the non-causal
detectors refuse to step."""
import json
import math
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from cook_ad.core import AnomalyDetector, OperatingPoint, Trial
from cook_ad.detectors.hsmm_hindsight import HindsightHSMMDetector
from cook_ad.detectors.hsmm_lag import LagHSMMDetector
from cook_ad.detectors.hsmm_live import LiveHSMMDetector
from cook_ad.detectors.narrate import narrate
from cook_ad.detectors.thresholds import BadSets
from cook_ad.hsmm import params
from cook_ad.hsmm.model import HSMM

K_R, K, V, N, D_MAX = 3, 5, 4, 6, 8
ALPHAS = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 1e-3]
OP = OperatingPoint(alpha=0.05, kappa=0.3)


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


def _trials(seed=1, n=4, t_range=(12, 30), recipe=None):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        t = int(rng.integers(*t_range))
        # runs of repeated tokens, so segments and switches actually happen
        verbs = np.repeat(rng.integers(0, V, t), rng.integers(1, 4, t))[:t]
        nouns = np.repeat(rng.integers(0, N, t), rng.integers(1, 4, t))[:t]
        out.append(Trial(f"t{i}", verbs, nouns, recipe))
    return out


@pytest.fixture(scope="module")
def setup():
    hsmm = HSMM(_model(), D_MAX)
    return hsmm, BadSets(hsmm, ALPHAS)


def _key(alarms):
    return sorted((a.tick, a.channel, a.known_at) for a in alarms)


def _scores(alarms):
    return {(a.tick, a.channel): a.score for a in alarms}


@pytest.mark.parametrize("lag", [0, 2, 5])
@pytest.mark.parametrize("recipe", [None, 1])
def test_fast_path_equals_step_loop(setup, lag, recipe):
    hsmm, bad = setup
    det = LiveHSMMDetector(hsmm, bad, OP) if lag == 0 else LagHSMMDetector(hsmm, bad, OP, lag)
    n_alarms = 0
    for trial in _trials(recipe=recipe):
        fast = det.detect_trial(trial)
        slow = AnomalyDetector.detect_trial(det, trial)
        assert _key(fast.alarms) == _key(slow.alarms)
        fs, ss = _scores(fast.alarms), _scores(slow.alarms)
        for k in fs:
            assert fs[k] == pytest.approx(ss[k], rel=1e-9, abs=1e-12)
        assert all(a.known_at >= a.tick for a in fast.alarms)
        assert all(a.known_at <= a.tick + lag for a in fast.alarms)
        n_alarms += len(fast.alarms)
    assert n_alarms > 0  # the comparison is not vacuous


def test_step_alarms_arrive_when_known(setup):
    """Every alarm returned by step() at tick t has known_at == t."""
    hsmm, bad = setup
    det = LagHSMMDetector(hsmm, bad, OP, 3)
    trial = _trials(n=1)[0]
    det.reset(trial.recipe)
    for t, (v, n) in enumerate(zip(trial.verbs, trial.nouns)):
        assert all(a.known_at == t for a in det.step(int(v), int(n)))
    assert all(a.known_at == len(trial.verbs) - 1 for a in det.finish())


def test_hindsight_detect_many_equals_detect_trial(setup):
    hsmm, bad = setup
    det = HindsightHSMMDetector(hsmm, bad, OP, chunk_size=3)
    trials = _trials(n=7)
    many = det.detect_many(trials)
    for trial, d in zip(trials, many):
        one = det.detect_trial(trial)
        assert _key(one.alarms) == _key(d.alarms)
        assert one.trace.r_hat == d.trace.r_hat
        assert one.trace.segments == d.trace.segments
        fs, ss = _scores(one.alarms), _scores(d.alarms)
        for k in fs:
            assert fs[k] == pytest.approx(ss[k], rel=1e-9, abs=1e-12)


def test_non_causal_detectors_refuse_to_step(setup):
    hsmm, bad = setup
    for det in (HindsightHSMMDetector(hsmm, bad, OP), LagHSMMDetector(hsmm, bad, OP, math.inf)):
        assert not det.causal
        det.reset()
        with pytest.raises(NotImplementedError):
            det.step(0, 0)


def test_lag_inf_equals_long_lag(setup):
    hsmm, bad = setup
    trial = _trials(n=1)[0]
    inf = LagHSMMDetector(hsmm, bad, OP, math.inf).detect_trial(trial)
    long = LagHSMMDetector(hsmm, bad, OP, len(trial.verbs)).detect_trial(trial)
    assert [(a.tick, a.channel) for a in inf.alarms] == [(a.tick, a.channel) for a in long.alarms]


def test_narrate_refuses_a_live_detection(setup):
    hsmm, bad = setup
    live = LiveHSMMDetector(hsmm, bad, OP).detect_trial(_trials(n=1)[0])
    with pytest.raises(NotImplementedError, match="live narration is planned"):
        narrate(live, LiveHSMMDetector(hsmm, bad, OP), {"verbs": {}, "nouns": {}})


EPIC_TENT = "dataset/processed/epic_tent"


@pytest.mark.skipif(not os.path.exists(f"{EPIC_TENT}/joint_params_clean.npz"),
                    reason="EPIC-Tent model not present")
def test_narrate_matches_the_pre_restructure_narration():
    """Same queries, word for word, as anomaly.narrate.narrate_joint on real sessions."""
    from cook_ad.anomaly import narrate as old_narrate, surprise
    from cook_ad.data.config import load_config
    from cook_ad.hsmm import joint_params

    d_max = load_config("configs/epic_tent.yaml")["duration"]["d_max_ticks"]
    vocab = json.load(open(f"{EPIC_TENT}/vocab.json"))
    seqs = json.load(open(f"{EPIC_TENT}/sequences.json"))[:3]
    hsmm = HSMM.load(f"{EPIC_TENT}/joint_params_clean.npz", d_max)
    alpha = 0.05
    det = HindsightHSMMDetector(hsmm, BadSets(hsmm, [alpha]), OperatingPoint(alpha))
    jp = joint_params.load_params(f"{EPIC_TENT}/joint_params_clean.npz")
    n_queries = 0
    for s in seqs:
        v, n = np.asarray(s["verb_ids"]), np.asarray(s["noun_ids"])
        trace, jlp, r_hat, ltm, _ = surprise.compute_trace_joint(jp, v, n, d_max)
        flags = surprise.flag_joint(trace, jlp, r_hat, ltm, alpha=alpha)
        pi_all = surprise.compute_pi_all_joint(jlp, r_hat, v, n, d_max)
        old = old_narrate.narrate_joint(trace, flags, vocab, jp, r_hat, v, n, jlp, ltm, pi_all,
                                        alpha=alpha)
        new = narrate(det.detect_trial(Trial(s["trial_id"], v, n)), det, vocab)
        assert [q[:7] for q in old] == [tuple(q) for q in new]
        n_queries += len(new)
    assert n_queries > 0
