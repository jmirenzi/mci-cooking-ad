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


# ---------------------------------------------------------------------------------------------
# LLM detectors, against a deterministic fake client

import hashlib  # noqa: E402

from cook_ad.detectors.llm_predictive import LLMPredictiveDetector  # noqa: E402
from cook_ad.detectors.llm_verdict import LLMVerdictDetector  # noqa: E402
from cook_ad.llm import prompts  # noqa: E402

LLM_VOCAB = {"verbs": {f"v{i}": i for i in range(V)}, "nouns": {f"n{i}": i for i in range(N)},
             "recipes": {}}


class _Lexicon:
    def verb(self, v):
        return f"v{int(v)}"

    def noun(self, n):
        return f"n{int(n)}"


class FakeClient:
    """Replies are a pure function of the messages, like a cache; records every request."""

    def __init__(self, kind):
        self.kind, self.requests, self.n_would_request = kind, [], 0

    def complete(self, messages):
        self.requests.append(json.dumps(messages))
        h = int(hashlib.sha256(json.dumps(messages).encode()).hexdigest(), 16)
        if self.kind == "predict":
            return f"DURATION: {1 + h % 5}\nNEXT: v{h % V} n{(h // 7) % N}"
        if h % 3:
            return "No Anomaly"
        return f"Substitution Anomaly. Correct move would have been v{h % V} n{h % N} for 3 seconds"

    def complete_many(self, message_lists):
        return [self.complete(m) for m in message_lists]


def _llm_case(det, client, trials):
    det.fit([])
    n_alarms = 0
    for trial in trials:
        client.requests = []
        fast = det.detect_trial(trial)
        fast_requests, client.requests = sorted(client.requests), []
        slow = AnomalyDetector.detect_trial(det, trial)
        assert sorted(client.requests) == fast_requests      # same requests: same cache keys
        assert _key(fast.alarms) == _key(slow.alarms)
        assert all(a.known_at >= a.tick for a in fast.alarms)
        n_alarms += len(fast.alarms)
    assert n_alarms > 0


def test_llm_predictive_fast_path_equals_step_loop():
    client = FakeClient("predict")
    _llm_case(LLMPredictiveDetector(client, LLM_VOCAB, _Lexicon()), client, _trials(n=6))


@pytest.mark.parametrize("protocol", ["incremental", "conversational"])
@pytest.mark.parametrize("unit", ["run", "tick"])
def test_llm_verdict_fast_path_equals_step_loop(protocol, unit):
    client = FakeClient("verdict")
    det = LLMVerdictDetector(client, LLM_VOCAB, _Lexicon(), protocol=protocol, unit=unit)
    _llm_case(det, client, _trials(n=3))


def test_llm_batch_verdict_refuses_to_step():
    det = LLMVerdictDetector(FakeClient("verdict"), LLM_VOCAB, _Lexicon(), protocol="batch")
    det.fit([])
    det.reset()
    assert not det.causal
    with pytest.raises(NotImplementedError):
        det.step(0, 0)


def test_llm_prompt_bytes_are_prompts_build_variant():
    det = LLMVerdictDetector(None, LLM_VOCAB, _Lexicon(), protocol="incremental", unit="tick",
                             variant="no-recipes")
    det.fit([])
    assert det.system_prompt == prompts.build_variant("no-recipes", LLM_VOCAB, None,
                                                      "incremental", "tick", exemplars=[])


BREAKFAST = "dataset/processed/breakfast"
GOLDEN_VERDICT = "runs/golden/streams_verdict.pkl"


@pytest.mark.skipif(not (os.path.exists(GOLDEN_VERDICT) and os.path.exists(f"{BREAKFAST}/llm_cache")),
                    reason="golden verdict streams or the Breakfast LLM cache not present")
def test_llm_verdict_step_loop_replays_from_cache():
    """The causal path, on real cached responses: zero uncached requests, golden alarms."""
    import pickle

    from cook_ad.detectors.narrate import Lexicon
    from cook_ad.hsmm import params as hparams
    from cook_ad.llm import client as llm_client

    class _Unpickler(pickle.Unpickler):   # the golden pickles name the pre-restructure Flag class
        def find_class(self, module, name):
            return super().find_class("cook_ad.eval.metric_g" if module.endswith("span_metrics")
                                      else module, name)

    golden = _Unpickler(open(GOLDEN_VERDICT, "rb")).load()
    vocab = json.load(open(f"{BREAKFAST}/vocab.json"))
    labels = json.load(open(f"{BREAKFAST}/labels.json"))
    lexicon = Lexicon(vocab, hparams.load(f"{BREAKFAST}/joint_params.npz"))
    client = llm_client.ChatClient(model="gemma3:27b", base_url="http://localhost:11435/v1",
                                   cache_dir=f"{BREAKFAST}/llm_cache/gemma3:27b", max_requests=0,
                                   concurrency=1)
    det = LLMVerdictDetector(client, vocab, lexicon, protocol="incremental", unit="tick",
                             variant="no-recipes", labels=labels)
    det.fit([])
    for i in range(2):
        traj = golden["pools"]["real"][i][0]
        d = AnomalyDetector.detect_trial(det, Trial("t", traj["verb_ids"], traj["noun_ids"]))
        expected = golden["streams"]["real/llm-no-recipes"]["healthy"][i]
        assert sorted((a.tick, a.channel) for a in d.alarms) == [(f.tick, f.channel) for f in expected]
    assert client.n_would_request == 0
