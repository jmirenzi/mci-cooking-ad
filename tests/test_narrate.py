"""detectors/narrate.py: one question per segment and kind, severity against the same threshold
that gated the flag, and coherent "I expected ..." replacements."""
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import logsumexp

jax.config.update("jax_enable_x64", True)

from _fixtures import stack_recipes, weak_limit_params  # noqa: E402
from cook_ad.core import OperatingPoint, Trial  # noqa: E402
from cook_ad.detectors import narrate as N_  # noqa: E402
from cook_ad.detectors.hsmm_hindsight import HindsightHSMMDetector  # noqa: E402
from cook_ad.detectors.thresholds import BadSets  # noqa: E402
from cook_ad.hsmm.model import HSMM  # noqa: E402
from cook_ad.synthetic import error_injection, generate  # noqa: E402

K, V, N, D_MAX = 4, 5, 5, 30
ALPHA = 5e-3


def _peaked():
    """Sharp verb/noun tables and moderate durations, so injections are unambiguous."""
    p = weak_limit_params(jax.random.PRNGKey(0), K, V, N, D_MAX)
    return p._replace(
        verb_counts=jnp.full((K, V), 0.2).at[jnp.arange(K), jnp.arange(K) % V].set(200.0),
        noun_counts=jnp.full((K, N), 0.2).at[jnp.arange(K), jnp.arange(K) % N].set(200.0),
        dur_r=jnp.full((K,), 6.0), dur_p=jnp.full((K,), 0.5))


VOCAB = {"verbs": {f"v{i}": i for i in range(V)}, "nouns": {f"n{i}": i for i in range(N)},
         "recipes": {}}


def _detector():
    hsmm = HSMM(stack_recipes([_peaked()]), D_MAX)
    return HindsightHSMMDetector(hsmm, BadSets(hsmm, [ALPHA]), OperatingPoint(ALPHA))


def _long_trajectory(seed):
    for offset in range(20):
        traj = generate.sample_trajectory(_peaked(), np.random.default_rng(seed + offset),
                                          max_ticks=150, d_max=D_MAX)
        if len(traj["segments"]) >= error_injection.MIN_SEGMENTS:
            return traj
    raise RuntimeError("could not find a trajectory with enough segments")


def test_missing_step_finds_a_bridge_only_when_it_beats_the_direct_jump():
    log_A = np.full((4, 4), np.log(0.01))
    log_A[0, 1] = np.log(0.9)
    log_A[1, 2] = np.log(0.9)
    log_A[np.arange(4), np.arange(4)] = -np.inf
    bridge, gain = N_.missing_step(log_A, a=0, c=2, min_gain=2.0)
    assert bridge == 1
    assert gain == pytest.approx(float(log_A[0, 1] + log_A[1, 2] - log_A[0, 2]))
    log_A[0, 2] = np.log(0.5)   # the direct jump is now plausible
    bridge, gain = N_.missing_step(log_A, a=0, c=2, min_gain=2.0)
    assert bridge is None and gain < 2.0


def test_one_stall_question_per_segment_not_per_tick():
    det = _detector()
    traj = _long_trajectory(seed=1)
    verbs, nouns, segs = list(traj["verb_ids"]), list(traj["noun_ids"]), traj["segments"]
    at = segs[0][1] + segs[1][1]       # stretch the second segment far past its usual length
    verbs = verbs[:at] + [verbs[at - 1]] * 40 + verbs[at:]
    nouns = nouns[:at] + [nouns[at - 1]] * 40 + nouns[at:]
    d = det.detect_trial(Trial("t", np.asarray(verbs), np.asarray(nouns)))
    spans = N_.segment_spans(d.trace)
    i, (_, start, end) = next((i, s) for i, s in enumerate(spans) if s[1] <= segs[0][1] < s[2])
    assert sum(1 for a in d.alarms if a.channel == "s_temporal" and start <= a.tick < end) > 5
    queries = N_.narrate(d, det, VOCAB)
    assert len([q for q in queries if q.channel == "s_temporal" and q.segment_index == i]) == 1


def test_severity_is_measured_against_the_dilution_corrected_threshold():
    """A question's severity ratio is s / threshold with the SAME per-tick threshold that gated
    its flag (z*'s quantile plus -log P(z* | o_<t)), not the raw per-state quantile."""
    det = _detector()
    for seed in range(10):
        traj = _long_trajectory(seed=200 + seed)
        sub = error_injection.inject("substitution", traj, np.random.default_rng(seed),
                                     det.hsmm.params)
        d = det.detect_trial(Trial("t", sub["verb_ids"], sub["noun_ids"]))
        nq = [q for q in N_.narrate(d, det, VOCAB) if q.channel == "s_noun"]
        if len(nq) == 1:
            break
    else:
        pytest.fail("substitution never produced a single s_noun question")
    q, tr = nq[0], d.trace
    th = det.thresholds(tr)["s_noun"][q.tick]
    raw = det.bad.hindsight(tr.r_hat, ALPHA).noun[tr.z_star[q.tick]]
    assert tr.pi_at_zstar[q.tick] < 1.0
    assert th == pytest.approx(raw - np.log(tr.pi_at_zstar[q.tick]))
    assert q.ratio == pytest.approx(tr.surprise["s_noun"][q.tick] / th)


# ---- "I expected ..." -----------------------------------------------------------------------

LOG_EMIT = np.log(np.array([[0.98, 0.01, 0.01], [0.05, 0.90, 0.05], [0.05, 0.05, 0.90]]))
PI = np.log(np.array([0.85, 0.10, 0.05]))   # mostly the idle-like state 0


def test_expected_given_follows_the_held_token_not_the_raw_mass():
    naive = int(np.argmax(np.asarray(logsumexp(PI[:, None] + LOG_EMIT, axis=0))))
    assert naive == 0                       # the raw mass points at idle's own noun
    assert N_.expected_given(PI, LOG_EMIT[:, 1], LOG_EMIT) == 1   # coherent with verb 1


def test_expected_pair_is_one_states_coherent_pair():
    assert N_.expected_pair(PI, LOG_EMIT, LOG_EMIT) == (0, 0)


def _item_query(s_verb, verb_flagged):
    lex = N_.Lexicon(VOCAB, _peaked())
    tr = types.SimpleNamespace(surprise={"s_noun": np.array([10.0]), "s_verb": np.array([s_verb])},
                               attribution=np.array(["item"], dtype=object),
                               verbs=np.array([1]), nouns=np.array([2]))
    ctx = N_._Ctx(tr, {"s_noun": np.array([True]), "s_verb": np.array([verb_flagged])},
                  {"s_noun": np.array([1.0]), "s_verb": np.array([1.0])}, PI[None, :], lex,
                  LOG_EMIT, LOG_EMIT, None, N_.MIN_BRIDGE_GAIN)
    (q,) = N_._item_query(ctx, 0, 0, 1)
    return q, lex


def test_a_wrong_object_question_holds_the_verb_when_only_the_noun_was_flagged():
    q, lex = _item_query(s_verb=0.0, verb_flagged=False)
    assert q.text == (f"Wait --, that's {lex.phrase(1, 2)} -- based on what I'd seen up to then, "
                      f"I expected {lex.phrase(1, 1)}.")


def test_a_wrong_object_question_replaces_both_when_both_were_flagged():
    q, lex = _item_query(s_verb=8.0, verb_flagged=True)
    v, n = N_.expected_pair(PI, LOG_EMIT, LOG_EMIT)
    assert v != 1
    assert q.text == (f"Wait --, that's {lex.phrase(1, 2)} -- based on what I'd seen up to then, "
                      f"I expected {lex.phrase(v, n)}.")
