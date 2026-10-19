"""The s_count channel (hsmm/visit_counts.py) in the hindsight and live HSMM detectors."""
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from cook_ad.core import OperatingPoint, Trial  # noqa: E402
from cook_ad.detectors.hsmm_hindsight import HindsightHSMMDetector  # noqa: E402
from cook_ad.detectors.hsmm_live import COUNT, LiveHSMMDetector  # noqa: E402
from cook_ad.detectors.thresholds import BadSets  # noqa: E402
from cook_ad.hsmm.model import HSMM  # noqa: E402
from cook_ad.hsmm.params import JointHSMMParams  # noqa: E402
from cook_ad.hsmm.visit_counts import VisitCounts  # noqa: E402

K, D_MAX, DUR = 4, 8, 3
ALPHAS = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 1e-3]


def _model():
    """One recipe, four subtasks done 0 1 2 3, each with its own verb and noun. Transitions are
    smooth enough that a repeat or a skip is ordinary to the transition channel."""
    trans = np.full((1, K, K), 20.0)
    for k in range(K - 1):
        trans[0, k, k + 1] = 60.0
    np.fill_diagonal(trans[0], 0.0)
    emit = np.full((K, K), 0.01) + 200.0 * np.eye(K)
    return HSMM(JointHSMMParams(
        init_counts=jnp.asarray([[60.0, 20.0, 20.0, 20.0]]), trans_counts=jnp.asarray(trans),
        verb_counts=jnp.asarray(emit), noun_counts=jnp.asarray(emit),
        dur_r=jnp.full((1, K), 30.0), dur_p=jnp.full((1, K), 30.0 / (30.0 + DUR - 1)),
        pi_counts=jnp.asarray([5.0])), D_MAX)


def _trial(states):
    v = np.repeat(states, DUR)
    return Trial("t", v, v.copy(), 0)


COUNTS = VisitCounts.fit([(0, [0, 1, 2, 3])] * 40)


def _hindsight(op):
    h = _model()
    return HindsightHSMMDetector(h, BadSets(h, ALPHAS), op, counts=COUNTS)


def test_hindsight_s_count_flags_a_repeat_and_a_skip_not_the_recipe():
    op = OperatingPoint({c: 1e-3 for c in ("s_emit", "s_verb", "s_noun", "s_temporal", "s_dur_two",
                                           "s_transition", "s_recipe_transition")} | {COUNT: 0.05})
    det = _hindsight(op)
    ok, rep, skip = det.detect_many([_trial([0, 1, 2, 3]), _trial([0, 1, 2, 1, 2, 3]),
                                     _trial([0, 1, 3])])
    assert not [a for a in ok.alarms if a.channel == COUNT]
    assert [a.tick for a in rep.alarms if a.channel == COUNT] == [3 * DUR, 4 * DUR]
    assert [a.tick for a in skip.alarms if a.channel == COUNT] == [2 * DUR]      # where 2 belonged


def test_without_counts_the_detector_is_unchanged():
    h = _model()
    det = HindsightHSMMDetector(h, BadSets(h, ALPHAS), OperatingPoint(0.05))
    assert COUNT not in det.channels and COUNT not in det.traces([_trial([0, 1, 2, 3])])[0].surprise


def _live(alpha=0.05):
    h = _model()
    op = OperatingPoint({c: 1e-3 for c in ("s_emit", "s_verb", "s_noun", "s_temporal", "s_dur_two",
                                           "s_transition", "s_recipe_transition")} | {COUNT: alpha}, 0.5)
    return LiveHSMMDetector(h, BadSets(h, ALPHAS), op, counts=COUNTS)


def test_live_s_count_is_causal_and_step_matches_trace():
    det = _live()
    for states, want in (([0, 1, 2, 3], []), ([0, 1, 2, 1, 2, 3], [3 * DUR, 4 * DUR]),
                         ([0, 1, 3], [2 * DUR])):
        tr = _trial(states)
        flagged = [a.tick for a in det.detect_trial(tr).alarms if a.channel == COUNT]
        det.reset(tr.recipe)
        streamed = []
        for v, n in zip(tr.verbs, tr.nouns):
            streamed += [a.tick for a in det.step(int(v), int(n)) if a.channel == COUNT]
        assert sorted(set(flagged)) == sorted(set(streamed)) == want, states


def test_live_counts_are_expected_visits():
    det = _live()
    tr = _trial([0, 1, 2, 3])
    det.reset(tr.recipe)
    for v, n in zip(tr.verbs, tr.nouns):
        det.step(int(v), int(n))
    np.testing.assert_allclose(det._count.c[0], np.ones(K), atol=0.05)
