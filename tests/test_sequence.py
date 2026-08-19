import numpy as np
import pytest

from cook_ad.anomaly import quantile, sequence

K = 5  # 5 states in a strict chain: 0(A) -> 1(B) -> 2(C) -> 3(D) -> 4(E)


class _Lex:
    """Fixed expected duration for every state, so repetition_ratio is exactly duration/5."""
    def expected_duration(self, k):
        return 5.0


def _chain_log_trans(chain, flat, k=K):
    log_trans = np.full((k, k), flat)
    for i in range(k - 1):
        log_trans[i, i + 1] = chain
    return log_trans


# A comparatively flat matrix: swapping two ADJACENT chain segments always leaves a one-state gap
# at both flanks (the state that got displaced would trivially bridge it), which is a genuine,
# unavoidable ambiguity for any LOCAL two-hop test -- not a test-construction bug. Keeping the
# chain/flat contrast modest here keeps that incidental flank signal well under the omission
# gate, so this scenario cleanly fires transposition and nothing else.
TRANSPOSITION_LOG_TRANS = _chain_log_trans(chain=-0.916, flat=-1.897)
TRANSPOSITION_THRESHOLDS = quantile.SequenceThresholds(transposition=1.5, repetition=1.8)

# A sharper matrix for the omission-only scenario, so the genuine bridging signal (state C
# explains the gap between B and D) clears the default min_bridge_gain comfortably.
OMISSION_LOG_TRANS = _chain_log_trans(chain=-0.0513, flat=-5.298)


def test_transposition_swap_fires_transposition_and_nothing_else():
    """A -> B -> D -> C -> E: segments 2 and 3 (C, D) were swapped. The swap junction must be
    named 'transposition' -- the whole point of Phase 4, since s_transition (surprise.py) cannot
    tell this apart from omission or repetition -- and no other junction or segment should fire."""
    segments = [(0, 0, 5), (1, 5, 10), (3, 10, 15), (2, 15, 20), (4, 20, 25)]
    verdicts = sequence.score_segments(
        segments, TRANSPOSITION_LOG_TRANS, _Lex(), TRANSPOSITION_THRESHOLDS, min_bridge_gain=2.0
    )
    assert len(verdicts) == 1
    v = verdicts[0]
    assert (v.segment_index, v.error_type) == (2, "transposition")
    assert v.gain > TRANSPOSITION_THRESHOLDS.transposition


def test_transposition_gain_matches_direct_local_score_arithmetic():
    """Sanity check transposition_gain in isolation against the local three-transition window."""
    a, b, prev, nxt = 3, 2, 1, 4
    states = [0, prev, a, b, nxt]
    gain = sequence.transposition_gain(TRANSPOSITION_LOG_TRANS, states, 2)
    old = (TRANSPOSITION_LOG_TRANS[a, b] + TRANSPOSITION_LOG_TRANS[prev, a]
           + TRANSPOSITION_LOG_TRANS[b, nxt])
    new = (TRANSPOSITION_LOG_TRANS[b, a] + TRANSPOSITION_LOG_TRANS[prev, b]
           + TRANSPOSITION_LOG_TRANS[a, nxt])
    assert gain == pytest.approx(new - old)


def test_omission_drop_fires_omission_and_names_the_bridge_state():
    """A -> B -> D -> E: state C (index 2) was dropped entirely. The junction must be named
    'omission' and identify C as the bridging state -- reusing narrate.missing_step directly."""
    segments = [(0, 0, 5), (1, 5, 10), (3, 10, 15), (4, 15, 20)]
    thresholds = quantile.SequenceThresholds(transposition=1.5, repetition=1.8)
    verdicts = sequence.score_segments(
        segments, OMISSION_LOG_TRANS, _Lex(), thresholds, min_bridge_gain=2.0
    )
    assert len(verdicts) == 1
    v = verdicts[0]
    assert (v.segment_index, v.error_type) == (1, "omission")
    assert v.detail == {"bridge_state": 2}
    assert v.gain > 2.0


def test_clean_chain_fires_nothing():
    """A -> B -> C -> D -> E in the correct order, uniform durations: no test should fire."""
    segments = [(0, 0, 5), (1, 5, 10), (2, 10, 15), (3, 15, 20), (4, 20, 25)]
    thresholds = quantile.SequenceThresholds(transposition=1.5, repetition=1.8)
    verdicts = sequence.score_segments(
        segments, TRANSPOSITION_LOG_TRANS, _Lex(), thresholds, min_bridge_gain=2.0
    )
    assert verdicts == []


def test_repetition_duplicate_fires_repetition_and_nothing_else():
    """Correct order A -> B -> C -> D -> E, but segment C's duration (12 ticks) is roughly double
    the fitted mean (5) -- the signature a Viterbi-merged duplicate leaves behind
    (synthetic.error_injection.inject_repetition, docs/synthetic.md)."""
    segments = [(0, 0, 5), (1, 5, 10), (2, 10, 22), (3, 22, 27), (4, 27, 32)]
    thresholds = quantile.SequenceThresholds(transposition=1.5, repetition=1.8)
    verdicts = sequence.score_segments(
        segments, TRANSPOSITION_LOG_TRANS, _Lex(), thresholds, min_bridge_gain=2.0
    )
    assert len(verdicts) == 1
    v = verdicts[0]
    assert (v.segment_index, v.error_type) == (2, "repetition")
    assert v.gain == pytest.approx(12 / 5)


def test_repetition_ratio_below_threshold_does_not_fire():
    thresholds = quantile.SequenceThresholds(transposition=1.5, repetition=1.8)
    segments = [(0, 0, 5), (1, 5, 10), (2, 10, 15), (3, 15, 20), (4, 20, 25)]  # all ratio 1.0
    verdicts = sequence.score_segments(
        segments, TRANSPOSITION_LOG_TRANS, _Lex(), thresholds, min_bridge_gain=2.0
    )
    assert verdicts == []


def test_transposition_wins_priority_over_omission_at_the_same_junction():
    """If BOTH tests would independently clear their thresholds at the same junction, the swap
    test wins and the omission test is skipped there -- mirroring element_metrics.CHANNEL_
    PRIORITY's argument for the tick-level channels: an ordering violation explains an odd local
    transition, not the other way round."""
    segments = [(0, 0, 5), (1, 5, 10), (3, 10, 15), (2, 15, 20), (4, 20, 25)]
    verdicts = sequence.score_segments(
        segments, TRANSPOSITION_LOG_TRANS, _Lex(), TRANSPOSITION_THRESHOLDS, min_bridge_gain=2.0
    )
    types_at_junction_2 = [v.error_type for v in verdicts if v.segment_index == 2]
    assert types_at_junction_2 == ["transposition"]


# ---- repetition_ratio -----------------------------------------------------------------------

def test_repetition_ratio_is_duration_over_expected():
    assert sequence.repetition_ratio(10, state=0, lexicon=_Lex()) == pytest.approx(2.0)
    assert sequence.repetition_ratio(5, state=0, lexicon=_Lex()) == pytest.approx(1.0)


# ---- quantile.sequence_thresholds -------------------------------------------------------------

def test_sequence_thresholds_empirical_quantile():
    """Threshold should sit at roughly the (1 - alpha) quantile of the healthy-trial sample --
    checked by requiring close to alpha fraction of samples to exceed it."""
    rng = np.random.default_rng(0)
    gains = rng.normal(0, 1, size=2000)
    ratios = rng.normal(1, 0.1, size=2000)
    thresholds = quantile.sequence_thresholds(gains, ratios, alpha=0.05)
    exceed = np.mean(gains > thresholds.transposition)
    assert exceed <= 0.05
    assert exceed > 0.03  # not wildly conservative either


def test_sequence_thresholds_empty_sample_is_infinite():
    thresholds = quantile.sequence_thresholds([], [], alpha=0.05)
    assert thresholds.transposition == float("inf")
    assert thresholds.repetition == float("inf")
