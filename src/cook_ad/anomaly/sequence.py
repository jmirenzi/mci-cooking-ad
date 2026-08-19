"""Retrospective segment-sequence detector (docs/anomaly.md, Phase 4).

Reads the Viterbi segment sequence z_1..z_J (narrate.segments_from_z) rather than the tick
stream -- independent of every per-tick channel in surprise.py, so it needs no interaction with
the existing detector and slots in as an additional scored arm alongside the HSMM and the LLM
(eval/element_metrics.py).

Two things this closes that the tick-level channels structurally cannot:

  1. **One verdict per event, not a scatter of tick flags.** A tick-level injection legitimately
     disturbs several ticks (entering the wrong step, the junction, exiting into what follows,
     plus a belief-state decay tail) -- this detector tests each segment junction/segment ONCE.
  2. **It names transposition.** `s_transition` (surprise.py) fires identically for omission,
     transposition and repetition -- it cannot tell them apart (docs/synthetic.md says so
     outright, and docs/llm.md measures the confusion diagonal at 0.00 for transposition). This
     detector runs three separate local edit tests, each naming its own error type:

Given decoded per-segment states z_1..z_J and a recipe's transition table `log_trans` (K, K), the
base score of the observed sequence is sum_j log A[z_j, z_{j+1}]. Three local edit tests, each
localised to one junction or segment index:

  transposition -- swap (z_j, z_{j+1}); gain = new local score - old local score, where "local"
                   is the (up to) three transitions that change: into z_j, the junction itself,
                   and out of z_{j+1}.
  omission      -- insert the best bridging state b between (z_j, z_{j+1}); reuses
                   `narrate.missing_step`, which already implements exactly this test.
  repetition    -- a segment's observed duration vs the fitted NB mean for its state (a ~2x
                   test); reuses `narrate.Lexicon.expected_duration`.

Thresholds are calibrated the same way every other channel is -- collect each test statistic's
distribution over HEALTHY trials and take the (1 - alpha) empirical quantile
(`quantile.sequence_thresholds`).

Non-causal by construction: the swap test at junction j needs segment j+1 to exist, so a verdict
at junction j cannot land until the segment AFTER the one it concerns has closed -- the same
~1-segment latency budget `s_dur_two` (a retrospective duration test) already accepts.
"""
from typing import NamedTuple

import numpy as np

from cook_ad.anomaly import narrate


class SequenceVerdict(NamedTuple):
    segment_index: int   # transposition/omission: the junction, indexed by the EARLIER segment.
                          # repetition: the segment itself.
    error_type: str       # "transposition" | "omission" | "repetition"
    gain: float            # transposition/omission: a log-probability gain, in nats.
                            # repetition: a duration ratio (observed / expected).
    detail: dict = None    # omission: {"bridge_state": b}; otherwise None.


def transposition_gain(log_trans, states, j):
    """Gain (nats) of swapping (states[j], states[j+1]) in place, over the LOCAL window of
    transitions that change: the junction itself, plus the incoming transition from states[j-1]
    (if j is not the first segment) and the outgoing transition to states[j+2] (if j+1 is not the
    last segment). Positive means the swapped order scores better than what was observed -- the
    observed order looks anomalous relative to what the recipe's transition table expects.
    """
    log_trans = np.asarray(log_trans)
    a, b = int(states[j]), int(states[j + 1])
    old = float(log_trans[a, b])
    new = float(log_trans[b, a])
    if j > 0:
        prev = int(states[j - 1])
        old += float(log_trans[prev, a])
        new += float(log_trans[prev, b])
    if j + 2 < len(states):
        nxt = int(states[j + 2])
        old += float(log_trans[b, nxt])
        new += float(log_trans[a, nxt])
    return new - old


def repetition_ratio(duration, state, lexicon):
    """Observed segment duration / the fitted NB mean for that state
    (`narrate.Lexicon.expected_duration`). A duplicated segment (`synthetic.error_injection.
    inject_repetition`) that the Viterbi decoder merges with its original produces one over-long
    run whose duration is roughly double the fitted mean (docs/synthetic.md) -- this is a coarse
    magnitude test on that ratio, not the full survival-function test `s_dur_two` already owns at
    tick resolution; it only needs to flag segments that are roughly integer multiples too long.
    """
    expected = lexicon.expected_duration(int(state))
    if expected <= 0:
        return 0.0
    return duration / expected


def score_segments(segments, log_trans, lexicon, thresholds,
                   min_bridge_gain=narrate.DEFAULT_MIN_BRIDGE_GAIN):
    """segments: narrate.segments_from_z(z_star) output, [(state, tick_start, tick_end), ...].
    log_trans: (K, K) log-probability transition table for the decoded recipe -- the same table
    narrate.py's order queries use (r_hat's own conditioned row set for the joint model, or the
    cascade's shared table).
    thresholds: quantile.SequenceThresholds (transposition / repetition), calibrated over healthy
    trials at this module's alpha.
    min_bridge_gain: gate for the omission test's bridging-gain -- narrate.missing_step's own
    parameter, passed through rather than duplicated.

    Returns a list of SequenceVerdict, at most one per (junction | segment). At a junction where
    both the swap test and the omission test would independently clear their thresholds,
    transposition wins and the omission test is skipped there: an ordering violation explains an
    odd local transition at that position, not the other way round -- the same priority argument
    element_metrics.CHANNEL_PRIORITY applies for the tick-level channels.
    """
    states = [s for s, _, _ in segments]
    n = len(states)
    verdicts = []

    for j in range(n - 1):
        gain = transposition_gain(log_trans, states, j)
        if gain > thresholds.transposition:
            verdicts.append(SequenceVerdict(j, "transposition", gain))
            continue
        a, c = states[j], states[j + 1]
        bridge, bridge_gain = narrate.missing_step(log_trans, a, c, min_bridge_gain)
        if bridge is not None:
            verdicts.append(SequenceVerdict(j, "omission", bridge_gain, {"bridge_state": bridge}))

    for idx, (state, start, end) in enumerate(segments):
        ratio = repetition_ratio(end - start, state, lexicon)
        if ratio > thresholds.repetition:
            verdicts.append(SequenceVerdict(idx, "repetition", ratio))

    return verdicts
