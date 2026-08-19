"""Step-level detection metrics -- the layer that makes the HSMM and the LLM comparable.

`eval/metrics.py` scores ticks. That is the right unit for the HSMM, whose seven channels emit a
value at every tick, but the LLM baseline reads a trial as a list of steps and answers once per
step, so it has no tick-level opinion to score. This module defines the common unit -- one
run-length-encoded step (llm/textify.py) -- converts BOTH detectors into per-step verdicts, and
scores them with the same code.

`eval/metrics.py`, `run_evaluation.py` and every figure they produce are untouched: the tick-level
numbers in docs/eval.md remain exactly what they were, and this sits alongside them.

Three things this measures that the tick-level layer structurally cannot:

  1. `type_confusion` -- the LLM names the anomaly type directly. The tick-level attribution
     matrix only proxies that through which channel fired.
  2. `correction_accuracy` -- both detectors propose what should have happened instead, so the
     proposal can be checked against the pre-injection step.
  3. `parse_failure_rate` -- an LLM arm only. A model that cannot hold the response format is a
     finding about that model.
"""
import numpy as np

from cook_ad.anomaly import surprise
from cook_ad.eval import metrics

ALL_CHANNELS = surprise.CHANNELS

# One step's tolerance, replacing the tick layer's latency_tol=5. Five ticks does not translate to
# a unit whose median width is ~11 ticks; one step is the smallest tolerance that still admits the
# structurally retrospective channels (an abandonment is only visible once the step closes).
DEFAULT_STEP_TOL = 1

# Duration counts as correct within 50%, floored at +/-2s so short steps are not held to an
# impossible standard. Durations here are integer seconds over steps whose median width is 11s.
DURATION_TOL_FRAC = 0.5
DURATION_TOL_MIN = 2

# Which anomaly type each surprise channel is evidence for. This is the HSMM's answer to the
# question the LLM answers in words, and it is deliberately many-to-one: omission, transposition
# and repetition ALL land on s_transition, because the cascade genuinely cannot separate them
# (docs/synthetic.md says so outright). That collapse shows up as an off-diagonal block in
# type_confusion, which is the honest result -- not something to tie-break away.
CHANNEL_TO_TYPE = {
    "s_transition": "omission",
    "s_recipe_transition": "omission",
    "s_dur_two": "abandonment",      # refined to 'repetition' when attributed 'stuck', below
    "s_temporal": "repetition",
    "s_noun": "substitution",
    "s_verb": "substitution",
    # s_emit is deliberately ABSENT. Measured on 20 real trials x 5 injections, it fires in-window
    # on 1.00 / 0.40 / 0.80 / 0.90 / 0.80 of substitution / abandonment / omission / transposition
    # / repetition detections -- i.e. on nearly everything. That is by construction: it is the
    # joint marginal -log P(v,n|.), so it carries no type information its two component channels
    # do not already carry. Mapping it to a type would manufacture a specific claim out of
    # deliberately non-specific evidence. A step where ONLY s_emit fires is therefore a detection
    # with NO predicted type, which lands in type_confusion's "none" column -- the honest reading
    # of "something is off here but the model cannot say what."
}

# Priority when several channels fire on one step, ordered by how specific each channel's evidence
# is to a single error type:
#   s_transition first -- an ordering violation EXPLAINS an odd duration at that position, not the
#     other way round, so it outranks the duration channels when both fire (measured: transposition
#     fires s_transition on 0.70 of detections vs s_dur_two on 0.30).
#   s_dur_two next -- it is signed, so it names a direction (left_early -> abandonment,
#     stuck -> repetition) rather than merely "unusual".
#   the emission channels last, since a boundary the causal filter has not caught up to yet fires
#     them as a side effect of every structural error, not only of item substitutions.
CHANNEL_PRIORITY = (
    "s_transition", "s_dur_two", "s_temporal", "s_noun", "s_verb", "s_recipe_transition", "s_emit",
)


class StepVerdict:
    """One detector's answer about one step."""

    __slots__ = ("step_index", "is_anomaly", "error_type", "correction", "parse_ok", "raw")

    def __init__(self, step_index, is_anomaly, error_type=None,
                 correction=None, parse_ok=True, raw=""):
        self.step_index = int(step_index)
        self.is_anomaly = bool(is_anomaly)
        self.error_type = error_type
        self.correction = correction
        self.parse_ok = bool(parse_ok)
        self.raw = raw

    def __repr__(self):
        return (f"StepVerdict(step={self.step_index}, anomaly={self.is_anomaly}, "
                f"type={self.error_type!r})")


def from_llm_verdicts(verdicts):
    """llm.detect.Verdict -> StepVerdict. The LLM emits exactly one verdict per step."""
    return [
        StepVerdict(v.step_index, v.is_anomaly, v.error_type, v.correction, v.parse_ok, v.raw)
        for v in verdicts
    ]


def from_sequence_verdicts(seq_verdicts, segments, steps):
    """anomaly.sequence.SequenceVerdict (Viterbi-segment-indexed) -> one StepVerdict per
    llm/textify.Step -- the common unit the HSMM and LLM arms are already scored in
    (llm/textify.py's run-length-encoded steps), so the sequence detector can be scored as a
    THIRD arm through evaluate_steps unchanged, landing in the same tables and figures.

    A textify step is flagged if its tick range overlaps the segment(s) a verdict implicates: the
    two segments either side of the junction for transposition/omission (segments[j],
    segments[j+1]), or the single named segment for repetition.
    """
    step_ticks = [(s.tick_start, s.tick_end) for s in steps]

    def _overlapping(t0, t1):
        return [i for i, (a, b) in enumerate(step_ticks) if a < t1 and b > t0]

    flagged = {}
    for v in seq_verdicts:
        if v.error_type == "repetition":
            _, start, end = segments[v.segment_index]
        else:
            _, start, _ = segments[v.segment_index]
            _, _, end = segments[v.segment_index + 1]
        for si in _overlapping(start, end):
            flagged.setdefault(si, v.error_type)

    return [
        StepVerdict(s.index, s.index in flagged, flagged.get(s.index))
        for s in steps
    ]


def _predicted_type(fired_channels, trace, tick_slice):
    """Map the channels that fired on a step to a single predicted anomaly type."""
    for ch in CHANNEL_PRIORITY:
        if ch not in fired_channels or ch not in CHANNEL_TO_TYPE:
            continue
        if ch == "s_dur_two" and trace is not None:
            # The two-sided duration channel is signed: temporal.py attributes it 'left_early'
            # (short) or 'stuck' (long). Those are evidence for different error types, so use the
            # direction rather than throwing it away.
            attribution = set(np.asarray(trace.temporal_attribution)[tick_slice].tolist())
            if "stuck" in attribution and "left_early" not in attribution:
                return "repetition"
            return "abandonment"
        return CHANNEL_TO_TYPE[ch]
    return None


def _hsmm_correction(trace, steps, step, tick_slice, lexicon):
    """The HSMM's 'I expected X instead', from trace.expected_verb/expected_noun at the step's
    first flagged tick, plus that state's fitted mean duration.

    Caveat, stated because narrate.py is emphatic about it: expected_verb/expected_noun are
    z_star's HINDSIGHT argmax, and narrate.py deliberately uses the causal
    surprise.conditional_expected instead when it renders a query for a user, since at a segment
    boundary the two can disagree and produce a self-contradictory sentence. The causal version
    needs pi_all, which SurpriseTrace does not carry; here the value is only being SCORED, never
    shown to anyone, so the hindsight argmax is the available and adequate quantity. Read
    correction_accuracy for the HSMM arm with that in mind.
    """
    if trace is None or lexicon is None:
        return None
    ticks = np.arange(len(np.asarray(trace.z_star)))[tick_slice]
    if ticks.size == 0:
        return None
    t = int(ticks[0])
    verb = lexicon.verb(int(np.asarray(trace.expected_verb)[t]))
    noun = lexicon.noun(int(np.asarray(trace.expected_noun)[t]))
    duration = int(round(lexicon.expected_duration(int(np.asarray(trace.z_star)[t]))))
    return (verb, noun, duration)


def step_verdicts_from_flags(flags, steps, trace=None, lexicon=None, channels=ALL_CHANNELS):
    """HSMM adapter: per-tick channel flags -> one StepVerdict per step.

    A step is `is_anomaly` if ANY tick inside it is flagged. No run-length requirement is applied
    -- see metrics.score_trial for why persistence filters by error type rather than by noise.
    Tick-level false alarms are controlled at the threshold (alpha), not here.
    """
    union = None
    for ch in channels:
        union = flags[ch].copy() if union is None else (union | flags[ch])

    verdicts = []
    for step in steps:
        sl = slice(step.tick_start, step.tick_end)
        fired = [ch for ch in channels if bool(np.any(flags[ch][sl]))]
        any_flag = bool(np.any(union[sl]))
        etype = _predicted_type(fired, trace, sl) if any_flag else None
        correction = _hsmm_correction(trace, steps, step, sl, lexicon) if any_flag else None
        verdicts.append(StepVerdict(step.index, any_flag, etype, correction,
                                    parse_ok=True, raw=",".join(fired)))
    return verdicts


def score_trial_steps(verdicts, gt_steps, tol_steps=DEFAULT_STEP_TOL, debris_steps=frozenset()):
    """(detected, latency_steps, flagged_out_of_window, predicted_type).

    `gt_steps` is the step-index list from textify.gt_steps_for_window. Detection is any
    `is_anomaly` verdict in [min(gt), max(gt) + tol_steps]; latency counts steps from min(gt).
    Any `is_anomaly` verdict outside that range is a false positive.

    `debris_steps` (textify.injection_touched_steps) are excluded from the out-of-window
    false-positive check: they are steps the injection itself created or reshaped but which are
    not the ground-truth anomaly, so a detector that flags them is neither wrong nor right about
    anything the injector actually tests. Debris is a third bucket -- excluded from scoring
    entirely, counted as neither a hit nor a false alarm.
    """
    if not gt_steps:
        out_of_window = any(v.is_anomaly for v in verdicts if v.step_index not in debris_steps)
        return False, None, out_of_window, None
    lo, hi = min(gt_steps), max(gt_steps) + tol_steps

    detected, latency, predicted = False, None, None
    for v in verdicts:
        if v.is_anomaly and lo <= v.step_index <= hi:
            detected = True
            latency = v.step_index - lo
            predicted = v.error_type
            break
    out_of_window = any(
        v.is_anomaly and not (lo <= v.step_index <= hi) and v.step_index not in debris_steps
        for v in verdicts
    )
    return detected, latency, out_of_window, predicted


def _any_flag(verdicts):
    return any(v.is_anomaly for v in verdicts)


def _correction_matches(predicted, truth):
    """(verb_noun_ok, duration_ok). Duration is judged separately because for abandonment the
    duration IS the correction -- the verb and noun are unchanged by that injector, so a combined
    score would read as a success for naming a step the model never had to identify."""
    if predicted is None or truth is None:
        return None, None
    verb_noun_ok = (predicted[0], predicted[1]) == (truth[0], truth[1])
    tol = max(DURATION_TOL_MIN, DURATION_TOL_FRAC * truth[2])
    duration_ok = abs(predicted[2] - truth[2]) <= tol
    return verb_noun_ok, duration_ok


def evaluate_steps(healthy_verdicts, degraded_by_type, tol_steps=DEFAULT_STEP_TOL,
                   error_types=None, artifact_steps=None):
    """healthy_verdicts: [[StepVerdict, ...], ...] for control trials with no injection.
    degraded_by_type: {error_type: [(verdicts, gt_steps, gt_correction), ...]}, where gt_correction
    is the pre-injection (verb, noun, duration) from textify.step_covering_tick, or None.
    artifact_steps: optional {error_type: [debris_step_set, ...]}, index-aligned with each error
    type's trial list in degraded_by_type -- textify.injection_touched_steps per trial. When
    absent (the default), no steps are treated as debris and this behaves exactly as before this
    parameter existed.
    Returns the same report shape as metrics.evaluate -- per_type / attribution / healthy -- so
    the two layers print through the same code, plus type_confusion, correction_accuracy,
    step_level, trial_located and parse_failure_rate.
    """
    types = list(error_types or degraded_by_type.keys())
    fp_healthy = sum(1 for v in healthy_verdicts if _any_flag(v))
    n_healthy = len(healthy_verdicts)

    # trial_located: one verdict per trial, scored on WHERE the flag landed (docs/eval.md 6).
    # The positive range is the whole injection-touched extent -- ground-truth window UNION
    # debris -- because every step in it is one the injection actually moved.
    loc_tp = loc_fn = loc_stray = 0
    loc_by_type = {}

    per_type, type_confusion, correction_accuracy = {}, {}, {}
    tp_steps = fp_steps = fn_steps = 0
    n_parsed = n_verdicts = 0
    # Total steps scored, degraded + healthy. Needed to state a CHANCE precision: precision has to
    # be read against the base rate of anomalous steps, or a detector that simply flags everything
    # looks respectable. tp+fp+fn cannot stand in for it -- that denominator varies per detector,
    # so it would give the two arms different chance baselines for the same data.
    n_steps_total = 0

    for etype in types:
        trials = degraded_by_type.get(etype, [])
        debris_per_trial = (artifact_steps or {}).get(etype, [])
        tp = fp_out = 0
        latencies = []
        confusion = dict.fromkeys([*types, "none"], 0)
        corr_total = corr_verb_noun = corr_duration = 0
        et_tp = et_fn = et_stray = 0

        for i, (verdicts, gt_steps, gt_correction) in enumerate(trials):
            debris = debris_per_trial[i] if i < len(debris_per_trial) else frozenset()
            n_verdicts += len(verdicts)
            n_parsed += sum(1 for v in verdicts if v.parse_ok)

            # trial_located. A flag inside the injection-touched range is a hit; a flag outside
            # it is a stray, charged as a false positive INDEPENDENTLY of the hit, so finding the
            # anomaly does not buy absolution for also firing elsewhere.
            in_range = set(gt_steps) | set(debris)
            hit = any(v.is_anomaly and v.step_index in in_range for v in verdicts)
            stray = any(v.is_anomaly and v.step_index not in in_range for v in verdicts)
            if in_range:
                et_tp += int(hit)
                et_fn += int(not hit)
            et_stray += int(stray)

            detected, latency, out_of_window, predicted = score_trial_steps(
                verdicts, gt_steps, tol_steps, debris_steps=debris
            )
            if detected:
                tp += 1
                latencies.append(latency)
                confusion[predicted if predicted in confusion else "none"] += 1
                hit = next(v for v in verdicts
                           if v.is_anomaly and min(gt_steps) <= v.step_index <= max(gt_steps) + tol_steps)
                verb_noun_ok, duration_ok = _correction_matches(hit.correction, gt_correction)
                if verb_noun_ok is not None:
                    corr_total += 1
                    corr_verb_noun += int(verb_noun_ok)
                    corr_duration += int(duration_ok)
            elif out_of_window:
                fp_out += 1

            # Step-level pooling: one ground-truth WINDOW is one test, not one per gt step -- a
            # transposition's swapped pair is a single anomaly event (2.01 gt steps/trial
            # measured), and flagging either half is a complete detection of it, not half of one.
            # Debris steps (created/reshaped by the injection but not the anomaly itself) are
            # excluded entirely rather than counted as either TP-eligible or FP-eligible.
            gt = set(gt_steps)
            scoreable = [v for v in verdicts if v.step_index not in gt and v.step_index not in debris]
            n_steps_total += len(scoreable) + (1 if gt_steps else 0)
            if gt_steps:
                tp_steps += int(detected)
                fn_steps += int(not detected)
            for v in scoreable:
                if v.is_anomaly:
                    fp_steps += 1

        n = len(trials)
        fp = fp_out + fp_healthy
        per_type[etype] = {
            "n": n,
            "recall": tp / n if n else 0.0,
            "precision": tp / (tp + fp) if (tp + fp) else 0.0,
            # Same argument as metrics.evaluate: fp_healthy is the SAME constant pooled into every
            # type's denominator, so it partly manufactures the "constant floor across types"
            # pattern. Reporting both separates the type-specific false alarms from the shared
            # ones. Preserved verbatim at step level (docs/eval.md 2).
            "precision_excl_healthy": tp / (tp + fp_out) if (tp + fp_out) else 0.0,
            "mean_latency": float(np.mean(latencies)) if latencies else float("nan"),
            "tp": tp,
            "fp_out_of_window": fp_out,
        }
        type_confusion[etype] = {k: (v / tp if tp else 0.0) for k, v in confusion.items()}
        correction_accuracy[etype] = {
            "n_scored": corr_total,
            "verb_noun_accuracy": corr_verb_noun / corr_total if corr_total else float("nan"),
            "duration_accuracy": corr_duration / corr_total if corr_total else float("nan"),
        }
        loc_tp += et_tp
        loc_fn += et_fn
        loc_stray += et_stray
        loc_by_type[etype] = {
            "n": n,
            "recall": et_tp / (et_tp + et_fn) if (et_tp + et_fn) else 0.0,
            "stray_rate": et_stray / n if n else 0.0,
            "tp": et_tp, "fn": et_fn, "stray": et_stray,
        }

    for verdicts in healthy_verdicts:
        n_steps_total += len(verdicts)
        n_verdicts += len(verdicts)
        n_parsed += sum(1 for v in verdicts if v.parse_ok)
        fp_steps += sum(1 for v in verdicts if v.is_anomaly)

    return {
        "unit": "step",
        "per_type": per_type,
        # Kept under the key metrics.evaluate uses so run_evaluation.py's printer and
        # eval.plotting work unchanged on this report; the rows are anomaly types, not channels.
        "attribution": type_confusion,
        "type_confusion": type_confusion,
        "correction_accuracy": correction_accuracy,
        "healthy": {
            # A healthy trial is a false positive if ANY of its steps was flagged. Both arms are
            # held to the same bar; there is no persistence correction on either side.
            "n": n_healthy, "false_positive_trials": fp_healthy,
            "false_positive_rate": fp_healthy / n_healthy if n_healthy else 0.0,
        },
        "step_level": {
            "tp": tp_steps, "fp": fp_steps, "fn": fn_steps,
            "precision": tp_steps / (tp_steps + fp_steps) if (tp_steps + fp_steps) else 0.0,
            "recall": tp_steps / (tp_steps + fn_steps) if (tp_steps + fn_steps) else 0.0,
            "n_steps": n_steps_total,
            # Precision a detector would get by flagging steps at random: the base rate of
            # ground-truth anomalous steps. Identical for every arm scored on the same pool.
            "chance_precision": ((tp_steps + fn_steps) / n_steps_total) if n_steps_total else 0.0,
        },
        # One verdict per trial, scored on WHERE the flag landed. `fp` pools two distinct events
        # -- a stray on a degraded trial and a flag on a healthy one -- so they are also reported
        # separately: `stray_rate` is per degraded trial, `healthy.false_positive_rate` per
        # healthy trial. FP/(FP+TN) is deliberately NOT reported, because a degraded trial can
        # contribute both a TP and an FP and so is not a member of a well-defined negative class.
        "trial_located": {
            "tp": loc_tp, "fn": loc_fn, "stray": loc_stray, "healthy_fp": fp_healthy,
            "precision": (loc_tp / (loc_tp + loc_stray + fp_healthy)
                          if (loc_tp + loc_stray + fp_healthy) else 0.0),
            "recall": loc_tp / (loc_tp + loc_fn) if (loc_tp + loc_fn) else 0.0,
            "stray_rate": loc_stray / (loc_tp + loc_fn) if (loc_tp + loc_fn) else 0.0,
            "healthy_fpr": fp_healthy / n_healthy if n_healthy else 0.0,
            "per_type": loc_by_type,
        },
        "parse_failure_rate": 1.0 - (n_parsed / n_verdicts) if n_verdicts else 0.0,
        "channels": [*types, "none"],
        "tol_steps": tol_steps,
    }
