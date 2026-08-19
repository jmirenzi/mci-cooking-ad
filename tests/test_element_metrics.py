import numpy as np
import pytest

from cook_ad.anomaly import surprise
from cook_ad.eval import element_metrics as em
from cook_ad.llm import textify
from cook_ad.llm.detect import Verdict

STEP_TICKS = 10
N_STEPS = 5


class _Lex:
    def verb(self, v):
        return f"v{int(v)}"

    def noun(self, n):
        return f"n{int(n)}"

    def expected_duration(self, k):
        return 10.0


def _steps(n=N_STEPS):
    verbs, nouns = [], []
    for i in range(n):
        verbs += [i] * STEP_TICKS
        nouns += [i] * STEP_TICKS
    return textify.steps_from_ids(verbs, nouns, _Lex())


def _llm(anomaly_at=(), etype="substitution", correction=None, unparsed=(), n=N_STEPS):
    return em.from_llm_verdicts([
        Verdict(i, i in anomaly_at, etype if i in anomaly_at else None,
                correction if i in anomaly_at else None, "", i not in unparsed)
        for i in range(n)
    ])


def _flags(ticks, channel="s_noun", t=N_STEPS * STEP_TICKS):
    d = {ch: np.zeros(t, dtype=bool) for ch in surprise.CHANNELS}
    for i in ticks:
        d[channel][i] = True
    return d


# ---- score_trial_steps ------------------------------------------------------------------------

def test_detection_latency_counts_steps_from_the_first_ground_truth_step():
    detected, latency, out_of_window, predicted = em.score_trial_steps(
        _llm(anomaly_at=(3,)), gt_steps=[2], tol_steps=1
    )
    assert (detected, latency, out_of_window) == (True, 1, False)
    assert predicted == "substitution"


def test_flag_beyond_tolerance_is_a_miss_and_a_false_positive():
    detected, _, out_of_window, _ = em.score_trial_steps(
        _llm(anomaly_at=(4,)), gt_steps=[0], tol_steps=1
    )
    assert detected is False
    assert out_of_window is True


def test_a_single_in_window_flag_is_a_detection():
    """No run-length requirement: one flagged step inside the window is a detection."""
    verdicts = [em.StepVerdict(i, is_anomaly=(i == 2)) for i in range(N_STEPS)]
    detected, latency, out_of_window, _ = em.score_trial_steps(verdicts, [2])
    assert (detected, latency, out_of_window) == (True, 0, False)


def test_a_single_out_of_window_flag_is_a_false_positive():
    verdicts = [em.StepVerdict(i, is_anomaly=(i == 4)) for i in range(N_STEPS)]
    _, _, out_of_window, _ = em.score_trial_steps(verdicts, [0])
    assert out_of_window is True


# ---- HSMM adapter -----------------------------------------------------------------------------

def test_any_flagged_tick_in_a_step_flags_that_step():
    """A step is anomalous if any tick inside it is flagged -- no run-length requirement."""
    steps = _steps()
    v = em.step_verdicts_from_flags(_flags([5, 25, 45]), steps)
    assert [x.is_anomaly for x in v] == [True, False, True, False, True]


def test_a_run_spanning_two_steps_flags_both():
    steps = _steps()
    v = em.step_verdicts_from_flags(_flags(range(20, 32)), steps)
    assert [x.is_anomaly for x in v] == [False, False, True, True, False]


def test_predicted_type_prefers_specific_channels_over_s_emit():
    """s_emit fires on nearly every error type, so it must never name a type on its own."""
    steps = _steps()
    only_emit = _flags(range(20, 32), channel="s_emit")
    v = em.step_verdicts_from_flags(only_emit, steps)
    assert v[2].is_anomaly is True
    assert v[2].error_type is None            # detected, but no type claimed

    both = _flags(range(20, 32), channel="s_emit")
    for t in range(20, 32):
        both["s_transition"][t] = True
    v = em.step_verdicts_from_flags(both, steps)
    assert v[2].error_type == "omission"


def test_duration_channel_direction_selects_the_type():
    """s_dur_two is signed: temporal.py attributes it 'left_early' or 'stuck', which are evidence
    for different error types."""
    steps = _steps()
    T = N_STEPS * STEP_TICKS

    class _Trace:
        def __init__(self, attribution):
            self.temporal_attribution = np.array(["none"] * T, dtype=object)
            self.temporal_attribution[25] = attribution
            self.z_star = np.zeros(T, dtype=int)
            self.expected_verb = np.zeros(T, dtype=int)
            self.expected_noun = np.zeros(T, dtype=int)

    f = _flags(range(20, 32), channel="s_dur_two")
    assert em.step_verdicts_from_flags(f, steps, _Trace("left_early"), _Lex())[2].error_type == "abandonment"
    assert em.step_verdicts_from_flags(f, steps, _Trace("stuck"), _Lex())[2].error_type == "repetition"


# ---- evaluate_steps ---------------------------------------------------------------------------

def test_recall_precision_and_confusion_arithmetic():
    report = em.evaluate_steps(
        healthy_verdicts=[_llm()],
        degraded_by_type={"substitution": [(_llm(anomaly_at=(2,)), [2], ("v2", "n2", 10))]},
    )
    m = report["per_type"]["substitution"]
    assert (m["recall"], m["precision"], m["mean_latency"]) == (1.0, 1.0, 0.0)
    assert report["type_confusion"]["substitution"]["substitution"] == 1.0
    assert report["healthy"]["false_positive_rate"] == 0.0
    assert report["unit"] == "step"


def test_healthy_flag_is_a_false_positive():
    report = em.evaluate_steps(
        healthy_verdicts=[_llm(anomaly_at=(1,))],
        degraded_by_type={"substitution": [(_llm(anomaly_at=(2,)), [2], None)]},
    )
    assert report["healthy"]["false_positive_rate"] == 1.0
    assert report["per_type"]["substitution"]["precision"] == 0.5
    # ...and excluding the shared healthy pool isolates the type-specific component
    assert report["per_type"]["substitution"]["precision_excl_healthy"] == 1.0


def test_wrong_type_still_counts_as_a_detection_but_lands_off_diagonal():
    report = em.evaluate_steps(
        healthy_verdicts=[],
        degraded_by_type={
            "omission": [(_llm(anomaly_at=(2,), etype="substitution"), [2], None)],
            "substitution": [],
        },
        error_types=["omission", "substitution"],
    )
    assert report["per_type"]["omission"]["recall"] == 1.0
    assert report["type_confusion"]["omission"]["substitution"] == 1.0
    assert report["type_confusion"]["omission"]["omission"] == 0.0


def test_detection_with_no_predicted_type_lands_in_the_none_column():
    report = em.evaluate_steps(
        healthy_verdicts=[],
        degraded_by_type={"omission": [(
            [em.StepVerdict(i, is_anomaly=(i == 2), error_type=None) for i in range(N_STEPS)],
            [2], None,
        )]},
    )
    assert report["type_confusion"]["omission"]["none"] == 1.0


def test_correction_accuracy_scores_tokens_and_duration_separately():
    """Separately because for abandonment the verb and noun are unchanged by the injector, so a
    combined score would credit naming a step the detector never had to identify."""
    truth = ("pour", "milk", 20)
    report = em.evaluate_steps(
        healthy_verdicts=[],
        degraded_by_type={"abandonment": [
            (_llm(anomaly_at=(2,), correction=("pour", "milk", 19)), [2], truth),   # both ok
            (_llm(anomaly_at=(2,), correction=("pour", "milk", 2)), [2], truth),    # duration off
            (_llm(anomaly_at=(2,), correction=("stir", "bowl", 20)), [2], truth),   # tokens off
        ]},
    )
    ca = report["correction_accuracy"]["abandonment"]
    assert ca["n_scored"] == 3
    assert ca["verb_noun_accuracy"] == pytest.approx(2 / 3)
    assert ca["duration_accuracy"] == pytest.approx(2 / 3)


def test_correction_accuracy_is_nan_when_nothing_was_scoreable():
    report = em.evaluate_steps(
        healthy_verdicts=[],
        degraded_by_type={"repetition": [(_llm(anomaly_at=(2,), correction=None), [2], None)]},
    )
    ca = report["correction_accuracy"]["repetition"]
    assert ca["n_scored"] == 0
    assert np.isnan(ca["verb_noun_accuracy"])


def test_parse_failure_rate_counts_every_verdict_from_both_pools():
    report = em.evaluate_steps(
        healthy_verdicts=[_llm(unparsed=(0,))],
        degraded_by_type={"substitution": [(_llm(anomaly_at=(2,), unparsed=(4,)), [2], None)]},
    )
    assert report["parse_failure_rate"] == pytest.approx(2 / (2 * N_STEPS))


def test_step_level_pooling_counts_every_step_as_one_test():
    report = em.evaluate_steps(
        healthy_verdicts=[_llm()],
        degraded_by_type={"substitution": [(_llm(anomaly_at=(2, 4)), [2], None)]},
    )
    sl = report["step_level"]
    assert sl["tp"] == 1        # step 2 is ground truth and was flagged
    assert sl["fn"] == 0
    assert sl["fp"] == 1        # step 4 was flagged and is not ground truth
    assert sl["precision"] == pytest.approx(0.5)
    assert sl["recall"] == 1.0


def test_report_shape_matches_the_tick_level_report():
    """evaluate_steps deliberately mirrors metrics.evaluate's keys so run_evaluation.py's printer
    and eval.plotting work on it unchanged."""
    report = em.evaluate_steps(
        healthy_verdicts=[_llm()],
        degraded_by_type={"substitution": [(_llm(anomaly_at=(2,)), [2], None)]},
    )
    assert {"per_type", "attribution", "healthy", "channels"} <= set(report)
    assert set(report["attribution"]) == set(report["per_type"])
    for row in report["attribution"].values():
        assert set(row) == set(report["channels"])


def test_chance_precision_is_a_property_of_the_data_not_the_detector():
    """Two detectors scored on the same pool must get the SAME chance baseline. Deriving it from
    tp+fp+fn would not: that denominator moves with how much each detector flags."""
    gt = [2]
    trigger_happy = _llm(anomaly_at=(0, 1, 2, 3, 4))
    conservative = _llm(anomaly_at=(2,))
    reports = [
        em.evaluate_steps(healthy_verdicts=[_llm()],
                          degraded_by_type={"substitution": [(v, gt, None)]})
        for v in (trigger_happy, conservative)
    ]
    a, b = (r["step_level"] for r in reports)
    assert a["n_steps"] == b["n_steps"] == 2 * N_STEPS
    assert a["chance_precision"] == b["chance_precision"] == pytest.approx(1 / (2 * N_STEPS))
    # ...while the achieved precisions differ sharply, which is the point of having the baseline
    assert a["precision"] < b["precision"]


# ---- from_sequence_verdicts (Phase 4 integration) ----------------------------------------------

def test_from_sequence_verdicts_flags_the_steps_the_junction_spans():
    from cook_ad.anomaly.sequence import SequenceVerdict
    steps = _steps()  # 5 steps, each 10 ticks: [0,10) [10,20) [20,30) [30,40) [40,50)
    segments = [(0, 0, 10), (1, 10, 20), (2, 20, 30), (3, 30, 40), (4, 40, 50)]
    seq_verdicts = [SequenceVerdict(1, "transposition", 5.0)]  # junction between segments 1, 2
    v = em.from_sequence_verdicts(seq_verdicts, segments, steps)
    flagged = {x.step_index for x in v if x.is_anomaly}
    assert flagged == {1, 2}
    assert next(x for x in v if x.step_index == 1).error_type == "transposition"


def test_from_sequence_verdicts_repetition_flags_only_its_own_segment():
    from cook_ad.anomaly.sequence import SequenceVerdict
    steps = _steps()
    segments = [(0, 0, 10), (1, 10, 20), (2, 20, 30), (3, 30, 40), (4, 40, 50)]
    seq_verdicts = [SequenceVerdict(2, "repetition", 2.4)]
    v = em.from_sequence_verdicts(seq_verdicts, segments, steps)
    flagged = {x.step_index for x in v if x.is_anomaly}
    assert flagged == {2}


def test_from_sequence_verdicts_no_verdicts_flags_nothing():
    steps = _steps()
    segments = [(0, 0, 10), (1, 10, 20), (2, 20, 30), (3, 30, 40), (4, 40, 50)]
    v = em.from_sequence_verdicts([], segments, steps)
    assert not any(x.is_anomaly for x in v)


# ---- relabel_with_sequence ----------------------------------------------------------------------

def test_relabel_never_changes_whether_a_step_is_flagged():
    """The containment property the whole design rests on: relabelling may only move `error_type`.
    If it could move `is_anomaly`, precision/recall/FPR would shift and the sequence tests would be
    raising alarms through the back door."""
    tick = [em.StepVerdict(i, is_anomaly=(i in (1, 3)),
                           error_type=("omission" if i in (1, 3) else None))
            for i in range(N_STEPS)]
    seq = [em.StepVerdict(i, is_anomaly=True, error_type="transposition") for i in range(N_STEPS)]
    out = em.relabel_with_sequence(tick, seq)
    assert [v.is_anomaly for v in out] == [v.is_anomaly for v in tick]


def test_relabel_corrects_transposition_on_flagged_steps_only():
    # mirrors step_verdicts_from_flags: an unflagged step carries no type
    tick = [em.StepVerdict(i, is_anomaly=(i == 2), error_type=("omission" if i == 2 else None))
            for i in range(N_STEPS)]
    seq = [em.StepVerdict(i, is_anomaly=(i in (2, 4)),
                          error_type=("transposition" if i in (2, 4) else None))
           for i in range(N_STEPS)]
    out = em.relabel_with_sequence(tick, seq, scope="step")
    assert out[2].error_type == "transposition"   # tick flagged it, sequence renamed it
    assert out[4].error_type is None              # sequence flagged it but the tick arm did not
    assert out[4].is_anomaly is False             # ...and it still raises no alarm


def test_relabel_only_overwrites_the_collapsed_channels_label():
    """`s_transition` resolves to "omission" via CHANNEL_TO_TYPE, so that is the one label a
    mis-read transposition wears. substitution/abandonment come from channels a swap cannot be
    confused with and are already accurate, so they must survive untouched."""
    seq = [em.StepVerdict(i, is_anomaly=True, error_type="transposition") for i in range(N_STEPS)]
    for src, expected in (("omission", "transposition"),      # overridable
                          ("substitution", "substitution"),   # not
                          ("abandonment", "abandonment"),     # not
                          ("repetition", "repetition")):      # not
        tick = [em.StepVerdict(i, is_anomaly=(i == 2), error_type=(src if i == 2 else None))
                for i in range(N_STEPS)]
        assert em.relabel_with_sequence(tick, seq)[2].error_type == expected


def test_relabel_trial_scope_matches_a_swap_found_on_a_different_step():
    """Step-exact matching misses the common case: one detector flags one half of the swapped
    pair, the other names the opposite half. Trial scope is sound because the harness injects
    exactly one anomaly per trial."""
    tick = [em.StepVerdict(i, is_anomaly=(i == 2), error_type=("omission" if i == 2 else None))
            for i in range(N_STEPS)]
    seq = [em.StepVerdict(i, is_anomaly=(i == 4),
                          error_type=("transposition" if i == 4 else None))
           for i in range(N_STEPS)]

    assert em.relabel_with_sequence(tick, seq, scope="trial")[2].error_type == "transposition"
    assert em.relabel_with_sequence(tick, seq, scope="step")[2].error_type == "omission"
    # containment holds under both scopes
    for sc in ("trial", "step"):
        out = em.relabel_with_sequence(tick, seq, scope=sc)
        assert [v.is_anomaly for v in out] == [v.is_anomaly for v in tick]


def test_relabel_trial_scope_is_inert_when_the_swap_test_never_fires():
    tick = [em.StepVerdict(i, is_anomaly=(i == 2), error_type=("omission" if i == 2 else None))
            for i in range(N_STEPS)]
    seq = [em.StepVerdict(i, is_anomaly=False) for i in range(N_STEPS)]
    assert em.relabel_with_sequence(tick, seq, scope="trial")[2].error_type == "omission"


def test_relabel_rejects_an_unknown_scope():
    with pytest.raises(ValueError, match="scope"):
        em.relabel_with_sequence([], [], scope="nonsense")


def test_relabel_ignores_sequence_types_outside_the_allowed_set():
    """Only transposition by default: the tick channels already name omission and repetition
    better, so deferring on those would swap better type evidence for worse."""
    tick = [em.StepVerdict(i, is_anomaly=(i == 2), error_type=("omission" if i == 2 else None))
            for i in range(N_STEPS)]
    for seq_type in ("omission", "repetition"):
        seq = [em.StepVerdict(i, is_anomaly=(i == 2), error_type=seq_type) for i in range(N_STEPS)]
        # only "transposition" is in `types`, so an omission/repetition verdict changes nothing
        assert em.relabel_with_sequence(tick, seq)[2].error_type == "omission"


def test_relabel_keeps_the_tick_type_when_the_sequence_test_is_silent():
    tick = [em.StepVerdict(i, is_anomaly=(i == 2), error_type="omission") for i in range(N_STEPS)]
    seq = [em.StepVerdict(i, is_anomaly=False) for i in range(N_STEPS)]
    assert em.relabel_with_sequence(tick, seq)[2].error_type == "omission"


# ---- debris exclusion (Phase 1 A) --------------------------------------------------------------

def test_debris_step_is_excluded_from_out_of_window_false_positive():
    """A step the injection created (debris, not ground truth) must not count as an out-of-
    window false positive even when flagged."""
    verdicts = [em.StepVerdict(i, is_anomaly=(i == 4)) for i in range(N_STEPS)]
    _, _, out_of_window, _ = em.score_trial_steps(verdicts, gt_steps=[0], debris_steps={4})
    assert out_of_window is False
    # without the debris exclusion the same flag would be a false positive
    _, _, out_of_window_uncorrected, _ = em.score_trial_steps(verdicts, gt_steps=[0])
    assert out_of_window_uncorrected is True


def test_debris_step_excluded_from_step_level_pooling():
    report = em.evaluate_steps(
        healthy_verdicts=[_llm()],
        degraded_by_type={"substitution": [(_llm(anomaly_at=(2, 4)), [2], None)]},
        artifact_steps={"substitution": [{4}]},
    )
    sl = report["step_level"]
    assert sl["tp"] == 1
    assert sl["fp"] == 0        # step 4 was debris, not scored
    assert sl["n_steps"] == 2 * N_STEPS - 1  # one fewer step scored (the debris one)


def test_artifact_steps_absent_behaves_as_before():
    kwargs = dict(healthy_verdicts=[_llm()],
                  degraded_by_type={"substitution": [(_llm(anomaly_at=(2, 4)), [2], None)]})
    without = em.evaluate_steps(**kwargs)
    with_none = em.evaluate_steps(artifact_steps=None, **kwargs)
    assert without["step_level"] == with_none["step_level"]


# ---- one-window-one-event (Phase 1 B) ------------------------------------------------------

def test_multi_step_window_scores_as_one_event_not_one_per_step():
    """A transposition-shaped ground truth spans two steps; flagging just one of them is a
    complete detection of the ONE event, and must not also book a false negative for its
    unflagged sibling."""
    verdicts = [em.StepVerdict(i, is_anomaly=(i == 2)) for i in range(N_STEPS)]
    report = em.evaluate_steps(
        healthy_verdicts=[],
        degraded_by_type={"transposition": [(verdicts, [2, 3], None)]},
    )
    sl = report["step_level"]
    assert sl["tp"] == 1
    assert sl["fn"] == 0        # NOT 1 -- step 3 (the unflagged sibling) is not a separate test
    assert report["per_type"]["transposition"]["recall"] == 1.0


def test_multi_step_window_with_neither_step_flagged_is_one_false_negative():
    verdicts = [em.StepVerdict(i, is_anomaly=False) for i in range(N_STEPS)]
    report = em.evaluate_steps(
        healthy_verdicts=[],
        degraded_by_type={"transposition": [(verdicts, [2, 3], None)]},
    )
    sl = report["step_level"]
    assert sl["tp"] == 0
    assert sl["fn"] == 1        # one event, one miss -- not two


# ---- flag-mask convention regression (Phase 1) ----------------------------------------------

def test_debris_exclusion_does_not_change_in_window_detection():
    """Debris is a third bucket -- excluded from FALSE-POSITIVE scoring only. It must never
    suppress an in-window detection, and a non-debris out-of-window flag must still count."""
    # in-window detection is unaffected by an unrelated debris step
    verdicts = [em.StepVerdict(i, is_anomaly=(i == 2)) for i in range(N_STEPS)]
    detected, _, _, _ = em.score_trial_steps(verdicts, gt_steps=[2], debris_steps={0})
    assert detected is True

    # out-of-window, not debris -> false positive
    verdicts = [em.StepVerdict(i, is_anomaly=(i == 4)) for i in range(N_STEPS)]
    _, _, out_of_window, _ = em.score_trial_steps(verdicts, gt_steps=[0], debris_steps=set())
    assert out_of_window is True


# ---- trial_located ----------------------------------------------------------------------------

def test_trial_located_hit_anywhere_in_range_counts_and_debris_is_in_range():
    """The positive range is gt UNION debris: a flag on a debris step is a hit, because the
    injection genuinely moved that step."""
    v_gt = [em.StepVerdict(i, is_anomaly=(i == 2)) for i in range(N_STEPS)]      # hits gt
    v_debris = [em.StepVerdict(i, is_anomaly=(i == 3)) for i in range(N_STEPS)]  # hits debris
    report = em.evaluate_steps(
        healthy_verdicts=[],
        degraded_by_type={"substitution": [(v_gt, [2], None), (v_debris, [2], None)]},
        artifact_steps={"substitution": [{3}, {3}]},
    )
    tl = report["trial_located"]
    assert tl["tp"] == 2 and tl["fn"] == 0
    assert tl["recall"] == 1.0
    assert tl["stray"] == 0          # neither flagged outside the range


def test_trial_located_miss_is_a_false_negative():
    verdicts = [em.StepVerdict(i, is_anomaly=False) for i in range(N_STEPS)]
    report = em.evaluate_steps(
        healthy_verdicts=[],
        degraded_by_type={"substitution": [(verdicts, [2], None)]},
    )
    tl = report["trial_located"]
    assert (tl["tp"], tl["fn"], tl["recall"]) == (0, 1, 0.0)


def test_trial_located_charges_a_stray_even_when_the_trial_was_also_hit():
    """Finding the anomaly must not buy absolution for also firing elsewhere -- that is the whole
    point of counting the stray independently of the hit."""
    verdicts = [em.StepVerdict(i, is_anomaly=(i in (2, 4))) for i in range(N_STEPS)]
    report = em.evaluate_steps(
        healthy_verdicts=[],
        degraded_by_type={"substitution": [(verdicts, [2], None)]},
    )
    tl = report["trial_located"]
    assert tl["tp"] == 1              # step 2 is in range
    assert tl["stray"] == 1           # step 4 is not -- charged anyway
    assert tl["recall"] == 1.0
    assert tl["precision"] == pytest.approx(0.5)   # 1 TP against 1 stray


def test_trial_located_pools_healthy_flags_into_precision_but_reports_rates_separately():
    verdicts = [em.StepVerdict(i, is_anomaly=(i == 2)) for i in range(N_STEPS)]
    report = em.evaluate_steps(
        healthy_verdicts=[_llm(anomaly_at=(1,)), _llm()],
        degraded_by_type={"substitution": [(verdicts, [2], None)]},
    )
    tl = report["trial_located"]
    assert tl["tp"] == 1 and tl["stray"] == 0 and tl["healthy_fp"] == 1
    assert tl["precision"] == pytest.approx(0.5)   # 1 TP against 1 healthy false alarm
    assert tl["healthy_fpr"] == pytest.approx(0.5)  # 1 of 2 healthy trials
    assert tl["stray_rate"] == 0.0                  # reported separately, not mixed in


# ---- healthy FPR ------------------------------------------------------------------------------

def test_healthy_false_positive_rate_counts_any_flagged_step():
    """A healthy trial is a false positive if ANY of its steps was flagged -- no run-length bar."""
    healthy = [
        em.from_llm_verdicts([Verdict(i, False, None, None, "", True) for i in range(N_STEPS)]),
        [em.StepVerdict(i, is_anomaly=(i == 2)) for i in range(N_STEPS)],
    ]
    report = em.evaluate_steps(healthy_verdicts=healthy, degraded_by_type={})
    assert report["healthy"]["false_positive_rate"] == 0.5


# ---- injection_touched_steps (textify) -------------------------------------------------------

def test_injection_touched_steps_flags_substitution_fragments_not_the_edited_step():
    """Substitution retags one tick inside a run, splitting it into [fragment, edited, fragment].
    Both fragments are debris (their duration only exists because of the split); the edited step
    itself is ground truth and must not be flagged as debris."""
    verbs = [0, 0, 0, 0, 0, 1, 1, 1]  # one run of v0 (8 ticks), then v1
    nouns = [0, 0, 0, 9, 0, 1, 1, 1]  # tick 3 retagged (noun 9), splitting the v0 run into 3
    steps = textify.steps_from_ids(verbs, nouns, _Lex())
    # steps: [0,3) v0/n0, [3,4) v0/n9 (edited), [4,5) v0/n0, [5,8) v1/n1
    tick_map = np.arange(len(verbs))
    edited_ticks = [3]
    gt_steps = textify.gt_steps_for_window(steps, (3, 3))
    assert gt_steps == [1]  # the edited step

    touched = textify.injection_touched_steps(steps, tick_map, edited_ticks, gt_steps)
    assert touched == {0, 2}  # both surviving fragments, not the edited step or the untouched v1 run


def test_injection_touched_steps_flags_interior_splice_not_natural_boundary():
    """A tick_map splice that lands exactly on an existing step boundary (the common case for
    abandonment/omission/transposition) is not debris; a splice INSIDE one step's own tick range
    (repetition's duplicate merging into an over-long run) is."""
    verbs = [0, 0, 1, 1]
    nouns = [0, 0, 1, 1]
    steps = textify.steps_from_ids(verbs, nouns, _Lex())  # [0,2) v0/n0, [2,4) v1/n1
    # splice lands exactly on the natural step boundary at tick 2 -- not debris
    tick_map_boundary = np.array([0, 1, 5, 6])
    assert textify.injection_touched_steps(steps, tick_map_boundary, [], []) == set()

    # splice INSIDE the second step's own range (between ticks 2 and 3) -- debris
    tick_map_interior = np.array([0, 1, 2, 9])
    assert textify.injection_touched_steps(steps, tick_map_interior, [], []) == {1}


def test_chance_precision_counts_healthy_steps_in_the_denominator():
    one = em.evaluate_steps(healthy_verdicts=[],
                            degraded_by_type={"substitution": [(_llm(anomaly_at=(2,)), [2], None)]})
    two = em.evaluate_steps(healthy_verdicts=[_llm(), _llm()],
                            degraded_by_type={"substitution": [(_llm(anomaly_at=(2,)), [2], None)]})
    assert two["step_level"]["n_steps"] == one["step_level"]["n_steps"] + 2 * N_STEPS
    assert two["step_level"]["chance_precision"] < one["step_level"]["chance_precision"]
