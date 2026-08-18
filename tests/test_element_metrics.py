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


def test_in_window_detection_ignores_persistence():
    """Inside the window there IS an injected anomaly, so a single non-persistent flag still
    counts -- otherwise genuine low-latency detections get delayed."""
    verdicts = [em.StepVerdict(i, is_anomaly=(i == 2), persistent=False) for i in range(N_STEPS)]
    detected, latency, out_of_window, _ = em.score_trial_steps(verdicts, [2])
    assert (detected, latency, out_of_window) == (True, 0, False)


def test_out_of_window_requires_persistence():
    verdicts = [em.StepVerdict(i, is_anomaly=(i == 4), persistent=False) for i in range(N_STEPS)]
    _, _, out_of_window, _ = em.score_trial_steps(verdicts, [0])
    assert out_of_window is False


# ---- HSMM adapter -----------------------------------------------------------------------------

def test_scattered_tick_flags_are_detections_but_not_persistent():
    """The reason step_verdicts_from_flags keeps two masks: a plain per-step OR over ~11 ticks at
    alpha=0.05 would report the HSMM as flagging nearly every healthy step."""
    steps = _steps()
    v = em.step_verdicts_from_flags(_flags([5, 25, 45]), steps, min_run=10)
    assert [x.is_anomaly for x in v] == [True, False, True, False, True]
    assert not any(x.persistent for x in v)


def test_a_long_run_is_both_a_detection_and_persistent():
    steps = _steps()
    v = em.step_verdicts_from_flags(_flags(range(20, 32)), steps, min_run=10)
    assert [x.persistent for x in v] == [False, False, True, True, False]


def test_predicted_type_prefers_specific_channels_over_s_emit():
    """s_emit fires on nearly every error type, so it must never name a type on its own."""
    steps = _steps()
    only_emit = _flags(range(20, 32), channel="s_emit")
    v = em.step_verdicts_from_flags(only_emit, steps, min_run=10)
    assert v[2].is_anomaly is True
    assert v[2].error_type is None            # detected, but no type claimed

    both = _flags(range(20, 32), channel="s_emit")
    for t in range(20, 32):
        both["s_transition"][t] = True
    v = em.step_verdicts_from_flags(both, steps, min_run=10)
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
