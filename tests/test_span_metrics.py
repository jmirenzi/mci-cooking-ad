"""The span-existence layer. Every test here pins a rule the spec states outright, because the
whole point of the layer is that its counting rules are not the tick layer's."""
import numpy as np
import pytest

from cook_ad.eval import element_metrics, span_metrics
from cook_ad.llm import textify

D = span_metrics.DEADLINE
M = span_metrics.MERGE_TOL


def _f(*ticks, strength=1.0, channel="x"):
    return [span_metrics.Flag(t, strength, channel) for t in ticks]


class _Lex:
    def verb(self, i):
        return f"v{int(i)}"

    def noun(self, i):
        return f"n{int(i)}"


# ---- existence matching ----------------------------------------------------------------------

def test_one_flag_anywhere_in_a_span_is_exactly_one_true_positive():
    """The defect the layer exists to fix: a 12-tick span flagged once and the same span flagged
    on all 12 ticks are the SAME detection, and neither is worth more than one."""
    once = span_metrics.score_trial(_f(10), [(10, 21)])
    every = span_metrics.score_trial(_f(*range(10, 22)), [(10, 21)])
    assert [r["hit"] for r in once["per_span"]] == [True]
    assert [r["hit"] for r in every["per_span"]] == [True]
    assert once["per_span"][0]["latency"] == every["per_span"][0]["latency"] == 0


def test_extra_flags_inside_a_satisfied_span_are_diagnostic_and_never_scored():
    r = span_metrics.score_trial(_f(*range(10, 22)), [(10, 21)])
    assert r["extra_in_span"] == 11
    assert r["fp_merged"] == [] and r["fp_raw"] == []


def test_a_span_with_no_satisfying_flag_is_a_false_negative_not_a_partial_hit():
    r = span_metrics.score_trial(_f(500), [(10, 21)])
    assert r["per_span"][0]["hit"] is False
    assert r["per_span"][0]["latency"] is None


def test_each_span_is_matched_independently_so_transposition_needs_two_flags():
    """Transposition labels both moved segments. One flag finds one of them, not both."""
    spans = [(10, 20), (21, 31)]
    one = span_metrics.score_trial(_f(10), spans)
    assert [r["hit"] for r in one["per_span"]] == [True, False]
    both = span_metrics.score_trial(_f(10, 25), spans)
    assert [r["hit"] for r in both["per_span"]] == [True, True]


# ---- the deadline ----------------------------------------------------------------------------

def test_the_deadline_extends_past_the_span_end_and_stops_one_tick_later():
    span = [(10, 20)]
    assert span_metrics.score_trial(_f(20 + D), span)["per_span"][0]["hit"] is True
    assert span_metrics.score_trial(_f(20 + D + 1), span)["per_span"][0]["hit"] is False


def test_a_flag_before_the_span_start_never_counts_however_close():
    """A deadline governs lateness only. Firing early is a false alarm that happens to be near
    an anomaly, not an early detection of it."""
    r = span_metrics.score_trial(_f(9), [(10, 20)])
    assert r["per_span"][0]["hit"] is False
    assert r["fp_merged"] == [9]


def test_the_deadline_admits_a_retrospective_channel_that_can_only_fire_at_the_step_close():
    """Why the deadline exists at all: an abandonment span is the onset tick, and a channel that
    only learns the step ended short once it closes fires strictly later than that."""
    onset = 40
    assert span_metrics.score_trial(_f(onset + 6), [(onset, onset)])["per_span"][0]["hit"] is True


# ---- merging ---------------------------------------------------------------------------------

def test_clustered_flags_merge_to_one_alarm_and_dispersed_ones_do_not():
    clustered = span_metrics.score_trial(_f(100, 102, 104, 106), [])
    dispersed = span_metrics.score_trial(_f(100, 200, 300, 400), [])
    assert len(clustered["fp_merged"]) == 1
    assert len(dispersed["fp_merged"]) == 4
    assert clustered["fp_raw"] == [100, 102, 104, 106]   # the noise diagnostic keeps both


def test_a_merged_alarm_is_represented_by_its_earliest_member_so_latency_survives():
    assert span_metrics.merge_flags([100, 103, 106, 109]) == [100]
    assert span_metrics.merge_flags([100, 100 + M, 100 + 2 * M]) == [100]
    assert span_metrics.merge_flags([100, 100 + M + 1]) == [100, 100 + M + 1]


def test_merging_does_not_apply_to_the_healthy_trial_false_positive_rate():
    """It is binary per trial and therefore already immune to clustering. One trial flagged four
    times in a row is one flagged trial either way."""
    rep = span_metrics.evaluate_spans([_f(5, 6, 7, 8), []], {})
    assert rep["healthy"]["false_positive_trials"] == 1
    assert rep["healthy"]["false_positive_rate"] == 0.5
    assert rep["healthy"]["flags_raw"] == 4 and rep["healthy"]["flags_merged"] == 1


# ---- pooling ---------------------------------------------------------------------------------

def test_recall_is_over_spans_so_span_width_cannot_change_it():
    """The structural defect being removed: a tick-summing metric makes a 1-tick abandonment and
    a 12-tick substitution worth different amounts. Here they are worth the same."""
    rep = span_metrics.evaluate_spans([], {
        "substitution": [(_f(10), [(10, 21)])],
        "abandonment": [(_f(40), [(40, 40)])],
    })
    assert rep["per_type"]["substitution"]["recall"] == 1.0
    assert rep["per_type"]["abandonment"]["recall"] == 1.0
    assert rep["pooled"]["recall"] == 1.0
    # and the width column is what exposes the asymmetry the metric controls for
    assert rep["per_type"]["substitution"]["median_span_width"] == 12.0
    assert rep["per_type"]["abandonment"]["median_span_width"] == 1.0


def test_the_healthy_false_positive_rate_is_never_folded_into_f1():
    """It is a different claim on a different denominator, and pooling it in would put the same
    constant into every error type's precision -- the self-deception precision_excl_healthy
    exists to avoid in the element layer."""
    spans = {"substitution": [(_f(10), [(10, 20)])]}
    clean = span_metrics.evaluate_spans([], spans)
    noisy = span_metrics.evaluate_spans([_f(1, 50, 900)] * 5, spans)
    assert clean["pooled"] == noisy["pooled"]
    assert noisy["healthy"]["false_positive_rate"] == 1.0


def test_latency_is_measured_from_the_span_start_over_hits_only():
    rep = span_metrics.evaluate_spans([], {
        "substitution": [(_f(13), [(10, 20)]), (_f(10), [(10, 20)]), (_f(999), [(10, 20)])],
    })
    assert rep["latency"]["n"] == 2                   # the miss contributes no latency
    assert rep["latency"]["median"] == 1.5


def test_true_negatives_are_not_reported_because_they_are_undefined_here():
    rep = span_metrics.evaluate_spans([], {"substitution": [(_f(10), [(10, 20)])]})
    assert "tn" not in rep["pooled"]
    assert "fpr" not in rep["pooled"] and "specificity" not in rep["pooled"]


def test_precision_charges_each_merged_false_positive_individually():
    rep = span_metrics.evaluate_spans([], {
        "substitution": [(_f(10, 500, 900), [(10, 20)])],
    })
    assert rep["pooled"]["tp"] == 1 and rep["pooled"]["fp"] == 2
    assert rep["pooled"]["precision"] == pytest.approx(1 / 3)


# ---- the strength threshold ------------------------------------------------------------------

def test_flags_below_the_strength_threshold_are_dropped():
    """The K-sampling seam: today every producer emits 1.0, but the layer already reads a float
    so a sampled arm becomes threshold-sweepable without an interface change."""
    weak = _f(10, strength=0.4)
    strong = _f(10, strength=0.6)
    assert span_metrics.score_trial(weak, [(10, 20)])["per_span"][0]["hit"] is False
    assert span_metrics.score_trial(strong, [(10, 20)])["per_span"][0]["hit"] is True


def test_bare_tick_integers_are_accepted_as_flags():
    assert span_metrics.score_trial([10], [(10, 20)])["per_span"][0]["hit"] is True


# ---- the arm adapters ------------------------------------------------------------------------

def test_hsmm_channel_masks_become_flags_at_the_tick_itself():
    mask = np.zeros(20, bool)
    mask[[3, 11]] = True
    flags = span_metrics.flags_from_channels({"s_noun": mask, "s_verb": np.zeros(20, bool)},
                                             ("s_noun", "s_verb"))
    assert [f.tick for f in flags] == [3, 11]
    assert all(f.channel == "s_noun" for f in flags)


def test_a_step_verdict_becomes_a_flag_at_the_first_tick_of_the_element_it_judges():
    steps = textify.steps_from_ids([0] * 10 + [1] * 10, [0] * 10 + [1] * 10, _Lex())
    verdicts = [element_metrics.StepVerdict(0, False), element_metrics.StepVerdict(1, True)]
    assert [f.tick for f in span_metrics.flags_from_verdicts(verdicts, steps)] == [10]


def test_at_the_tick_unit_the_verdict_adapter_is_the_identity():
    ticks = textify.ticks_from_ids([0] * 20, [0] * 20, _Lex())
    verdicts = [element_metrics.StepVerdict(i, i in (4, 17)) for i in range(20)]
    assert [f.tick for f in span_metrics.flags_from_verdicts(verdicts, ticks)] == [4, 17]


# ---- merging is a choice, not a neutral cleanup ----------------------------------------------

def test_merge_tol_zero_disables_merging_but_still_dedupes_one_tick():
    """Two channels firing on the SAME tick is one alarm at that tick under any tolerance --
    that is deduplication, not merging."""
    assert span_metrics.merge_flags([100, 101, 102], merge_tol=0) == [100, 101, 102]
    assert span_metrics.merge_flags([100, 100, 100], merge_tol=0) == [100]


def test_merging_lets_a_saturated_detector_buy_recall_almost_free():
    """The exploit that makes span F1 unsafe to quote alone: flag every tick and the whole trial
    collapses to ONE alarm, while true positives are still counted on the raw flags."""
    flags = list(range(200))
    spans = [(50, 60)]
    merged = span_metrics.score_trial(flags, spans, merge_tol=span_metrics.MERGE_TOL)
    unmerged = span_metrics.score_trial(flags, spans, merge_tol=0)
    assert merged["per_span"][0]["hit"] and unmerged["per_span"][0]["hit"]
    assert len(merged["fp_merged"]) == 1              # ~one alarm for the whole trial
    assert len(unmerged["fp_merged"]) == len(unmerged["fp_raw"]) > 100


def test_merging_penalises_a_boundary_sparse_detector_relative_to_a_tick_dense_one():
    """Both detectors fire over the same region. The one emitting a single flag per step is
    charged more merged false positives than the one that saturates it, which is backwards."""
    spans = [(500, 510)]
    sparse = list(range(0, 200, 20))      # one flag per step boundary, 20 ticks apart
    dense = list(range(0, 200))           # every tick
    assert len(span_metrics.score_trial(sparse, spans)["fp_merged"]) == 10
    assert len(span_metrics.score_trial(dense, spans)["fp_merged"]) == 1


# ---------------------------------------------------------------------------
# evaluate_spans_multi -- trials carrying several anomalies of mixed type
# ---------------------------------------------------------------------------

def test_multi_does_not_charge_a_hit_on_one_type_as_a_miss_on_another():
    """The whole reason this sibling exists. Passing one trial's flags to evaluate_spans once
    per type counts its false positives once per type, and charges a flag that correctly hit a
    type-a span as a false positive against type b."""
    flags = _f(10)
    trials = [(flags, [((10, 12), "a"), ((50, 52), "b")])]

    multi = span_metrics.evaluate_spans_multi([], trials)
    assert multi["pooled"]["tp"] == 1 and multi["pooled"]["fn"] == 1
    assert multi["pooled"]["fp"] == 0, "the flag hit a real span; it is not a false positive"
    assert multi["per_type"]["a"]["recall"] == 1.0
    assert multi["per_type"]["b"]["recall"] == 0.0

    naive = span_metrics.evaluate_spans(
        [], {"a": [(flags, [(10, 12)])], "b": [(flags, [(50, 52)])]}
    )
    assert naive["pooled"]["fp"] == 1, "the shape this sibling exists to avoid"


def test_multi_counts_each_trials_false_positives_once():
    trials = [(_f(0, 1, 2), [((80, 82), "a"), ((90, 92), "b")])]
    report = span_metrics.evaluate_spans_multi([], trials, merge_tol=5)
    assert report["pooled"]["fp"] == 1, "0,1,2 merge into one alarm, counted once for the trial"
    assert report["false_positives"]["raw"] == 3


def test_multi_agrees_with_evaluate_spans_on_one_anomaly_per_trial():
    """On an injected corpus the two shapes describe the same thing, so they must agree."""
    trials = [(_f(10), [((10, 14), "a")]), (_f(99), [((30, 34), "a")])]
    multi = span_metrics.evaluate_spans_multi([_f(7)], trials)
    single = span_metrics.evaluate_spans([_f(7)], {"a": [(_f(10), [(10, 14)]),
                                                        (_f(99), [(30, 34)])]})
    for field in ("precision", "recall", "f1", "tp", "fn", "fp"):
        assert multi["pooled"][field] == single["pooled"][field]
    assert multi["healthy"] == single["healthy"]
    assert multi["per_type"]["a"]["recall"] == single["per_type"]["a"]["recall"]


def test_multi_attributes_per_span_rows_by_position_not_by_order():
    """score_trial returns per_span in input order; the attribution relies on that."""
    trials = [(_f(0, 40), [((40, 41), "late"), ((0, 1), "early")])]
    report = span_metrics.evaluate_spans_multi([], trials)
    assert report["per_type"]["late"]["tp"] == 1
    assert report["per_type"]["early"]["tp"] == 1


def test_multi_reports_healthy_fpr_on_its_own_denominator():
    report = span_metrics.evaluate_spans_multi(
        [_f(), _f(3), _f(4, 5)], [(_f(10), [((10, 12), "a")])]
    )
    assert report["healthy"]["n"] == 3
    assert report["healthy"]["false_positive_trials"] == 2
    assert report["healthy"]["false_positive_rate"] == pytest.approx(2 / 3)
    assert report["pooled"]["f1"] == 1.0, "healthy alarms are never folded into F1"


def test_multi_error_types_selects_what_recall_is_measured_over():
    """An unlisted type contributes NEITHER tp nor fn -- it is outside the question being asked
    -- but its span still suppresses false positives, because it is a real anomaly. That is what
    lets run_epic_tent_eval report PREGO's four-type subset without charging a flag that caught
    a real `misuse` as a false alarm."""
    # Well clear of each other's DEADLINE (10 ticks), so each flag can only satisfy its own span.
    trials = [(_f(10, 60), [((10, 12), "a"), ((60, 62), "unlisted")])]
    report = span_metrics.evaluate_spans_multi([], trials, error_types=("a", "b"))
    assert set(report["per_type"]) == {"a", "b"}
    assert report["per_type"]["b"]["n_spans"] == 0
    assert report["pooled"]["tp"] == 1 and report["pooled"]["fn"] == 0
    assert report["pooled"]["fp"] == 0, "the flag at 60 hit a real span, listed or not"


def test_multi_charges_a_flag_as_fp_when_its_span_is_removed_entirely():
    """The other reading: drop the span from the input and the same flag becomes a false
    positive. The caller chooses which question it is asking; the metric does not guess."""
    trials = [(_f(10, 60), [((10, 12), "a")])]
    report = span_metrics.evaluate_spans_multi([], trials, error_types=("a",))
    assert report["pooled"]["tp"] == 1 and report["pooled"]["fp"] == 1



# ---- metric G (onsets) ----------------------------------------------------------------------

def test_metric_g_kills_the_saturation_exploit():
    """flag-every-tick: D finds every span; G only the ones its single onset (tick 0) reaches."""
    T = 200
    spans = [(40, 50), (120, 130)]
    every = list(range(T))
    d = span_metrics.score_trial(every, spans)
    g = span_metrics.score_trial(every, spans, onset=True)
    assert all(r["hit"] for r in d["per_span"])
    assert not any(r["hit"] for r in g["per_span"])
    assert g["fp_merged"] == d["fp_merged"] == [0]


def test_metric_g_equals_d_when_every_alarm_starts_inside_its_span():
    flags = [41, 42, 43, 121]
    spans = [(40, 50), (120, 130)]
    d = span_metrics.score_trial(flags, spans)
    g = span_metrics.score_trial(flags, spans, onset=True)
    assert [r["hit"] for r in d["per_span"]] == [r["hit"] for r in g["per_span"]] == [True, True]
    assert d["fp_merged"] == g["fp_merged"] == []


def test_dense_onsets_match_merge_flags_per_column():
    rng = np.random.default_rng(0)
    f = rng.random((300, 7)) < np.array([0.01, 0.05, 0.1, 0.3, 0.6, 0.9, 1.0])
    on = span_metrics.alarm_onsets(f)
    for j in range(f.shape[1]):
        assert list(np.flatnonzero(on[:, j])) == span_metrics.merge_flags(np.flatnonzero(f[:, j]).tolist())


def test_dense_scorer_matches_score_trial_with_onsets():
    rng = np.random.default_rng(1)
    T = 400
    spans = [((30, 45), "a"), ((100, 100), "b"), ((250, 290), "a")]
    f = rng.random((T, 5)) < np.array([0.005, 0.02, 0.05, 0.2, 1.0])
    dense = span_metrics.score_onsets_dense(f, spans)
    for j in range(f.shape[1]):
        r = span_metrics.score_trial(np.flatnonzero(f[:, j]).tolist(), [s for s, _ in spans], onset=True)
        assert [row["hit"] for row in r["per_span"]] == list(dense["hits"][:, j])
        assert len(r["fp_merged"]) == dense["fp"][j]



def test_run_merge_makes_flag_every_tick_equal_flag_every_run():
    verb = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 0])
    noun = np.zeros(10, dtype=int)
    rs = span_metrics.run_starts(verb, noun)
    assert list(rs) == [0, 3, 5, 9]
    every_tick = list(range(10))
    assert span_metrics.merge_flags(every_tick, run_starts=rs) == [0, 3, 5, 9]
    assert span_metrics.merge_flags(list(rs), run_starts=rs) == [0, 3, 5, 9]
    # a flag continuing across a run boundary is a new alarm; two flags in one run are one
    assert span_metrics.merge_flags([1, 2, 3], run_starts=rs) == [1, 3]


def test_dense_run_onsets_match_merge_flags():
    rng = np.random.default_rng(3)
    verb = rng.integers(0, 3, 300).repeat(1)
    noun = np.zeros(300, dtype=int)
    rs = span_metrics.run_starts(verb, noun)
    f = rng.random((300, 6)) < np.array([0.01, 0.05, 0.2, 0.5, 0.9, 1.0])
    on = span_metrics.alarm_onsets(f, run_starts=rs)
    for j in range(f.shape[1]):
        assert list(np.flatnonzero(on[:, j])) == span_metrics.merge_flags(
            np.flatnonzero(f[:, j]).tolist(), run_starts=rs)
