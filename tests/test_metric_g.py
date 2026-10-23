"""eval/metric_g.py: alarms are the first flag of each run of constant (verb, noun); only alarm
onsets find spans (within [start, end + DEADLINE]) or count as false positives."""
import numpy as np
import pytest

from cook_ad.eval import metric_g as G

D = G.DEADLINE


def _f(*ticks, strength=1.0, channel="s_emit"):
    return [G.Flag(t, strength, channel) for t in ticks]


def _starts(n, every=1):
    """Run starts for a stream of n ticks whose runs are `every` ticks long."""
    return np.arange(0, n, every)


def test_flags_inside_one_run_are_one_alarm_found_once():
    rs = _starts(100, every=50)              # runs [0, 50) and [50, 100)
    r = G.score_trial(_f(*range(10, 22)), [(10, 21)], rs)
    assert r["per_span"][0]["hit"] and r["per_span"][0]["latency"] == 0
    assert r["fp_merged"] == [] and r["extra_in_span"] == 0


def test_a_span_with_no_onset_in_its_window_is_a_false_negative():
    r = G.score_trial(_f(500), [(10, 21)], _starts(600))
    assert not r["per_span"][0]["hit"] and r["fp_merged"] == [500]


def test_each_span_needs_its_own_alarm():
    spans, rs = [(10, 14), (25, 29)], _starts(40)
    assert sum(s["hit"] for s in G.score_trial(_f(10), spans, rs)["per_span"]) == 1
    assert sum(s["hit"] for s in G.score_trial(_f(10, 25), spans, rs)["per_span"]) == 2


def test_the_deadline_extends_past_the_span_end_and_stops_one_tick_later():
    rs = _starts(100)
    assert G.score_trial(_f(20 + D), [(10, 20)], rs)["per_span"][0]["hit"]
    assert not G.score_trial(_f(20 + D + 1), [(10, 20)], rs)["per_span"][0]["hit"]


def test_an_onset_before_the_span_never_counts_however_close():
    rs = _starts(100)
    r = G.score_trial(_f(9), [(10, 20)], rs)
    assert not r["per_span"][0]["hit"] and r["fp_merged"] == [9]


def test_only_the_alarm_onset_can_find_a_span():
    """An alarm that starts before a span and runs into it is a false positive, not a hit --
    the saturation exploit metric D allowed."""
    rs = np.array([0, 40])                    # one run covering the span's start
    r = G.score_trial(_f(*range(5, 30)), [(10, 20)], rs)
    assert not r["per_span"][0]["hit"] and r["fp_merged"] == [5]


def test_alarms_are_runs_and_start_at_their_first_flag():
    rs = np.array([0, 100, 200, 300])
    assert G.alarm_ticks([103, 106, 150, 210, 299], rs) == [103, 210]


def test_flag_every_tick_scores_exactly_flag_every_run():
    rng = np.random.default_rng(0)
    verb = np.repeat(rng.integers(0, 5, 30), rng.integers(1, 9, 30))
    noun = np.zeros_like(verb)
    rs = G.run_starts(verb, noun)
    spans = [(int(a), int(a) + 3) for a in rs[::4]]
    every_tick = G.score_trial(list(range(len(verb))), spans, rs)
    every_run = G.score_trial(list(rs), spans, rs)
    assert every_tick["per_span"] == every_run["per_span"]
    assert every_tick["fp_merged"] == every_run["fp_merged"]


def test_weak_flags_are_dropped_and_bare_ticks_are_accepted():
    rs = _starts(50)
    assert not G.score_trial(_f(10, strength=0.2), [(10, 20)], rs)["per_span"][0]["hit"]
    assert G.score_trial([10], [(10, 20)], rs)["per_span"][0]["hit"]


def test_channel_masks_become_one_flag_per_tick_and_channel():
    flags = G.flags_from_channels({"s_emit": np.array([0, 1, 0, 1], bool),
                                   "s_noun": np.array([0, 1, 0, 0], bool)})
    assert [(f.tick, f.channel) for f in flags] == [(1, "s_emit"), (1, "s_noun"), (3, "s_emit")]


# ---- pooled ---------------------------------------------------------------------------------


def _pool(trials, healthy=(), types=None):
    starts = [_starts(1000) for _ in trials]
    return G.evaluate(list(healthy), trials, [_starts(1000) for _ in healthy], starts, types)


def test_recall_is_over_spans_so_span_width_cannot_change_it():
    rep = _pool([(_f(10), [((10, 11), "a")]), (_f(10), [((10, 400), "b")])])
    assert rep["per_type"]["a"]["recall"] == rep["per_type"]["b"]["recall"] == 1.0


def test_the_healthy_population_is_reported_but_never_folded_into_f1():
    trials = [(_f(10), [((10, 20), "a")])]
    clean, noisy = _pool(trials), _pool(trials, healthy=[_f(1, 50, 900)] * 5)
    assert clean["pooled"] == noisy["pooled"]
    assert noisy["healthy"]["false_positive_rate"] == 1.0
    assert noisy["healthy"]["flags_merged"] == 15


def test_latency_is_measured_from_the_span_start_over_hits_only():
    rep = _pool([(_f(13), [((10, 20), "a")]), (_f(999), [((10, 20), "a")])])
    assert rep["latency"]["n"] == 1 and rep["latency"]["median"] == 3


def test_precision_charges_every_false_alarm_and_reports_no_true_negatives():
    rep = _pool([(_f(10, 300, 600), [((10, 20), "a")])])
    assert rep["pooled"]["fp"] == 2 and rep["pooled"]["precision"] == pytest.approx(1 / 3)
    assert "tn" not in rep["pooled"]


def test_a_hit_on_one_type_is_not_a_miss_on_another_and_fps_count_once_per_trial():
    rep = _pool([(_f(10, 30, 500), [((10, 14), "a"), ((30, 34), "b")])])
    assert rep["per_type"]["a"]["recall"] == rep["per_type"]["b"]["recall"] == 1.0
    assert rep["pooled"]["fp"] == 1


def test_error_types_selects_recall_and_a_removed_span_becomes_a_false_alarm():
    trials = [(_f(10, 30), [((10, 14), "a"), ((30, 34), "b")])]
    starts = [_starts(1000)]
    only_a = G.evaluate([], [(trials[0][0], [trials[0][1][0]])], [], starts, ("a",))
    assert set(only_a["per_type"]) == {"a"} and only_a["pooled"]["fp"] == 1


# ---- dense (sweeps) -------------------------------------------------------------------------


def test_dense_onsets_and_scores_match_the_sparse_path():
    rng = np.random.default_rng(1)
    verb = np.repeat(rng.integers(0, 4, 25), rng.integers(1, 6, 25))
    noun = np.zeros_like(verb)
    rs, t = G.run_starts(verb, noun), len(verb)
    f = rng.random((t, 4)) < 0.15
    spans = [(5, 9), (40, 41)]
    on = G.alarm_onsets(f, rs)
    dense = G.score_dense(f, spans, G.correct_runs(verb, noun, spans), rs)
    for j in range(f.shape[1]):
        ticks = np.flatnonzero(f[:, j]).tolist()
        assert list(np.flatnonzero(on[:, j])) == G.alarm_ticks(ticks, rs)
        sparse = G.score_trial(ticks, spans, rs)
        assert list(dense["hits"][:, j]) == [s["hit"] for s in sparse["per_span"]]
        assert dense["fp"][j] == len(sparse["fp_merged"])


def test_grid_score_pools_like_the_per_trial_counts():
    verb, noun = np.repeat([0, 1, 2, 3], 10), np.zeros(40, int)
    flags = np.zeros((40, 2), bool)
    flags[[0, 12, 25], 0] = True          # column 0: alarms in runs 0, 1, 2
    flags[:, 1] = True                    # column 1: every tick
    g = G.GridScore()
    g.add("a", flags, [(10, 12)], verb, noun)
    g.add("healthy", flags, [], verb, noun)
    r = g.result()
    assert list(r["recall"]) == [1.0, 1.0]
    # column 0 FPs: ticks 0, 25; column 1: 0, 30 (20 is a second onset inside the window)
    assert list(r["precision"]) == pytest.approx([1 / 3, 1 / 3])
    assert list(r["healthy_trial_fpr"]) == [1.0, 1.0]
    assert r["run_fpr"][1] == 1.0


def test_alarm_rate_uses_the_tick_length():
    """600 ticks at 0.5 s are 5 minutes: one alarm in them is 2 per 10 minutes."""
    verbs, nouns = np.arange(600) // 60, np.zeros(600, dtype=int)
    flags = np.zeros((600, 1), dtype=bool)
    flags[0] = True
    for ts, rate in ((1.0, 1.0), (0.5, 2.0)):
        g = G.GridScore(tick_seconds=ts)
        g.add("healthy", flags, [], verbs, nouns)
        g.add("omission", np.zeros((600, 1), dtype=bool), [(100, 110)], verbs, nouns)
        assert float(g.result()["alarms_per_10min"][0]) == pytest.approx(rate)
