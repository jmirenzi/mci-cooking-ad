"""Metric H (eval/metric_h.py): any flag in the window hits; one miss per run with a flag outside."""
import numpy as np

from cook_ad.eval import metric_g, metric_h


def _trial():
    v = np.zeros(30, int)
    v[5:25] = 1                       # idle, one 20-tick run, idle
    n = np.zeros(30, int)
    return metric_g.run_starts(v, n), [(0, 4), (25, 29)]


def _score(ticks, delay=0, spans=((15, 24),)):
    starts, correct = _trial()
    f = np.zeros((30, 1), bool)
    f[list(ticks), 0] = True
    return metric_h.score_dense(f, list(spans), correct, starts, delay)


def test_flags_from_the_copy_on_are_one_hit_no_miss():
    r = _score(range(15, 25))
    assert r["hits"].ravel().tolist() == [True] and r["fp"].tolist() == [0]


def test_a_stray_flag_costs_one_miss_but_does_not_hide_the_hit():
    r = _score([8] + list(range(15, 25)))       # G scores this FP + FN
    assert r["hits"].ravel().tolist() == [True] and r["fp"].tolist() == [1]


def test_flags_starting_one_tick_early_are_hit_plus_miss():
    r = _score(range(14, 25))
    assert r["hits"].ravel().tolist() == [True] and r["fp"].tolist() == [1]


def test_many_flags_outside_one_run_are_one_miss():
    assert _score(range(5, 12))["fp"].tolist() == [1]


def test_delay_widens_the_window_past_the_span_end():
    assert _score([27], delay=0, spans=((15, 24),))["hits"].ravel().tolist() == [False]
    assert _score([27], delay=3, spans=((15, 24),))["hits"].ravel().tolist() == [True]


def test_flag_every_tick_hits_everything_and_misses_each_run_once():
    r = _score(range(30))
    assert r["hits"].all() and r["fp"].tolist() == [3]      # idle, the long run, idle


def test_alarms_count_stretches_so_flicker_is_charged():
    starts, _ = _trial()
    f = np.zeros((30, 1), bool)
    f[[6, 7, 9, 10], 0] = True
    assert metric_h._stretch_onsets(f, starts).sum() == 2
