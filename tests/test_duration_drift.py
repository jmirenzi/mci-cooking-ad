import pytest

from cook_ad.lifecycle import duration_drift

D_MAX = 20
STATE = 0
FILLER_STATE = 1
K_SUBTASK = 2


def _segments_by_trial(durations_list):
    """Each trial has a target-state segment followed by a filler segment, so the target
    duration is never the trial's final (right-censored) segment and is retained."""
    return [[(STATE, d), (FILLER_STATE, 1)] for d in durations_list]


def _state_row(rows):
    return next(r for r in rows if r["state"] == STATE)


def test_duration_drift_gates_on_significance_not_just_kl():
    frozen_small = [4, 5, 6]
    recent_small = [8, 9, 10]
    rows_small = duration_drift.duration_drift(
        _segments_by_trial(recent_small), _segments_by_trial(frozen_small),
        K_SUBTASK, D_MAX, min_instances=3, min_delta_ticks=2.0, alpha=0.05,
    )
    row_small = _state_row(rows_small)
    assert row_small["kl"] > 0.0
    assert row_small["p_value"] == pytest.approx(0.05)  # n=3 vs n=3: smallest achievable p is exactly alpha
    assert row_small["reportable"] is False  # strict p < alpha fails at the boundary

    frozen_large = frozen_small * 4
    recent_large = recent_small * 4
    rows_large = duration_drift.duration_drift(
        _segments_by_trial(recent_large), _segments_by_trial(frozen_large),
        K_SUBTASK, D_MAX, min_instances=3, min_delta_ticks=2.0, alpha=0.05,
    )
    row_large = _state_row(rows_large)
    assert row_large["kl"] > 0.0
    assert row_large["p_value"] < 0.05
    assert row_large["reportable"] is True


def test_duration_drift_direction_matches_sign_of_delta():
    frozen = [4, 5, 6] * 4
    slower = [8, 9, 10] * 4
    faster_recent = [4, 5, 6] * 4
    faster_frozen = [8, 9, 10] * 4

    rows_slower = duration_drift.duration_drift(
        _segments_by_trial(slower), _segments_by_trial(frozen),
        K_SUBTASK, D_MAX, min_instances=3, min_delta_ticks=2.0, alpha=0.05,
    )
    row_slower = _state_row(rows_slower)
    assert row_slower["delta_mean"] > 0
    assert row_slower["direction"] == "slower"

    rows_faster = duration_drift.duration_drift(
        _segments_by_trial(faster_recent), _segments_by_trial(faster_frozen),
        K_SUBTASK, D_MAX, min_instances=3, min_delta_ticks=2.0, alpha=0.05,
    )
    row_faster = _state_row(rows_faster)
    assert row_faster["delta_mean"] < 0
    assert row_faster["direction"] == "faster"
