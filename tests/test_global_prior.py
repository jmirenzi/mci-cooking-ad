import numpy as np
import pytest

from cook_ad.hsmm.params import JointHSMMParams, _row_normalize
from cook_ad.hsmm.train.global_prior import (apply_global_prior, count_global, run_length,
                                             state_pairs)


def _params():
    """K = 4 states anchored on pairs (0,0), (1,1), (2,2), plus one flat headroom state; 1 recipe.
    The person has only ever done 0 -> 1 -> 2."""
    k, v = 4, 3
    verb = np.full((k, v), 0.1)
    noun = np.full((k, v), 0.1)
    for s in range(3):
        verb[s, s] += 50
        noun[s, s] += 50
    trans = np.zeros((1, k, k))
    trans[0, 0, 1] = 50.0
    trans[0, 1, 2] = 50.0
    init = np.zeros((1, k))
    init[0, 0] = 5.0
    return JointHSMMParams(init, trans, verb, noun, np.ones((1, k)), np.full((1, k), 0.5),
                           np.ones(1))


def _seq(pairs, reps=2):
    v, n = [], []
    for a, b in pairs:
        v += [a] * reps
        n += [b] * reps
    return {"verb_ids": v, "noun_ids": n}


def test_run_length():
    assert run_length([0, 0, 1, 1, 0], [0, 0, 1, 1, 0]) == [(0, 0, 2), (1, 1, 2), (0, 0, 1)]


def test_counts_and_state_pairs():
    g = count_global([_seq([(0, 0), (1, 1), (2, 2)]), _seq([(0, 0), (2, 2), (1, 1)])], ["r", "r"])
    assert g["init"]["r"] == {"0_0": 2} and g["trans"]["r"]["0_0"] == {"1_1": 1, "2_2": 1}
    assert state_pairs(_params()) == [(0, 0), (1, 1), (2, 2), None]


def test_lambda_zero_is_identity_and_prior_lifts_only_population_transitions():
    p = _params()
    g = count_global([_seq([(0, 0), (2, 2), (1, 1)])], ["r"])   # the population also does 0 -> 2
    assert apply_global_prior(p, g, ["r"], 0.0) is p
    q = apply_global_prior(p, g, ["r"], 10.0)
    before = np.exp(np.asarray(_row_normalize(p.trans_counts[0], mask_diag=True)))
    after = np.exp(np.asarray(_row_normalize(q.trans_counts[0], mask_diag=True)))
    assert before[0, 2] < 1e-9 < 0.1 < after[0, 2]                 # 0 -> 2: seen by the population
    assert after[1, 0] < 1e-9                                      # 1 -> 0: seen by nobody
    assert after[0, 1] > after[0, 2]                               # the person's own habit still wins
    assert np.all(np.diagonal(q.trans_counts, axis1=1, axis2=2) == 0)


def test_unknown_recipe_and_replace():
    p = _params()
    g = count_global([_seq([(0, 0), (2, 2)])], ["other"])
    q = apply_global_prior(p, g, ["r"], 10.0)
    np.testing.assert_array_equal(np.maximum(np.asarray(p.trans_counts), 1.0) * (1 - np.eye(4)),
                                  q.trans_counts)
    g = count_global([_seq([(0, 0), (2, 2)])], ["r"])
    only = apply_global_prior(p, g, ["r"], 10.0, replace=True)
    assert only.trans_counts[0, 0, 1] == 1.0 and only.trans_counts[0, 0, 2] == 11.0


def test_fading_prior_shrinks_with_the_persons_own_count():
    p = _params()
    g = count_global([_seq([(0, 0), (2, 2), (1, 1), (0, 0)])], ["r"])   # 0->2, 2->1, 1->0
    # the person left state 0 fifty times (0 -> 1) and never left state 2
    person = count_global([_seq([(0, 0), (1, 1)])] * 50, ["r"] * 50)
    fixed = apply_global_prior(p, g, ["r"], 10.0)
    fade = apply_global_prior(p, g, ["r"], 10.0, person_counts=person, n0=5.0)
    # row 0 (well observed): prior mass 10 / (1 + 50/5) ~ 0.91 instead of 10
    assert fade.trans_counts[0, 0, 2] - 1.0 == pytest.approx(10.0 / 11.0)
    assert fixed.trans_counts[0, 0, 2] - 1.0 == pytest.approx(10.0)
    # row 2 (never left by the person): the full prior
    assert fade.trans_counts[0, 2, 1] == fixed.trans_counts[0, 2, 1]
