"""detectors/thresholds.py: the exact discrete alpha-tails every bad set is built from."""
import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

from cook_ad.detectors.thresholds import (
    excess_quantile_threshold,
    joint_quantile_threshold,
    recipe_excess_thresholds,
)
from cook_ad.detectors.thresholds import tail_threshold as _tail_threshold


def _largest_achievable_mass(scores, probs, alpha):
    """Independent brute-force oracle: sort descending by score, collapse ties, take the
    largest achievable cumulative mass <= alpha. Used to check _tail_threshold's claim without
    just re-running the same code path."""
    scores = np.asarray(scores, dtype=np.float64)
    probs = np.asarray(probs, dtype=np.float64)
    support = probs > 0
    scores, probs = scores[support], probs[support]
    order = np.argsort(-scores)
    scores, probs = scores[order], probs[order]

    groups = []  # (score, mass)
    for s, p in zip(scores, probs):
        if groups and groups[-1][0] == s:
            groups[-1] = (s, groups[-1][1] + p)
        else:
            groups.append((s, p))

    achieved = 0.0
    for _, mass in groups:
        if achieved + mass <= alpha:
            achieved += mass
        else:
            break
    return achieved


def _assert_largest_achievable(scores, probs, alpha):
    t = _tail_threshold(scores, probs, alpha)
    scores = np.asarray(scores)
    probs = np.asarray(probs)
    support = probs > 0
    actual = probs[support][scores[support] > t].sum()
    expected = _largest_achievable_mass(scores, probs, alpha)
    assert actual <= alpha + 1e-9
    assert actual == pytest.approx(expected, abs=1e-9)


def test_quantile_threshold_achieves_correct_tail_mass():
    # plain case, no ties
    probs = np.array([0.5, 0.3, 0.15, 0.05])
    scores = -np.log(probs)
    _assert_largest_achievable(scores, probs, alpha=0.1)

    # tie case: a tied group must be included/excluded as one atomic unit under strict '>'
    probs = np.array([0.5, 0.1, 0.1, 0.1, 0.1, 0.1])
    scores = -np.log(probs)
    _assert_largest_achievable(scores, probs, alpha=0.25)
    t = _tail_threshold(scores, probs, alpha=0.25)
    # the five tied 0.1 tokens collectively carry 0.5 > alpha, so none of them are achievable
    assert probs[scores > t].sum() == pytest.approx(0.0)

    # degenerate case: uniform support, alpha too small for even the smallest atomic group
    probs = np.full(8, 1.0 / 8)
    scores = -np.log(probs)
    t = _tail_threshold(scores, probs, alpha=0.05)
    assert probs[scores > t].sum() == pytest.approx(0.0)
    assert t == pytest.approx(scores[0])  # falls back to the (single, tied) max score itself


def test_joint_threshold_matches_bruteforce():
    rng = np.random.default_rng(0)
    K, V, N = 3, 4, 3
    pv = rng.dirichlet(np.ones(V), size=K)
    pn = rng.dirichlet(np.ones(N), size=K)
    log_v = np.log(pv)
    log_n = np.log(pn)
    alpha = 0.1

    t = joint_quantile_threshold(log_v, log_n, alpha)
    for k in range(K):
        log_joint = log_v[k][:, None] + log_n[k][None, :]
        probs = np.exp(log_joint).ravel()
        scores = -log_joint.ravel()
        expected = _largest_achievable_mass(scores, probs, alpha)
        actual = probs[scores > t[k]].sum()
        assert actual == pytest.approx(expected, abs=1e-9)
        assert actual <= alpha + 1e-9


def test_excess_threshold_tail_mass():
    # state 2's recipe-conditioned row is FLATTER than the marginal -> negative threshold
    trans_r = np.array([
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.4, 0.4, 0.2],
    ])
    trans_marginal = np.array([
        [0.0, 0.9, 0.1],
        [0.9, 0.0, 0.1],
        [0.1, 0.1, 0.8],
    ])
    log_trans_r = np.log(trans_r, out=np.full_like(trans_r, -np.inf), where=trans_r > 0)
    log_trans_marginal = np.log(trans_marginal, out=np.full_like(trans_marginal, -np.inf), where=trans_marginal > 0)
    alpha = 0.3

    t = excess_quantile_threshold(log_trans_r, log_trans_marginal, alpha)
    assert t[2] < 0.0

    for j in range(3):
        probs = trans_r[j]
        with np.errstate(invalid="ignore"):  # -inf - -inf on the masked diagonal; dropped by probs>0 below
            scores = log_trans_marginal[j] - log_trans_r[j]
        expected = _largest_achievable_mass(scores, probs, alpha)
        actual = probs[probs > 0][scores[probs > 0] > t[j]].sum()
        assert actual == pytest.approx(expected, abs=1e-9)
        assert actual <= alpha + 1e-9


def test_recipe_excess_thresholds_are_inf_at_one_recipe_and_untouched_otherwise():
    """Gated on the NUMBER of recipes, not on a per-row equality test: the per-row test also
    matches 112/1024 rows of the Breakfast k_recipe=16 fit and would have moved its numbers."""
    rng = np.random.default_rng(0)
    rows = rng.dirichlet(np.ones(4), size=4)
    log_r = np.log(rows)
    log_m = np.log(0.5 * rows + 0.5 * rng.dirichlet(np.ones(4), size=4))

    one = recipe_excess_thresholds(log_r, log_m, 0.05, k_recipe=1)
    assert np.all(np.isinf(one)) and np.all(one > 0)
    two = recipe_excess_thresholds(log_r, log_m, 0.05, k_recipe=2)
    np.testing.assert_array_equal(two, excess_quantile_threshold(log_r, log_m, 0.05))
