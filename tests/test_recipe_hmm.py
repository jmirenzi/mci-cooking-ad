import jax
import numpy as np
import pytest

from cook_ad.recipe import recipe_hmm

jax.config.update("jax_enable_x64", True)


def test_adjusted_rand_identical_is_one():
    labels = np.array([0, 0, 1, 1, 2, 2, 2])
    assert np.isclose(recipe_hmm.adjusted_rand(labels, labels), 1.0)


def test_adjusted_rand_permuted_labels_still_one():
    a = np.array([0, 0, 1, 1, 2, 2])
    b = np.array(["x", "x", "y", "y", "z", "z"])
    assert np.isclose(recipe_hmm.adjusted_rand(a, b), 1.0)


def test_adjusted_rand_independent_near_zero():
    rng = np.random.default_rng(0)
    n = 4000
    a = rng.integers(0, 5, size=n)
    b = rng.integers(0, 5, size=n)
    ari = recipe_hmm.adjusted_rand(a, b)
    assert abs(ari) < 0.05


def test_adjusted_rand_single_cluster_degenerate():
    a = np.zeros(10, dtype=int)
    b = np.zeros(10, dtype=int)
    assert recipe_hmm.adjusted_rand(a, b) == 1.0


def test_matched_accuracy_perfect_up_to_permutation():
    true = np.array([0, 0, 1, 1, 2, 2])
    pred = np.array([5, 5, 3, 3, 9, 9])  # arbitrary cluster ids, same partition
    acc, table, pred_vals, true_vals = recipe_hmm.matched_accuracy(pred, true)
    assert acc == 1.0


def test_matched_accuracy_partial():
    true = np.array([0, 0, 0, 1, 1, 1])
    pred = np.array([0, 0, 1, 1, 1, 1])  # one true-0 mislabeled as cluster 1
    acc, table, pred_vals, true_vals = recipe_hmm.matched_accuracy(pred, true)
    assert np.isclose(acc, 5 / 6)


def test_effective_k_counts_distinct_used_clusters():
    assert recipe_hmm.effective_k([0, 0, 1, 2, 2, 2]) == 3
    assert recipe_hmm.effective_k([4, 4, 4]) == 1


def _synthetic_recipe_segments(rng, k_recipe, k_subtask, n_trials_per_recipe, seg_len_range=(6, 14)):
    """Generate segment-symbol sequences from k_recipe distinct, well-separated emission
    profiles over subtask symbols, so a correctly-fit recipe HMM should recover the true
    recipe id per trial (up to permutation) with high ARI -- the oracle for this synthetic
    test. `n_trials_per_recipe` must be large enough that finite-sample within-block
    frequency noise doesn't give a weak-limit model an incentive to split one true recipe
    into two active states purely by chance (verified directly: 20 trials/recipe at
    k_recipe=6 let exactly this happen once; 60 trials/recipe with less weak-limit headroom
    recovers cleanly).
    """
    # Each recipe gets its own disjoint half-open block of subtask symbols so recipe identity
    # is unambiguous from the emitted symbols alone (mirrors distinct recipes using distinct
    # subtask vocabularies in Breakfast, e.g. "add_teabag" only appears in tea).
    block = k_subtask // k_recipe
    assert block >= 2, "need at least 2 subtask symbols per recipe for a real transition"

    seg_sequences = []
    true_recipes = []
    for true_r in range(k_recipe):
        symbols = list(range(true_r * block, true_r * block + block))
        for _ in range(n_trials_per_recipe):
            length = rng.integers(seg_len_range[0], seg_len_range[1] + 1)
            seq = rng.choice(symbols, size=length).tolist()
            seg_sequences.append(seq)
            true_recipes.append(true_r)
    return seg_sequences, np.array(true_recipes)


def test_recipe_em_recovers_synthetic_clusters():
    rng = np.random.default_rng(42)
    k_recipe, k_subtask = 3, 12
    seg_sequences, true_recipes = _synthetic_recipe_segments(rng, k_recipe, k_subtask, n_trials_per_recipe=60)

    best_params, best_loglik, history = recipe_hmm.run_em(
        jax.random.PRNGKey(0),
        seg_sequences,
        k_recipe=5,  # weak-limit nominal > true 3, mirrors real usage
        k_subtask=k_subtask,
        n_restarts=5,
        max_iters=60,
        tol=1e-4,
        progress=False,
    )

    assert np.isfinite(float(best_loglik))
    for restart_history in history:
        losses = np.array(restart_history)
        # Loglik must climb sharply overall and settle near its ceiling -- not collapse or
        # diverge. Tiny (~1e-2) non-monotonic wobbles are tolerated near convergence: the
        # alpha/K<1 init/trans prior is non-log-concave (same caveat as the subtask HSMM's
        # MAP-EM, see hsmm/params.py), so the floored MAP closed-form isn't guaranteed to be
        # an exact ascent step right at the flooring boundary.
        assert losses[-1] > losses[0]
        assert np.all(np.diff(losses) >= -1e-2)

    obs_ids, mask = recipe_hmm.pad_segment_batch(seg_sequences)
    pred_recipes = recipe_hmm.decode_recipe(best_params, obs_ids, mask)

    ari = recipe_hmm.adjusted_rand(pred_recipes, true_recipes)
    assert ari > 0.9

    eff_k = recipe_hmm.effective_k(pred_recipes)
    assert eff_k == k_recipe
