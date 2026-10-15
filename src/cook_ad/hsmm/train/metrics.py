"""Clustering scores reported by the training scripts (recipe and subtask recovery against the
labels, which training never reads)."""
import numpy as np
from scipy.optimize import linear_sum_assignment

def effective_k(recipe_ids, min_frac=0.02):
    """Number of recipe clusters holding a non-negligible share of trials -- the weak-limit
    report metric ('effective K, not nominal K'), read off the decode rather than raw
    pseudocounts. A plain distinct-label count would over-report: a handful of trials
    stray-assigned to an otherwise-unused nominal state (a few misclassifications out of
    hundreds) is noise, not a genuinely occupied mode, so clusters below `min_frac` of the
    dataset are excluded."""
    recipe_ids = np.asarray(recipe_ids)
    _, counts = np.unique(recipe_ids, return_counts=True)
    threshold = min_frac * len(recipe_ids)
    return int(np.sum(counts >= threshold))


def _contingency_table(labels_a, labels_b):
    a_vals, a_inv = np.unique(labels_a, return_inverse=True)
    b_vals, b_inv = np.unique(labels_b, return_inverse=True)
    table = np.zeros((len(a_vals), len(b_vals)), dtype=np.int64)
    np.add.at(table, (a_inv, b_inv), 1)
    return table, a_vals, b_vals


def adjusted_rand(labels_a, labels_b):
    """Hand-rolled Adjusted Rand Index (no scikit-learn dependency) from a contingency
    table. Standard sklearn convention: return 1.0 when the max-index/expected-index
    denominator is 0 (degenerate labelings, e.g. everything in one cluster on both sides)."""
    labels_a = np.asarray(labels_a)
    labels_b = np.asarray(labels_b)
    table, _, _ = _contingency_table(labels_a, labels_b)

    def comb2(n):
        return n * (n - 1) / 2.0

    sum_comb = np.sum(comb2(table))
    a_sums = table.sum(axis=1)
    b_sums = table.sum(axis=0)
    sum_comb_a = np.sum(comb2(a_sums))
    sum_comb_b = np.sum(comb2(b_sums))

    n = labels_a.shape[0]
    total_comb = comb2(n)
    expected = (sum_comb_a * sum_comb_b / total_comb) if total_comb > 0 else 0.0
    max_index = 0.5 * (sum_comb_a + sum_comb_b)
    denom = max_index - expected
    if denom == 0:
        return 1.0
    return (sum_comb - expected) / denom


def matched_accuracy(pred_clusters, true_labels):
    """Hungarian-matched cluster accuracy: best one-to-one mapping of predicted cluster ids
    to true labels (via scipy's linear_sum_assignment maximizing matched count), then the
    fraction of trials whose predicted cluster maps to their true label. Returns (accuracy,
    contingency_table, pred_vals, true_vals) so the mapping can be inspected."""
    pred = np.asarray(pred_clusters)
    true = np.asarray(true_labels)
    table, pred_vals, true_vals = _contingency_table(pred, true)

    row_ind, col_ind = linear_sum_assignment(-table)
    matched = table[row_ind, col_ind].sum()
    accuracy = matched / len(true)
    return accuracy, table, pred_vals, true_vals
