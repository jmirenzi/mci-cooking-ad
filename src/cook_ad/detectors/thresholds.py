"""class BadSets: the alpha-tail sets every HSMM channel is tested against (docs/detectors.md §2).

A bad set is a set of outcomes carrying at most alpha of the model's own probability. It is a
property of the model; data only chooses alpha (and kappa).
"""
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from cook_ad.hsmm import durations


# ---------------------------------------------------------------------------------------------
# Exact discrete tails


def tail_threshold(scores, probs, alpha):
    """Smallest t with P(score > t) <= alpha over a discrete support; ties stay together."""
    scores = np.asarray(scores, dtype=np.float64)
    probs = np.asarray(probs, dtype=np.float64)
    support = probs > 0
    scores, probs = scores[support], probs[support]
    order = np.argsort(-scores)
    scores, probs = scores[order], probs[order]
    uniq_scores, first_idx = np.unique(-scores, return_index=True)   # descending scores
    group_probs = np.add.reduceat(probs, first_idx)
    uniq_scores = -uniq_scores
    m = np.searchsorted(np.cumsum(group_probs), alpha, side="right")
    return float(uniq_scores[min(m, len(uniq_scores) - 1)])


def categorical_quantile_threshold(log_probs, alpha):
    """(K,) thresholds on -log P(x | k), one per row of a (K, X) log table."""
    log_probs = np.asarray(log_probs, dtype=np.float64)
    probs, scores = np.exp(log_probs), -log_probs
    return np.array([tail_threshold(scores[i], probs[i], alpha) for i in range(log_probs.shape[0])])


def joint_quantile_threshold(log_emit_v, log_emit_n, alpha):
    """(K,) thresholds on -log P(v, n | k) = -log P(v | k) - log P(n | k)."""
    log_emit_v = np.asarray(log_emit_v, dtype=np.float64)
    log_emit_n = np.asarray(log_emit_n, dtype=np.float64)
    out = np.empty(log_emit_v.shape[0])
    for i in range(log_emit_v.shape[0]):
        log_joint = log_emit_v[i][:, None] + log_emit_n[i][None, :]
        out[i] = tail_threshold(-log_joint.ravel(), np.exp(log_joint).ravel(), alpha)
    return out


def transition_quantile_threshold(log_A, alpha):
    """(K,) thresholds on -log A[k, i], one per from-state k."""
    return categorical_quantile_threshold(log_A, alpha)


def excess_quantile_threshold(log_A_r, log_A_marginal, alpha):
    """(K,) thresholds on the recipe excess log A_marg[k, i] - log A_r[k, i], i ~ A_r[k, :]."""
    log_A_r = np.asarray(log_A_r, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        scores = np.asarray(log_A_marginal, dtype=np.float64) - log_A_r
    probs = np.exp(log_A_r)
    return np.array([tail_threshold(scores[i], probs[i], alpha) for i in range(log_A_r.shape[0])])


def recipe_excess_thresholds(log_A_r, log_A_marginal, alpha, k_recipe):
    """excess_quantile_threshold, or +inf when there is one recipe (docs/detectors.md §2.3)."""
    if k_recipe == 1:
        return np.full(np.asarray(log_A_r).shape[0], np.inf)
    return excess_quantile_threshold(log_A_r, log_A_marginal, alpha)


# ---------------------------------------------------------------------------------------------
# The sets


class HindsightThresholds(NamedTuple):
    """Per-state thresholds for one (r_hat, alpha); flag where the surprise exceeds them."""
    emit: np.ndarray        # (K,)
    verb: np.ndarray        # (K,)
    noun: np.ndarray        # (K,)
    transition: np.ndarray  # (K,) by from-state
    recipe: np.ndarray      # (K,) by from-state


def first_true(masks):
    """(G, ...) nested bool masks -> index of the first True along axis 0 (G where none)."""
    masks = np.asarray(masks)
    if not np.all(masks[:-1] <= masks[1:]):
        raise ValueError("bad sets are not nested in alpha -- the binning would be wrong")
    return np.where(masks.any(axis=0), masks.argmax(axis=0), masks.shape[0]).astype(np.int32)


class BadSets:
    """Bad sets of one model over an alpha grid.

    Live/Lag read them as bins: a cell's bin is the first grid alpha at which it is bad, so one
    segment-sum gives the bad mass at every alpha. Hindsight reads per-(r_hat, alpha) thresholds."""

    def __init__(self, hsmm, alphas):
        self.hsmm = hsmm
        self.alphas = np.sort(np.asarray(alphas, dtype=np.float64))
        t = hsmm.tables
        self.log_A_marginal = np.asarray(t.log_A_marginal, dtype=np.float64)
        self.stall_bin = jnp.asarray(self._stall_bin())
        self.early_bin = jnp.asarray(self._early_bin())
        trans, recipe = self._transition_bins()
        self.trans_bin, self.recipe_bin = jnp.asarray(trans), jnp.asarray(recipe)
        self._emission, self._by_recipe = {}, {}

    def _stall_bin(self):
        """Elapsed j+1 is a stall at alpha when P(D >= j+1) < alpha (docs/detectors.md §3.3)."""
        surv = np.asarray(self.hsmm.tables.log_surv, dtype=np.float64)
        return np.searchsorted(self.alphas, np.exp(surv), side="right").astype(np.int32)

    def _early_bin(self):
        """A segment ending after d ticks is early at alpha when P(D <= d) < alpha / 2."""
        p = self.hsmm.params
        d = np.arange(1, self.hsmm.d_max + 1, dtype=np.float64)
        log_cdf = durations.nb_log_cdf_np(d[None, None, :],
                                          np.asarray(p.dur_r, dtype=np.float64)[..., None],
                                          np.asarray(p.dur_p, dtype=np.float64)[..., None])
        return np.searchsorted(self.alphas, 2.0 * np.exp(log_cdf), side="right").astype(np.int32)

    def _transition_bins(self):
        """(K_R, K, K) bins for k -> i in the alpha-tail of A[r, k, :] and of r's excess."""
        log_A = np.asarray(self.hsmm.tables.log_A, dtype=np.float64)
        k_r = log_A.shape[0]
        trans = np.zeros((len(self.alphas),) + log_A.shape, dtype=bool)
        recipe = np.zeros_like(trans)
        with np.errstate(invalid="ignore"):
            excess = self.log_A_marginal[None] - log_A
        real = np.isfinite(log_A)
        for g, a in enumerate(self.alphas):
            for r in range(k_r):
                tau = transition_quantile_threshold(log_A[r], a)
                trans[g, r] = real[r] & (-log_A[r] > tau[:, None])
                tau = recipe_excess_thresholds(log_A[r], self.log_A_marginal, a, k_r)
                recipe[g, r] = real[r] & (excess[r] > tau[:, None])
        return first_true(trans), first_true(recipe)

    def alpha_index(self, alpha):
        """Position of `alpha` in the grid (it must be one of the grid's values)."""
        hit = np.flatnonzero(np.isclose(self.alphas, alpha))
        if not hit.size:
            raise ValueError(f"alpha={alpha} is not on this BadSets grid {list(self.alphas)}")
        return int(hit[0])

    def hindsight(self, r_hat, alpha) -> HindsightThresholds:
        """Thresholds for one trial's recipe r_hat at alpha (memoised)."""
        t, r = self.hsmm.tables, int(r_hat)
        if alpha not in self._emission:
            self._emission[alpha] = (
                joint_quantile_threshold(t.log_emit_v, t.log_emit_n, alpha),
                categorical_quantile_threshold(t.log_emit_v, alpha),
                categorical_quantile_threshold(t.log_emit_n, alpha))
        if (alpha, r) not in self._by_recipe:
            log_A_r = t.log_A[r]
            self._by_recipe[alpha, r] = (
                transition_quantile_threshold(log_A_r, alpha),
                recipe_excess_thresholds(log_A_r, self.log_A_marginal, alpha,
                                         self.hsmm.k_recipe))
        return HindsightThresholds(*self._emission[alpha], *self._by_recipe[alpha, r])
