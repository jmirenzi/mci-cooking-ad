from typing import NamedTuple

import numpy as np

from cook_ad.detectors import thresholds as _t
from cook_ad.hsmm.params import FLOOR

# Every public function here takes already-normalized log-probability tables (the output of
# params.to_log_probs / recipe_hmm.to_log_probs, which floor via cook_ad.hsmm.params.FLOOR
# before their own log()). Nothing in this module takes a fresh log of a raw probability, so
# there is no second floor constant to introduce.


# Restructure shims: the joint-model tails live in detectors/thresholds.py. The pair, sequence and
# cascade helpers below are deleted in Phase E.
_tail_threshold = _t.tail_threshold
categorical_quantile_threshold = _t.categorical_quantile_threshold
joint_quantile_threshold = _t.joint_quantile_threshold
transition_quantile_threshold = _t.transition_quantile_threshold
excess_quantile_threshold = _t.excess_quantile_threshold


def recipe_excess_thresholds(joint_log_probs, log_trans_r, log_trans_marginal, alpha):
    return _t.recipe_excess_thresholds(log_trans_r, log_trans_marginal, alpha,
                                       joint_log_probs.log_trans.shape[0])


def pair_quantile_threshold(pi_all, log_emit_v, log_emit_n, alpha, chunk=64):
    """Per-tick (T,) thresholds for the pair channel s_pair = s_emit - s_verb - s_noun.

    Per tick, not per state, and it has to be: under a single state the emission is a product by
    construction, so PMI(v,n | Z=k) is identically 0 and a per-state null would be degenerate.
    All of s_pair's signal comes from the predictive MIXTURE, which is not a product even though
    every component is. Exact discrete tail as elsewhere in this module, over the tick's (V,N)
    mixture joint. O(T * V * N) with one sort per tick, so this channel dominates the runtime of
    a sweep.
    """
    pi_all = np.asarray(pi_all, dtype=np.float64)
    p_v = np.exp(np.asarray(log_emit_v, dtype=np.float64))   # (K,V)
    p_n = np.exp(np.asarray(log_emit_n, dtype=np.float64))   # (K,N)
    out = np.empty(pi_all.shape[0], dtype=np.float64)

    for start in range(0, pi_all.shape[0], chunk):
        w = np.exp(pi_all[start : start + chunk])                    # (t,K)
        joint = np.einsum("tk,kv,kn->tvn", w, p_v, p_n)              # (t,V,N) mixture joint
        joint = np.maximum(joint, 0.0)
        total = joint.sum(axis=(1, 2), keepdims=True)
        joint = joint / np.maximum(total, FLOOR)                     # pi_all may not sum to 1 exactly
        marg_v = joint.sum(axis=2)[:, :, None]
        marg_n = joint.sum(axis=1)[:, None, :]
        # -PMI, so larger = less compatible, matching every other channel's convention.
        with np.errstate(divide="ignore", invalid="ignore"):
            score = np.log(np.maximum(marg_v, FLOOR)) + np.log(np.maximum(marg_n, FLOOR)) \
                - np.log(np.maximum(joint, FLOOR))
        for i in range(joint.shape[0]):
            out[start + i] = _tail_threshold(score[i].ravel(), joint[i].ravel(), alpha)
    return out


class SequenceThresholds(NamedTuple):
    transposition: float  # nats
    repetition: float     # duration ratio


def sequence_thresholds(transposition_gains, repetition_ratios, alpha):
    """Empirical (1 - alpha) quantile thresholds for anomaly/sequence.py's two magnitude tests,
    calibrated the same way every other channel is: collect the statistic's distribution over
    HEALTHY trials and take the largest achievable tail mass <= alpha (_tail_threshold) -- here
    over an EMPIRICAL sample rather than a fitted discrete distribution, so each observed value
    gets equal mass 1/N. `_tail_threshold` then returns the empirical (1 - alpha) quantile under
    the same strict '>' flag convention used everywhere else in this module.

    The omission test has no threshold here: it reuses `narrate.missing_step`'s own fixed
    `min_gain` bridging-gain gate directly, not a healthy-trial quantile (anomaly/sequence.py).
    """
    def _empirical(values):
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            return float("inf")
        probs = np.full(values.shape, 1.0 / values.size)
        return _tail_threshold(values, probs, alpha)

    return SequenceThresholds(
        transposition=_empirical(transposition_gains),
        repetition=_empirical(repetition_ratios),
    )


class ThresholdTables(NamedTuple):
    emit: np.ndarray        # (K,)
    verb: np.ndarray        # (K,)
    noun: np.ndarray        # (K,)
    transition: np.ndarray  # (K,), indexed by FROM-state
    recipe: np.ndarray      # (K,) cascade: indexed by FROM-recipe; joint: indexed by FROM-state


def threshold_tables(log_probs, recipe_log_trans, alpha):
    """Cascade-model bundle: all five quantile tables from one HSMMLogProbs plus the recipe
    transition matrix, at a given alpha."""
    emit = joint_quantile_threshold(log_probs.log_emit_v, log_probs.log_emit_n, alpha)
    verb = categorical_quantile_threshold(log_probs.log_emit_v, alpha)
    noun = categorical_quantile_threshold(log_probs.log_emit_n, alpha)
    transition = transition_quantile_threshold(log_probs.log_trans, alpha)
    recipe = transition_quantile_threshold(recipe_log_trans, alpha)
    return ThresholdTables(emit, verb, noun, transition, recipe)


def threshold_tables_joint(joint_log_probs, r_hat, log_trans_marginal, alpha):
    """Joint-model bundle for one trial's MAP recipe r_hat: emission tables are shared
    (unindexed by recipe), s_transition's table comes from that recipe's own transition row,
    and the repurposed s_recipe_transition channel uses excess_quantile_threshold against the
    pi-weighted marginal (joint_params.marginal_log_trans)."""
    log_trans_r = joint_log_probs.log_trans[r_hat]
    emit = joint_quantile_threshold(joint_log_probs.log_emit_v, joint_log_probs.log_emit_n, alpha)
    verb = categorical_quantile_threshold(joint_log_probs.log_emit_v, alpha)
    noun = categorical_quantile_threshold(joint_log_probs.log_emit_n, alpha)
    transition = transition_quantile_threshold(log_trans_r, alpha)
    recipe = recipe_excess_thresholds(joint_log_probs, log_trans_r, log_trans_marginal, alpha)
    return ThresholdTables(emit, verb, noun, transition, recipe)


class JointThresholdCache:
    """Memoised `threshold_tables_joint`, for sweeps that re-flag one model many times.

    The three emission tables depend only on alpha, not on r_hat, yet are rebuilt on every call
    and `joint_quantile_threshold` sorts a V*N support K times each; a sweep of A alphas x N
    trials x G groups pays that A*N*G times for A distinct values. Cache key is (alpha, r_hat)
    only, so build one per fitted model -- cross-model reuse is rejected below.
    """

    def __init__(self, joint_log_probs, log_trans_marginal):
        self._lp = joint_log_probs
        self._ltm = log_trans_marginal
        self._emission = {}   # alpha -> (emit, verb, noun)
        self._recipe = {}     # (alpha, r_hat) -> (transition, recipe)

    def tables(self, joint_log_probs, r_hat, log_trans_marginal, alpha):
        if joint_log_probs is not self._lp or log_trans_marginal is not self._ltm:
            raise ValueError(
                "JointThresholdCache was built for a different model's tables; build a new "
                "cache per fitted model rather than reusing one across refits."
            )
        r_hat = int(r_hat)
        if alpha not in self._emission:
            self._emission[alpha] = (
                joint_quantile_threshold(joint_log_probs.log_emit_v, joint_log_probs.log_emit_n, alpha),
                categorical_quantile_threshold(joint_log_probs.log_emit_v, alpha),
                categorical_quantile_threshold(joint_log_probs.log_emit_n, alpha),
            )
        key = (alpha, r_hat)
        if key not in self._recipe:
            log_trans_r = joint_log_probs.log_trans[r_hat]
            self._recipe[key] = (
                transition_quantile_threshold(log_trans_r, alpha),
                recipe_excess_thresholds(joint_log_probs, log_trans_r, log_trans_marginal,
                                         alpha),
            )
        emit, verb, noun = self._emission[alpha]
        transition, recipe = self._recipe[key]
        return ThresholdTables(emit, verb, noun, transition, recipe)
