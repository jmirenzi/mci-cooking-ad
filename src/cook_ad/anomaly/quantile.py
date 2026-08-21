from typing import NamedTuple

import numpy as np

# Every public function here takes already-normalized log-probability tables (the output of
# params.to_log_probs / recipe_hmm.to_log_probs, which floor via cook_ad.hsmm.params.FLOOR
# before their own log()). Nothing in this module takes a fresh log of a raw probability, so
# there is no second floor constant to introduce.


def _tail_threshold(scores, probs, alpha):
    """scores, probs: 1D arrays over one state's discrete support (aligned index-for-index).
    Returns the threshold t achieving the LARGEST cumulative tail mass P(score > t) that is
    still <= alpha -- the most sensitive threshold consistent with the alpha bound, for the
    "strict greater-than" flag rule used everywhere in surprise.flag. Zero-probability entries
    (e.g. the masked self-transition diagonal) carry no null mass and are dropped from the
    support before sorting.

    Step function, not exact: alpha cannot be hit exactly, only the nearest achievable point
    at or below it -- report achieved tail mass, never claim exact alpha coverage.
    """
    scores = np.asarray(scores, dtype=np.float64)
    probs = np.asarray(probs, dtype=np.float64)
    support = probs > 0
    scores = scores[support]
    probs = probs[support]

    order = np.argsort(-scores)
    scores = scores[order]
    probs = probs[order]

    # Collapse ties: a threshold set at a tied score excludes the WHOLE tie group under
    # strict '>', so the achievable tail mass is indexed by distinct score values, not by
    # raw position in the sorted array.
    uniq_scores, first_idx = np.unique(-scores, return_index=True)
    # np.unique sorts ascending on -scores, i.e. descending on scores -- matches `order` above.
    group_probs = np.add.reduceat(probs, first_idx)
    uniq_scores = -uniq_scores  # back to descending scores

    cum = np.cumsum(group_probs)
    # Largest m (number of leading groups EXCLUDED from the tail) with cumulative mass <= alpha.
    m = np.searchsorted(cum, alpha, side="right")
    m = min(m, len(uniq_scores) - 1)
    return float(uniq_scores[m])


def categorical_quantile_threshold(log_probs, alpha):
    """log_probs: (K, V) log P(token | state), already normalized/floored (params.to_log_probs).
    Returns (K,) thresholds t[k] such that P(-log P(token) > t[k] | Z=k) <= alpha, the largest
    achievable tail mass at or below alpha (see _tail_threshold), using the exact discrete tail.
    Mirrors durations.duration_tables in role, exact instead of NB-parametric."""
    log_probs = np.asarray(log_probs, dtype=np.float64)
    probs = np.exp(log_probs)
    scores = -log_probs
    k = log_probs.shape[0]
    return np.array([_tail_threshold(scores[i], probs[i], alpha) for i in range(k)])


def joint_quantile_threshold(log_emit_v, log_emit_n, alpha):
    """log_emit_v: (K, V), log_emit_n: (K, N). Returns (K,) thresholds for the joint surprise
    -log P(v, n | Z=k) = -log P(v|k) - log P(n|k), using the emissions' conditional
    independence given state. Built from the outer-product joint distribution per state (V*N
    support, a few hundred to a few thousand entries, cheap to sort). Used for s_emit."""
    log_emit_v = np.asarray(log_emit_v, dtype=np.float64)
    log_emit_n = np.asarray(log_emit_n, dtype=np.float64)
    k = log_emit_v.shape[0]
    thresholds = np.empty(k)
    for i in range(k):
        log_joint = log_emit_v[i][:, None] + log_emit_n[i][None, :]  # (V,N)
        probs = np.exp(log_joint).ravel()
        scores = -log_joint.ravel()
        thresholds[i] = _tail_threshold(scores, probs, alpha)
    return thresholds


def transition_quantile_threshold(log_trans, alpha):
    """log_trans: (K, K), self-transition already masked to -inf where applicable. Returns
    (K,) thresholds, one per from-state, over that state's transition row. Used for
    s_transition. Same function, applied to recipe_log_trans, covers s_recipe_transition."""
    log_trans = np.asarray(log_trans, dtype=np.float64)
    probs = np.exp(log_trans)
    scores = -log_trans
    k = log_trans.shape[0]
    return np.array([_tail_threshold(scores[i], probs[i], alpha) for i in range(k)])


def excess_quantile_threshold(log_trans_r, log_trans_marginal, alpha):
    """log_trans_r, log_trans_marginal: (K, K), both from-state rows sum to 1 (or are all
    -inf for a structurally banned row). Returns (K,) thresholds, one per from-state, for the
    joint model's repurposed s_recipe_transition channel: the SIGNED excess
    log P_marginal(i|j) - log P_r(i|j) for a transition j->i, NOT a neg-log-prob. Its exact
    null given from-state j is scored over i ~ P_r(.|j) (the recipe-conditioned transition
    actually taken), since that is the distribution the observed transition is drawn from.

    Unlike the other three quantile functions, this threshold CAN be negative (when the
    recipe-conditioned row is flatter than the marginal row, most transitions score below 0).
    Callers must mask non-boundary ticks explicitly rather than relying on a positive
    threshold making `0 > t` trivially False (see surprise.flag_joint)."""
    log_trans_r = np.asarray(log_trans_r, dtype=np.float64)
    log_trans_marginal = np.asarray(log_trans_marginal, dtype=np.float64)
    probs = np.exp(log_trans_r)
    # Structurally banned entries are -inf in both operands (-inf - -inf = NaN); harmless
    # since probs==0 there and _tail_threshold drops zero-probability entries before use, but
    # suppress the spurious warning explicitly rather than let it leak to callers.
    with np.errstate(invalid="ignore"):
        scores = log_trans_marginal - log_trans_r
    k = log_trans_r.shape[0]
    return np.array([_tail_threshold(scores[i], probs[i], alpha) for i in range(k)])


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
    recipe = excess_quantile_threshold(log_trans_r, log_trans_marginal, alpha)
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
                excess_quantile_threshold(log_trans_r, log_trans_marginal, alpha),
            )
        emit, verb, noun = self._emission[alpha]
        transition, recipe = self._recipe[key]
        return ThresholdTables(emit, verb, noun, transition, recipe)
