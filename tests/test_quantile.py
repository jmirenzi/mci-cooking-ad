import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from cook_ad.anomaly import surprise
from cook_ad.anomaly.quantile import (
    _tail_threshold,
    categorical_quantile_threshold,
    excess_quantile_threshold,
    joint_quantile_threshold,
    transition_quantile_threshold,
)
from cook_ad.eval import batch
from cook_ad.hsmm import joint_params, params
from cook_ad.synthetic import generate


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


def test_flag_calibrates_across_entropy():
    """Two states share one achievable score (-log(0.01) ~= 4.605 nats). State A collects six
    tied tokens at that score (collective mass 0.06 > alpha=0.05): a fixed 4.0-nat cutoff would
    flag all six, but the calibrated threshold correctly refuses (flagging any of them would
    exceed the alpha budget). State B has only ONE token at that score (individually well under
    alpha): the fixed cutoff flags it identically to state A, and the calibrated threshold
    agrees -- it's genuinely in state B's tail."""
    alpha = 0.05
    probs_a = np.array([0.01] * 6 + [0.94])
    probs_b = np.array([0.01] + [0.99 / 6] * 6)
    log_probs = np.log(np.stack([probs_a, probs_b]))

    t = categorical_quantile_threshold(log_probs, alpha)
    shared_score = -np.log(0.01)

    OLD_FIXED_THRESHOLD = 4.0  # the pre-calibration scalar this module replaces
    assert shared_score > OLD_FIXED_THRESHOLD  # old rule would have flagged both identically

    assert not (shared_score > t[0])  # state A: correctly refuses (over-budget tied group)
    assert shared_score > t[1]        # state B: correctly flags (genuinely rare there)


def _build_transition_trace(segments, log_trans):
    t_true = sum(d for _, d in segments)
    z_star = np.concatenate([np.full(d, s, dtype=np.int64) for s, d in segments])
    from_state = surprise._scatter_from_previous(segments, [s for s, _ in segments])
    from_recipe = np.full(t_true, -1, dtype=np.int64)
    s_transition, _ = surprise.transition_surprise(segments, log_trans)
    zeros = np.zeros(t_true)
    return surprise.SurpriseTrace(
        s_emit=zeros, s_verb=zeros, s_noun=zeros, s_temporal=zeros,
        s_dur_long=zeros, s_dur_short=zeros, s_dur_two=zeros,
        s_transition=s_transition, s_recipe_transition=zeros,
        pit=np.full(t_true, np.nan), z_star=z_star,
        expected_verb=np.zeros(t_true, dtype=np.int64), expected_noun=np.zeros(t_true, dtype=np.int64),
        expected_next_state=np.full(t_true, -1, dtype=np.int64),
        expected_next_recipe=np.full(t_true, -1, dtype=np.int64),
        attribution=np.full(t_true, "none", dtype=object),
        temporal_attribution=np.full(t_true, "none", dtype=object),
        from_state=from_state, from_recipe=from_recipe, belief_concentration=np.ones(t_true),
        pi_at_zstar=np.ones(t_true),
    )


def test_transition_threshold_indexed_by_from_state():
    """State 0's row is peaked with one genuinely rare target (prob 0.01 < alpha): that
    transition should flag. State 1's row is diffuse with no individually-rare target: the
    same-alpha calibrated threshold should NOT flag its (also somewhat surprising) transition.
    Mirrors test_transition_surprise_flags_unexpected_boundary's pattern but through
    surprise.flag(), confirming the FROM-state's own row quantile drives the channel."""
    K = 4
    V = N = 3
    alpha = 0.05
    trans = np.array([
        [0.0, 0.97, 0.02, 0.01],
        [0.34, 0.0, 0.33, 0.33],
        [0.34, 0.33, 0.0, 0.33],
        [0.34, 0.33, 0.33, 0.0],
    ])
    log_trans = np.log(trans, out=np.full_like(trans, -np.inf), where=trans > 0)
    uniform = np.log(np.full((K, V), 1.0 / V))
    log_probs = params.HSMMLogProbs(
        log_init=jnp.log(jnp.full(K, 1.0 / K)),
        log_trans=jnp.array(log_trans),
        log_emit_v=jnp.array(uniform),
        log_emit_n=jnp.array(uniform),
        log_dur_pmf=jnp.zeros((K, 5)),
        log_dur_survival=jnp.zeros((K, 5)),
    )
    recipe_log_trans = jnp.array(log_trans)  # unused: from_recipe stays -1 throughout

    trace_peaked = _build_transition_trace([(0, 3), (3, 2)], log_trans)
    flags_peaked = surprise.flag(trace_peaked, log_probs, recipe_log_trans, alpha=alpha)
    assert flags_peaked["s_transition"][3]  # FROM state0's rare target (prob 0.01) flags
    assert not flags_peaked["s_transition"][[0, 1, 2, 4]].any()  # non-boundary ticks never flag

    trace_diffuse = _build_transition_trace([(1, 3), (2, 2)], log_trans)
    flags_diffuse = surprise.flag(trace_diffuse, log_probs, recipe_log_trans, alpha=alpha)
    assert not flags_diffuse["s_transition"][3]  # FROM state1's common target does not flag
    assert not flags_diffuse["s_transition"][[0, 1, 2, 4]].any()


def test_flag_masks_non_boundary_ticks_under_negative_threshold():
    """Regression guard for the joint model's repurposed s_recipe_transition channel: its
    threshold can be negative (quantile.excess_quantile_threshold), so `0 > threshold` is no
    longer trivially False off segment boundaries the way it was under the old fixed-positive
    thresholds. Without the explicit from_state != -1 mask, every non-boundary tick (value 0)
    would wrongly flag whenever the threshold is negative."""
    K = 3
    V = N = 3
    alpha = 0.3
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
    uniform = np.log(np.full((K, V), 1.0 / V))

    # sanity: the excess threshold at from-state 2 really is negative for this construction,
    # so `0 > threshold` would be True there if the boundary mask were ever dropped.
    t = excess_quantile_threshold(log_trans_r, log_trans_marginal, alpha)
    assert t[2] < 0.0

    joint_log_probs = joint_params.JointHSMMLogProbs(
        log_pi=jnp.log(jnp.array([1.0])),
        log_init=jnp.array(np.log(np.full((1, K), 1.0 / K))),
        log_trans=jnp.array(log_trans_r)[None, :, :],
        log_emit_v=jnp.array(uniform),
        log_emit_n=jnp.array(uniform),
        log_dur_pmf=jnp.zeros((1, K, 5)),
        log_dur_survival=jnp.zeros((1, K, 5)),
    )
    r_hat = 0

    segments = [(2, 3), (0, 2)]  # boundary FROM state2 (the negative-threshold row) -> state0
    t_true = sum(d for _, d in segments)
    z_star = np.concatenate([np.full(d, s, dtype=np.int64) for s, d in segments])
    from_state = surprise._scatter_from_previous(segments, [s for s, _ in segments])
    from_recipe = np.full(t_true, -1, dtype=np.int64)
    s_transition, _ = surprise.transition_surprise(segments, log_trans_r)
    s_transition_marginal, _ = surprise.transition_surprise(segments, log_trans_marginal)
    s_recipe_transition = s_transition - s_transition_marginal
    zeros = np.zeros(t_true)
    trace = surprise.SurpriseTrace(
        s_emit=zeros, s_verb=zeros, s_noun=zeros, s_temporal=zeros,
        s_dur_long=zeros, s_dur_short=zeros, s_dur_two=zeros,
        s_transition=s_transition, s_recipe_transition=s_recipe_transition,
        pit=np.full(t_true, np.nan), z_star=z_star,
        expected_verb=np.zeros(t_true, dtype=np.int64), expected_noun=np.zeros(t_true, dtype=np.int64),
        expected_next_state=np.full(t_true, -1, dtype=np.int64),
        expected_next_recipe=np.full(t_true, -1, dtype=np.int64),
        attribution=np.full(t_true, "none", dtype=object),
        temporal_attribution=np.full(t_true, "none", dtype=object),
        from_state=from_state, from_recipe=from_recipe, belief_concentration=np.ones(t_true),
        pi_at_zstar=np.ones(t_true),
    )

    flags = surprise.flag_joint(trace, joint_log_probs, r_hat, jnp.array(log_trans_marginal), alpha=alpha)
    # every non-boundary tick has s_recipe_transition == 0; under the negative threshold at
    # from_state 2, an unmasked `0 > t[2]` would be True -- assert it never fires there.
    non_boundary = np.ones(t_true, dtype=bool)
    non_boundary[3] = False  # tick 3 is the only segment-start tick with a predecessor
    assert not flags["s_recipe_transition"][non_boundary].any()


def test_thresholds_override_rejects_recalibrated_channels():
    """The `thresholds` override now only accepts the two duration-channel scalars; a scalar
    override on any of the five per-state channels would reintroduce exactly the bug this
    module fixes, so it must raise rather than silently apply."""
    with pytest.raises(KeyError):
        surprise._duration_thresholds(alpha=surprise.DEFAULT_ALPHA, thresholds={"s_noun": 4.0})

    # the two duration channels are still overridable
    resolved = surprise._duration_thresholds(alpha=surprise.DEFAULT_ALPHA, thresholds={"s_temporal": 1.0})
    assert resolved["s_temporal"] == 1.0


def _peaked_joint_params(k_recipe=2, k=4, v=5, n=5, d_max=15):
    """Sharp per-state verb/noun tables (shared across recipes, like the real joint model) and
    a distinct per-recipe transition structure -- enough for compute_traces_joint to produce a
    genuine, non-degenerate segmentation."""
    per_recipe = [
        params.init_weak_limit_params(jax.random.PRNGKey(r), k, v, n, d_max) for r in range(k_recipe)
    ]
    verb = jnp.full((k, v), 0.2).at[jnp.arange(k), jnp.arange(k) % v].set(200.0)
    noun = jnp.full((k, n), 0.2).at[jnp.arange(k), jnp.arange(k) % n].set(200.0)
    return joint_params.JointHSMMParams(
        init_counts=jnp.stack([p.init_counts for p in per_recipe]),
        trans_counts=jnp.stack([p.trans_counts for p in per_recipe]),
        verb_counts=verb,
        noun_counts=noun,
        dur_r=jnp.stack([jnp.full((k,), 6.0) for _ in per_recipe]),
        dur_p=jnp.stack([jnp.full((k,), 0.5) for _ in per_recipe]),
        pi_counts=jnp.ones(k_recipe),
    )


def test_compute_traces_joint_end_to_end_flag_joint():
    """Closes a real coverage gap: nothing previously exercised eval.batch.compute_traces_joint
    together with flag_joint on a genuine assemble_trace_joint output -- the joint model's flag
    path was covered only by hand-built synthetic traces above. Runs the actual joint driver
    end to end and checks the dilution-correction machinery (pi_at_zstar, emission_thresholds)
    behaves sanely on real (non-hand-built) output: pi_at_zstar is a valid probability and is
    always <= belief_concentration (the mixture weight AT z_star can never exceed the mixture's
    OWN max weight, by definition)."""
    jp = _peaked_joint_params()
    rng = np.random.default_rng(3)
    trials = generate.generate_healthy_joint(jp, n=4, rng=rng, max_ticks=60, d_max=15)

    traces, log_probs, r_hat, log_trans_marginal = batch.compute_traces_joint(jp, trials, d_max=15)
    assert len(traces) == 4

    for i, trace in enumerate(traces):
        flags = surprise.flag_joint(trace, log_probs, int(r_hat[i]), log_trans_marginal)
        assert set(flags) == set(surprise.CHANNELS)
        assert np.all((trace.pi_at_zstar > 0) & (trace.pi_at_zstar <= 1.0 + 1e-9))
        assert np.all(trace.pi_at_zstar <= trace.belief_concentration + 1e-9)
        assert np.all(trace.from_recipe == -1)  # joint trace never populates from_recipe
