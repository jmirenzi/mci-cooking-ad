import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cook_ad.hsmm import durations, em, joint_em, joint_params, params
from cook_ad.recipe import recipe_hmm
from cook_ad.synthetic import generate

jax.config.update("jax_enable_x64", True)

K, V, N, D_MAX = 4, 5, 6, 10


def _random_sequences(rng, n_trials=6, t_range=(10, 20)):
    sequences = []
    for _ in range(n_trials):
        t = rng.integers(*t_range)
        sequences.append({
            "verb_ids": rng.integers(0, V, size=t).tolist(),
            "noun_ids": rng.integers(0, N, size=t).tolist(),
        })
    return sequences


def _single_recipe_joint_params(hp: params.HSMMParams) -> joint_params.JointHSMMParams:
    """K_R=1 joint params wrapping a single HSMMParams -- the equivalence fixture used by
    several tests below."""
    return joint_params.JointHSMMParams(
        init_counts=hp.init_counts[None, :],
        trans_counts=hp.trans_counts[None, :, :],
        verb_counts=hp.verb_counts,
        noun_counts=hp.noun_counts,
        dur_r=hp.dur_r[None, :],
        dur_p=hp.dur_p[None, :],
        pi_counts=jnp.array([1.0]),
    )


def test_to_log_probs_joint_matches_single_recipe_slice():
    """Each recipe's slice of JointHSMMLogProbs must equal to_log_probs of an HSMMParams built
    from that recipe alone -- catches the K_R-axis vmap bugs the plan calls out explicitly
    (trans's mask_diag eye(), duration_tables' 1-D dur_r assumption)."""
    k_r = 3
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, k_r)
    per_recipe = [params.init_weak_limit_params(keys[r], K, V, N, D_MAX) for r in range(k_r)]

    jp = joint_params.JointHSMMParams(
        init_counts=jnp.stack([p.init_counts for p in per_recipe]),
        trans_counts=jnp.stack([p.trans_counts for p in per_recipe]),
        verb_counts=per_recipe[0].verb_counts,
        noun_counts=per_recipe[0].noun_counts,
        dur_r=jnp.stack([p.dur_r for p in per_recipe]),
        dur_p=jnp.stack([p.dur_p for p in per_recipe]),
        pi_counts=jnp.ones(k_r),
    )
    joint_log_probs = joint_params.to_log_probs_joint(jp, D_MAX)

    for r in range(k_r):
        hp_r = params.HSMMParams(
            per_recipe[r].init_counts, per_recipe[r].trans_counts,
            per_recipe[0].verb_counts, per_recipe[0].noun_counts,
            per_recipe[r].dur_r, per_recipe[r].dur_p,
        )
        expected = params.to_log_probs(hp_r, D_MAX)
        assert jnp.allclose(joint_log_probs.log_init[r], expected.log_init, atol=1e-8)
        assert jnp.allclose(joint_log_probs.log_trans[r], expected.log_trans, atol=1e-8)
        assert jnp.allclose(joint_log_probs.log_dur_pmf[r], expected.log_dur_pmf, atol=1e-8)
        assert jnp.allclose(joint_log_probs.log_dur_survival[r], expected.log_dur_survival, atol=1e-8)

    expected_emit = params.to_log_probs(
        params.HSMMParams(per_recipe[0].init_counts, per_recipe[0].trans_counts,
                           per_recipe[0].verb_counts, per_recipe[0].noun_counts,
                           per_recipe[0].dur_r, per_recipe[0].dur_p), D_MAX,
    )
    assert jnp.allclose(joint_log_probs.log_emit_v, expected_emit.log_emit_v, atol=1e-8)
    assert jnp.allclose(joint_log_probs.log_emit_n, expected_emit.log_emit_n, atol=1e-8)


def test_k_r_1_e_step_matches_cascade_em():
    """The strongest regression test: at K_R=1, the joint E-step's weighted sufficient stats
    and objective must equal hsmm.em.e_step's, since rho is forced to 1 for the only recipe.
    """
    rng = np.random.default_rng(0)
    sequences = _random_sequences(rng)
    key = jax.random.PRNGKey(1)
    hp = params.init_weak_limit_params(key, K, V, N, D_MAX)
    verb_ids, noun_ids, mask = em.pad_batch(sequences)

    stats_single, ll_single = em.e_step(hp, verb_ids, noun_ids, mask, D_MAX, chunk_size=4)
    jp = _single_recipe_joint_params(hp)
    stats_joint, ll_joint = joint_em.e_step(jp, verb_ids, noun_ids, mask, D_MAX, chunk_size=4)

    assert jnp.allclose(ll_single, ll_joint, atol=1e-6)
    assert jnp.allclose(stats_single["init_counts"], stats_joint["init_counts"][0], atol=1e-6)
    assert jnp.allclose(stats_single["trans_counts"], stats_joint["trans_counts"][0], atol=1e-6)
    assert jnp.allclose(stats_single["verb_counts"], stats_joint["verb_counts"], atol=1e-6)
    assert jnp.allclose(stats_single["noun_counts"], stats_joint["noun_counts"], atol=1e-6)
    assert jnp.allclose(stats_single["xi_dur"], stats_joint["xi_dur"][0], atol=1e-6)
    assert jnp.allclose(stats_single["cens"], stats_joint["cens"][0], atol=1e-6)
    assert jnp.isclose(stats_joint["pi_counts"][0], len(sequences), atol=1e-6)


def test_k_r_1_m_step_matches_cascade_em():
    rng = np.random.default_rng(0)
    sequences = _random_sequences(rng)
    key = jax.random.PRNGKey(2)
    hp = params.init_weak_limit_params(key, K, V, N, D_MAX)
    verb_ids, noun_ids, mask = em.pad_batch(sequences)

    stats_single, _ = em.e_step(hp, verb_ids, noun_ids, mask, D_MAX, chunk_size=4)
    hp2 = em.m_step(hp, stats_single, 0.5, 0.5, float(V), float(N), D_MAX)

    jp = _single_recipe_joint_params(hp)
    stats_joint, _ = joint_em.e_step(jp, verb_ids, noun_ids, mask, D_MAX, chunk_size=4)
    jp2, _global_r, _global_p = joint_em.m_step(
        jp, stats_joint, alpha_pi=1.0, alpha_init=0.5, alpha_trans=0.5,
        alpha_emit_v=float(V), alpha_emit_n=float(N), kappa=0.0, d_max=D_MAX,
    )

    assert jnp.allclose(hp2.init_counts, jp2.init_counts[0], atol=1e-6)
    assert jnp.allclose(hp2.trans_counts, jp2.trans_counts[0], atol=1e-6)
    assert jnp.allclose(hp2.verb_counts, jp2.verb_counts, atol=1e-6)
    assert jnp.allclose(hp2.noun_counts, jp2.noun_counts, atol=1e-6)
    # kappa=0.0 at K_R=1: fit_durations_shrunk's shrink step is a no-op, so this should
    # reproduce em.py's plain censoring-imputation + Newton duration fit exactly.
    assert jnp.allclose(hp2.dur_r, jp2.dur_r[0], atol=1e-6)
    assert jnp.allclose(hp2.dur_p, jp2.dur_p[0], atol=1e-6)


def test_fit_durations_shrunk_kappa_zero_matches_plain_fit():
    key = jax.random.PRNGKey(3)
    ks = jax.random.split(key, 4)
    xi_dur = jax.random.uniform(ks[0], (1, K, D_MAX)) * 5
    cens = jax.random.uniform(ks[1], (1, K, D_MAX)) * 0.5
    r_old = jax.random.uniform(ks[2], (1, K), minval=1.0, maxval=5.0)
    p_old = jax.random.uniform(ks[3], (1, K), minval=0.2, maxval=0.8)

    dur_r, dur_p, global_r, global_p = durations.fit_durations_shrunk(xi_dur, cens, r_old, p_old, D_MAX, kappa=0.0)

    n_hat = durations.impute_censored_histogram(xi_dur[0], cens[0], r_old[0], p_old[0], D_MAX)
    n_tot, s_hat = durations.duration_stats_from_histogram(n_hat, D_MAX)
    r_ref = durations.newton_update_r(n_hat, n_tot, s_hat, r_old[0], n_iters=5)
    p_ref = durations.update_p_given_r(n_tot, s_hat, r_ref)

    assert jnp.allclose(dur_r[0], r_ref, atol=1e-6)
    assert jnp.allclose(dur_p[0], p_ref, atol=1e-6)
    # global_r/global_p are the pooled fit BEFORE shrinkage/re-fit; at K_R=1 that pooled fit
    # is over the same single cell's histogram (no other recipes to pool with).
    assert global_r.shape == (K,)
    assert global_p.shape == (K,)


def test_global_damping_reduces_swing_and_matches_ema_formula():
    """Regression guard for the mechanism found on a real full-scale checkpoint: a near-empty
    state's pooled global duration fit can swing by an order of magnitude between M-steps
    (observed directly: r=471->38 in one call), and because it's shared via kappa*pmf_global
    across every recipe's copy of that state, the swing perturbs K_R cells at once. Simulates
    two consecutive M-step calls with DIFFERENT sparse histograms for the same state (standing
    in for the noisy pooled data at real scale) and checks: (1) global_damping=0.0 exactly
    matches the undamped/legacy fit (already covered above, re-asserted here for the two-call
    case), (2) damping>0 moves LESS from the previous global fit than the undamped fit does,
    (3) the damped result matches the documented EMA formula exactly, not just qualitatively."""
    key = jax.random.PRNGKey(11)
    ks = jax.random.split(key, 4)

    # call 1: establish a baseline global fit from one sparse histogram
    xi_dur_1 = jax.random.uniform(ks[0], (1, K, D_MAX)) * 0.3  # sparse: near-empty state
    cens_1 = jnp.zeros((1, K, D_MAX))
    r_old = jnp.full((1, K), 5.0)
    p_old = jnp.full((1, K), 0.5)
    _, _, global_r_1, global_p_1 = durations.fit_durations_shrunk(
        xi_dur_1, cens_1, r_old, p_old, D_MAX, kappa=5.0,
    )

    # call 2: a DELIBERATELY DIFFERENT sparse histogram for the same state (simulates the
    # E-step's tiny amount of real data shifting between iterations)
    xi_dur_2 = jax.random.uniform(ks[1], (1, K, D_MAX)) * 0.3 + 2.0
    cens_2 = jnp.zeros((1, K, D_MAX))

    _, _, global_r_2_undamped, global_p_2_undamped = durations.fit_durations_shrunk(
        xi_dur_2, cens_2, global_r_1[None, :], global_p_1[None, :], D_MAX, kappa=5.0,
    )
    damping = 0.7
    _, _, global_r_2_damped, global_p_2_damped = durations.fit_durations_shrunk(
        xi_dur_2, cens_2, global_r_1[None, :], global_p_1[None, :], D_MAX, kappa=5.0,
        prev_global_r=global_r_1, prev_global_p=global_p_1, global_damping=damping,
    )

    swing_undamped = jnp.abs(global_r_2_undamped - global_r_1)
    swing_damped = jnp.abs(global_r_2_damped - global_r_1)
    assert jnp.all(swing_damped <= swing_undamped + 1e-6), (
        "damping should never move the global fit FARTHER from the previous estimate than the undamped fit"
    )
    assert jnp.mean(swing_damped) < jnp.mean(swing_undamped), "damping should measurably reduce the swing on average"

    # exact EMA formula check, not just directional
    expected_r = damping * global_r_1 + (1.0 - damping) * global_r_2_undamped
    expected_p = damping * global_p_1 + (1.0 - damping) * global_p_2_undamped
    assert jnp.allclose(global_r_2_damped, expected_r, atol=1e-6)
    assert jnp.allclose(global_p_2_damped, expected_p, atol=1e-6)


def test_objective_approximately_non_decreasing():
    """The shrinkage duration fit is not an exact M-step, so small dips are legitimate (the
    reason run_joint_em warns rather than asserts) -- but any systematic/large decrease would
    mean the weighted sufficient stats are wrong. Bound total backward slack loosely."""
    rng = np.random.default_rng(0)
    sequences = _random_sequences(rng, n_trials=8)
    key = jax.random.PRNGKey(4)
    k_r = 2
    keys = jax.random.split(key, k_r)
    per_recipe = [params.init_weak_limit_params(keys[r], K, V, N, D_MAX) for r in range(k_r)]
    jp0 = joint_params.JointHSMMParams(
        init_counts=jnp.stack([p.init_counts for p in per_recipe]),
        trans_counts=jnp.stack([p.trans_counts for p in per_recipe]),
        verb_counts=per_recipe[0].verb_counts,
        noun_counts=per_recipe[0].noun_counts,
        dur_r=jnp.stack([p.dur_r for p in per_recipe]),
        dur_p=jnp.stack([p.dur_p for p in per_recipe]),
        pi_counts=jnp.ones(k_r),
    )

    best, obj, history, converged = joint_em.run_joint_em(
        jp0, sequences, D_MAX, alpha_pi=1.0, kappa=1.0, max_iters=20, tol=1e-4, chunk_size=4,
    )

    decreases = [history[i] - history[i + 1] for i in range(len(history) - 1) if history[i + 1] < history[i]]
    assert not decreases or max(decreases) < 1.0, f"large objective decrease found: {decreases}"


def test_recipe_recovery_from_well_separated_dynamics():
    """Two recipes sharing emissions but with near-deterministic, opposite-direction subtask
    cycles should be recovered by joint EM (started from the true generative params, a
    warm-ish start standing in for the cascade warm start) with high recipe ARI."""
    k = 4
    key = jax.random.PRNGKey(5)
    shared = params.init_weak_limit_params(key, k, V, N, D_MAX, alpha_emit_v=float(V), alpha_emit_n=float(N))

    forward = jnp.zeros((k, k)).at[jnp.arange(k), (jnp.arange(k) + 1) % k].set(10.0)
    backward = jnp.zeros((k, k)).at[jnp.arange(k), (jnp.arange(k) - 1) % k].set(10.0)
    hp1 = params.HSMMParams(shared.init_counts, forward, shared.verb_counts, shared.noun_counts,
                             shared.dur_r, shared.dur_p)
    hp2 = params.HSMMParams(shared.init_counts, backward, shared.verb_counts, shared.noun_counts,
                             shared.dur_r, shared.dur_p)

    np_rng = np.random.default_rng(7)
    sequences, true_recipes = [], []
    for i in range(16):
        hp = hp1 if i % 2 == 0 else hp2
        traj = generate.sample_trajectory(hp, np_rng, max_ticks=30, d_max=D_MAX)
        sequences.append({"verb_ids": traj["verb_ids"].tolist(), "noun_ids": traj["noun_ids"].tolist()})
        true_recipes.append(0 if i % 2 == 0 else 1)

    jp0 = joint_params.JointHSMMParams(
        init_counts=jnp.stack([hp1.init_counts, hp2.init_counts]),
        trans_counts=jnp.stack([hp1.trans_counts, hp2.trans_counts]),
        verb_counts=shared.verb_counts,
        noun_counts=shared.noun_counts,
        dur_r=jnp.stack([hp1.dur_r, hp2.dur_r]),
        dur_p=jnp.stack([hp1.dur_p, hp2.dur_p]),
        pi_counts=jnp.array([1.0, 1.0]),
    )

    best, obj, history, converged = joint_em.run_joint_em(
        jp0, sequences, D_MAX, alpha_pi=1.0, kappa=1.0, max_iters=25, tol=1e-4, chunk_size=4,
    )

    verb_ids, noun_ids, mask = joint_em.pad_batch(sequences)
    r_hat, rho, trial_ll = joint_em.infer_recipe(best, verb_ids, noun_ids, mask, D_MAX, chunk_size=4)

    ari = recipe_hmm.adjusted_rand(np.asarray(r_hat), np.asarray(true_recipes))
    assert ari > 0.8, f"expected near-perfect recipe recovery, got ARI={ari}"


def test_dead_recipe_stays_finite():
    """A recipe with near-zero pi_counts and (after the E-step) near-zero responsibility must
    stay finite through the M-step: init/trans via the Dirichlet floor, durations via the
    kappa*pmf_global shrinkage term."""
    rng = np.random.default_rng(0)
    sequences = _random_sequences(rng, n_trials=6)
    key = jax.random.PRNGKey(6)
    hp = params.init_weak_limit_params(key, K, V, N, D_MAX)

    k_r = 3
    init_counts = jnp.tile(hp.init_counts[None, :], (k_r, 1))
    trans_counts = jnp.tile(hp.trans_counts[None, :, :], (k_r, 1, 1))
    dur_r = jnp.tile(hp.dur_r[None, :], (k_r, 1))
    dur_p = jnp.tile(hp.dur_p[None, :], (k_r, 1))
    pi_counts = jnp.array([5.0, 5.0, 1e-6])  # third recipe effectively dead from iteration 0

    jp = joint_params.JointHSMMParams(init_counts, trans_counts, hp.verb_counts, hp.noun_counts,
                                       dur_r, dur_p, pi_counts)
    verb_ids, noun_ids, mask = joint_em.pad_batch(sequences)
    stats, _ = joint_em.e_step(jp, verb_ids, noun_ids, mask, D_MAX, chunk_size=4)
    jp2, _global_r, _global_p = joint_em.m_step(
        jp, stats, alpha_pi=1.0, alpha_init=0.5, alpha_trans=0.5,
        alpha_emit_v=float(V), alpha_emit_n=float(N), kappa=5.0, d_max=D_MAX,
    )

    for field in jp2:
        if field is None:
            continue   # kernel_v/kernel_n: absent == identity
        assert jnp.all(jnp.isfinite(field))
    assert jp2.pi_counts[2] < jp2.pi_counts[0]


# --- Phase 1: rank-1 recipe modulation of the noun emission (noun_tilt) -----------------------
#
# P_r(n|k) = softmax_n(log_emit_n[k,n] + noun_tilt[r,n]) -- a proper per-recipe reweighting of
# the SHARED noun table, giving the recipe latent direct access to noun content instead of only
# an indirect signal via which states a recipe favours. See docs/recipe.md "Noun tilt" for the
# derivation of why the per-trial cost stays O(chunk*K_R*T), not the O(chunk*K_R) *sequence*
# recompute a naive per-recipe emission would need.

def _shared_dynamics_two_recipe_params(key, noun_tilt=None):
    """K_R=2 JointHSMMParams whose init/trans/duration tables are IDENTICAL between recipes --
    isolates whatever separability comes purely from noun_tilt. Without a tilt, log_z_r is
    bit-identical for r=0 and r=1 on any sequence here (same tables, same shared emissions), so
    recipe recovery from dynamics alone is impossible by construction, not just hard."""
    hp = params.init_weak_limit_params(key, K, V, N, D_MAX)
    return joint_params.JointHSMMParams(
        init_counts=jnp.stack([hp.init_counts, hp.init_counts]),
        trans_counts=jnp.stack([hp.trans_counts, hp.trans_counts]),
        verb_counts=hp.verb_counts,
        noun_counts=hp.noun_counts,
        dur_r=jnp.stack([hp.dur_r, hp.dur_r]),
        dur_p=jnp.stack([hp.dur_p, hp.dur_p]),
        pi_counts=jnp.array([1.0, 1.0]),
        noun_tilt=noun_tilt,
    )


def _single_noun_sequences(labels, noun_by_label, t=15, v_vocab=V, seed=0):
    """One synthetic trial per label: verb_ids random (irrelevant to the recipe signal),
    noun_ids constant at the label's own noun id every tick -- disjoint noun vocabularies."""
    rng = np.random.default_rng(seed)
    sequences = []
    for label in labels:
        sequences.append({
            "verb_ids": rng.integers(0, v_vocab, size=t).tolist(),
            "noun_ids": [noun_by_label[label]] * t,
        })
    return sequences


def test_log_emit_n_recipe_none_tilt_matches_shared_table():
    """noun_tilt=None must reproduce the shared log_emit_n exactly -- no per-recipe
    renormalization applied when there is nothing to renormalize."""
    key = jax.random.PRNGKey(25)
    hp = params.init_weak_limit_params(key, K, V, N, D_MAX)
    jp = _single_recipe_joint_params(hp)  # noun_tilt defaults to None
    log_probs = joint_params.to_log_probs_joint(jp, D_MAX)
    recovered = joint_params.log_emit_n_recipe(log_probs, 0)
    assert jnp.allclose(recovered, log_probs.log_emit_n, atol=1e-8)


def test_log_emit_n_recipe_sums_to_one_when_tilted():
    """Every recipe's tilted noun table must still be a valid per-state distribution over
    nouns -- the -log_tilt_norm[r,k] term exists precisely to guarantee this."""
    key = jax.random.PRNGKey(26)
    tilt = jax.random.normal(jax.random.PRNGKey(27), (2, N))
    jp = _shared_dynamics_two_recipe_params(key, noun_tilt=tilt)
    log_probs = joint_params.to_log_probs_joint(jp, D_MAX)
    for r in range(2):
        table = joint_params.log_emit_n_recipe(log_probs, r)
        row_sums = jnp.exp(table).sum(axis=-1)
        assert jnp.allclose(row_sums, 1.0, atol=1e-6)


def test_select_recipe_and_collapse_to_marginal_warn_when_tilted():
    """A tilted model reaching the anomaly-scoring path (select_recipe/collapse_to_marginal,
    which both return only the SHARED emission counts) must be flagged, not silently
    mis-scored -- propagating the tilt into surprise/quantile/narrate is out of scope here."""
    key = jax.random.PRNGKey(28)
    jp = _shared_dynamics_two_recipe_params(key, noun_tilt=jnp.zeros((2, N)))
    with pytest.warns(UserWarning, match="tilt"):
        joint_params.select_recipe(jp, 0)
    with pytest.warns(UserWarning, match="tilt"):
        joint_params.collapse_to_marginal(jp)


def test_noun_tilt_zero_matches_none_in_e_step():
    """noun_tilt=zeros must be bit-identical to noun_tilt=None through the full E-step: the
    safety net for the claim that an inert tilt costs nothing behaviourally, only a broadcast
    add that was already inside the (chunk,K_R,T,K) shape gamma occupies."""
    rng = np.random.default_rng(1)
    sequences = _random_sequences(rng, n_trials=5)
    key = jax.random.PRNGKey(20)
    k_r = 2
    keys = jax.random.split(key, k_r)
    per_recipe = [params.init_weak_limit_params(keys[r], K, V, N, D_MAX) for r in range(k_r)]
    common = dict(
        init_counts=jnp.stack([p.init_counts for p in per_recipe]),
        trans_counts=jnp.stack([p.trans_counts for p in per_recipe]),
        verb_counts=per_recipe[0].verb_counts,
        noun_counts=per_recipe[0].noun_counts,
        dur_r=jnp.stack([p.dur_r for p in per_recipe]),
        dur_p=jnp.stack([p.dur_p for p in per_recipe]),
        pi_counts=jnp.ones(k_r),
    )
    jp_none = joint_params.JointHSMMParams(**common, noun_tilt=None)
    jp_zero = joint_params.JointHSMMParams(**common, noun_tilt=jnp.zeros((k_r, N)))

    verb_ids, noun_ids, mask = joint_em.pad_batch(sequences)
    stats_none, ll_none = joint_em.e_step(jp_none, verb_ids, noun_ids, mask, D_MAX, chunk_size=4)
    stats_zero, ll_zero = joint_em.e_step(jp_zero, verb_ids, noun_ids, mask, D_MAX, chunk_size=4)

    assert jnp.allclose(ll_none, ll_zero, atol=1e-6)
    for name in ("init_counts", "trans_counts", "verb_counts", "noun_counts",
                 "xi_dur", "cens", "pi_counts"):
        assert jnp.allclose(stats_none[name], stats_zero[name], atol=1e-6), name


def test_noun_tilt_recovers_recipe_from_noun_content_alone():
    """The decisive test. Two recipes share IDENTICAL init/trans/duration tables, so without a
    tilt log_z_r is bit-identical for r=0/1 on any sequence -- dynamics-only recipe recovery is
    impossible here BY CONSTRUCTION, not merely hard. Nouns are the only signal available: each
    trial emits one recipe-specific noun on every tick. Confirms both halves: (1) the
    dynamics-only assignment really is degenerate, and (2) the tilt alone fixes it."""
    key = jax.random.PRNGKey(21)
    true_labels = [0, 1] * 8  # 16 trials, alternating
    noun_by_label = {0: 0, 1: 1}
    sequences = _single_noun_sequences(true_labels, noun_by_label)
    verb_ids, noun_ids, mask = joint_em.pad_batch(sequences)

    jp_none = _shared_dynamics_two_recipe_params(key, noun_tilt=None)
    r_hat_none, _, _ = joint_em.infer_recipe(jp_none, verb_ids, noun_ids, mask, D_MAX, chunk_size=4)
    assert len(set(np.asarray(r_hat_none).tolist())) == 1, (
        "dynamics-only assignment must be degenerate (every trial -> one recipe) when the "
        "per-recipe tables are literally identical"
    )

    tilt = jnp.zeros((2, N)).at[0, 0].set(8.0).at[1, 1].set(8.0)
    jp_tilt = _shared_dynamics_two_recipe_params(key, noun_tilt=tilt)
    r_hat_tilt, _, _ = joint_em.infer_recipe(jp_tilt, verb_ids, noun_ids, mask, D_MAX, chunk_size=4)
    ari = recipe_hmm.adjusted_rand(np.asarray(r_hat_tilt), np.asarray(true_labels))
    assert ari > 0.8, f"expected the noun tilt alone to recover the recipe split, got ARI={ari}"


def test_infer_recipe_matches_e_step_objective_with_tilt():
    """Decode-time log_z (_recipe_logz_chunk, via forward_pass) must equal training-time log_z
    (e_step, via combine_sufficient_stats) once a tilt is active -- otherwise infer_recipe's
    r_hat stops matching the rho the E-step actually computed."""
    key = jax.random.PRNGKey(22)
    rng = np.random.default_rng(2)
    sequences = _random_sequences(rng, n_trials=6)
    tilt = jax.random.normal(jax.random.PRNGKey(23), (2, N)) * 0.5
    jp = _shared_dynamics_two_recipe_params(key, noun_tilt=tilt)

    verb_ids, noun_ids, mask = joint_em.pad_batch(sequences)
    _, ll_e_step = joint_em.e_step(jp, verb_ids, noun_ids, mask, D_MAX, chunk_size=4)
    _, _, trial_ll = joint_em.infer_recipe(jp, verb_ids, noun_ids, mask, D_MAX, chunk_size=4)

    assert jnp.allclose(ll_e_step, jnp.sum(trial_ll), atol=1e-6)


def test_tilt_m_step_objective_approximately_non_decreasing():
    """The tilt's GIS update (m_step's tilt_steps>0 branch) is a coordinate step, not an exact
    M-step -- the shared noun_counts update in the same call ignores the logZ_r[k] coupling.
    Mirrors test_objective_approximately_non_decreasing's loose bound, widened slightly since
    this stacks two approximate updates (duration shrinkage + tilt GIS) per iteration."""
    rng = np.random.default_rng(3)
    sequences = _random_sequences(rng, n_trials=8)
    key = jax.random.PRNGKey(24)
    k_r = 2
    keys = jax.random.split(key, k_r)
    per_recipe = [params.init_weak_limit_params(keys[r], K, V, N, D_MAX) for r in range(k_r)]
    jp0 = joint_params.JointHSMMParams(
        init_counts=jnp.stack([p.init_counts for p in per_recipe]),
        trans_counts=jnp.stack([p.trans_counts for p in per_recipe]),
        verb_counts=per_recipe[0].verb_counts,
        noun_counts=per_recipe[0].noun_counts,
        dur_r=jnp.stack([p.dur_r for p in per_recipe]),
        dur_p=jnp.stack([p.dur_p for p in per_recipe]),
        pi_counts=jnp.ones(k_r),
        noun_tilt=jnp.zeros((k_r, N)),
    )

    best, obj, history, converged = joint_em.run_joint_em(
        jp0, sequences, D_MAX, alpha_pi=1.0, kappa=1.0, max_iters=20, tol=1e-4, chunk_size=4,
        tilt_steps=1,
    )

    decreases = [history[i] - history[i + 1] for i in range(len(history) - 1) if history[i + 1] < history[i]]
    assert not decreases or max(decreases) < 2.0, f"large objective decrease found: {decreases}"


# --- Phase 2: the lam anchoring dial ------------------------------------------------------------
#
# log_rho = log_pi + log_z + lam * log_prior_r(trial). lam=0 is today; a small lam biases the
# recipe assignment toward an external per-trial guess while letting the likelihood override it;
# a large lam effectively freezes the assignment to that guess. "Freeze for N iterations then
# release" becomes a schedule over lam rather than separate machinery -- see make_lam_schedule.

def test_lam_zero_matches_no_prior_regardless_of_prior_content():
    """lam=0.0 must ignore recipe_log_prior entirely, whatever it contains -- the dial's `0 =
    today` behaviour has to hold exactly, not just approximately, since it is the default."""
    rng = np.random.default_rng(4)
    sequences = _random_sequences(rng, n_trials=5)
    key = jax.random.PRNGKey(30)
    jp = _shared_dynamics_two_recipe_params(key, noun_tilt=None)
    verb_ids, noun_ids, mask = joint_em.pad_batch(sequences)

    stats_default, ll_default = joint_em.e_step(jp, verb_ids, noun_ids, mask, D_MAX, chunk_size=4)
    prior = jax.random.normal(jax.random.PRNGKey(31), (5, 2)) * 10.0  # large but lam=0 nulls it
    stats_prior, ll_prior = joint_em.e_step(
        jp, verb_ids, noun_ids, mask, D_MAX, chunk_size=4, recipe_log_prior=prior, lam=0.0,
    )

    assert jnp.allclose(ll_default, ll_prior, atol=1e-6)
    for name in ("pi_counts", "init_counts", "trans_counts"):
        assert jnp.allclose(stats_default[name], stats_prior[name], atol=1e-6), name


def test_large_lam_freezes_assignment_to_the_prior():
    """A large enough lam must make r_hat exactly track the prior's argmax, overriding whatever
    the (here: degenerate, since dynamics are identical) likelihood alone would say."""
    key = jax.random.PRNGKey(32)
    rng = np.random.default_rng(5)
    sequences = _random_sequences(rng, n_trials=8)
    jp = _shared_dynamics_two_recipe_params(key, noun_tilt=None)
    verb_ids, noun_ids, mask = joint_em.pad_batch(sequences)

    true_labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    prior = jnp.full((8, 2), -50.0).at[jnp.arange(8), jnp.asarray(true_labels)].set(0.0)

    r_hat, _, _ = joint_em.infer_recipe(
        jp, verb_ids, noun_ids, mask, D_MAX, chunk_size=4, recipe_log_prior=prior, lam=1e6,
    )
    assert np.array_equal(np.asarray(r_hat), true_labels)


def test_lam_prior_respects_the_length_reorder():
    """e_step reorders trials by length before chunking (_length_order) and must slice
    recipe_log_prior with the SAME permutation, not the pre-sort [start:end] range -- otherwise
    a per-trial prior silently applies to the wrong trial once chunking reorders. K_R=n_trials
    with a one-hot-per-trial-index prior and a large lam makes any misalignment show up
    directly: correct slicing gives pi_counts == all-ones (each recipe claims exactly its own
    trial); a start:end/idx mismatch collides some recipes and starves others."""
    n_trials = 6
    rng = np.random.default_rng(6)
    sequences = _random_sequences(rng, n_trials=n_trials, t_range=(5, 40))  # wide range -> real reordering
    key = jax.random.PRNGKey(33)
    keys = jax.random.split(key, n_trials)
    per_recipe = [params.init_weak_limit_params(keys[r], K, V, N, D_MAX) for r in range(n_trials)]
    jp = joint_params.JointHSMMParams(
        init_counts=jnp.stack([p.init_counts for p in per_recipe]),
        trans_counts=jnp.stack([p.trans_counts for p in per_recipe]),
        verb_counts=per_recipe[0].verb_counts,
        noun_counts=per_recipe[0].noun_counts,
        dur_r=jnp.stack([p.dur_r for p in per_recipe]),
        dur_p=jnp.stack([p.dur_p for p in per_recipe]),
        pi_counts=jnp.ones(n_trials),
    )
    verb_ids, noun_ids, mask = joint_em.pad_batch(sequences)
    prior = jnp.full((n_trials, n_trials), -50.0).at[jnp.arange(n_trials), jnp.arange(n_trials)].set(0.0)

    stats, _ = joint_em.e_step(
        jp, verb_ids, noun_ids, mask, D_MAX, chunk_size=2, recipe_log_prior=prior, lam=1e6,
    )
    assert jnp.allclose(stats["pi_counts"], jnp.ones(n_trials), atol=1e-3), stats["pi_counts"]


def test_infer_recipe_lam_does_not_reorder():
    """infer_recipe (unlike e_step) does not sort by length, so recipe_log_prior must be sliced
    plain [start:end] there -- this is the deliberate asymmetry _e_step_chunk's docstring notes.
    A trivial identity check: with chunk_size < n_trials (forcing >1 chunk) and lam=0, r_hat
    must be invariant to trial order in the input arrays -- i.e. slicing by [start:end] is
    self-consistent with pad_batch's own (unsorted) order."""
    key = jax.random.PRNGKey(34)
    rng = np.random.default_rng(7)
    sequences = _random_sequences(rng, n_trials=8, t_range=(5, 30))
    jp = _shared_dynamics_two_recipe_params(key, noun_tilt=None)
    verb_ids, noun_ids, mask = joint_em.pad_batch(sequences)

    r_hat_full, _, ll_full = joint_em.infer_recipe(jp, verb_ids, noun_ids, mask, D_MAX, chunk_size=8)
    r_hat_chunked, _, ll_chunked = joint_em.infer_recipe(jp, verb_ids, noun_ids, mask, D_MAX, chunk_size=3)
    assert jnp.array_equal(r_hat_full, r_hat_chunked)
    assert jnp.allclose(ll_full, ll_chunked, atol=1e-6)


def test_objective_non_decreasing_at_constant_lam():
    """At CONSTANT lam, sum_i logsumexp_r(log_pi_r + logz_ir + lam*logprior_ir) is a valid EM
    objective (see run_joint_em's docstring: absorbing exp(lam*log_prior_ir) as a theta-
    independent per-(trial,recipe) constant into the generative model keeps EM's monotonicity
    argument intact for an EXACT M-step).

    Needs global_damping here, unlike the plain test_objective_approximately_non_decreasing:
    a nonzero lam sharpens how much responsibility can swing between iterations (the prior
    pulls trials toward specific recipes harder than the likelihood alone would), and an
    undamped near-empty duration cell's pooled global fit is already documented
    (durations.fit_durations_shrunk) to swing by an order of magnitude on exactly that kind of
    swing -- confirmed directly: this same setup undamped produces a >6-point single-iteration
    drop, a PRE-EXISTING instability unrelated to lam, and global_damping=0.7 (production's own
    default whenever recipe responsibility is expected to move a lot) makes it disappear
    entirely. Damped, this is held to the same <1.0 bound as the undialed test."""
    rng = np.random.default_rng(8)
    sequences = _random_sequences(rng, n_trials=8)
    key = jax.random.PRNGKey(35)
    k_r = 2
    keys = jax.random.split(key, k_r)
    per_recipe = [params.init_weak_limit_params(keys[r], K, V, N, D_MAX) for r in range(k_r)]
    jp0 = joint_params.JointHSMMParams(
        init_counts=jnp.stack([p.init_counts for p in per_recipe]),
        trans_counts=jnp.stack([p.trans_counts for p in per_recipe]),
        verb_counts=per_recipe[0].verb_counts,
        noun_counts=per_recipe[0].noun_counts,
        dur_r=jnp.stack([p.dur_r for p in per_recipe]),
        dur_p=jnp.stack([p.dur_p for p in per_recipe]),
        pi_counts=jnp.ones(k_r),
    )
    prior = jax.random.normal(jax.random.PRNGKey(36), (8, k_r))

    best, obj, history, converged = joint_em.run_joint_em(
        jp0, sequences, D_MAX, alpha_pi=1.0, kappa=1.0, max_iters=20, tol=1e-4, chunk_size=4,
        recipe_log_prior=prior, lam=0.5, global_damping=0.7,
    )

    decreases = [history[i] - history[i + 1] for i in range(len(history) - 1) if history[i + 1] < history[i]]
    assert not decreases or max(decreases) < 1.0, f"large objective decrease found: {decreases}"


def test_make_lam_schedule_variants():
    """const/geom/linear/freeze specs must parse and evaluate as documented."""
    const = joint_em.make_lam_schedule("const:2.5")
    assert const(0) == 2.5 and const(10) == 2.5

    geom = joint_em.make_lam_schedule("geom:5,0.5")
    assert geom(0) == 5.0
    assert abs(geom(1) - 2.5) < 1e-9
    assert abs(geom(2) - 1.25) < 1e-9

    linear = joint_em.make_lam_schedule("linear:10,5")
    assert linear(0) == 10.0
    assert abs(linear(5) - 0.0) < 1e-9
    assert linear(10) == 0.0  # clamped, does not go negative

    freeze = joint_em.make_lam_schedule("freeze:1000.0,3")
    assert freeze(0) == 1000.0 and freeze(2) == 1000.0
    assert freeze(3) == 0.0 and freeze(10) == 0.0


def test_run_joint_em_lam_schedule_overrides_constant_lam():
    """run_joint_em must call lam_schedule(iteration) each iteration when given, rather than
    the constant `lam` -- checked indirectly via the freeze-then-release shape: pi_counts after
    a run with freeze:1e6,100 (frozen for the whole run) must match a run with lam=1e6 fixed."""
    rng = np.random.default_rng(9)
    sequences = _random_sequences(rng, n_trials=6)
    key = jax.random.PRNGKey(37)
    k_r = 2
    keys = jax.random.split(key, k_r)
    per_recipe = [params.init_weak_limit_params(keys[r], K, V, N, D_MAX) for r in range(k_r)]
    jp0 = joint_params.JointHSMMParams(
        init_counts=jnp.stack([p.init_counts for p in per_recipe]),
        trans_counts=jnp.stack([p.trans_counts for p in per_recipe]),
        verb_counts=per_recipe[0].verb_counts,
        noun_counts=per_recipe[0].noun_counts,
        dur_r=jnp.stack([p.dur_r for p in per_recipe]),
        dur_p=jnp.stack([p.dur_p for p in per_recipe]),
        pi_counts=jnp.ones(k_r),
    )
    prior = jnp.full((6, k_r), -50.0).at[jnp.arange(6), jnp.arange(6) % k_r].set(0.0)

    best_fixed, _, _, _ = joint_em.run_joint_em(
        jp0, sequences, D_MAX, alpha_pi=1.0, kappa=1.0, max_iters=3, tol=1e-8, chunk_size=4,
        recipe_log_prior=prior, lam=1e6,
    )
    best_sched, _, _, _ = joint_em.run_joint_em(
        jp0, sequences, D_MAX, alpha_pi=1.0, kappa=1.0, max_iters=3, tol=1e-8, chunk_size=4,
        recipe_log_prior=prior, lam_schedule=joint_em.make_lam_schedule("freeze:1e6,100"),
    )
    assert jnp.allclose(best_fixed.pi_counts, best_sched.pi_counts, atol=1e-3)
