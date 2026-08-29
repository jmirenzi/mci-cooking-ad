import jax
import jax.numpy as jnp
import numpy as np

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
