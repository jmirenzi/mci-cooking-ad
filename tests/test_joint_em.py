import jax
import jax.numpy as jnp
import numpy as np

from _fixtures import one_recipe_tables, stack_recipes, weak_limit_params
from cook_ad.hsmm import params
from cook_ad.hsmm.train import durations_fit, joint_em, metrics
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


def test_to_log_tables_matches_single_recipe_slice():
    """Each recipe's slice of the joint log tables equals that recipe's tables computed alone --
    catches K_R-axis vmap bugs (trans's mask_diag eye(), duration_tables' 1-D dur_r)."""
    keys = jax.random.split(jax.random.PRNGKey(0), 3)
    views = [weak_limit_params(k, K, V, N, D_MAX) for k in keys]
    t = params.to_log_tables(stack_recipes(views), D_MAX)
    for r, view in enumerate(views):
        ref = one_recipe_tables(view, D_MAX)
        for name in ("log_init", "log_A", "log_pmf", "log_surv"):
            np.testing.assert_allclose(getattr(t, name)[r], getattr(ref, name), atol=1e-12)
    np.testing.assert_allclose(t.log_emit_v, one_recipe_tables(views[0], D_MAX).log_emit_v)


def test_fit_durations_shrunk_kappa_zero_matches_plain_fit():
    key = jax.random.PRNGKey(3)
    ks = jax.random.split(key, 4)
    xi_dur = jax.random.uniform(ks[0], (1, K, D_MAX)) * 5
    cens = jax.random.uniform(ks[1], (1, K, D_MAX)) * 0.5
    r_old = jax.random.uniform(ks[2], (1, K), minval=1.0, maxval=5.0)
    p_old = jax.random.uniform(ks[3], (1, K), minval=0.2, maxval=0.8)

    dur_r, dur_p, global_r, global_p = durations_fit.fit_durations_shrunk(xi_dur, cens, r_old, p_old, D_MAX, kappa=0.0)

    n_hat = durations_fit.impute_censored_histogram(xi_dur[0], cens[0], r_old[0], p_old[0], D_MAX)
    n_tot, s_hat = durations_fit.duration_stats_from_histogram(n_hat, D_MAX)
    r_ref = durations_fit.newton_update_r(n_hat, n_tot, s_hat, r_old[0], n_iters=5)
    p_ref = durations_fit.update_p_given_r(n_tot, s_hat, r_ref)

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
    _, _, global_r_1, global_p_1 = durations_fit.fit_durations_shrunk(
        xi_dur_1, cens_1, r_old, p_old, D_MAX, kappa=5.0,
    )

    # call 2: a DELIBERATELY DIFFERENT sparse histogram for the same state (simulates the
    # E-step's tiny amount of real data shifting between iterations)
    xi_dur_2 = jax.random.uniform(ks[1], (1, K, D_MAX)) * 0.3 + 2.0
    cens_2 = jnp.zeros((1, K, D_MAX))

    _, _, global_r_2_undamped, global_p_2_undamped = durations_fit.fit_durations_shrunk(
        xi_dur_2, cens_2, global_r_1[None, :], global_p_1[None, :], D_MAX, kappa=5.0,
    )
    damping = 0.7
    _, _, global_r_2_damped, global_p_2_damped = durations_fit.fit_durations_shrunk(
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
    per_recipe = [weak_limit_params(keys[r], K, V, N, D_MAX) for r in range(k_r)]
    jp0 = stack_recipes(per_recipe)

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
    shared = weak_limit_params(key, k, V, N, D_MAX, alpha_emit_v=float(V), alpha_emit_n=float(N))

    forward = jnp.zeros((k, k)).at[jnp.arange(k), (jnp.arange(k) + 1) % k].set(10.0)
    backward = jnp.zeros((k, k)).at[jnp.arange(k), (jnp.arange(k) - 1) % k].set(10.0)
    hp1 = params.RecipeView(shared.init_counts, forward, shared.verb_counts, shared.noun_counts,
                             shared.dur_r, shared.dur_p)
    hp2 = params.RecipeView(shared.init_counts, backward, shared.verb_counts, shared.noun_counts,
                             shared.dur_r, shared.dur_p)

    np_rng = np.random.default_rng(7)
    sequences, true_recipes = [], []
    for i in range(16):
        hp = hp1 if i % 2 == 0 else hp2
        traj = generate.sample_trajectory(hp, np_rng, max_ticks=30, d_max=D_MAX)
        sequences.append({"verb_ids": traj["verb_ids"].tolist(), "noun_ids": traj["noun_ids"].tolist()})
        true_recipes.append(0 if i % 2 == 0 else 1)

    jp0 = params.JointHSMMParams(
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

    ari = metrics.adjusted_rand(np.asarray(r_hat), np.asarray(true_recipes))
    assert ari > 0.8, f"expected near-perfect recipe recovery, got ARI={ari}"


def test_dead_recipe_stays_finite():
    """A recipe with near-zero pi_counts and (after the E-step) near-zero responsibility must
    stay finite through the M-step: init/trans via the Dirichlet floor, durations via the
    kappa*pmf_global shrinkage term."""
    rng = np.random.default_rng(0)
    sequences = _random_sequences(rng, n_trials=6)
    key = jax.random.PRNGKey(6)
    hp = weak_limit_params(key, K, V, N, D_MAX)

    k_r = 3
    init_counts = jnp.tile(hp.init_counts[None, :], (k_r, 1))
    trans_counts = jnp.tile(hp.trans_counts[None, :, :], (k_r, 1, 1))
    dur_r = jnp.tile(hp.dur_r[None, :], (k_r, 1))
    dur_p = jnp.tile(hp.dur_p[None, :], (k_r, 1))
    pi_counts = jnp.array([5.0, 5.0, 1e-6])  # third recipe effectively dead from iteration 0

    jp = params.JointHSMMParams(init_counts, trans_counts, hp.verb_counts, hp.noun_counts,
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


def _shared_dynamics_two_recipe_params(key):
    """K_R=2 with IDENTICAL init/trans/duration tables between recipes: log P(o | r) is the same
    for both, so only an external prior can separate them."""
    hp = weak_limit_params(key, K, V, N, D_MAX)
    return stack_recipes([hp, hp])


def test_lam_zero_matches_no_prior_regardless_of_prior_content():
    """lam=0.0 must ignore recipe_log_prior entirely, whatever it contains -- the dial's `0 =
    today` behaviour has to hold exactly, not just approximately, since it is the default."""
    rng = np.random.default_rng(4)
    sequences = _random_sequences(rng, n_trials=5)
    key = jax.random.PRNGKey(30)
    jp = _shared_dynamics_two_recipe_params(key)
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
    jp = _shared_dynamics_two_recipe_params(key)
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
    per_recipe = [weak_limit_params(keys[r], K, V, N, D_MAX) for r in range(n_trials)]
    jp = stack_recipes(per_recipe)
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
    jp = _shared_dynamics_two_recipe_params(key)
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
    (durations_fit.fit_durations_shrunk) to swing by an order of magnitude on exactly that kind of
    swing -- confirmed directly: this same setup undamped produces a >6-point single-iteration
    drop, a PRE-EXISTING instability unrelated to lam, and global_damping=0.7 (production's own
    default whenever recipe responsibility is expected to move a lot) makes it disappear
    entirely. Damped, this is held to the same <1.0 bound as the undialed test."""
    rng = np.random.default_rng(8)
    sequences = _random_sequences(rng, n_trials=8)
    key = jax.random.PRNGKey(35)
    k_r = 2
    keys = jax.random.split(key, k_r)
    per_recipe = [weak_limit_params(keys[r], K, V, N, D_MAX) for r in range(k_r)]
    jp0 = stack_recipes(per_recipe)
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
    per_recipe = [weak_limit_params(keys[r], K, V, N, D_MAX) for r in range(k_r)]
    jp0 = stack_recipes(per_recipe)
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
