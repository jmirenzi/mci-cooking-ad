"""The latent-intended-token emission: fixed similarity kernel S, marginal P(n|k) = (B S)[k,n].

The load-bearing test here is `test_objective_non_decreasing_with_kernel`. Composing a kernel
into the emission is only legitimate because it is an exact latent-variable model, so EM stays
monotone; the ad-hoc alternative (smoothing counts directly, `counts @ S` in the M-step) is not
a generative model and silently breaks that. Both produce a smoothed emission and look alike in
a diff -- this test is what tells them apart.
"""
import jax
import jax.numpy as jnp
import numpy as np

from cook_ad.anomaly import quantile
from cook_ad.hsmm import em, joint_em, joint_params, kernel, params

jax.config.update("jax_enable_x64", True)

K, V, N, D_MAX = 4, 5, 6, 10


def _embeddings(rng, w, d=8):
    return rng.normal(size=(w, d))


def _params(seed=0):
    return params.init_weak_limit_params(jax.random.PRNGKey(seed), K, V, N, D_MAX)


def _sequences(rng, n_trials=8, t_range=(10, 20)):
    return [{"verb_ids": rng.integers(0, V, size=(t := int(rng.integers(*t_range)))).tolist(),
             "noun_ids": rng.integers(0, N, size=t).tolist()} for _ in range(n_trials)]


# --- kernel construction -------------------------------------------------------------------

def test_kernel_is_row_stochastic_and_strictly_positive():
    rng = np.random.default_rng(0)
    s = kernel.similarity_kernel(_embeddings(rng, N), tau=0.05, lam=0.2)
    assert np.allclose(s.sum(axis=1), 1.0)
    assert (s > 0).all(), "epsilon floor must leave no zero -- an observation is never -inf"


def test_lam_zero_is_the_identity_for_any_tau():
    rng = np.random.default_rng(0)
    e = _embeddings(rng, N)
    for tau in (1e-3, 0.05, 1.0, 1e6):
        assert np.array_equal(kernel.similarity_kernel(e, tau=tau, lam=0.0), np.eye(N))


def test_lam_controls_mass_and_tau_controls_only_its_shape():
    """The reason the kernel is not a single softmax over the whole row: leaked mass and
    semantic sharpness have to move independently. Self-mass must depend on lam alone."""
    rng = np.random.default_rng(0)
    e = _embeddings(rng, N)
    for lam in (0.05, 0.15, 0.4):
        diags = [np.diag(kernel.similarity_kernel(e, tau=t, lam=lam)) for t in (0.02, 0.1, 1.0)]
        for d in diags[1:]:
            assert np.allclose(d, diags[0], atol=1e-9)
        assert np.allclose(diags[0], 1 - lam, atol=1e-3)


def test_large_tau_approaches_the_uniform_ablation():
    rng = np.random.default_rng(0)
    e = _embeddings(rng, N)
    hot = kernel.similarity_kernel(e, tau=1e6, lam=0.2)
    uni = kernel.similarity_kernel(e, tau=0.05, lam=0.2, uniform=True)
    assert np.allclose(hot, uni, atol=1e-6)


# --- composition ---------------------------------------------------------------------------

def test_identity_kernel_reproduces_the_plain_categorical_bit_for_bit():
    """S = I must be the same CODE PATH as the baseline, not merely close to it. This fails
    loudly if log S floors its structural zeros instead of keeping them at -inf: _row_normalize
    already floors an unobserved cell to ~-36 nats, which log(1e-12) = -27.6 would swamp."""
    p = _params()
    base = params.normalize_categoricals(p)
    with_i = params.normalize_categoricals(p._replace(kernel_v=jnp.eye(V), kernel_n=jnp.eye(N)))
    for a, b in zip(base, with_i):
        assert jnp.array_equal(a, b)


def test_composed_emission_normalises_and_stays_finite():
    rng = np.random.default_rng(0)
    p = _params()._replace(kernel_n=kernel.similarity_kernel(_embeddings(rng, N), 0.05, 0.2))
    _, _, _, log_emit_n = params.normalize_categoricals(p)
    assert np.allclose(np.exp(np.asarray(log_emit_n)).sum(axis=1), 1.0)
    assert np.isfinite(np.asarray(log_emit_n)).all()


def test_quantile_thresholds_stay_finite_under_a_kernel():
    """quantile.py's exact discrete tail is only valid on a normalised row over an enumerable
    support -- the property the whole latent-token formulation exists to preserve."""
    rng = np.random.default_rng(0)
    p = _params()._replace(kernel_n=kernel.similarity_kernel(_embeddings(rng, N), 0.05, 0.2))
    lp = params.to_log_probs(p, D_MAX)
    assert np.isfinite(quantile.categorical_quantile_threshold(lp.log_emit_n, 0.05)).all()
    assert np.isfinite(quantile.joint_quantile_threshold(lp.log_emit_v, lp.log_emit_n, 0.05)).all()


def test_smoothing_lifts_the_near_neighbour_far_more_than_a_distant_token():
    """The entire premise: the kernel must be SEMANTIC, not uniform. smooth_params.py's
    objection is to uniform mass-shifting, and this is the property that answers it."""
    e = np.zeros((3, 2))
    e[0], e[1], e[2] = [1.0, 0.0], [0.98, 0.2], [-1.0, 0.0]   # 0 near 1, 0 opposite 2
    s = kernel.similarity_kernel(e, tau=0.05, lam=0.2)
    assert s[0, 1] > 100 * s[0, 2]


# --- the M-step ----------------------------------------------------------------------------

def test_latent_counts_conserve_total_mass():
    """R[k,m,v] = P(m|v,k) sums to 1 over m, so remapping observed counts to latent ones moves
    mass between columns but never creates or destroys it."""
    rng = np.random.default_rng(0)
    c = jnp.asarray(rng.random((K, N)) * 10)
    log_b = params._row_normalize(_params().noun_counts)
    hat = params.latent_counts(c, log_b, kernel.similarity_kernel(_embeddings(rng, N), 0.05, 0.2))
    assert np.allclose(np.asarray(hat).sum(axis=1), np.asarray(c).sum(axis=1))
    assert (np.asarray(hat) >= 0).all()


def test_latent_counts_is_a_no_op_without_a_kernel():
    rng = np.random.default_rng(0)
    c = jnp.asarray(rng.random((K, N)) * 10)
    log_b = params._row_normalize(_params().noun_counts)
    assert jnp.array_equal(params.latent_counts(c, log_b, None), c)
    assert np.allclose(params.latent_counts(c, log_b, jnp.eye(N)), c)


def test_m_step_carries_the_kernel_through():
    """A kernel dropped by the M-step would silently revert the model to the baseline after
    one iteration while every other check still passed."""
    rng = np.random.default_rng(0)
    p = _params()._replace(kernel_n=kernel.similarity_kernel(_embeddings(rng, N), 0.05, 0.2))
    v, n, mask = em.pad_batch(_sequences(rng))
    stats, _ = em.e_step(p, v, n, mask, D_MAX, chunk_size=4)
    p2 = em.m_step(p, stats, 0.5, 0.5, float(V), float(N), D_MAX)
    assert p2.kernel_n is not None and np.allclose(p2.kernel_n, p.kernel_n)


def test_objective_non_decreasing_with_kernel():
    """Standard latent-variable EM, so the joint objective must be non-decreasing -- the same
    bound test_objective_approximately_non_decreasing applies to the unkernelled model, and the
    check that would catch an ad-hoc `counts @ S` M-step if someone reimplemented it that way.
    """
    rng = np.random.default_rng(0)
    keys = jax.random.split(jax.random.PRNGKey(4), 2)
    per_recipe = [params.init_weak_limit_params(k, K, V, N, D_MAX) for k in keys]
    jp0 = joint_params.JointHSMMParams(
        init_counts=jnp.stack([p.init_counts for p in per_recipe]),
        trans_counts=jnp.stack([p.trans_counts for p in per_recipe]),
        verb_counts=per_recipe[0].verb_counts,
        noun_counts=per_recipe[0].noun_counts,
        dur_r=jnp.stack([p.dur_r for p in per_recipe]),
        dur_p=jnp.stack([p.dur_p for p in per_recipe]),
        pi_counts=jnp.ones(2),
        kernel_n=jnp.asarray(kernel.similarity_kernel(_embeddings(rng, N), 0.05, 0.2)),
    )
    _, _, history, _ = joint_em.run_joint_em(
        jp0, _sequences(rng), D_MAX, alpha_pi=1.0, kappa=1.0, max_iters=20, tol=1e-4, chunk_size=4,
    )
    drops = [history[i] - history[i + 1] for i in range(len(history) - 1) if history[i + 1] < history[i]]
    assert not drops or max(drops) < 1.0, f"large objective decrease under a kernel: {drops}"


def test_legacy_npz_round_trips_without_a_kernel(tmp_path):
    p = _params()
    path = tmp_path / "p.npz"
    params.save_params(p, path)
    with np.load(str(path) if str(path).endswith(".npz") else str(path) + ".npz") as d:
        assert "kernel_v" not in d and "kernel_n" not in d
    back = params.load_params(str(path) + ".npz" if not str(path).endswith(".npz") else str(path))
    assert back.kernel_v is None and back.kernel_n is None
