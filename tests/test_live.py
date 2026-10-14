"""The live (causal) channels of docs/live.md: exactness of the filter, and rule R2 -- with a
point-mass belief on the hindsight answer, every live flag reduces to the shipped one."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import logsumexp

jax.config.update("jax_enable_x64", True)

from cook_ad.anomaly import live, quantile, surprise
from cook_ad.eval import batch
from cook_ad.hsmm import emissions, joint_em, joint_params, messages

K_R, K, V, N, D_MAX = 2, 4, 3, 5, 6
ALPHAS = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 1e-3]


def _model(seed=0, k_r=K_R):
    rng = np.random.default_rng(seed)
    trans = rng.gamma(1.0, 3.0, size=(k_r, K, K)) + 1.0
    for r in range(k_r):
        np.fill_diagonal(trans[r], 0.0)
    return joint_params.JointHSMMParams(
        init_counts=jnp.asarray(rng.gamma(1.0, 3.0, size=(k_r, K)) + 1.0),
        trans_counts=jnp.asarray(trans),
        verb_counts=jnp.asarray(rng.gamma(0.5, 4.0, size=(K, V)) + 1.0),
        noun_counts=jnp.asarray(rng.gamma(0.5, 4.0, size=(K, N)) + 1.0),
        dur_r=jnp.asarray(rng.uniform(1.0, 6.0, size=(k_r, K))),
        dur_p=jnp.asarray(rng.uniform(0.3, 0.8, size=(k_r, K))),
        pi_counts=jnp.asarray(rng.gamma(2.0, 2.0, size=(k_r,)) + 1.0),
    )


def _trials(seed=1, n=5, t_range=(8, 15)):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        t = int(rng.integers(*t_range))
        out.append({"verb_ids": rng.integers(0, V, t), "noun_ids": rng.integers(0, N, t)})
    return out


def _loglik(tb, s):
    v = jnp.asarray(s["verb_ids"])
    n = jnp.asarray(s["noun_ids"])
    return emissions.sequence_loglik(v, n, tb.log_emit_v, tb.log_emit_n, jnp.ones(len(v), bool))


def test_filter_loglik_equals_the_joint_marginal():
    """sum_t log P(o_t|o_<t) from the live filter == infer_recipe's per-trial log P(o)."""
    jp = _model()
    tb = live.build_tables(jp, D_MAX, ALPHAS)
    trials = _trials()
    verb, noun, mask = batch._pad(trials, max(len(s["verb_ids"]) for s in trials))
    _, _, trial_ll = joint_em.infer_recipe(jp, verb, noun, mask, D_MAX)
    for i, s in enumerate(trials):
        trace = live.compute_live_trace(tb, s["verb_ids"], s["noun_ids"])
        assert trace.log_norm.sum() == pytest.approx(float(trial_ll[i]), rel=1e-9, abs=1e-9)


def test_predictive_belief_matches_messages_at_one_recipe():
    """At K_R = 1 the live pi_tilde is exactly messages.predictive_occupancy."""
    jp = _model(k_r=1)
    tb = live.build_tables(jp, D_MAX, ALPHAS)
    lp = joint_params.to_log_probs_joint(jp, D_MAX)
    for s in _trials():
        ll = _loglik(tb, s)
        mask = jnp.ones(ll.shape[0], bool)
        ref = messages.predictive_occupancy(ll, lp.log_init[0], lp.log_trans[0], lp.log_dur_pmf[0],
                                            lp.log_dur_survival[0], mask, D_MAX)
        _, _, log_pt, _, _ = live.run_filter(tb, ll, mask)
        np.testing.assert_allclose(np.exp(log_pt), np.exp(ref), atol=1e-10)


def test_full_lag_reproduces_the_smoother():
    """With a lag covering the whole trial, the stall posterior's subtask marginal is the
    smoothed gamma, and the total switch probability sums to the expected transition count."""
    jp = _model(k_r=1)
    tb = live.build_tables(jp, D_MAX, ALPHAS)
    lp = joint_params.to_log_probs_joint(jp, D_MAX)
    for s in _trials():
        ll = _loglik(tb, s)
        t = ll.shape[0]
        mask = jnp.ones(t, bool)
        logb_all, _, _, _, _ = live.run_filter(tb, ll, mask)
        beta = live.fixed_lag_beta(tb, ll, t, t)
        post = logb_all + beta
        post = post - logsumexp(post, axis=(1, 2, 3), keepdims=True)
        gamma_live = np.exp(logsumexp(post, axis=(1, 3)))
        xi_trans, _, _, gamma, _ = messages.combine_sufficient_stats(
            ll, mask, lp.log_init[0], lp.log_trans[0], lp.log_dur_pmf[0], lp.log_dur_survival[0], D_MAX)
        np.testing.assert_allclose(gamma_live, np.asarray(gamma), atol=1e-9)

        trace = live.compute_live_trace(tb, s["verb_ids"], s["noun_ids"], lags=(t,))
        assert trace.switch[t].sum() == pytest.approx(float(np.asarray(xi_trans).sum()), abs=1e-9)


def test_event_probabilities_are_probabilities_and_nested_in_alpha():
    jp = _model()
    tb = live.build_tables(jp, D_MAX, ALPHAS)
    for s in _trials():
        trace = live.compute_live_trace(tb, s["verb_ids"], s["noun_ids"], lags=(0, 3))
        for lag in (0, 3):
            for name in live.EVENT_CHANNELS:
                x = trace.events[lag][name]
                assert np.all(x >= -1e-12) and np.all(x <= 1 + 1e-9)
                assert np.all(np.diff(x, axis=1) >= -1e-12)   # alphas ascending -> mass grows
            assert np.all(trace.switch[lag] <= 1 + 1e-9)
        for p in (trace.p_emit, trace.p_verb, trace.p_noun):
            assert np.all(p > 0) and np.all(p <= 1 + 1e-9)


def test_emission_pvalue_collapses_to_the_hindsight_flag():
    """R2 for the emission channels: under a point-mass belief on state z the p-value flag
    (p <= alpha) equals the shipped rule s > tau_z for every observation and alpha. The dilution
    term -log pi(z*) is 0 at a point mass."""
    jp = _model()
    lp = joint_params.to_log_probs_joint(jp, D_MAX)
    ev, en = lp.log_emit_v, lp.log_emit_n
    vv, nn = np.meshgrid(np.arange(V), np.arange(N), indexing="ij")
    vv, nn = jnp.asarray(vv.ravel()), jnp.asarray(nn.ravel())
    for z in range(K):
        one_hot = jnp.where(jnp.arange(K) == z, 0.0, -jnp.inf)
        pt = jnp.tile(one_hot, (vv.shape[0], 1))
        p_emit, p_verb, p_noun = live.emission_pvalues(pt, ev, en, vv, nn)
        s_emit = -(np.asarray(ev)[z, vv] + np.asarray(en)[z, nn])
        for a in ALPHAS:
            tau_e = quantile.joint_quantile_threshold(ev, en, a)[z]
            tau_v = quantile.categorical_quantile_threshold(ev, a)[z]
            tau_n = quantile.categorical_quantile_threshold(en, a)[z]
            np.testing.assert_array_equal(np.asarray(p_emit) <= a, surprise._exceeds(s_emit, tau_e))
            np.testing.assert_array_equal(np.asarray(p_verb) <= a,
                                          surprise._exceeds(-np.asarray(ev)[z, vv], tau_v))
            np.testing.assert_array_equal(np.asarray(p_noun) <= a,
                                          surprise._exceeds(-np.asarray(en)[z, nn], tau_n))


def test_event_bad_sets_collapse_to_the_hindsight_flags():
    """R2 for the hidden-event channels: a point-mass belief at the hindsight answer
    (r_hat, z*, e*) puts all its event mass on one cell, so the live flag at alpha_g is exactly
    "that cell's bin <= g". Checked against flag_joint on real hindsight traces."""
    jp = _model(seed=3)
    tb = live.build_tables(jp, D_MAX, ALPHAS)
    trials = _trials(seed=4, n=8, t_range=(12, 25))
    traces, lp, r_hat, ltm = batch.compute_traces_joint(jp, trials, D_MAX, chunk_size=4)
    alphas = tb.alphas
    stall_bin, early_bin = np.asarray(tb.stall_bin), np.asarray(tb.early_bin)
    trans_bin, recipe_bin = np.asarray(tb.trans_bin), np.asarray(tb.recipe_bin)
    for tr, r in zip(traces, np.asarray(r_hat)):
        r = int(r)
        z = tr.z_star
        starts = np.flatnonzero(np.r_[True, z[1:] != z[:-1]])
        ends = np.r_[starts[1:] - 1, len(z) - 1]
        elapsed = np.empty(len(z), dtype=int)
        for a, b in zip(starts, ends):
            elapsed[a : b + 1] = np.arange(1, b - a + 2)
        for g, a in enumerate(alphas):
            f = surprise.flag_joint(tr, lp, r, ltm, alpha=a)
            np.testing.assert_array_equal(stall_bin[r, z, elapsed - 1] <= g, f["s_temporal"])
            for m in range(1, len(starts)):
                t0, k, i = starts[m], z[starts[m] - 1], z[starts[m]]
                assert (trans_bin[r, k, i] <= g) == bool(f["s_transition"][t0])
                assert (recipe_bin[r, k, i] <= g) == bool(f["s_recipe_transition"][t0])
                # The hindsight early flag sits on the ended step's LAST tick (one tick before it
                # is knowable); the live one on the next step's first tick.
                d_prev = ends[m - 1] - starts[m - 1] + 1
                short_flag = bool(f["s_dur_two"][ends[m - 1]]) and \
                    tr.temporal_attribution[ends[m - 1]] == "left_early"
                assert (early_bin[r, k, d_prev - 1] <= g) == short_flag


def test_split_belief_sums_to_the_predictive_belief_and_never_flags_more():
    """stay + switch = pi_tilde, t=0 is all "switch", and at a point mass the absent hypothesis
    drops out of the max so the split p-value equals the plain one -- R2 holds unchanged."""
    jp = _model()
    tb = live.build_tables(jp, D_MAX, ALPHAS)
    for s in _trials():
        ll = _loglik(tb, s)
        mask = jnp.ones(ll.shape[0], bool)
        _, _, log_pt, log_stay, log_switch = live.run_filter(tb, ll, mask)
        np.testing.assert_allclose(np.exp(np.logaddexp(log_stay, log_switch)), np.exp(log_pt), atol=1e-12)
        trace = live.compute_live_trace(tb, s["verb_ids"], s["noun_ids"])
        assert trace.p_emit[0] == pytest.approx(trace.p_mix[0][0])  # t=0: all mass is "switch"

    lp = joint_params.to_log_probs_joint(jp, D_MAX)
    vv = jnp.arange(V).repeat(N)
    nn = jnp.tile(jnp.arange(N), V)
    one_hot = jnp.where(jnp.arange(K) == 1, 0.0, -jnp.inf)
    absent = jnp.full((K,), -jnp.inf)
    pt = jnp.tile(one_hot, (vv.shape[0], 1))
    gone = jnp.tile(absent, (vv.shape[0], 1))
    mixed = live.emission_pvalues(pt, lp.log_emit_v, lp.log_emit_n, vv, nn)
    for split in (live.split_emission_pvalues(pt, gone, lp.log_emit_v, lp.log_emit_n, vv, nn),
                  live.split_emission_pvalues(gone, pt, lp.log_emit_v, lp.log_emit_n, vv, nn)):
        for a, b in zip(split, mixed):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-12)


def test_full_beta_equals_a_lag_covering_the_trial():
    jp = _model()
    tb = live.build_tables(jp, D_MAX, ALPHAS)
    for s in _trials():
        ll = _loglik(tb, s)
        t = ll.shape[0]
        a = live.full_beta(tb, ll, jnp.ones(t, bool))
        b = live.fixed_lag_beta(tb, ll, t, t)
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-10)


def test_pinned_recipe_equals_a_one_hot_prior_and_matches_its_own_likelihood():
    """Knowing the recipe: pin_recipe (slice to r) and a one-hot prior give the same channels,
    and the filter's likelihood is that recipe's own forward pass."""
    jp = _model()
    tb = live.build_tables(jp, D_MAX, ALPHAS)
    lp = joint_params.to_log_probs_joint(jp, D_MAX)
    for s in _trials(n=3):
        for r in range(K_R):
            pinned = live.compute_live_trace(tb, s["verb_ids"], s["noun_ids"], lags=(0, 2), recipe=r)
            prior = np.where(np.arange(K_R) == r, 0.0, -np.inf)
            soft = live.compute_live_trace(tb, s["verb_ids"], s["noun_ids"], lags=(0, 2), recipe=prior)
            np.testing.assert_allclose(pinned.p_emit, soft.p_emit, atol=1e-12)
            for lag in (0, 2):
                for name in live.EVENT_CHANNELS:
                    np.testing.assert_allclose(pinned.events[lag][name], soft.events[lag][name], atol=1e-12)
            ll = _loglik(tb, s)
            ln, _ = messages.forward_pass(ll, lp.log_init[r], lp.log_trans[r], lp.log_dur_pmf[r],
                                          lp.log_dur_survival[r], jnp.ones(ll.shape[0], bool), D_MAX)
            assert pinned.log_norm.sum() == pytest.approx(float(jnp.sum(ln)), abs=1e-9)
