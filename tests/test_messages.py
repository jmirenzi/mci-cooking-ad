import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cook_ad.hsmm import emissions, messages, params

jax.config.update("jax_enable_x64", True)


def _random_log_probs(rng, k, d_max):
    init = rng.dirichlet(np.ones(k) * 2)
    trans = rng.dirichlet(np.ones(k) * 2, size=k)
    np.fill_diagonal(trans, 0)
    trans = trans / trans.sum(axis=1, keepdims=True)
    dur = rng.dirichlet(np.ones(d_max) * 2, size=k)
    log_init = np.log(init)
    with np.errstate(divide="ignore"):
        log_trans = np.log(trans)  # -inf on the (structurally banned) diagonal, by construction
    log_dur_pmf = np.log(dur)
    survival = np.cumsum(dur[:, ::-1], axis=1)[:, ::-1]
    log_dur_survival = np.log(survival)
    return log_init, log_trans, log_dur_pmf, log_dur_survival


def _brute_force_segmentations(loglik, log_init, log_trans, log_dur_pmf, log_dur_survival, t_true, d_max, k):
    """Direct enumeration of every valid segmentation of ticks 0..t_true-1: no consecutive
    repeated states, durations 1..d_max, the final segment weighted by survival (censored),
    every other segment weighted by pmf (exact). This is the from-scratch, obviously-correct
    oracle the scan-based implementation is checked against.
    """
    segmentations = []

    def recurse(start, prev_state, prob_so_far, is_first, segs):
        if start == t_true:
            segmentations.append((prob_so_far, list(segs)))
            return
        for state in range(k):
            if not is_first and state == prev_state:
                continue
            trans_p = np.exp(log_init[state]) if is_first else np.exp(log_trans[prev_state, state])
            if trans_p == 0:
                continue
            max_d = min(d_max, t_true - start)
            for d in range(1, max_d + 1):
                end = start + d - 1
                seg_ll = np.exp(sum(loglik[start : end + 1, state]))
                is_censored = end == t_true - 1
                dur_p = np.exp(log_dur_survival[state, d - 1] if is_censored else log_dur_pmf[state, d - 1])
                segs.append((start, end, state, d, is_censored))
                recurse(end + 1, state, prob_so_far * trans_p * dur_p * seg_ll, False, segs)
                segs.pop()

    recurse(0, None, 1.0, True, [])
    return segmentations


def _brute_force_stats(segmentations, t_true, d_max, k):
    total = sum(prob for prob, _ in segmentations)
    xi_trans = np.zeros((k, k))
    xi_dur = np.zeros((k, d_max))
    cens = np.zeros((k, d_max))
    gamma = np.zeros((t_true, k))
    for prob, segs in segmentations:
        weight = prob / total
        for i, (start, end, state, d, is_censored) in enumerate(segs):
            gamma[start : end + 1, state] += weight
            if is_censored:
                cens[state, d - 1] += weight
            else:
                xi_dur[state, d - 1] += weight
            if i + 1 < len(segs):
                xi_trans[state, segs[i + 1][2]] += weight
    return total, xi_trans, xi_dur, cens, gamma


@pytest.mark.parametrize(
    "t, k, d_max, t_true",
    [
        (6, 3, 3, 6),   # no padding
        (8, 3, 3, 5),   # padded, right-censored short trial
        (7, 4, 4, 7),   # larger K/D_max, no padding
    ],
)
def test_forward_backward_matches_bruteforce(t, k, d_max, t_true):
    rng = np.random.default_rng(0)
    log_init, log_trans, log_dur_pmf, log_dur_survival = _random_log_probs(rng, k, d_max)
    loglik_np = rng.standard_normal((t, k)) * 0.4
    mask_np = np.array([True] * t_true + [False] * (t - t_true))

    segmentations = _brute_force_segmentations(
        loglik_np, log_init, log_trans, log_dur_pmf, log_dur_survival, t_true, d_max, k
    )
    bf_total, bf_xi_trans, bf_xi_dur, bf_cens, bf_gamma = _brute_force_stats(segmentations, t_true, d_max, k)

    loglik = jnp.array(loglik_np)
    mask = jnp.array(mask_np)
    log_init_j, log_trans_j = jnp.array(log_init), jnp.array(log_trans)
    log_dur_pmf_j, log_dur_survival_j = jnp.array(log_dur_pmf), jnp.array(log_dur_survival)

    log_norm, _ = messages.forward_pass(
        loglik, log_init_j, log_trans_j, log_dur_pmf_j, log_dur_survival_j, mask, d_max
    )
    assert jnp.isclose(jnp.exp(jnp.sum(log_norm)), bf_total, rtol=1e-8)

    xi_trans, xi_dur, cens, gamma, log_z = messages.combine_sufficient_stats(
        loglik, mask, log_init_j, log_trans_j, log_dur_pmf_j, log_dur_survival_j, d_max
    )
    assert jnp.isclose(jnp.exp(log_z), bf_total, rtol=1e-8)
    np.testing.assert_allclose(np.array(xi_trans), bf_xi_trans, atol=1e-8)
    np.testing.assert_allclose(np.array(xi_dur), bf_xi_dur, atol=1e-8)
    np.testing.assert_allclose(np.array(cens), bf_cens, atol=1e-8)
    np.testing.assert_allclose(np.array(gamma)[:t_true], bf_gamma, atol=1e-8)
    np.testing.assert_allclose(np.array(gamma)[:t_true].sum(axis=1), 1.0, atol=1e-8)


def test_starved_mode_no_nan():
    """A dying weak-limit mode (near-zero counts) must stay finite everywhere and become
    genuinely inert (near-zero posterior occupancy/transition involvement) once the other
    states have real, well-separated data to explain -- not merely NaN-free.
    """
    k, vocab_verbs, vocab_nouns, d_max, t = 6, 15, 36, 20, 60
    starved = 0

    key_i, key_t = jax.random.split(jax.random.PRNGKey(7))
    init_counts = jax.random.uniform(key_i, (k,), minval=5.0, maxval=50.0).at[starved].set(0.0)

    trans_counts = jax.random.uniform(key_t, (k, k), minval=5.0, maxval=50.0)
    trans_counts = trans_counts * (1.0 - jnp.eye(k))
    trans_counts = trans_counts.at[starved, :].set(0.0).at[:, starved].set(0.0)

    verb_counts = jnp.full((k, vocab_verbs), 0.5).at[jnp.arange(1, k), jnp.arange(1, k) % vocab_verbs].set(200.0)
    verb_counts = verb_counts.at[starved, :].set(0.0)
    noun_counts = jnp.full((k, vocab_nouns), 0.5).at[jnp.arange(1, k), jnp.arange(1, k) % vocab_nouns].set(200.0)
    noun_counts = noun_counts.at[starved, :].set(0.0)

    p = params.HSMMParams(
        init_counts, trans_counts, verb_counts, noun_counts, jnp.full((k,), 5.0), jnp.full((k,), 0.3)
    )
    log_probs = params.to_log_probs(p, d_max)

    active_states = jax.random.randint(jax.random.PRNGKey(99), (t,), 1, k)
    verb_ids, noun_ids = active_states % vocab_verbs, active_states % vocab_nouns
    mask = jnp.ones((t,), dtype=bool)

    loglik = emissions.sequence_loglik(verb_ids, noun_ids, log_probs.log_emit_v, log_probs.log_emit_n, mask)
    xi_trans, xi_dur, cens, gamma, log_z = messages.combine_sufficient_stats(
        loglik, mask, log_probs.log_init, log_probs.log_trans,
        log_probs.log_dur_pmf, log_probs.log_dur_survival, d_max,
    )

    for arr in (xi_trans, xi_dur, cens, gamma, log_z):
        assert jnp.all(jnp.isfinite(arr))

    assert jnp.max(gamma[:, starved]) < 1e-6
    assert jnp.sum(xi_trans[starved, :]) < 1e-6
    assert jnp.sum(xi_trans[:, starved]) < 1e-6
