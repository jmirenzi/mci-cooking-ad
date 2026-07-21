import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cook_ad.hsmm import emissions, params
from cook_ad.recipe import segmentize

jax.config.update("jax_enable_x64", True)


def _random_log_probs(rng, k, d_max):
    init = rng.dirichlet(np.ones(k) * 2)
    trans = rng.dirichlet(np.ones(k) * 2, size=k)
    np.fill_diagonal(trans, 0)
    trans = trans / trans.sum(axis=1, keepdims=True)
    dur = rng.dirichlet(np.ones(d_max) * 2, size=k)
    log_init = np.log(init)
    with np.errstate(divide="ignore"):
        log_trans = np.log(trans)
    log_dur_pmf = np.log(dur)
    survival = np.cumsum(dur[:, ::-1], axis=1)[:, ::-1]
    log_dur_survival = np.log(survival)
    return log_init, log_trans, log_dur_pmf, log_dur_survival


def _brute_force_map(loglik, log_init, log_trans, log_dur_pmf, log_dur_survival, t_true, d_max, k):
    """Direct enumeration of every valid segmentation (no consecutive repeats, durations
    1..d_max, final segment survival-weighted, all others pmf-weighted), keeping the single
    highest-log-probability one -- the from-scratch oracle for the MAP segmentation, mirroring
    test_messages.py's `_brute_force_segmentations` but tracking the best instead of the sum.
    """
    best = {"logprob": -np.inf, "segs": None}

    def recurse(start, prev_state, logprob_so_far, is_first, segs):
        if start == t_true:
            if logprob_so_far > best["logprob"]:
                best["logprob"] = logprob_so_far
                best["segs"] = list(segs)
            return
        for state in range(k):
            if not is_first and state == prev_state:
                continue
            trans_lp = log_init[state] if is_first else log_trans[prev_state, state]
            if not np.isfinite(trans_lp):
                continue
            max_d = min(d_max, t_true - start)
            for d in range(1, max_d + 1):
                end = start + d - 1
                seg_ll = loglik[start : end + 1, state].sum()
                is_censored = end == t_true - 1
                dur_lp = log_dur_survival[state, d - 1] if is_censored else log_dur_pmf[state, d - 1]
                total = logprob_so_far + trans_lp + dur_lp + seg_ll
                segs.append((state, d))
                recurse(end + 1, state, total, False, segs)
                segs.pop()

    recurse(0, None, 0.0, True, [])
    return best["logprob"], best["segs"]


def _score_segments(segments, loglik, log_init, log_trans, log_dur_pmf, log_dur_survival, t_true):
    logprob = 0.0
    pos = 0
    prev_state = None
    for i, (state, d) in enumerate(segments):
        end = pos + d - 1
        logprob += log_init[state] if i == 0 else log_trans[prev_state, state]
        is_censored = end == t_true - 1
        logprob += log_dur_survival[state, d - 1] if is_censored else log_dur_pmf[state, d - 1]
        logprob += loglik[pos : end + 1, state].sum()
        prev_state = state
        pos = end + 1
    return logprob


@pytest.mark.parametrize(
    "t, k, d_max, t_true",
    [
        (6, 3, 3, 6),   # no padding
        (8, 3, 3, 5),   # padded, right-censored short trial
        (7, 4, 4, 7),   # larger K/D_max, no padding
    ],
)
def test_viterbi_matches_bruteforce(t, k, d_max, t_true):
    rng = np.random.default_rng(1)
    log_init, log_trans, log_dur_pmf, log_dur_survival = _random_log_probs(rng, k, d_max)
    loglik_np = rng.standard_normal((t, k)) * 0.4
    mask_np = np.array([True] * t_true + [False] * (t - t_true))

    bf_logprob, bf_segs = _brute_force_map(
        loglik_np, log_init, log_trans, log_dur_pmf, log_dur_survival, t_true, d_max, k
    )

    loglik = jnp.array(loglik_np)
    mask = jnp.array(mask_np)
    log_init_j, log_trans_j = jnp.array(log_init), jnp.array(log_trans)
    log_dur_pmf_j, log_dur_survival_j = jnp.array(log_dur_pmf), jnp.array(log_dur_survival)

    _, dur_bp_all, asurv_all, dur_bp_surv_all, prev_bp_all = segmentize.viterbi_decode(
        loglik, log_init_j, log_trans_j, log_dur_pmf_j, log_dur_survival_j, mask, d_max
    )

    k_star = int(jnp.argmax(asurv_all[t_true - 1, :]))
    assert jnp.isclose(asurv_all[t_true - 1, k_star], bf_logprob, atol=1e-8)

    decoded = segmentize.traceback(
        t_true, k_star, np.array(dur_bp_all), np.array(dur_bp_surv_all), np.array(prev_bp_all)
    )
    decoded_logprob = _score_segments(
        decoded, loglik_np, log_init, log_trans, log_dur_pmf, log_dur_survival, t_true
    )
    assert np.isclose(decoded_logprob, bf_logprob, atol=1e-8)
    assert decoded == bf_segs

    per_tick = segmentize.segments_to_per_tick(decoded, t_true)
    assert per_tick.shape == (t_true,)
    assert sum(d for _, d in decoded) == t_true


def test_starved_mode_no_nan_or_inf():
    """A dying weak-limit mode must never poison the max-product scan with NaN/Inf, even
    when it's structurally disconnected (zero init/trans/emission mass) -- mirrors
    test_messages.py's starved-mode check, adapted to the max-product outputs.
    """
    k, vocab_verbs, vocab_nouns, d_max, t = 6, 15, 36, 20, 40
    starved = 0

    key_i, key_t = jax.random.split(jax.random.PRNGKey(3))
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

    active_states = jax.random.randint(jax.random.PRNGKey(11), (t,), 1, k)
    verb_ids, noun_ids = active_states % vocab_verbs, active_states % vocab_nouns
    mask = jnp.ones((t,), dtype=bool)

    loglik = emissions.sequence_loglik(verb_ids, noun_ids, log_probs.log_emit_v, log_probs.log_emit_n, mask)
    astar_all, dur_bp_all, asurv_all, dur_bp_surv_all, prev_bp_all = segmentize.viterbi_decode(
        loglik, log_probs.log_init, log_probs.log_trans,
        log_probs.log_dur_pmf, log_probs.log_dur_survival, mask, d_max,
    )

    assert jnp.all(jnp.isfinite(astar_all[:, 1:]))
    assert jnp.all(jnp.isfinite(asurv_all[:, 1:]))
    k_star = int(jnp.argmax(asurv_all[-1, :]))
    assert k_star != starved

    decoded = segmentize.traceback(t, k_star, np.array(dur_bp_all), np.array(dur_bp_surv_all), np.array(prev_bp_all))
    assert all(state != starved for state, _ in decoded)
    assert sum(d for _, d in decoded) == t
