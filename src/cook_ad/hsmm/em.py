import functools

import jax
import jax.numpy as jnp
import numpy as np

from cook_ad.hsmm import durations, emissions, messages, params

jax.config.update("jax_enable_x64", True)


def pad_batch(sequences, t_max=None):
    """sequences: list of {"verb_ids": [...], "noun_ids": [...]} (Phase 1's sequences.json).
    Returns (verb_ids, noun_ids: (N,T_max) int32, mask: (N,T_max) bool). Pads with the
    valid dummy id 0 -- never read meaningfully, gated out everywhere via `mask`.
    `t_max` is derived from the data if not given, never hardcoded.
    """
    lengths = [len(seq["verb_ids"]) for seq in sequences]
    if t_max is None:
        t_max = max(lengths)

    n = len(sequences)
    verb_ids = np.zeros((n, t_max), dtype=np.int32)
    noun_ids = np.zeros((n, t_max), dtype=np.int32)
    mask = np.zeros((n, t_max), dtype=bool)
    for i, seq in enumerate(sequences):
        length = len(seq["verb_ids"])
        verb_ids[i, :length] = seq["verb_ids"]
        noun_ids[i, :length] = seq["noun_ids"]
        mask[i, :length] = True

    return jnp.array(verb_ids), jnp.array(noun_ids), jnp.array(mask)


@functools.partial(jax.jit, static_argnames=("d_max",))
def _e_step_chunk(log_init, log_trans, log_emit_v, log_emit_n, log_dur_pmf, log_dur_survival,
                   verb_ids, noun_ids, mask, d_max):
    """The batched E-step for one chunk of sequences: vmap emissions + forward/backward/
    combination across the chunk's batch axis, then sum sufficient stats over it.
    """
    loglik = jax.vmap(emissions.sequence_loglik, in_axes=(0, 0, None, None, 0))(
        verb_ids, noun_ids, log_emit_v, log_emit_n, mask
    )

    combine = jax.vmap(messages.combine_sufficient_stats, in_axes=(0, 0, None, None, None, None, None))
    xi_trans, xi_dur, cens, gamma, log_z = combine(
        loglik, mask, log_init, log_trans, log_dur_pmf, log_dur_survival, d_max
    )

    gamma_masked = jnp.where(mask[:, :, None], gamma, 0.0)
    n_verb, n_noun = log_emit_v.shape[1], log_emit_n.shape[1]
    verb_onehot = jax.nn.one_hot(verb_ids, n_verb, dtype=gamma.dtype)
    noun_onehot = jax.nn.one_hot(noun_ids, n_noun, dtype=gamma.dtype)

    stats = {
        "init_counts": jnp.sum(gamma_masked[:, 0, :], axis=0),
        "trans_counts": jnp.sum(xi_trans, axis=0),
        "verb_counts": jnp.einsum("ntk,ntv->kv", gamma_masked, verb_onehot),
        "noun_counts": jnp.einsum("ntk,ntv->kv", gamma_masked, noun_onehot),
        "xi_dur": jnp.sum(xi_dur, axis=0),
        "cens": jnp.sum(cens, axis=0),
    }
    return stats, jnp.sum(log_z)


def e_step(hsmm_params, verb_ids, noun_ids, mask, d_max, temperature=1.0, chunk_size=8):
    """to_log_probs -> (if annealing) temper emission/duration terms -> chunked vmap of
    emissions + forward/backward/combination across the batch axis -> sum sufficient stats.

    Chunked rather than one vmap over the full batch: at K=64, T_max~650, a full 503-
    sequence vmap allocates well over 10GB for the (T,K,K) transition-posterior and
    duration-histogram intermediates (measured directly -- triggers a real OOM), even
    after those were already reorganized as per-sequence scans (see messages.py). Chunking
    the batch axis bounds peak memory to O(chunk_size) regardless of total dataset size.
    """
    log_probs = params.to_log_probs(hsmm_params, d_max)
    inv_temp = 1.0 / temperature
    log_emit_v = log_probs.log_emit_v * inv_temp
    log_emit_n = log_probs.log_emit_n * inv_temp
    log_dur_pmf = log_probs.log_dur_pmf * inv_temp
    log_dur_survival = log_probs.log_dur_survival * inv_temp

    n = verb_ids.shape[0]
    total_stats = None
    total_loglik = 0.0
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk_stats, chunk_loglik = _e_step_chunk(
            log_probs.log_init, log_probs.log_trans, log_emit_v, log_emit_n,
            log_dur_pmf, log_dur_survival,
            verb_ids[start:end], noun_ids[start:end], mask[start:end], d_max,
        )
        if total_stats is None:
            total_stats = chunk_stats
        else:
            total_stats = {name: total_stats[name] + chunk_stats[name] for name in total_stats}
        total_loglik = total_loglik + chunk_loglik

    return total_stats, total_loglik


@functools.partial(jax.jit, static_argnames=("d_max",))
def m_step(hsmm_params, stats, alpha_init, alpha_trans, alpha_emit_v, alpha_emit_n, d_max):
    """new_*_counts = alpha/width + this E-step's expected data counts, for the four
    Dirichlet families -- a fresh posterior each iteration (standard batch MAP-EM), not an
    accumulation across iterations. Duration goes through the censoring imputation and
    Newton refinement in durations.py.
    """
    k = hsmm_params.init_counts.shape[0]
    n_verb = hsmm_params.verb_counts.shape[1]
    n_noun = hsmm_params.noun_counts.shape[1]

    new_init_counts = alpha_init / k + stats["init_counts"]
    new_trans_counts = (alpha_trans / k + stats["trans_counts"]) * (1.0 - jnp.eye(k))
    new_verb_counts = alpha_emit_v / n_verb + stats["verb_counts"]
    new_noun_counts = alpha_emit_n / n_noun + stats["noun_counts"]

    n_hat = durations.impute_censored_histogram(
        stats["xi_dur"], stats["cens"], hsmm_params.dur_r, hsmm_params.dur_p, d_max
    )
    n_hat_total, s_hat = durations.duration_stats_from_histogram(n_hat, d_max)
    new_r = durations.newton_update_r(n_hat, n_hat_total, s_hat, hsmm_params.dur_r, n_iters=5)
    new_p = durations.update_p_given_r(n_hat_total, s_hat, new_r)

    return params.HSMMParams(new_init_counts, new_trans_counts, new_verb_counts, new_noun_counts, new_r, new_p)


def _anneal_schedule(iteration, t_start=2.0, decay=0.9):
    return max(1.0, t_start * decay**iteration)


def run_em(
    key,
    sequences,
    k_subtask,
    d_max,
    vocab_verbs,
    vocab_nouns,
    alpha_init=0.5,
    alpha_trans=0.5,
    alpha_emit_v=None,
    alpha_emit_n=None,
    n_restarts=10,
    max_iters=100,
    tol=1e-4,
    annealing=False,
    chunk_size=8,
):
    """Plain Python loop over restarts (embarrassingly parallel via vmap later if profiling
    says so -- not built up front). Each restart: fresh random init, iterate E/M to
    convergence or max_iters, keep the best by final total log-likelihood (no held-out
    metric exists at this phase). Returns (best_params, best_loglik, history).
    """
    if alpha_emit_v is None:
        alpha_emit_v = float(vocab_verbs)
    if alpha_emit_n is None:
        alpha_emit_n = float(vocab_nouns)

    verb_ids, noun_ids, mask = pad_batch(sequences)

    best_params, best_loglik = None, -jnp.inf
    history = []

    for restart_key in jax.random.split(key, n_restarts):
        p = params.init_weak_limit_params(
            restart_key,
            k_subtask,
            vocab_verbs,
            vocab_nouns,
            d_max,
            alpha_init=alpha_init,
            alpha_trans=alpha_trans,
            alpha_emit_v=alpha_emit_v,
            alpha_emit_n=alpha_emit_n,
        )
        prev_loglik = -jnp.inf
        loglik = -jnp.inf
        restart_history = []
        for iteration in range(max_iters):
            temperature = _anneal_schedule(iteration) if annealing else 1.0
            stats, loglik = e_step(p, verb_ids, noun_ids, mask, d_max, temperature, chunk_size)
            p = m_step(p, stats, alpha_init, alpha_trans, alpha_emit_v, alpha_emit_n, d_max)
            restart_history.append(float(loglik))
            if abs(float(loglik) - float(prev_loglik)) < tol:
                break
            prev_loglik = loglik

        history.append(restart_history)
        if float(loglik) > float(best_loglik):
            best_params, best_loglik = p, loglik

    return best_params, best_loglik, history
