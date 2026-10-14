import functools

import jax
import jax.numpy as jnp
import numpy as np

from cook_ad.hsmm import emissions, kernels, params


# Restructure shim: the Viterbi kernel lives in hsmm/kernels.py.
viterbi_decode = kernels.viterbi_decode


@functools.partial(jax.jit, static_argnames=("d_max",))
def _viterbi_batch(log_init, log_trans, log_dur_pmf, log_dur_survival, loglik, mask, d_max):
    return jax.vmap(viterbi_decode, in_axes=(0, None, None, None, None, 0, None))(
        loglik, log_init, log_trans, log_dur_pmf, log_dur_survival, mask, d_max
    )


@functools.partial(jax.jit, static_argnames=("d_max",))
def _viterbi_batch_conditioned(log_init, log_trans, log_dur_pmf, log_dur_survival, loglik, mask, d_max):
    """Same as _viterbi_batch, but the four dynamics tables also vary per trial (axis 0) --
    used when each trial has its own MAP-recipe tables rather than one shared set."""
    return jax.vmap(viterbi_decode, in_axes=(0, 0, 0, 0, 0, 0, None))(
        loglik, log_init, log_trans, log_dur_pmf, log_dur_survival, mask, d_max
    )


traceback = kernels.traceback


def segments_to_per_tick(segments, t_true):
    subtask_per_tick = np.zeros(t_true, dtype=np.int64)
    pos = 0
    for subtask_id, d in segments:
        subtask_per_tick[pos : pos + d] = subtask_id
        pos += d
    return subtask_per_tick


def _assemble_segment_results(dur_bp_all, dur_bp_surv_all, prev_bp_all, asurv_all, mask):
    """Shared numpy traceback loop: batched Viterbi backpointer arrays -> per-sequence
    {"segments", "subtask_per_tick"} dicts. Factored out so both the shared-table path
    (segment_all_from_log_probs) and the per-trial-conditioned path (segment_all_conditioned)
    reuse the exact same assembly logic."""
    dur_bp_np = np.asarray(dur_bp_all)
    dur_bp_surv_np = np.asarray(dur_bp_surv_all)
    prev_bp_np = np.asarray(prev_bp_all)
    asurv_np = np.asarray(asurv_all)
    mask_np = np.asarray(mask)

    results = []
    n = mask_np.shape[0]
    for i in range(n):
        t_true = int(mask_np[i].sum())
        k_star = int(np.argmax(asurv_np[i, t_true - 1, :]))
        segments = traceback(t_true, k_star, dur_bp_np[i], dur_bp_surv_np[i], prev_bp_np[i])
        results.append({
            "segments": segments,
            "subtask_per_tick": segments_to_per_tick(segments, t_true),
        })
    return results


def segment_all_from_log_probs(log_probs, verb_ids, noun_ids, mask, d_max):
    """Batched driver: emissions -> Viterbi decode (jit+vmap) -> per-sequence numpy
    traceback, given already-normalized HSMMLogProbs (one shared set of tables for the whole
    batch). Returns a list of {"segments": [(subtask_id,duration),...],
    "subtask_per_tick": (T_true,) int64 array} aligned with the input batch order.
    """
    loglik = jax.vmap(emissions.sequence_loglik, in_axes=(0, 0, None, None, 0))(
        verb_ids, noun_ids, log_probs.log_emit_v, log_probs.log_emit_n, mask
    )
    _, dur_bp_all, asurv_all, dur_bp_surv_all, prev_bp_all = _viterbi_batch(
        log_probs.log_init, log_probs.log_trans, log_probs.log_dur_pmf, log_probs.log_dur_survival,
        loglik, mask, d_max,
    )
    return _assemble_segment_results(dur_bp_all, dur_bp_surv_all, prev_bp_all, asurv_all, mask)


def segment_all(hsmm_params, verb_ids, noun_ids, mask, d_max):
    """Batched driver: HSMMParams -> normalize -> segment_all_from_log_probs. Unchanged
    interface/behavior for existing callers."""
    log_probs = params.to_log_probs(hsmm_params, d_max)
    return segment_all_from_log_probs(log_probs, verb_ids, noun_ids, mask, d_max)


def segment_all_conditioned(joint_log_probs, r_hat, verb_ids, noun_ids, mask, d_max):
    """Recipe-conditioned batched Viterbi: each trial i is decoded under its own MAP recipe's
    init/trans/duration tables (gathered from the JointHSMMLogProbs's K_R axis via r_hat),
    while emissions stay shared across recipes. Mirrors segment_all_from_log_probs but with
    per-trial dynamics tables instead of one shared set.
    """
    r_hat = jnp.asarray(r_hat)
    loglik = jax.vmap(emissions.sequence_loglik, in_axes=(0, 0, None, None, 0))(
        verb_ids, noun_ids, joint_log_probs.log_emit_v, joint_log_probs.log_emit_n, mask
    )

    log_init_i = joint_log_probs.log_init[r_hat]                  # (N,K)
    log_trans_i = joint_log_probs.log_trans[r_hat]                # (N,K,K)
    log_dur_pmf_i = joint_log_probs.log_dur_pmf[r_hat]             # (N,K,D)
    log_dur_survival_i = joint_log_probs.log_dur_survival[r_hat]   # (N,K,D)

    _, dur_bp_all, asurv_all, dur_bp_surv_all, prev_bp_all = _viterbi_batch_conditioned(
        log_init_i, log_trans_i, log_dur_pmf_i, log_dur_survival_i, loglik, mask, d_max,
    )
    return _assemble_segment_results(dur_bp_all, dur_bp_surv_all, prev_bp_all, asurv_all, mask)
