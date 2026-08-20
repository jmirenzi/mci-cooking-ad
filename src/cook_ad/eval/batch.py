import functools

import jax
import jax.numpy as jnp
import numpy as np

from cook_ad.anomaly import surprise
from cook_ad.hsmm import emissions, joint_em, joint_params, messages, params
from cook_ad.recipe import recipe_hmm, segmentize


@functools.partial(jax.jit, static_argnames=("d_max",))
def _batched_pi(verb_ids, noun_ids, mask, log_init, log_trans, log_emit_v, log_emit_n,
                log_dur_pmf, log_dur_survival, d_max):
    loglik = jax.vmap(emissions.sequence_loglik, in_axes=(0, 0, None, None, 0))(
        verb_ids, noun_ids, log_emit_v, log_emit_n, mask
    )
    return jax.vmap(messages.predictive_occupancy, in_axes=(0, None, None, None, None, 0, None))(
        loglik, log_init, log_trans, log_dur_pmf, log_dur_survival, mask, d_max
    )


@jax.jit
def _batched_recipe_gamma(obs_ids, mask, rlog_init, rlog_trans, rlog_emit):
    gamma, _, _ = jax.vmap(recipe_hmm._forward_backward, in_axes=(0, 0, None, None, None))(
        obs_ids, mask, rlog_init, rlog_trans, rlog_emit
    )
    return gamma


@functools.partial(jax.jit, static_argnames=("d_max",))
def _batched_pi_conditioned(verb_ids, noun_ids, mask, log_init, log_trans, log_emit_v, log_emit_n,
                             log_dur_pmf, log_dur_survival, d_max):
    """Same as _batched_pi, but log_init/log_trans/log_dur_pmf/log_dur_survival vary per trial
    (already gathered to that trial's MAP recipe, axis 0) instead of being one shared table
    broadcast across the batch."""
    loglik = jax.vmap(emissions.sequence_loglik, in_axes=(0, 0, None, None, 0))(
        verb_ids, noun_ids, log_emit_v, log_emit_n, mask
    )
    return jax.vmap(messages.predictive_occupancy, in_axes=(0, 0, 0, 0, 0, 0, None))(
        loglik, log_init, log_trans, log_dur_pmf, log_dur_survival, mask, d_max
    )


def _pad(sequences, t_max):
    n = len(sequences)
    verb = np.zeros((n, t_max), dtype=np.int32)
    noun = np.zeros((n, t_max), dtype=np.int32)
    mask = np.zeros((n, t_max), dtype=bool)
    for i, s in enumerate(sequences):
        length = len(s["verb_ids"])
        verb[i, :length] = s["verb_ids"]
        noun[i, :length] = s["noun_ids"]
        mask[i, :length] = True
    return jnp.asarray(verb), jnp.asarray(noun), jnp.asarray(mask)


def compute_traces(hsmm_params, recipe_params, sequences, d_max, chunk_size=16):
    """Batched analogue of surprise.compute_trace over many trials. The three jax-heavy ops --
    predictive_occupancy, Viterbi segmentation, and the recipe forward-backward -- are run
    vmapped over a padded batch (padded to a single global T_max so the big ops compile ONCE,
    not per distinct sequence length as the single-trial path would). The cheap per-trial numpy
    assembly is then done via surprise.assemble_trace. Chunked to bound peak memory. Returns
    (traces, log_probs, rlog_trans): a list of SurpriseTrace aligned with `sequences`, plus the
    HSMMLogProbs and recipe transition matrix built here -- otherwise discarded -- which
    surprise.flag() needs and would have to rebuild.

    sequences: list of dicts with `verb_ids`/`noun_ids` (numpy int arrays, per-trial true length).
    """
    log_probs = params.to_log_probs(hsmm_params, d_max)
    rlog_init, rlog_trans, rlog_emit = recipe_hmm.to_log_probs(recipe_params)
    t_max = max(len(s["verb_ids"]) for s in sequences)

    traces = []
    for start in range(0, len(sequences), chunk_size):
        chunk = sequences[start : start + chunk_size]
        verb, noun, mask = _pad(chunk, t_max)

        pi = _batched_pi(
            verb, noun, mask, log_probs.log_init, log_probs.log_trans,
            log_probs.log_emit_v, log_probs.log_emit_n,
            log_probs.log_dur_pmf, log_probs.log_dur_survival, d_max,
        )
        seg_results = segmentize.segment_all(hsmm_params, verb, noun, mask, d_max)

        seg_symbols = [[s for s, _ in r["segments"]] for r in seg_results]
        obs_ids, seg_mask = recipe_hmm.pad_segment_batch(seg_symbols)
        gammas = np.asarray(_batched_recipe_gamma(obs_ids, seg_mask, rlog_init, rlog_trans, rlog_emit))
        pi = np.asarray(pi)

        for i, s in enumerate(chunk):
            length = len(s["verb_ids"])
            n_seg = len(seg_results[i]["segments"])
            seg_recipe_ids = np.argmax(gammas[i][:n_seg], axis=-1)
            traces.append(
                surprise.assemble_trace(
                    hsmm_params, log_probs, rlog_trans, pi[i][:length],
                    s["verb_ids"], s["noun_ids"], seg_results[i], seg_recipe_ids,
                )
            )
    return traces, log_probs, rlog_trans


def compute_traces_joint(joint_hsmm_params, sequences, d_max, chunk_size=16, r_hat=None):
    """Joint-model analogue of compute_traces. Recipe inference is now a direct EM readout
    (joint_em.infer_recipe) rather than a second forward-backward pass over segment symbols,
    so it's computed once for the whole dataset up front -- it's cheap (forward-pass only, see
    joint_em._recipe_logz_chunk), and every trial's r_hat is needed before that trial's
    recipe-conditioned segmentation/occupancy can run. Segmentation and the predictive-
    occupancy prior are both now recipe-conditioned: each trial is decoded under its own
    r_hat's tables (segment_all_conditioned / _batched_pi_conditioned) rather than one shared
    set. Returns (traces, log_probs, r_hat, log_trans_marginal): a list of SurpriseTrace aligned
    with `sequences`, the JointHSMMLogProbs, the whole-dataset r_hat array (index per trial),
    and the pi-weighted marginal transition matrix -- all four are what surprise.flag_joint()
    needs per trial and would otherwise have to be rebuilt.

    `r_hat`: optionally supply the per-trial recipe assignment instead of inferring it. The
    detector never should -- it has to read the recipe off the trial in front of it -- but an
    ORACLE analysis does: the MAP recipe is re-inferred from the degraded stream, and on a
    transposition it flips away from the trial's real recipe most of the time, which quietly
    relicenses the very transition the injection created. Pinning r_hat to the healthy decode's
    value is how that loss is sized (tools_oracle_recipe.py).
    """
    log_probs = joint_params.to_log_probs_joint(joint_hsmm_params, d_max)
    t_max = max(len(s["verb_ids"]) for s in sequences)

    verb_all, noun_all, mask_all = _pad(sequences, t_max)
    if r_hat is None:
        r_hat, _, _ = joint_em.infer_recipe(
            joint_hsmm_params, verb_all, noun_all, mask_all, d_max, chunk_size=chunk_size
        )
    else:
        r_hat = jnp.asarray(r_hat)
    log_trans_marginal = joint_params.marginal_log_trans(log_probs)

    traces = []
    for start in range(0, len(sequences), chunk_size):
        chunk = sequences[start : start + chunk_size]
        verb, noun, mask = _pad(chunk, t_max)
        r_hat_chunk = r_hat[start : start + chunk_size]

        log_init_i = log_probs.log_init[r_hat_chunk]
        log_trans_i = log_probs.log_trans[r_hat_chunk]
        log_dur_pmf_i = log_probs.log_dur_pmf[r_hat_chunk]
        log_dur_survival_i = log_probs.log_dur_survival[r_hat_chunk]

        pi = _batched_pi_conditioned(
            verb, noun, mask, log_init_i, log_trans_i, log_probs.log_emit_v, log_probs.log_emit_n,
            log_dur_pmf_i, log_dur_survival_i, d_max,
        )
        seg_results = segmentize.segment_all_conditioned(log_probs, r_hat_chunk, verb, noun, mask, d_max)
        pi = np.asarray(pi)

        for i, s in enumerate(chunk):
            length = len(s["verb_ids"])
            traces.append(
                surprise.assemble_trace_joint(
                    joint_hsmm_params, log_probs, int(r_hat_chunk[i]), log_trans_marginal,
                    pi[i][:length], s["verb_ids"], s["noun_ids"], seg_results[i],
                )
            )
    return traces, log_probs, r_hat, log_trans_marginal
