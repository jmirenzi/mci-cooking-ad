import functools

import jax
import jax.numpy as jnp
import numpy as np

from cook_ad.anomaly import surprise
from cook_ad.hsmm import emissions, messages, params
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
    assembly is then done via surprise.assemble_trace. Chunked to bound peak memory. Returns a
    list of SurpriseTrace aligned with `sequences`.

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
    return traces
