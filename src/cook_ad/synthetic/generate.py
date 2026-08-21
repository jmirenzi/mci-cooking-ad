import jax.numpy as jnp
import numpy as np
from scipy.stats import nbinom

from cook_ad.hsmm import joint_em, joint_params, params
from cook_ad.recipe import segmentize


def _prob_tables(hsmm_params):
    log_init, log_trans, log_emit_v, log_emit_n = params.normalize_categoricals(hsmm_params)
    return (
        np.asarray(jnp.exp(log_init)),
        np.asarray(jnp.exp(log_trans)),   # diagonal is 0 (self-transitions banned)
        np.asarray(jnp.exp(log_emit_v)),
        np.asarray(jnp.exp(log_emit_n)),
    )


def _segments_to_arrays(segments, verb_by_seg, noun_by_seg):
    verb_ids = np.concatenate(verb_by_seg) if verb_by_seg else np.zeros(0, dtype=np.int64)
    noun_ids = np.concatenate(noun_by_seg) if noun_by_seg else np.zeros(0, dtype=np.int64)
    subtask_per_tick = np.concatenate(
        [np.full(d, s, dtype=np.int64) for s, d in segments]
    ) if segments else np.zeros(0, dtype=np.int64)
    return verb_ids, noun_ids, subtask_per_tick


def sample_trajectory(hsmm_params, rng, max_ticks, d_max):
    """Ancestral sample from the frozen HSMM: a healthy trajectory with EXACT ground truth,
    since we know every segment/state/duration because we sampled them. s0 ~ init; per segment
    d ~ NB(dur_r[s], dur_p[s]) clamped to [1, d_max], emit d ticks of v ~ P(v|s) and n ~ P(n|s)
    independently (the product-model assumption), then s' ~ trans[s] (diagonal already 0). The
    final segment is trimmed so the sequence is exactly max_ticks. Returns a trajectory dict.
    """
    init_p, trans_p, verb_p, noun_p = _prob_tables(hsmm_params)
    dur_r = np.asarray(hsmm_params.dur_r)
    dur_p = np.asarray(hsmm_params.dur_p)
    k = init_p.shape[0]

    segments = []
    verb_by_seg = []
    noun_by_seg = []
    total = 0
    state = int(rng.choice(k, p=init_p))

    while total < max_ticks:
        d = int(nbinom.rvs(dur_r[state], dur_p[state], random_state=rng)) + 1
        d = min(max(d, 1), d_max, max_ticks - total)  # trim last segment to hit max_ticks exactly
        verbs = rng.choice(verb_p.shape[1], size=d, p=verb_p[state])
        nouns = rng.choice(noun_p.shape[1], size=d, p=noun_p[state])

        segments.append((state, d))
        verb_by_seg.append(verbs)
        noun_by_seg.append(nouns)
        total += d
        state = int(rng.choice(k, p=trans_p[state]))

    verb_ids, noun_ids, subtask_per_tick = _segments_to_arrays(segments, verb_by_seg, noun_by_seg)
    return {
        "verb_ids": verb_ids,
        "noun_ids": noun_ids,
        "segments": segments,
        "subtask_per_tick": subtask_per_tick,
    }


def generate_healthy(hsmm_params, n, rng, max_ticks, d_max):
    return [sample_trajectory(hsmm_params, rng, max_ticks, d_max) for _ in range(n)]


def trajectory_from_real(hsmm_params, verb_ids, noun_ids, d_max):
    """Adapter: turn a real Breakfast (verb_ids, noun_ids) trial into the same trajectory dict
    the synthetic sampler produces, by Viterbi-segmenting it with the fitted model. Lets the
    injectors and metrics run unchanged on both healthy sources; here the ground-truth segment
    boundaries are the decoded ones (fuzzier than the synthetic sampler's exact boundaries).
    """
    verb_ids = np.asarray(verb_ids, dtype=np.int64)
    noun_ids = np.asarray(noun_ids, dtype=np.int64)
    mask = jnp.ones((1, verb_ids.shape[0]), dtype=bool)
    result = segmentize.segment_all(
        hsmm_params, jnp.asarray(verb_ids)[None, :], jnp.asarray(noun_ids)[None, :], mask, d_max
    )[0]
    return {
        "verb_ids": verb_ids,
        "noun_ids": noun_ids,
        "segments": result["segments"],
        "subtask_per_tick": result["subtask_per_tick"],
    }


def _prob_tables_joint(joint_hsmm_params, d_max):
    """Categorical prob tables for every recipe, reusing joint_params.to_log_probs_joint
    (its duration tables are unused here -- sampling draws from dur_r/dur_p directly via
    scipy, the same convention _prob_tables/sample_trajectory already use)."""
    log_probs = joint_params.to_log_probs_joint(joint_hsmm_params, d_max)
    return (
        np.asarray(jnp.exp(log_probs.log_pi)),
        np.asarray(jnp.exp(log_probs.log_init)),
        np.asarray(jnp.exp(log_probs.log_trans)),
        np.asarray(jnp.exp(log_probs.log_emit_v)),
        np.asarray(jnp.exp(log_probs.log_emit_n)),
    )


def sample_trajectory_joint(joint_hsmm_params, rng, max_ticks, d_max):
    """Joint-model analogue of sample_trajectory: draws R ~ pi once per trial, then samples
    from that recipe's own init/trans/duration and the shared emissions -- so synthetic
    healthy trials are genuinely recipe-structured, with the sampled recipe id returned as
    ground truth (trajectory dicts gain a "recipe_id" key; existing consumers that only read
    verb_ids/noun_ids/segments, e.g. synthetic.error_injection, are unaffected).
    """
    pi_p, init_p, trans_p, verb_p, noun_p = _prob_tables_joint(joint_hsmm_params, d_max)
    dur_r = np.asarray(joint_hsmm_params.dur_r)
    dur_p = np.asarray(joint_hsmm_params.dur_p)
    k_recipe, k = init_p.shape

    recipe = int(rng.choice(k_recipe, p=pi_p))

    segments = []
    verb_by_seg = []
    noun_by_seg = []
    total = 0
    state = int(rng.choice(k, p=init_p[recipe]))

    while total < max_ticks:
        d = int(nbinom.rvs(dur_r[recipe, state], dur_p[recipe, state], random_state=rng)) + 1
        d = min(max(d, 1), d_max, max_ticks - total)
        verbs = rng.choice(verb_p.shape[1], size=d, p=verb_p[state])
        nouns = rng.choice(noun_p.shape[1], size=d, p=noun_p[state])

        segments.append((state, d))
        verb_by_seg.append(verbs)
        noun_by_seg.append(nouns)
        total += d
        state = int(rng.choice(k, p=trans_p[recipe, state]))

    verb_ids, noun_ids, subtask_per_tick = _segments_to_arrays(segments, verb_by_seg, noun_by_seg)
    return {
        "verb_ids": verb_ids,
        "noun_ids": noun_ids,
        "segments": segments,
        "subtask_per_tick": subtask_per_tick,
        "recipe_id": recipe,
    }


def generate_healthy_joint(joint_hsmm_params, n, rng, max_ticks, d_max):
    return [sample_trajectory_joint(joint_hsmm_params, rng, max_ticks, d_max) for _ in range(n)]


def trajectory_from_real_joint(joint_hsmm_params, verb_ids, noun_ids, d_max):
    """Adapter: joint-model analogue of trajectory_from_real. Infers the trial's MAP recipe
    via joint_em.infer_recipe, then Viterbi-segments under that recipe's own tables
    (segmentize.segment_all_conditioned) instead of the cascade's single shared table set.
    """
    verb_ids = np.asarray(verb_ids, dtype=np.int64)
    noun_ids = np.asarray(noun_ids, dtype=np.int64)
    verb_j = jnp.asarray(verb_ids)[None, :]
    noun_j = jnp.asarray(noun_ids)[None, :]
    mask = jnp.ones((1, verb_ids.shape[0]), dtype=bool)

    r_hat, _, _ = joint_em.infer_recipe(joint_hsmm_params, verb_j, noun_j, mask, d_max, chunk_size=1)
    log_probs = joint_params.to_log_probs_joint(joint_hsmm_params, d_max)
    result = segmentize.segment_all_conditioned(log_probs, r_hat, verb_j, noun_j, mask, d_max)[0]
    return {
        "verb_ids": verb_ids,
        "noun_ids": noun_ids,
        "segments": result["segments"],
        "subtask_per_tick": result["subtask_per_tick"],
        "recipe_id": int(r_hat[0]),
    }


def trajectories_from_real_joint(joint_hsmm_params, sequences, d_max, chunk_size=8):
    """Batched `trajectory_from_real_joint`: same result, one JAX compile instead of one per
    distinct trial length. `sequences` is sequences.json's layout.
    """
    from cook_ad.hsmm import em as _em

    verb_ids, noun_ids, mask = _em.pad_batch(sequences)
    r_hat, _, _ = joint_em.infer_recipe(
        joint_hsmm_params, verb_ids, noun_ids, mask, d_max, chunk_size=chunk_size
    )
    log_probs = joint_params.to_log_probs_joint(joint_hsmm_params, d_max)
    results = segmentize.segment_all_conditioned(log_probs, r_hat, verb_ids, noun_ids, mask, d_max)

    out = []
    for i, seq in enumerate(sequences):
        out.append({
            "verb_ids": np.asarray(seq["verb_ids"], dtype=np.int64),
            "noun_ids": np.asarray(seq["noun_ids"], dtype=np.int64),
            "segments": results[i]["segments"],
            "subtask_per_tick": results[i]["subtask_per_tick"],
            "recipe_id": int(r_hat[i]),
        })
    return out
