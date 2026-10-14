"""Trajectories: ancestral samples from a fitted model, and real sequences decoded into the
segment form the error injectors need (synthetic/error_injection.py)."""
import jax.numpy as jnp
import numpy as np
from scipy.stats import nbinom

from cook_ad.hsmm import batch
from cook_ad.hsmm.params import FLOOR, JointHSMMParams, _row_normalize, to_log_tables


def _prob_tables(params):
    """One recipe's (init, trans, verb, noun) probabilities; `params` has unbatched counts
    (e.g. hsmm.params.RecipeView)."""
    return (
        np.asarray(jnp.exp(_row_normalize(params.init_counts, FLOOR))),
        np.asarray(jnp.exp(_row_normalize(params.trans_counts, FLOOR, mask_diag=True))),
        np.asarray(jnp.exp(_row_normalize(params.verb_counts, FLOOR))),
        np.asarray(jnp.exp(_row_normalize(params.noun_counts, FLOOR))),
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


def _prob_tables_joint(params, d_max):
    """Categorical prob tables for every recipe (durations are sampled from dur_r/dur_p)."""
    t = to_log_tables(JointHSMMParams(*params[:7]), d_max)
    return tuple(np.asarray(jnp.exp(x)) for x in (t.log_pi, t.log_init, t.log_A, t.log_emit_v,
                                                  t.log_emit_n))


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


def generate_healthy_joint(params, n, rng, max_ticks, d_max):
    return [sample_trajectory_joint(params, rng, max_ticks, d_max) for _ in range(n)]


def trajectories_from_real_joint(params, sequences, d_max, chunk_size=8, r_hat=None):
    """Real sequences (sequences.json layout) -> trajectory dicts carrying the MAP segmentation
    under each trial's recipe: inferred, or the given `r_hat` (e.g. dish labels)."""
    tables = to_log_tables(JointHSMMParams(*params[:7]), d_max)
    verb_ids, noun_ids, mask = batch.pad([s["verb_ids"] for s in sequences],
                                         [s["noun_ids"] for s in sequences])
    if r_hat is None:
        r_hat = batch.infer_recipe(tables, verb_ids, noun_ids, mask, d_max, chunk_size)[0]
    else:
        r_hat = jnp.asarray(np.asarray(r_hat, dtype=np.int64))
    segments = batch.viterbi(tables, r_hat, verb_ids, noun_ids, mask, d_max)
    return [{
        "verb_ids": np.asarray(seq["verb_ids"], dtype=np.int64),
        "noun_ids": np.asarray(seq["noun_ids"], dtype=np.int64),
        "segments": segs,
        "subtask_per_tick": np.repeat(np.array([k for k, _ in segs], dtype=np.int64),
                                      [d for _, d in segs]),
        "recipe_id": int(r_hat[i]),
    } for i, (seq, segs) in enumerate(zip(sequences, segments))]
