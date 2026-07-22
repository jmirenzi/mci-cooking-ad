import jax.numpy as jnp
import numpy as np
from scipy.stats import nbinom

from cook_ad.hsmm import params
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
