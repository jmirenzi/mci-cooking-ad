from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

from cook_ad.anomaly import temporal
from cook_ad.hsmm import emissions, messages, params
from cook_ad.recipe import recipe_hmm, segmentize

# Emission/transition channels are raw neg-log-probs; their thresholds are plain nats.
# The temporal channels (s_temporal / s_dur_two) are -log(tail probability), so a single
# global threshold -log(alpha) is automatically per-state calibrated (see temporal.py) -- it
# fires exactly when the duration reaches that state's alpha tail quantile. Not hand-tuned per
# state; that per-state calibration is precisely the point of the tail-probability framing.
DEFAULT_ALPHA = 0.05

DEFAULT_THRESHOLDS = {
    "s_emit": 6.0,
    "s_verb": 4.0,
    "s_noun": 4.0,
    "s_temporal": -float(np.log(DEFAULT_ALPHA)),
    "s_dur_two": -float(np.log(DEFAULT_ALPHA)),
    "s_transition": 4.0,
    "s_recipe_transition": 4.0,
}

DEFAULT_ATTRIBUTION_MARGIN = 2.0


class SurpriseTrace(NamedTuple):
    s_emit: np.ndarray               # (T,) observational anomaly, -log P(v,n|Z_t,o_{<t})
    s_verb: np.ndarray               # (T,) verb-isolated channel
    s_noun: np.ndarray                # (T,) noun-isolated channel
    s_temporal: np.ndarray            # (T,) live 'stuck' surprise -log P(D>=d_elapsed), monotone within a segment
    s_dur_long: np.ndarray            # (T,) retrospective -log P(D>=d), at segment-end ticks (0 elsewhere)
    s_dur_short: np.ndarray           # (T,) retrospective -log P(D<=d), at segment-end ticks (0 elsewhere)
    s_dur_two: np.ndarray             # (T,) retrospective two-sided p-value surprise, at segment-end ticks
    s_transition: np.ndarray          # (T,) subtask-transition surprise, 0 except segment starts
    s_recipe_transition: np.ndarray   # (T,) recipe-transition surprise, 0 except segment starts
    pit: np.ndarray                    # (T,) mid-PIT duration coordinate at segment-end ticks (NaN elsewhere), calibration diagnostic
    z_star: np.ndarray                 # (T,) believed subtask (Viterbi) per tick
    expected_verb: np.ndarray          # (T,) argmax_v P(v|z_star) per tick
    expected_noun: np.ndarray          # (T,) argmax_n P(n|z_star) per tick
    expected_next_state: np.ndarray    # (T,) argmax next subtask at segment starts, -1 elsewhere
    expected_next_recipe: np.ndarray   # (T,) argmax next recipe at segment starts, -1 elsewhere
    attribution: np.ndarray            # (T,) "item"/"action"/"none" per tick (emission attribution)
    temporal_attribution: np.ndarray   # (T,) "stuck"/"left_early"/"none" at segment-end ticks


def emission_surprise(pi_all, log_emit_v, log_emit_n, verb_ids, noun_ids):
    """pi_all: (T,K) log P(Z_t=k|o_{<t}). log_emit_v/n: (K,V)/(K,N). verb_ids/noun_ids: (T,).

    Marginalizes verb, noun, and the joint under the SAME predictive weighting pi_all, which
    is what keeps S_verb/S_noun on a shared scale and makes the item-vs-action attribution
    meaningful (Model_descript.md's Product Model / conditional-independence argument).
    """
    term_v = log_emit_v[:, verb_ids].T  # (T,K)
    term_n = log_emit_n[:, noun_ids].T  # (T,K)
    s_emit = -logsumexp(pi_all + term_v + term_n, axis=-1)
    s_verb = -logsumexp(pi_all + term_v, axis=-1)
    s_noun = -logsumexp(pi_all + term_n, axis=-1)
    return s_emit, s_verb, s_noun


def attribute(s_verb, s_noun, margin=DEFAULT_ATTRIBUTION_MARGIN):
    """S_verb << S_noun => the item is anomalous, not the action (and vice versa), per
    Model_descript.md's decomposition. Ties within `margin` nats are left unattributed."""
    s_verb_np = np.asarray(s_verb)
    s_noun_np = np.asarray(s_noun)
    labels = np.full(s_verb_np.shape[0], "none", dtype=object)
    labels[(s_noun_np - s_verb_np) > margin] = "item"
    labels[(s_verb_np - s_noun_np) > margin] = "action"
    return labels


def _scatter_segment_end(segments, per_segment_values, fill=0.0, dtype=np.float64):
    """Place one value per segment at that segment's LAST tick (segment-completion events),
    filling all other ticks with `fill` -- the retrospective/PIT analogue of how
    transition_surprise places values at segment *start* ticks."""
    t_true = sum(d for _, d in segments)
    out = np.full(t_true, fill, dtype=dtype)
    pos = 0
    for (state, d), value in zip(segments, per_segment_values):
        out[pos + d - 1] = value
        pos += d
    return out


def transition_surprise(segments, log_trans):
    """segments: [(state, duration), ...] covering a full trial. Returns (s_transition,
    expected_next_state), each (T,): nonzero/valid only at a segment's first tick (0 and -1
    elsewhere), since a subtask transition is a segment-boundary event, not a per-tick one."""
    t_true = sum(d for _, d in segments)
    s_trans = np.zeros(t_true, dtype=np.float64)
    expected_next = np.full(t_true, -1, dtype=np.int64)
    log_trans_np = np.asarray(log_trans)

    pos = 0
    prev_state = None
    for state, d in segments:
        if prev_state is not None:
            s_trans[pos] = float(-log_trans_np[prev_state, state])
            expected_next[pos] = int(np.argmax(log_trans_np[prev_state]))
        pos += d
        prev_state = state
    return s_trans, expected_next


def _segment_recipe_path(recipe_params, subtask_symbols):
    """Per-segment (not per-trial majority-voted) recipe argmax -- the segment-indexed
    sequence the recipe HMM actually operates over, reusing recipe_hmm's own
    forward-backward rather than re-deriving it."""
    obs_ids, mask = recipe_hmm.pad_segment_batch([subtask_symbols])
    log_init, log_trans, log_emit = recipe_hmm.to_log_probs(recipe_params)

    gamma, _, _ = jax.vmap(recipe_hmm._forward_backward, in_axes=(0, 0, None, None, None))(
        obs_ids, mask, log_init, log_trans, log_emit
    )
    return np.asarray(jnp.argmax(gamma[0], axis=-1))


def recipe_transition_surprise(segments, seg_recipe_ids, recipe_log_trans):
    """Recipe-level analogue of transition_surprise: evaluated at every segment boundary
    (recipe self-transitions are allowed and expected, unlike the subtask HSMM), since each
    segment is one tick of the recipe HMM's own sequence."""
    t_true = sum(d for _, d in segments)
    s_trans = np.zeros(t_true, dtype=np.float64)
    expected_next = np.full(t_true, -1, dtype=np.int64)
    log_trans_np = np.asarray(recipe_log_trans)

    pos = 0
    prev_recipe = None
    for (state, d), recipe_id in zip(segments, seg_recipe_ids):
        if prev_recipe is not None:
            s_trans[pos] = float(-log_trans_np[prev_recipe, recipe_id])
            expected_next[pos] = int(np.argmax(log_trans_np[prev_recipe]))
        pos += d
        prev_recipe = int(recipe_id)
    return s_trans, expected_next


def flag(trace, alpha=DEFAULT_ALPHA, thresholds=None):
    """Per-channel boolean flags. Temporal channels default to the tail-probability threshold
    -log(alpha) (per-state calibrated); emission/transition channels use raw-nat thresholds.
    Rigorous precision/recall tuning against injected error types is Phase 6."""
    resolved = {**DEFAULT_THRESHOLDS}
    if alpha != DEFAULT_ALPHA:
        resolved["s_temporal"] = -float(np.log(alpha))
        resolved["s_dur_two"] = -float(np.log(alpha))
    resolved.update(thresholds or {})
    return {name: np.asarray(getattr(trace, name)) > resolved[name] for name in resolved}


def assemble_trace(hsmm_params, log_probs, recipe_log_trans, pi_all, verb_ids, noun_ids,
                   seg_result, seg_recipe_ids):
    """Pure per-trial assembly of a SurpriseTrace from already-computed jax quantities
    (pi_all, the Viterbi seg_result, and the per-segment recipe ids). Everything here is cheap
    numpy over one trial's true length -- factored out so both the single-trial `compute_trace`
    and the batched `eval.batch.compute_traces` reuse the exact same channel logic. pi_all/
    verb_ids/noun_ids are the trial's true-length (T,) / (T,K) arrays."""
    segments = seg_result["segments"]
    z_star = seg_result["subtask_per_tick"]

    s_emit, s_verb, s_noun = emission_surprise(
        jnp.asarray(pi_all), log_probs.log_emit_v, log_probs.log_emit_n,
        jnp.asarray(verb_ids), jnp.asarray(noun_ids),
    )

    log_emit_v_np = np.asarray(log_probs.log_emit_v)
    log_emit_n_np = np.asarray(log_probs.log_emit_n)
    expected_verb = np.argmax(log_emit_v_np[z_star], axis=-1)
    expected_noun = np.argmax(log_emit_n_np[z_star], axis=-1)

    s_temporal = temporal.live_stall_surprise(segments, log_probs.log_dur_survival, log_probs.log_dur_survival.shape[1])
    s_transition, expected_next_state = transition_surprise(segments, log_probs.log_trans)

    s_long, s_short, s_two, temporal_attr = temporal.completed_segment_surprise(
        segments, hsmm_params.dur_r, hsmm_params.dur_p
    )
    pit_per_seg = temporal.pit_coordinate(segments, hsmm_params.dur_r, hsmm_params.dur_p)
    s_dur_long = _scatter_segment_end(segments, s_long)
    s_dur_short = _scatter_segment_end(segments, s_short)
    s_dur_two = _scatter_segment_end(segments, s_two)
    pit = _scatter_segment_end(segments, pit_per_seg, fill=np.nan)
    temporal_attribution = _scatter_segment_end(segments, temporal_attr, fill="none", dtype=object)

    s_recipe_transition, expected_next_recipe = recipe_transition_surprise(
        segments, seg_recipe_ids, recipe_log_trans
    )

    return SurpriseTrace(
        s_emit=np.asarray(s_emit),
        s_verb=np.asarray(s_verb),
        s_noun=np.asarray(s_noun),
        s_temporal=s_temporal,
        s_dur_long=s_dur_long,
        s_dur_short=s_dur_short,
        s_dur_two=s_dur_two,
        s_transition=s_transition,
        s_recipe_transition=s_recipe_transition,
        pit=pit,
        z_star=z_star,
        expected_verb=expected_verb,
        expected_noun=expected_noun,
        expected_next_state=expected_next_state,
        expected_next_recipe=expected_next_recipe,
        attribution=attribute(s_verb, s_noun),
        temporal_attribution=temporal_attribution,
    )


def compute_trace(hsmm_params, recipe_params, verb_ids, noun_ids, d_max):
    """Driver: single trial (v,n) stream -> a full SurpriseTrace. verb_ids/noun_ids: (T,) int
    arrays, no padding (mask is all-True; this is a per-trial analysis tool). For many trials
    use eval.batch.compute_traces, which batches the jax-heavy ops to avoid per-trial recompiles."""
    verb_ids = jnp.asarray(verb_ids)
    noun_ids = jnp.asarray(noun_ids)
    t = verb_ids.shape[0]
    mask = jnp.ones((t,), dtype=bool)

    log_probs = params.to_log_probs(hsmm_params, d_max)
    loglik = emissions.sequence_loglik(
        verb_ids, noun_ids, log_probs.log_emit_v, log_probs.log_emit_n, mask
    )
    pi_all = messages.predictive_occupancy(
        loglik, log_probs.log_init, log_probs.log_trans,
        log_probs.log_dur_pmf, log_probs.log_dur_survival, mask, d_max,
    )
    seg_result = segmentize.segment_all(
        hsmm_params, verb_ids[None, :], noun_ids[None, :], mask[None, :], d_max
    )[0]
    seg_recipe_ids = _segment_recipe_path(recipe_params, [s for s, _ in seg_result["segments"]])
    _, recipe_log_trans, _ = recipe_hmm.to_log_probs(recipe_params)

    return assemble_trace(hsmm_params, log_probs, recipe_log_trans, pi_all, verb_ids, noun_ids,
                          seg_result, seg_recipe_ids)
