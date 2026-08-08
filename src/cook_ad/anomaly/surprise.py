from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

from cook_ad.anomaly import quantile, temporal
from cook_ad.hsmm import emissions, joint_em, joint_params, messages, params
from cook_ad.recipe import recipe_hmm, segmentize

# The temporal channels (s_temporal / s_dur_two) are -log(tail probability) against a
# parametric survival function, so a single global threshold -log(alpha) is automatically
# per-state calibrated (see temporal.py) -- it fires exactly when the duration reaches that
# state's alpha tail quantile. The five categorical/transition channels (s_emit, s_verb,
# s_noun, s_transition, s_recipe_transition) get the SAME per-state alpha-quantile treatment,
# but computed exactly (quantile.py) rather than parametrically, since their support is a
# finite, known vocab. Neither family is hand-tuned per state; that per-state calibration is
# precisely the point of the tail-probability framing.
DEFAULT_ALPHA = 0.05

# Only the two duration channels keep a fixed-scalar threshold (already a tail-probability
# cutoff, not a raw nat value). The five categorical/transition channels are recomputed as
# (K,)/(K_R,) tables per flag()/flag_joint() call -- see quantile.py -- and are NOT in this
# dict; a scalar override on a per-state channel would reintroduce exactly the miscalibration
# this module fixes.
DEFAULT_THRESHOLDS = {
    "s_temporal": -float(np.log(DEFAULT_ALPHA)),
    "s_dur_two": -float(np.log(DEFAULT_ALPHA)),
}

# Canonical ordered channel list -- callers that previously derived this from
# DEFAULT_THRESHOLDS.keys() (when it held all seven channels) should use this instead.
CHANNELS = (
    "s_emit", "s_verb", "s_noun", "s_temporal", "s_dur_two", "s_transition", "s_recipe_transition",
)

# s_emit/s_verb/s_noun are computed against the pi_all MIXTURE (emission_surprise) but
# calibrated per-state against z_star's own SINGLE-state distribution (quantile.py). Since
#     mixture = sum_k pi_k P(o|k) >= pi_{z*} P(o|z*)
#     => s_emit <= -log(pi_{z*}) + s_pure          where s_pure = -log P(o|z*),
# the mixture can only inflate surprise above z*'s own view by AT MOST -log(pi_{z*}). Adding
# that same per-tick offset to the per-state quantile threshold (emission_thresholds, below)
# cancels the dilution exactly: an observation inside z*'s own alpha-quantile can never be
# flagged by mixture dilution alone -- a one-sided correctness guarantee, not a heuristic.
#
# An earlier version of this fix used a flat floor (max(quantile_threshold, 6.0/4.0/4.0) --
# the original fixed nats) instead of this per-tick correction. Measured directly: the flat
# floor sat ABOVE the maximum achievable quantile threshold on every state at BOTH the mini
# (K=20, max 2.60/2.52) and full-scale (K=64, max 3.30/2.87) checkpoints, making the emission
# calibration entirely dead code -- every tick fell back to the old fixed-nat behavior
# regardless of entropy. The per-tick dilution correction replaces it: it is tight only by the
# amount of ACTUAL dilution present at that tick (pi_{z*} < 1), not a blanket safety margin.
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
    from_state: np.ndarray             # (T,) previous segment's subtask id at segment-start ticks, -1 elsewhere
    from_recipe: np.ndarray            # (T,) previous segment's recipe id at segment-start ticks (cascade only), -1 elsewhere
    belief_concentration: np.ndarray   # (T,) max_k P(Z_t=k|o_{<t}), diagnostic for the z_star-indexed threshold approximation
    pi_at_zstar: np.ndarray             # (T,) P(Z_t=z_star_t|o_{<t}) -- the mixture weight ON z_star specifically, used to cancel dilution in emission_thresholds


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


def conditional_expected(pi_all_t, log_emit_fixed_col, log_emit_target):
    """argmax_x logsumexp_k(pi_all_t[k] + log_emit_fixed_col[k] + log_emit_target[k,x]), for
    one tick: the same pi_all-weighted mixture s_verb/s_noun are scored against
    (emission_surprise), evaluated over every entry of the channel being explained, but
    reweighted by ALSO requiring compatibility with a second, held-fixed observed token
    (log_emit_fixed_col = the OTHER channel's emission column at ITS observed value).

    Marginalizing over pi_all with no such conditioning can pick a value that is individually
    plausible in isolation but incoherent paired with the token actually being held constant --
    e.g. the noun a mostly-idle belief would expect, paired with a verb that idle state barely
    supports at all ("pour kitchen"). Conditioning down-weights states incompatible with the
    fixed token before taking the argmax, so the result is compatible with it by construction.

    pi_all_t: (K,) log P(Z_t=k|o_{<t}) at one tick. log_emit_fixed_col: (K,), the other
    channel's emission column at its observed token. log_emit_target: (K, X), the channel
    being explained. Returns a single int index into X.
    """
    pi_all_t = jnp.asarray(pi_all_t)
    log_emit_fixed_col = jnp.asarray(log_emit_fixed_col)
    log_emit_target = jnp.asarray(log_emit_target)
    mixture_log = logsumexp(
        pi_all_t[:, None] + log_emit_fixed_col[:, None] + log_emit_target, axis=0
    )  # (X,)
    return int(jnp.argmax(mixture_log))


def joint_expected(pi_all_t, log_emit_v, log_emit_n):
    """argmax_{v,n} logsumexp_k(pi_all_t[k] + log_emit_v[k,v] + log_emit_n[k,n]), for one tick:
    the fully joint expectation under the live belief, letting BOTH verb and noun vary.

    Used when both channels are independently flagged at the same tick: treating either
    observed token as a trustworthy anchor for the other (conditional_expected's premise) is
    then unjustified, since both are themselves in question. Returns the single best (v, n)
    pair, not two independently-argmaxed halves, so it can never invent a combination the
    model's own joint mixture doesn't actually support.

    pi_all_t: (K,). log_emit_v: (K,V). log_emit_n: (K,N). Returns (expected_verb, expected_noun).
    """
    pi_all_t = jnp.asarray(pi_all_t)
    log_emit_v = jnp.asarray(log_emit_v)
    log_emit_n = jnp.asarray(log_emit_n)
    joint = logsumexp(
        pi_all_t[:, None, None] + log_emit_v[:, :, None] + log_emit_n[:, None, :], axis=0
    )  # (V, N)
    v, n = np.unravel_index(int(jnp.argmax(joint)), joint.shape)
    return int(v), int(n)


def severity(value, threshold):
    """ratio = value / threshold, bucketed low/medium/high. Shared by narrate.py (per-query
    severity, one call per narrated card) and flagged_tick_severity below (per-FLAGGED-tick
    severity, including ticks that never became a narrated query) -- the same bucketing so a
    tick's marker color always matches what its own query card, if any, would say."""
    value = float(value)
    threshold = float(threshold)
    if not np.isfinite(threshold) or threshold <= 0:
        return "high", float("inf")
    ratio = value / threshold
    if ratio < 1.5:
        label = "low"
    elif ratio < 3.0:
        label = "medium"
    else:
        label = "high"
    return label, ratio


def flagged_tick_severity(trace, flags, tables, alpha=DEFAULT_ALPHA, thresholds=None):
    """For every tick flagged in `flags` (surprise.flag/flag_joint's output), the severity label
    of that flag -- built from the SAME per-tick value/threshold pairing _base_flags gated on,
    so a tick's severity always matches why it was flagged. Returns {channel: {tick: label}},
    restricted to the five channels render_anomaly_png.py's CHANNEL_ROW knows how to place
    (s_emit/s_verb/s_noun/s_temporal/s_dur_two/s_transition) -- s_recipe_transition is omitted,
    mirroring narrate.py's own caveat that recipe clusters have no learned name to render against.
    """
    resolved = _duration_thresholds(alpha, thresholds)
    emit_thresh, verb_thresh, noun_thresh = emission_thresholds(trace, tables)
    from_state_safe = np.where(trace.from_state != -1, trace.from_state, 0)
    t_dur = np.full(np.asarray(trace.s_temporal).shape, resolved["s_temporal"])
    d_dur = np.full(np.asarray(trace.s_dur_two).shape, resolved["s_dur_two"])

    per_channel = {
        "s_emit": (trace.s_emit, emit_thresh),
        "s_verb": (trace.s_verb, verb_thresh),
        "s_noun": (trace.s_noun, noun_thresh),
        "s_temporal": (trace.s_temporal, t_dur),
        "s_dur_two": (trace.s_dur_two, d_dur),
        "s_transition": (trace.s_transition, tables.transition[from_state_safe]),
    }

    out = {}
    for ch, (values, thresh) in per_channel.items():
        values = np.asarray(values)
        thresh = np.asarray(thresh)
        out[ch] = {
            int(t): severity(values[t], thresh[t])[0] for t in np.flatnonzero(np.asarray(flags[ch]))
        }
    return out


def compute_pi_all(log_probs, verb_ids, noun_ids, d_max):
    """Standalone predictive-occupancy computation -- the same pi_all compute_trace builds
    internally, factored out so callers that need it directly (conditional_expected/
    joint_expected need a specific tick's pi_all, which compute_trace does not expose) don't
    have to duplicate or re-derive it."""
    verb_ids = jnp.asarray(verb_ids)
    noun_ids = jnp.asarray(noun_ids)
    t = verb_ids.shape[0]
    mask = jnp.ones((t,), dtype=bool)
    loglik = emissions.sequence_loglik(
        verb_ids, noun_ids, log_probs.log_emit_v, log_probs.log_emit_n, mask
    )
    return messages.predictive_occupancy(
        loglik, log_probs.log_init, log_probs.log_trans,
        log_probs.log_dur_pmf, log_probs.log_dur_survival, mask, d_max,
    )


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


def _scatter_from_previous(segments, per_segment_ids, fill=-1, dtype=np.int64):
    """At each segment's FIRST tick, place the id of the PREVIOUS segment (per_segment_ids[i-1]);
    `fill` at the very first segment's first tick (no predecessor) and at all non-boundary
    ticks. Mirrors _scatter_segment_end's placement convention but at segment starts -- used to
    index the per-state/per-recipe quantile threshold tables by the FROM side of a transition
    (surprise.flag / flag_joint), not the TO side z_star already indexes."""
    t_true = sum(d for _, d in segments)
    out = np.full(t_true, fill, dtype=dtype)
    pos = 0
    for i, (_, d) in enumerate(segments):
        if i > 0:
            out[pos] = per_segment_ids[i - 1]
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


def _duration_thresholds(alpha, thresholds):
    """Resolve the two duration-channel scalar thresholds, honoring an alpha override and a
    `thresholds` override restricted to {s_temporal, s_dur_two} (see DEFAULT_THRESHOLDS)."""
    resolved = {**DEFAULT_THRESHOLDS}
    if alpha != DEFAULT_ALPHA:
        resolved["s_temporal"] = -float(np.log(alpha))
        resolved["s_dur_two"] = -float(np.log(alpha))
    if thresholds:
        unknown = set(thresholds) - set(DEFAULT_THRESHOLDS)
        if unknown:
            raise KeyError(
                f"thresholds override only accepts {sorted(DEFAULT_THRESHOLDS)}; got "
                f"{sorted(unknown)}. The categorical/transition channels are now per-state "
                "quantile tables (quantile.py), not scalar overrides -- pass alpha instead."
            )
        resolved.update(thresholds)
    return resolved


def emission_thresholds(trace, tables):
    """Per-tick (T,) thresholds for s_emit/s_verb/s_noun: the z_star-indexed quantile table,
    corrected for pi_all-mixture dilution by adding -log(pi_at_zstar) at each tick (see the
    module-level comment above CHANNELS for the one-sided guarantee this provides). Callers
    that flag ticks (_base_flags) and callers that narrate them (narrate.narrate) must divide
    by the SAME per-tick threshold -- narrate() previously recomputed threshold_tables()
    directly with no correction at all, so severity on peaked states (raw quantile as low as
    ~0.002 nats) divided by a threshold with no dilution allowance, pushing ratio arbitrarily
    high and rendering nearly every emission query "high" regardless of how surprising the
    observation actually was relative to what got flagged.

    Returns (emit_thresh, verb_thresh, noun_thresh), each (T,)."""
    z = trace.z_star
    offset = -np.log(np.asarray(trace.pi_at_zstar))
    return tables.emit[z] + offset, tables.verb[z] + offset, tables.noun[z] + offset


def _base_flags(trace, tables, alpha, thresholds):
    """s_emit/s_verb/s_noun (z_star-indexed, dilution-corrected per tick -- see
    emission_thresholds), s_temporal/s_dur_two (duration-channel scalars), and s_transition
    (from_state-indexed, masked to segment-start ticks). Shared by flag()/flag_joint(); each
    caller adds its own s_recipe_transition, since the cascade indexes it by from_recipe and
    the joint model repurposes it as a from_state-indexed signed excess (see
    assemble_trace_joint)."""
    resolved = _duration_thresholds(alpha, thresholds)
    emit_thresh, verb_thresh, noun_thresh = emission_thresholds(trace, tables)

    from_state_valid = trace.from_state != -1
    from_state_safe = np.where(from_state_valid, trace.from_state, 0)

    return {
        "s_emit": trace.s_emit > emit_thresh,
        "s_verb": trace.s_verb > verb_thresh,
        "s_noun": trace.s_noun > noun_thresh,
        "s_temporal": np.asarray(trace.s_temporal) > resolved["s_temporal"],
        "s_dur_two": np.asarray(trace.s_dur_two) > resolved["s_dur_two"],
        # Boundary mask is load-bearing, not cosmetic: quantile thresholds are not guaranteed
        # positive (excess_quantile_threshold in particular), so `0 > t` off-boundary is no
        # longer trivially False the way it was under the old fixed-positive-nat thresholds.
        "s_transition": from_state_valid & (trace.s_transition > tables.transition[from_state_safe]),
    }


def flag(trace, log_probs, recipe_log_trans, alpha=DEFAULT_ALPHA, thresholds=None):
    """Per-channel boolean flags for the cascade model. s_emit/s_verb/s_noun/s_transition/
    s_recipe_transition use exact per-state alpha-quantile thresholds (quantile.py), indexed
    by trace.z_star (emission channels) or the FROM side of a transition -- trace.from_state
    for s_transition, trace.from_recipe for s_recipe_transition -- at segment-start ticks
    only. s_temporal/s_dur_two keep the parametric tail-probability threshold -log(alpha).

    `log_probs`/`recipe_log_trans`: the HSMMLogProbs and recipe transition matrix already
    built for this checkpoint (params.to_log_probs / recipe_hmm.to_log_probs); pass the same
    objects used to build `trace` (compute_trace / eval.batch.compute_traces return them).

    `thresholds` overrides ONLY {s_temporal, s_dur_two} -- see DEFAULT_THRESHOLDS; a scalar
    override on any of the five per-state channels would reintroduce the miscalibration this
    module fixes and raises KeyError.
    """
    tables = quantile.threshold_tables(log_probs, recipe_log_trans, alpha)
    flags = _base_flags(trace, tables, alpha, thresholds)

    from_recipe_valid = trace.from_recipe != -1
    from_recipe_safe = np.where(from_recipe_valid, trace.from_recipe, 0)
    flags["s_recipe_transition"] = from_recipe_valid & (
        trace.s_recipe_transition > tables.recipe[from_recipe_safe]
    )
    return flags


def flag_joint(trace, joint_log_probs, r_hat, log_trans_marginal, alpha=DEFAULT_ALPHA, thresholds=None):
    """Joint-model analogue of flag(). s_recipe_transition is the repurposed signed-excess
    channel (see assemble_trace_joint: log P_marginal - log P_r for the observed transition)
    and is indexed by trace.from_state under the trial's own MAP recipe r_hat via
    quantile.excess_quantile_threshold -- NOT trace.from_recipe, which the joint trace leaves
    at -1 throughout (one recipe per trial, not a per-segment path). That table's threshold
    can be negative, so the boundary mask (from_state != -1) is load-bearing here.
    """
    tables = quantile.threshold_tables_joint(joint_log_probs, r_hat, log_trans_marginal, alpha)
    flags = _base_flags(trace, tables, alpha, thresholds)

    from_state_valid = trace.from_state != -1
    from_state_safe = np.where(from_state_valid, trace.from_state, 0)
    flags["s_recipe_transition"] = from_state_valid & (
        trace.s_recipe_transition > tables.recipe[from_state_safe]
    )
    return flags


def belief_diagnostic(traces, cutoff=0.8):
    """Required diagnostic for the z_star-indexed threshold approximation (flag()/flag_joint()
    index by the hard Viterbi state, which is only exact when the filtered belief is
    concentrated there). Returns (pooled_fraction_below_cutoff, per_trial_mean_fraction): the
    fraction of ticks across all `traces` with belief_concentration < cutoff, and the mean of
    that same fraction computed per trial. A large fraction means the z_star approximation is
    frequently invoked when the mixture is diffuse -- a known limitation, not silently
    absorbed; per-tick mixture-quantile scoring (out of scope here) is the eventual fix."""
    per_trial = []
    total_below = 0
    total_ticks = 0
    for trace in traces:
        bc = np.asarray(trace.belief_concentration)
        below = bc < cutoff
        per_trial.append(float(below.mean()) if bc.size else 0.0)
        total_below += int(below.sum())
        total_ticks += bc.size
    pooled = total_below / total_ticks if total_ticks else 0.0
    return pooled, float(np.mean(per_trial)) if per_trial else 0.0


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

    from_state = _scatter_from_previous(segments, [s for s, _ in segments])
    from_recipe = _scatter_from_previous(segments, list(seg_recipe_ids))
    pi_all_np = np.exp(np.asarray(pi_all))
    belief_concentration = np.max(pi_all_np, axis=-1)
    pi_at_zstar = pi_all_np[np.arange(z_star.shape[0]), z_star]

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
        from_state=from_state,
        from_recipe=from_recipe,
        belief_concentration=belief_concentration,
        pi_at_zstar=pi_at_zstar,
    )


def assemble_trace_joint(joint_hsmm_params, joint_log_probs, r_hat, log_trans_marginal, pi_all,
                          verb_ids, noun_ids, seg_result):
    """Joint-model analogue of assemble_trace: identical per-channel logic, but every
    recipe-conditioned table (trans, duration) is sliced from the K_R-indexed
    joint_log_probs/joint_hsmm_params at this trial's MAP recipe r_hat, while emissions stay
    shared (unindexed). `pi_all` must already be the trial's predictive occupancy computed
    under recipe r_hat's own tables (the caller's job -- messages.predictive_occupancy takes
    one recipe's tables at a time).

    s_transition is now recipe-conditioned (log_trans[r_hat] in place of the cascade's single
    shared log_trans). s_recipe_transition is repurposed as the recipe-attributable excess:
    the gap between the recipe-conditioned transition surprise and the same transition scored
    under the pi-weighted marginal transition matrix (joint_params.marginal_log_trans) -- it
    fires when a transition is ordinary in general but wrong for this trial's specific recipe.
    Reuses transition_surprise verbatim for both terms rather than a new function.
    expected_next_recipe is repurposed to hold r_hat at the same segment-boundary ticks
    expected_next_state is valid at (there is only one recipe per trial now, not a
    per-segment path), -1 elsewhere.
    """
    segments = seg_result["segments"]
    z_star = seg_result["subtask_per_tick"]

    log_trans_r = joint_log_probs.log_trans[r_hat]
    log_dur_survival_r = joint_log_probs.log_dur_survival[r_hat]
    dur_r_r = joint_hsmm_params.dur_r[r_hat]
    dur_p_r = joint_hsmm_params.dur_p[r_hat]

    s_emit, s_verb, s_noun = emission_surprise(
        jnp.asarray(pi_all), joint_log_probs.log_emit_v, joint_log_probs.log_emit_n,
        jnp.asarray(verb_ids), jnp.asarray(noun_ids),
    )

    log_emit_v_np = np.asarray(joint_log_probs.log_emit_v)
    log_emit_n_np = np.asarray(joint_log_probs.log_emit_n)
    expected_verb = np.argmax(log_emit_v_np[z_star], axis=-1)
    expected_noun = np.argmax(log_emit_n_np[z_star], axis=-1)

    s_temporal = temporal.live_stall_surprise(segments, log_dur_survival_r, log_dur_survival_r.shape[1])
    s_transition, expected_next_state = transition_surprise(segments, log_trans_r)
    s_transition_marginal, _ = transition_surprise(segments, log_trans_marginal)
    s_recipe_transition = s_transition - s_transition_marginal
    expected_next_recipe = np.where(expected_next_state != -1, int(r_hat), -1).astype(np.int64)

    s_long, s_short, s_two, temporal_attr = temporal.completed_segment_surprise(segments, dur_r_r, dur_p_r)
    pit_per_seg = temporal.pit_coordinate(segments, dur_r_r, dur_p_r)
    s_dur_long = _scatter_segment_end(segments, s_long)
    s_dur_short = _scatter_segment_end(segments, s_short)
    s_dur_two = _scatter_segment_end(segments, s_two)
    pit = _scatter_segment_end(segments, pit_per_seg, fill=np.nan)
    temporal_attribution = _scatter_segment_end(segments, temporal_attr, fill="none", dtype=object)

    from_state = _scatter_from_previous(segments, [s for s, _ in segments])
    from_recipe = np.full_like(from_state, -1)  # recipe channel is indexed by from_state under r_hat, not from_recipe
    pi_all_np = np.exp(np.asarray(pi_all))
    belief_concentration = np.max(pi_all_np, axis=-1)
    pi_at_zstar = pi_all_np[np.arange(z_star.shape[0]), z_star]

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
        from_state=from_state,
        from_recipe=from_recipe,
        belief_concentration=belief_concentration,
        pi_at_zstar=pi_at_zstar,
    )


def compute_trace(hsmm_params, recipe_params, verb_ids, noun_ids, d_max):
    """Driver: single trial (v,n) stream -> a full SurpriseTrace. verb_ids/noun_ids: (T,) int
    arrays, no padding (mask is all-True; this is a per-trial analysis tool). For many trials
    use eval.batch.compute_traces, which batches the jax-heavy ops to avoid per-trial recompiles.

    Returns (trace, log_probs, recipe_log_trans): the latter two are exactly what flag() needs
    and are otherwise built internally and discarded, so callers that also need to call flag()
    get them for free instead of re-deriving via a second params.to_log_probs/
    recipe_hmm.to_log_probs call."""
    verb_ids = jnp.asarray(verb_ids)
    noun_ids = jnp.asarray(noun_ids)
    t = verb_ids.shape[0]
    mask = jnp.ones((t,), dtype=bool)

    log_probs = params.to_log_probs(hsmm_params, d_max)
    pi_all = compute_pi_all(log_probs, verb_ids, noun_ids, d_max)
    seg_result = segmentize.segment_all(
        hsmm_params, verb_ids[None, :], noun_ids[None, :], mask[None, :], d_max
    )[0]
    seg_recipe_ids = _segment_recipe_path(recipe_params, [s for s, _ in seg_result["segments"]])
    _, recipe_log_trans, _ = recipe_hmm.to_log_probs(recipe_params)

    trace = assemble_trace(hsmm_params, log_probs, recipe_log_trans, pi_all, verb_ids, noun_ids,
                            seg_result, seg_recipe_ids)
    return trace, log_probs, recipe_log_trans


def compute_pi_all_joint(joint_log_probs, r_hat, verb_ids, noun_ids, d_max):
    """Joint analogue of compute_pi_all: predictive occupancy under recipe r_hat's own
    init/trans/duration tables (emissions stay shared, unindexed)."""
    verb_ids = jnp.asarray(verb_ids)
    noun_ids = jnp.asarray(noun_ids)
    t = verb_ids.shape[0]
    mask = jnp.ones((t,), dtype=bool)
    loglik = emissions.sequence_loglik(
        verb_ids, noun_ids, joint_log_probs.log_emit_v, joint_log_probs.log_emit_n, mask
    )
    return messages.predictive_occupancy(
        loglik, joint_log_probs.log_init[r_hat], joint_log_probs.log_trans[r_hat],
        joint_log_probs.log_dur_pmf[r_hat], joint_log_probs.log_dur_survival[r_hat], mask, d_max,
    )


def compute_trace_joint(joint_hsmm_params, verb_ids, noun_ids, d_max):
    """Joint analogue of compute_trace: single trial (v,n) stream -> a full SurpriseTrace, but
    first infers the trial's own MAP recipe (joint_em.infer_recipe) and scores everything under
    that recipe's conditioned tables, mirroring the batched eval.batch.compute_traces_joint but
    for one trial without padding. For many trials use compute_traces_joint instead.

    Returns (trace, joint_log_probs, r_hat, log_trans_marginal, rho): the middle three are
    exactly what flag_joint()/narrate_joint() need and are otherwise built internally and
    discarded, so callers get them for free instead of re-deriving via a second
    infer_recipe/to_log_probs_joint call. r_hat is a plain int (not a length-1 array) since this
    is a single-trial driver. rho is this trial's full (K_R,) recipe posterior -- callers that
    only need r_hat's own confidence read rho[r_hat]; kept as the full vector rather than a
    scalar since a flat second-best margin is sometimes more informative than the top value alone.
    """
    verb_ids = jnp.asarray(verb_ids)
    noun_ids = jnp.asarray(noun_ids)
    t = verb_ids.shape[0]
    mask = jnp.ones((t,), dtype=bool)

    joint_log_probs = joint_params.to_log_probs_joint(joint_hsmm_params, d_max)
    r_hat_arr, rho_arr, _ = joint_em.infer_recipe(
        joint_hsmm_params, verb_ids[None, :], noun_ids[None, :], mask[None, :], d_max, chunk_size=1
    )
    r_hat = int(r_hat_arr[0])
    rho = rho_arr[0]

    pi_all = compute_pi_all_joint(joint_log_probs, r_hat, verb_ids, noun_ids, d_max)
    seg_result = segmentize.segment_all_conditioned(
        joint_log_probs, jnp.array([r_hat]), verb_ids[None, :], noun_ids[None, :], mask[None, :], d_max
    )[0]
    log_trans_marginal = joint_params.marginal_log_trans(joint_log_probs)

    trace = assemble_trace_joint(
        joint_hsmm_params, joint_log_probs, r_hat, log_trans_marginal, pi_all, verb_ids, noun_ids, seg_result
    )
    return trace, joint_log_probs, r_hat, log_trans_marginal, rho
