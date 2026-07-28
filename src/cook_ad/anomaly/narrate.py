from typing import NamedTuple

import numpy as np

from cook_ad.anomaly import quantile, surprise
from cook_ad.hsmm import params
from cook_ad.lifecycle.online_update import PreferenceEvent

# Template-based, not generative: every sentence below is a direct read of a model quantity
# (an argmax over an emission/transition row, a fitted NB mean), so every query is auditable
# back to the parameter that produced it. This is the reason categorical emissions were chosen
# over embeddings in the first place -- routing any of this through a language model would
# throw that away.
#
# Two things this module deliberately does NOT do, stated here rather than left implicit:
#   1. Emission queries explain "I expected X" using surprise.conditional_expected/
#      joint_expected -- both computed from the SAME pi_all-weighted mixture s_noun/s_verb are
#      scored against, not z_star's single-state hindsight argmax (expected_noun/expected_verb).
#      Right at a segment boundary the causal filter hasn't caught up to yet, the two can
#      disagree, and using the hindsight value there renders a self-contradictory "that's cup,
#      normally you use cup." Naively marginalizing the live mixture over just the flagged word
#      has its own failure mode: it can pick a replacement that is individually plausible but
#      incoherent paired with the OTHER (held-constant, presumed-fine) word -- "pour kitchen."
#      conditional_expected fixes that by reweighting on the held word before picking a
#      replacement for the flagged one; joint_expected (used when BOTH channels are
#      independently flagged, so neither word can be trusted as an anchor) picks the single
#      best (verb, noun) pair jointly instead. Either way, log_emit_n/log_emit_v are not
#      recipe-conditioned in the cascade, so "I expected jelly" means "at this step, across all
#      recipes, you usually use jelly" -- not "in a PB sandwich you use jelly." State this
#      caveat whenever a query is presented.
#   2. There is no s_recipe_transition renderer. K_recipe is a weak-limit nominal (larger than
#      the real recipe count) and there is no learned cluster->name map anywhere in the repo
#      (recipe_hmm's cluster alignment exists only for scoring). Naming a recipe cluster here
#      would print a name that does not correspond to anything.

DEFAULT_MIN_BRIDGE_GAIN = 2.0

SEVERITY_HEDGE = {"low": "I think", "medium": "I noticed", "high": "Wait --"}


class Lexicon:
    """vocab.json + fitted HSMMParams -> human-readable names."""

    def __init__(self, vocab, hsmm_params, sil_verb="stall", sil_noun="kitchen"):
        self._id_to_verb = {i: v for v, i in vocab["verbs"].items()}
        self._id_to_noun = {i: n for n, i in vocab["nouns"].items()}
        self._sil_verb = sil_verb
        self._sil_noun = sil_noun
        _, _, log_emit_v, log_emit_n = params.normalize_categoricals(hsmm_params)
        self._log_emit_v = np.asarray(log_emit_v)
        self._log_emit_n = np.asarray(log_emit_n)
        self._dur_r = np.asarray(hsmm_params.dur_r)
        self._dur_p = np.asarray(hsmm_params.dur_p)

    def verb(self, v):
        return self._id_to_verb[int(v)]

    def noun(self, n):
        return self._id_to_noun[int(n)]

    def subtask(self, k):
        """Names a subtask by its modal (verb, noun) pair, e.g. 'pour milk'."""
        k = int(k)
        v = int(np.argmax(self._log_emit_v[k]))
        n = int(np.argmax(self._log_emit_n[k]))
        return self.phrase(v, n)

    def phrase(self, v, n):
        """Names a specific (verb, noun) pair, e.g. 'pour water'. Both SIL -> 'idle', matching
        subtask(). A SIL verb paired with a real noun drops to just the noun ('bowl', not
        'stall bowl') -- that reads naturally as "no specific action, but this object is
        present". The reverse does NOT get the same treatment: an action verb needs some object
        to read as a complete phrase, so a SIL noun paired with a real verb is left as
        '{verb} kitchen' rather than a dangling '{verb}' -- accepting a literal SIL-token leak
        as the lesser problem next to an objectless sentence.
        """
        v_name = self.verb(v)
        n_name = self.noun(n)
        if v_name == self._sil_verb and n_name == self._sil_noun:
            return "idle"
        if v_name == self._sil_verb:
            return n_name
        return f"{v_name} {n_name}"

    def expected_duration(self, k):
        """E[D|Z=k] under the fitted NB. Duration support is d=1,2,... on d'=d-1 (durations.py),
        so mean = 1 + r(1-p)/p, not the unshifted r(1-p)/p."""
        r = float(self._dur_r[int(k)])
        p = float(self._dur_p[int(k)])
        return 1.0 + r * (1.0 - p) / p


class Query(NamedTuple):
    tick: int
    segment_index: int
    channel: str
    kind: str
    severity: str
    ratio: float
    text: str
    event: object  # PreferenceEvent | None -- routes directly into state_manager.handle_confirmation


def segments_from_z(z_star):
    """z_star -> [(state, start, end_exclusive), ...]. Plain run-length encode: self-transitions
    are structurally banned (params.py zeroes the trans_counts diagonal), so a run of constant
    z_star is exactly one Viterbi segment -- lossless."""
    z = np.asarray(z_star)
    segments = []
    if z.size == 0:
        return segments
    start = 0
    for t in range(1, z.size + 1):
        if t == z.size or z[t] != z[start]:
            segments.append((int(z[start]), start, t))
            start = t
    return segments


def missing_step(log_trans, a, c, min_gain=DEFAULT_MIN_BRIDGE_GAIN):
    """b* = argmax_b log P(b|a) + log P(c|b). Returns b* only if the two-hop path beats the
    direct jump by min_gain nats. Turns a one-step 'normally next is B' template into 'you
    skipped B'. O(K) per call."""
    log_trans = np.asarray(log_trans)
    k = log_trans.shape[0]
    two_hop = log_trans[a] + log_trans[:, c]
    mask = np.ones(k, dtype=bool)
    mask[a] = False
    mask[c] = False
    two_hop = np.where(mask, two_hop, -np.inf)
    if not np.any(np.isfinite(two_hop)):
        return None, 0.0
    b = int(np.argmax(two_hop))
    gain = float(two_hop[b] - log_trans[a, c])
    if not np.isfinite(gain) or gain < min_gain:
        return None, gain
    return b, gain


def _severity(value, threshold):
    """ratio = value / threshold, bucketed low/medium/high -> hedge phrase. Query intensity
    scales with discrepancy, per the design doc."""
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


def _emission_queries(trace, flags, segments, verb_ids, noun_ids, lexicon, tables,
                      pi_all, log_emit_v, log_emit_n):
    queries = []
    s_noun_flags = np.asarray(flags["s_noun"])
    s_verb_flags = np.asarray(flags["s_verb"])
    noun_flagged = s_noun_flags & (trace.attribution == "item")
    verb_flagged = s_verb_flags & (trace.attribution == "action")

    for seg_idx, (state, start, end) in enumerate(segments):
        n_ticks = np.flatnonzero(noun_flagged[start:end]) + start
        if n_ticks.size:
            tick = int(n_ticks[np.argmax(np.asarray(trace.s_noun)[n_ticks])])
            observed_verb = int(verb_ids[tick])
            observed_noun = int(noun_ids[tick])
            severity, ratio = _severity(trace.s_noun[tick], tables.noun[state])
            hedge = SEVERITY_HEDGE[severity]
            # attribution="item" means s_noun dominates s_verb by margin -- the verb is the
            # BIGGER culprit's opposite, not necessarily clean. If s_verb is ALSO independently
            # flagged here, the observed verb can't be trusted as an anchor for picking a
            # replacement noun (that's what produces incoherent pairs like "pour kitchen"), so
            # fall back to the unconstrained joint pick instead of conditioning on it.
            if s_verb_flags[tick]:
                expected_verb, expected_noun = surprise.joint_expected(pi_all[tick], log_emit_v, log_emit_n)
            else:
                expected_verb = observed_verb
                expected_noun = surprise.conditional_expected(
                    pi_all[tick], log_emit_v[:, observed_verb], log_emit_n
                )
            observed_phrase = lexicon.phrase(observed_verb, observed_noun)
            expected_phrase = lexicon.phrase(expected_verb, expected_noun)
            text = (
                f"{hedge}, that's {observed_phrase} -- based on what I'd seen up to then, "
                f"I expected {expected_phrase}."
            )
            queries.append(Query(
                tick, seg_idx, "s_noun", "item_substitution", severity, ratio, text,
                PreferenceEvent("noun", state, observed_noun),
            ))

        v_ticks = np.flatnonzero(verb_flagged[start:end]) + start
        if v_ticks.size:
            tick = int(v_ticks[np.argmax(np.asarray(trace.s_verb)[v_ticks])])
            observed_verb = int(verb_ids[tick])
            observed_noun = int(noun_ids[tick])
            severity, ratio = _severity(trace.s_verb[tick], tables.verb[state])
            hedge = SEVERITY_HEDGE[severity]
            # Mirrors the noun branch above: don't anchor on the observed noun if s_noun is
            # ALSO independently flagged at this tick.
            if s_noun_flags[tick]:
                expected_verb, expected_noun = surprise.joint_expected(pi_all[tick], log_emit_v, log_emit_n)
            else:
                expected_noun = observed_noun
                expected_verb = surprise.conditional_expected(
                    pi_all[tick], log_emit_n[:, observed_noun], log_emit_v
                )
            observed_phrase = lexicon.phrase(observed_verb, observed_noun)
            expected_phrase = lexicon.phrase(expected_verb, expected_noun)
            text = (
                f"{hedge}, that looks like '{observed_phrase}' -- based on what I'd seen up "
                f"to then, I expected '{expected_phrase}'."
            )
            queries.append(Query(
                tick, seg_idx, "s_verb", "wrong_action", severity, ratio, text,
                PreferenceEvent("verb", state, observed_verb),
            ))

    return queries


def _stall_queries(trace, flags, segments, lexicon, dur_threshold):
    """Live 'are you stuck' phrasing, fires at the first threshold crossing in the segment
    (s_temporal is monotone within a segment; the peak is always the last tick, which is not
    when a live system would speak)."""
    queries = []
    temporal_flagged = np.asarray(flags["s_temporal"])

    for seg_idx, (state, start, end) in enumerate(segments):
        rel = np.flatnonzero(temporal_flagged[start:end])
        if rel.size == 0:
            continue
        tick = start + int(rel[0])
        elapsed = tick - start + 1
        expected = lexicon.expected_duration(state)
        severity, ratio = _severity(trace.s_temporal[tick], dur_threshold)
        hedge = SEVERITY_HEDGE[severity]
        text = (
            f"{hedge}, are you stuck on {lexicon.subtask(state)}? You've been on it for "
            f"{elapsed} ticks; it usually takes about {expected:.0f}."
        )
        queries.append(Query(tick, seg_idx, "s_temporal", "stall", severity, ratio, text, None))

    return queries


def _completed_duration_queries(trace, flags, segments, lexicon, dur_threshold):
    """Retrospective duration surprise, evaluated at a segment's last tick. Direction from
    temporal_attribution: 'left_early' -> abandonment phrasing, 'stuck' -> retrospective stall."""
    queries = []
    dur_flagged = np.asarray(flags["s_dur_two"])

    for seg_idx, (state, start, end) in enumerate(segments):
        tick = end - 1
        if not dur_flagged[tick]:
            continue
        d = end - start
        expected = lexicon.expected_duration(state)
        severity, ratio = _severity(trace.s_dur_two[tick], dur_threshold)
        hedge = SEVERITY_HEDGE[severity]
        attr = trace.temporal_attribution[tick]

        if attr == "left_early":
            kind = "left_early"
            text = (
                f"{hedge}, {lexicon.subtask(state)} only took {d} ticks -- you usually spend "
                f"about {expected:.0f}. Did you skip part of it?"
            )
        elif attr == "stuck":
            kind = "retro_stall"
            text = (
                f"{hedge}, {lexicon.subtask(state)} took {d} ticks -- longer than your usual "
                f"{expected:.0f}. Everything okay?"
            )
        else:
            continue

        queries.append(Query(tick, seg_idx, "s_dur_two", kind, severity, ratio, text, None))

    return queries


def _order_queries(trace, flags, segments, lexicon, log_trans, tables, min_gain):
    """Tries missing_step first, falls back to the plain one-step 'normally after A you do B'
    template if no bridge clears min_gain."""
    queries = []
    trans_flagged = np.asarray(flags["s_transition"])

    for seg_idx, (state, start, end) in enumerate(segments):
        if seg_idx == 0 or not trans_flagged[start]:
            continue
        a = int(trace.from_state[start])
        c = state
        severity, ratio = _severity(trace.s_transition[start], tables.transition[a])
        hedge = SEVERITY_HEDGE[severity]

        bridge, _ = missing_step(log_trans, a, c, min_gain)
        if bridge is not None:
            kind = "missing_step"
            text = (
                f"{hedge}, you went from {lexicon.subtask(a)} straight to {lexicon.subtask(c)} "
                f"-- did you skip {lexicon.subtask(bridge)}?"
            )
        else:
            expected_next = int(trace.expected_next_state[start])
            kind = "out_of_order"
            text = (
                f"{hedge}, that's not the usual order -- after {lexicon.subtask(a)} you "
                f"normally do {lexicon.subtask(expected_next)}, not {lexicon.subtask(c)}."
            )

        queries.append(Query(
            start, seg_idx, "s_transition", kind, severity, ratio, text,
            PreferenceEvent("trans", a, c),
        ))

    return queries


def narrate(trace, flags, vocab, hsmm_params, verb_ids, noun_ids, log_probs, recipe_log_trans,
            pi_all, alpha=surprise.DEFAULT_ALPHA, min_gain=DEFAULT_MIN_BRIDGE_GAIN):
    """Dispatches to per-channel renderers, returns chronologically sorted.

    `log_probs`/`recipe_log_trans` are the same objects `surprise.compute_trace`/`flag` build
    for this trace (reused here, not rebuilt) -- pass what `compute_trace` returned. `pi_all`
    is NOT part of `compute_trace`'s return value; get it via `surprise.compute_pi_all(log_probs,
    verb_ids, noun_ids, d_max)` on the same (verb_ids, noun_ids, d_max) used to build `trace`.
    `alpha` must match the alpha used to build `flags`, since severity is computed against the
    same per-state quantile tables `flag()` used internally.
    """
    verb_ids = np.asarray(verb_ids)
    noun_ids = np.asarray(noun_ids)
    pi_all = np.asarray(pi_all)
    lexicon = Lexicon(vocab, hsmm_params)
    tables = quantile.threshold_tables(log_probs, recipe_log_trans, alpha)
    log_trans = np.asarray(log_probs.log_trans)
    dur_threshold = -float(np.log(alpha))
    segments = segments_from_z(trace.z_star)

    queries = []
    queries += _emission_queries(
        trace, flags, segments, verb_ids, noun_ids, lexicon, tables,
        pi_all, log_probs.log_emit_v, log_probs.log_emit_n,
    )
    queries += _stall_queries(trace, flags, segments, lexicon, dur_threshold)
    queries += _completed_duration_queries(trace, flags, segments, lexicon, dur_threshold)
    queries += _order_queries(trace, flags, segments, lexicon, log_trans, tables, min_gain)
    queries.sort(key=lambda q: (q.tick, q.channel))
    return queries
