"""narrate(): Hindsight alarms -> templated questions for the user (docs/detectors.md §6).

Every sentence is a direct read of a model quantity (an argmax over an emission or transition
row, a fitted NB mean), so each question is auditable back to the parameter behind it.
"""
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

from cook_ad.core import Detection
from cook_ad.detectors.hsmm_hindsight import HindsightTrace
from cook_ad.hsmm import kernels
from cook_ad.hsmm.params import _row_normalize, select_recipe

MIN_BRIDGE_GAIN = 2.0   # nats a two-hop path a -> b -> c must beat a -> c by to say "you skipped b"
HEDGE = {"low": "I think", "medium": "I noticed", "high": "Wait --"}


class Query(NamedTuple):
    tick: int
    segment_index: int
    channel: str
    kind: str
    severity: str
    ratio: float
    text: str


class Lexicon:
    """vocab.json + one recipe's parameters -> names for tokens and subtasks."""

    def __init__(self, vocab, params, sil_verb="stall", sil_noun="kitchen"):
        if getattr(params, "kernel_v", None) is not None or getattr(params, "kernel_n", None) is not None:
            raise NotImplementedError("Lexicon does not support emission kernels")
        self._verb = {i: v for v, i in vocab["verbs"].items()}
        self._noun = {i: n for n, i in vocab["nouns"].items()}
        self._sil = (sil_verb, sil_noun)
        self._log_emit_v = np.asarray(_row_normalize(params.verb_counts))
        self._log_emit_n = np.asarray(_row_normalize(params.noun_counts))
        self._dur_r, self._dur_p = np.asarray(params.dur_r), np.asarray(params.dur_p)

    def verb(self, v):
        return self._verb[int(v)]

    def noun(self, n):
        return self._noun[int(n)]

    def subtask(self, k):
        """A subtask's modal (verb, noun), e.g. 'pour milk'."""
        k = int(k)
        return self.phrase(np.argmax(self._log_emit_v[k]), np.argmax(self._log_emit_n[k]))

    def phrase(self, v, n):
        """'pour water'; both idle tokens -> 'idle'; an idle verb with an object -> the object."""
        v_name, n_name = self.verb(v), self.noun(n)
        if (v_name, n_name) == self._sil:
            return "idle"
        return n_name if v_name == self._sil[0] else f"{v_name} {n_name}"

    def expected_duration(self, k):
        """E[D | k] = 1 + r (1 - p) / p (durations start at 1 tick)."""
        r, p = float(self._dur_r[int(k)]), float(self._dur_p[int(k)])
        return 1.0 + r * (1.0 - p) / p


# ---------------------------------------------------------------------------------------------
# Model reads


def severity(value, threshold):
    """(label, value / threshold): low < 1.5 <= medium < 3 <= high."""
    value, threshold = float(value), float(threshold)
    if not np.isfinite(threshold) or threshold <= 0:
        return "high", float("inf")
    ratio = value / threshold
    return ("low" if ratio < 1.5 else "medium" if ratio < 3.0 else "high"), ratio


def expected_given(pi_t, log_emit_held, log_emit_target):
    """argmax_x sum_i pi_t(i) P(held token | i) P(x | i): the replacement for the flagged token
    that fits the forecast AND the token that was not flagged."""
    mix = logsumexp(jnp.asarray(pi_t)[:, None] + jnp.asarray(log_emit_held)[:, None]
                    + jnp.asarray(log_emit_target), axis=0)
    return int(jnp.argmax(mix))


def expected_pair(pi_t, log_emit_v, log_emit_n):
    """argmax_(v, n) sum_i pi_t(i) P(v | i) P(n | i): used when both tokens were flagged."""
    joint = logsumexp(jnp.asarray(pi_t)[:, None, None] + jnp.asarray(log_emit_v)[:, :, None]
                      + jnp.asarray(log_emit_n)[:, None, :], axis=0)
    v, n = np.unravel_index(int(jnp.argmax(joint)), joint.shape)
    return int(v), int(n)


def missing_step(log_A, a, c, min_gain=MIN_BRIDGE_GAIN):
    """b = argmax_b log A[a, b] + log A[b, c], if that beats log A[a, c] by min_gain; else None."""
    log_A = np.asarray(log_A)
    two_hop = log_A[a] + log_A[:, c]
    keep = np.ones(log_A.shape[0], dtype=bool)
    keep[[a, c]] = False
    two_hop = np.where(keep, two_hop, -np.inf)
    if not np.any(np.isfinite(two_hop)):
        return None, 0.0
    b = int(np.argmax(two_hop))
    gain = float(two_hop[b] - log_A[a, c])
    return (b, gain) if np.isfinite(gain) and gain >= min_gain else (None, gain)


def segment_spans(trace: HindsightTrace):
    """[(subtask, start, end_exclusive)] of the Viterbi path."""
    ends = np.cumsum([d for _, d in trace.segments])
    return [(int(k), int(e - d), int(e)) for (k, d), e in zip(trace.segments, ends)]


# ---------------------------------------------------------------------------------------------
# One renderer per kind of question


class _Ctx(NamedTuple):
    trace: HindsightTrace
    flags: dict
    th: dict
    pi_tilde: np.ndarray
    lexicon: Lexicon
    log_emit_v: np.ndarray
    log_emit_n: np.ndarray
    log_A: np.ndarray
    min_gain: float


def _item_query(c, seg_idx, start, end):
    tr, f = c.trace, c.flags
    ticks = np.flatnonzero((f["s_noun"] & (tr.attribution == "item"))[start:end]) + start
    if not ticks.size:
        return []
    t = int(ticks[np.argmax(tr.surprise["s_noun"][ticks])])
    v, n = int(tr.verbs[t]), int(tr.nouns[t])
    if f["s_verb"][t]:
        ev, en = expected_pair(c.pi_tilde[t], c.log_emit_v, c.log_emit_n)
    else:
        ev, en = v, expected_given(c.pi_tilde[t], c.log_emit_v[:, v], c.log_emit_n)
    sev, ratio = severity(tr.surprise["s_noun"][t], c.th["s_noun"][t])
    text = (f"{HEDGE[sev]}, that's {c.lexicon.phrase(v, n)} -- based on what I'd seen up to "
            f"then, I expected {c.lexicon.phrase(ev, en)}.")
    return [Query(t, seg_idx, "s_noun", "item_substitution", sev, ratio, text)]


def _action_query(c, seg_idx, start, end):
    tr, f = c.trace, c.flags
    ticks = np.flatnonzero((f["s_verb"] & (tr.attribution == "action"))[start:end]) + start
    if not ticks.size:
        return []
    t = int(ticks[np.argmax(tr.surprise["s_verb"][ticks])])
    v, n = int(tr.verbs[t]), int(tr.nouns[t])
    if f["s_noun"][t]:
        ev, en = expected_pair(c.pi_tilde[t], c.log_emit_v, c.log_emit_n)
    else:
        ev, en = expected_given(c.pi_tilde[t], c.log_emit_n[:, n], c.log_emit_v), n
    sev, ratio = severity(tr.surprise["s_verb"][t], c.th["s_verb"][t])
    text = (f"{HEDGE[sev]}, that looks like '{c.lexicon.phrase(v, n)}' -- based on what I'd seen "
            f"up to then, I expected '{c.lexicon.phrase(ev, en)}'.")
    return [Query(t, seg_idx, "s_verb", "wrong_action", sev, ratio, text)]


def _stall_query(c, seg_idx, k, start, end):
    """At the first tick past the threshold: when a live system would have asked."""
    rel = np.flatnonzero(c.flags["s_temporal"][start:end])
    if not rel.size:
        return []
    t = start + int(rel[0])
    sev, ratio = severity(c.trace.surprise["s_temporal"][t], c.th["s_temporal"][t])
    text = (f"{HEDGE[sev]}, are you stuck on {c.lexicon.subtask(k)}? You've been on it for "
            f"{t - start + 1} ticks; it usually takes about {c.lexicon.expected_duration(k):.0f}.")
    return [Query(t, seg_idx, "s_temporal", "stall", sev, ratio, text)]


def _duration_query(c, seg_idx, k, start, end):
    t = end - 1
    attr = c.trace.temporal_attribution[t]
    if not c.flags["s_dur_two"][t] or attr not in ("left_early", "stuck"):
        return []
    sev, ratio = severity(c.trace.surprise["s_dur_two"][t], c.th["s_dur_two"][t])
    name, mean = c.lexicon.subtask(k), c.lexicon.expected_duration(k)
    if attr == "left_early":
        kind = "left_early"
        text = (f"{HEDGE[sev]}, {name} only took {end - start} ticks -- you usually spend "
                f"about {mean:.0f}. Did you skip part of it?")
    else:
        kind = "retro_stall"
        text = (f"{HEDGE[sev]}, {name} took {end - start} ticks -- longer than your usual "
                f"{mean:.0f}. Everything okay?")
    return [Query(t, seg_idx, "s_dur_two", kind, sev, ratio, text)]


def _order_query(c, seg_idx, k, start):
    if seg_idx == 0 or not c.flags["s_transition"][start]:
        return []
    a, lex = int(c.trace.from_state[start]), c.lexicon
    sev, ratio = severity(c.trace.surprise["s_transition"][start], c.th["s_transition"][start])
    bridge, _ = missing_step(c.log_A, a, k, c.min_gain)
    if bridge is not None:
        kind = "missing_step"
        text = (f"{HEDGE[sev]}, you went from {lex.subtask(a)} straight to {lex.subtask(k)} -- "
                f"did you skip {lex.subtask(bridge)}?")
    else:
        kind = "out_of_order"
        text = (f"{HEDGE[sev]}, that's not the usual order -- after {lex.subtask(a)} you normally "
                f"do {lex.subtask(c.trace.expected_next_state[start])}, not {lex.subtask(k)}.")
    return [Query(start, seg_idx, "s_transition", kind, sev, ratio, text)]


# ---------------------------------------------------------------------------------------------


def narrate(detection: Detection, detector, vocab, op=None, min_gain=MIN_BRIDGE_GAIN,
            sil_verb="stall", sil_noun="kitchen") -> list[Query]:
    """One question per flagged segment and kind, in time order. Needs a Viterbi segmentation,
    so only a HindsightHSMMDetector's detection can be narrated. s_recipe_transition is never
    narrated: recipe clusters have no names."""
    tr = detection.trace
    if not isinstance(tr, HindsightTrace):
        raise NotImplementedError("narration needs a Viterbi segmentation; live narration is "
                                  "planned — see docs/detectors.md")
    hsmm, r = detector.hsmm, tr.r_hat
    t = hsmm.tables
    n = len(tr.z_star)
    flags = {ch: np.zeros(n, dtype=bool) for ch in tr.surprise}
    for a in detection.alarms:
        flags[a.channel][a.tick] = True
    verbs, nouns, mask = jnp.asarray(tr.verbs), jnp.asarray(tr.nouns), jnp.ones(n, dtype=bool)
    e = kernels.sequence_loglik(verbs, nouns, t.log_emit_v, t.log_emit_n, mask)
    pi_tilde = np.asarray(kernels.predictive_occupancy(
        e, t.log_init[r], t.log_A[r], t.log_pmf[r], t.log_surv[r], mask, hsmm.d_max))
    ctx = _Ctx(tr, flags, detector.thresholds(tr, op), pi_tilde,
               Lexicon(vocab, select_recipe(hsmm.params, r), sil_verb, sil_noun),
               np.asarray(t.log_emit_v), np.asarray(t.log_emit_n), np.asarray(t.log_A[r]), min_gain)
    spans = segment_spans(tr)
    queries = [q for i, (_, s, e_) in enumerate(spans) for q in _item_query(ctx, i, s, e_)
               + _action_query(ctx, i, s, e_)]
    queries += [q for i, (k, s, e_) in enumerate(spans) for q in _stall_query(ctx, i, k, s, e_)]
    queries += [q for i, (k, s, e_) in enumerate(spans) for q in _duration_query(ctx, i, k, s, e_)]
    queries += [q for i, (k, s, _) in enumerate(spans) for q in _order_query(ctx, i, k, s)]
    return sorted(queries, key=lambda q: (q.tick, q.channel))
