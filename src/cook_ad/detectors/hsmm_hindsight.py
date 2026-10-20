"""class HindsightHSMMDetector: alarms read off the whole-trial decode (docs/detectors.md §5).

r_hat, then the Viterbi segments under r_hat, then each channel's surprise along that path,
tested against the alpha-tail thresholds of BadSets.hindsight(r_hat, alpha).
"""
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from scipy.special import logsumexp

from cook_ad.core import Alarm, AnomalyDetector, Detection, OperatingPoint, Trial
from cook_ad.hsmm import batch, durations

COUNT = "s_count"   # optional eighth channel (hsmm/visit_counts.py), only with fitted counts
CHANNELS = ("s_emit", "s_verb", "s_noun", "s_temporal", "s_dur_two", "s_transition",
            "s_recipe_transition")
EMISSION_FLAG_ATOL = 1e-6   # emission surprise must beat its threshold by more than float noise
ATTRIBUTION_MARGIN = 2.0    # nats by which s_noun must beat s_verb (or vice versa) to attribute
LOG2 = float(np.log(2.0))


class HindsightTrace(NamedTuple):
    """One trial's decode and channel surprises (true length T)."""
    r_hat: int
    segments: list                  # [(subtask, duration)], oldest first
    z_star: np.ndarray              # (T,) subtask on the MAP path
    verbs: np.ndarray               # (T,)
    nouns: np.ndarray               # (T,)
    surprise: dict                  # channel -> (T,) surprise in nats
    pi_at_zstar: np.ndarray         # (T,) P(z_t = z*_t | o_<t), the emission dilution
    from_state: np.ndarray          # (T,) previous subtask at segment starts, -1 elsewhere
    expected_verb: np.ndarray       # (T,) argmax_v P(v | z*_t)
    expected_noun: np.ndarray       # (T,)
    expected_next_state: np.ndarray  # (T,) argmax_i A[r_hat, previous, i] at segment starts
    attribution: np.ndarray         # (T,) "item" / "action" / "none"
    temporal_attribution: np.ndarray  # (T,) "stuck" / "left_early" / "none" at segment ends


def _at_segment_ends(segments, values, fill=0.0, dtype=np.float64):
    out = np.full(sum(d for _, d in segments), fill, dtype=dtype)
    ends = np.cumsum([d for _, d in segments]) - 1
    out[ends] = values
    return out


def table_tails(tables, r, k, d):
    """(log P(D >= d), log P(D <= d)) of states k at lengths d, from the duration tables."""
    log_surv = np.asarray(tables.log_surv[r], dtype=float)
    D = log_surv.shape[-1]
    d = np.minimum(np.asarray(d), D)
    ls = log_surv[k, d - 1]
    nxt = np.where(d < D, log_surv[k, np.minimum(d, D - 1)], -np.inf)       # log P(D >= d + 1)
    with np.errstate(divide="ignore"):
        lc = np.log1p(-np.exp(nxt))
    return ls, lc


def _from_previous(segments):
    """At each segment's first tick, the previous segment's subtask; -1 elsewhere."""
    out = np.full(sum(d for _, d in segments), -1, dtype=np.int64)
    starts = np.cumsum([0] + [d for _, d in segments[:-1]])
    out[starts[1:]] = [k for k, _ in segments[:-1]]
    return out


class HindsightHSMMDetector(AnomalyDetector):
    """Not causal: needs the whole trial before any alarm."""
    name = "hsmm-hindsight"
    causal = False

    def __init__(self, hsmm, bad_sets, op: OperatingPoint, chunk_size=8, counts=None):
        """`counts`: hsmm/visit_counts.VisitCounts fitted on the training decodes; adds the
        s_count channel (§5.7). Without it the detector is exactly the seven-channel one."""
        self.hsmm, self.bad, self.op, self.chunk_size = hsmm, bad_sets, op, chunk_size
        self.counts = counts
        self.channels = CHANNELS + (COUNT,) if counts is not None else CHANNELS

    def reset(self, recipe=None):
        pass

    # ---- channels (docs/detectors.md §5.1-5.5)

    def emission_surprise(self, pi_tilde, verbs, nouns):
        """-log P(o_t | o_<t) for (v, n), v alone and n alone, under the forecast mixture."""
        t = self.hsmm.tables
        lv = np.asarray(t.log_emit_v)[:, verbs].T
        ln = np.asarray(t.log_emit_n)[:, nouns].T
        return (-logsumexp(pi_tilde + lv + ln, axis=-1), -logsumexp(pi_tilde + lv, axis=-1),
                -logsumexp(pi_tilde + ln, axis=-1))

    def stall_surprise(self, segments, r):
        """-log P(D >= elapsed) at every tick of every segment."""
        log_surv = np.asarray(self.hsmm.tables.log_surv[r])
        d_max = log_surv.shape[1]
        return np.concatenate([-log_surv[k, np.minimum(np.arange(d), d_max - 1)]
                               for k, d in segments])

    def duration_surprise(self, segments, r):
        """Two-sided -log p of each closed segment's length, at its last tick; the last segment
        is still running, so it is not scored. -> (s_dur_two, 'stuck' / 'left_early')."""
        n = len(segments) - 1
        s_two, attr = np.zeros(n + 1), np.full(n + 1, "none", dtype=object)
        if n > 0:
            k = np.array([s for s, _ in segments[:n]])
            d = np.array([float(d) for _, d in segments[:n]])
            if getattr(self.hsmm, "table_durations", False):    # nonparametric hazard tables
                log_surv, log_cdf = table_tails(self.hsmm.tables, r, k, d.astype(int))
            else:
                p = self.hsmm.params
                rr, pp = np.asarray(p.dur_r[r])[k].astype(float), np.asarray(p.dur_p[r])[k].astype(float)
                log_surv = durations.nb_log_survival_np(d, rr, pp)
                log_cdf = durations.nb_log_cdf_np(d, rr, pp)
            s_two[:n] = np.maximum(0.0, -(LOG2 + np.minimum(log_surv, log_cdf)))
            attr[:n] = np.where(log_surv < log_cdf, "stuck", "left_early")
        return (_at_segment_ends(segments, s_two),
                _at_segment_ends(segments, attr, fill="none", dtype=object))

    def transition_surprise(self, segments, log_A):
        """-log A[k, i] at each segment start, and argmax_i A[k, i]."""
        log_A = np.asarray(log_A)
        s = np.zeros(sum(d for _, d in segments))
        expected = np.full(s.shape, -1, dtype=np.int64)
        pos = segments[0][1]
        for (k, _), (i, d) in zip(segments[:-1], segments[1:]):
            s[pos], expected[pos] = float(-log_A[k, i]), int(np.argmax(log_A[k]))
            pos += d
        return s, expected

    def recipe_transition_surprise(self, segments, s_transition):
        """log A_marg[k, i] - log A_r[k, i]: ordinary in general, wrong for this recipe. Zero
        when there is one recipe (docs/detectors.md §2.3)."""
        if self.hsmm.k_recipe == 1:
            return np.zeros_like(s_transition)
        return s_transition - self.transition_surprise(segments, self.hsmm.tables.log_A_marginal)[0]

    def count_surprise(self, segments, r):
        """-log p of each visit's count and of each missing subtask (§5.7), at the segment start
        where it shows; the largest where two land on one tick."""
        s = np.zeros(sum(d for _, d in segments))
        for t, p in self.counts.path_pvalues(int(r), segments):
            s[t] = max(s[t], -float(np.log(p)))
        return s

    # ---- one trial's trace

    def assemble(self, r, segments, pi_tilde, verbs, nouns) -> HindsightTrace:
        t = self.hsmm.tables
        z = np.repeat([k for k, _ in segments], [d for _, d in segments])
        s_emit, s_verb, s_noun = self.emission_surprise(pi_tilde, verbs, nouns)
        s_trans, expected_next = self.transition_surprise(segments, t.log_A[r])
        s_dur_two, temporal_attr = self.duration_surprise(segments, r)
        surprise = {"s_emit": s_emit, "s_verb": s_verb, "s_noun": s_noun,
                    "s_temporal": self.stall_surprise(segments, r), "s_dur_two": s_dur_two,
                    "s_transition": s_trans,
                    "s_recipe_transition": self.recipe_transition_surprise(segments, s_trans)}
        if self.counts is not None:
            surprise[COUNT] = self.count_surprise(segments, r)
        attribution = np.full(len(z), "none", dtype=object)
        attribution[(s_noun - s_verb) > ATTRIBUTION_MARGIN] = "item"
        attribution[(s_verb - s_noun) > ATTRIBUTION_MARGIN] = "action"
        return HindsightTrace(
            int(r), segments, z, np.asarray(verbs), np.asarray(nouns), surprise,
            np.exp(pi_tilde)[np.arange(len(z)), z], _from_previous(segments),
            np.argmax(np.asarray(t.log_emit_v)[z], axis=-1),
            np.argmax(np.asarray(t.log_emit_n)[z], axis=-1), expected_next, attribution,
            temporal_attr)

    # ---- thresholds

    def thresholds(self, trace: HindsightTrace, op: OperatingPoint | None = None):
        """{channel: (T,) threshold}; segment-start-only channels are +inf elsewhere."""
        op = op or self.op
        th = {c: self.bad.hindsight(trace.r_hat, op.alpha_for(c)) for c in CHANNELS}
        z, at_start = trace.z_star, trace.from_state != -1
        prev = np.where(at_start, trace.from_state, 0)
        dilution = -np.log(trace.pi_at_zstar)
        return {
            "s_emit": th["s_emit"].emit[z] + dilution,
            "s_verb": th["s_verb"].verb[z] + dilution,
            "s_noun": th["s_noun"].noun[z] + dilution,
            "s_temporal": np.full(len(z), -float(np.log(op.alpha_for("s_temporal")))),
            "s_dur_two": np.full(len(z), -float(np.log(op.alpha_for("s_dur_two")))),
            "s_transition": np.where(at_start, th["s_transition"].transition[prev], np.inf),
            "s_recipe_transition": np.where(at_start, th["s_recipe_transition"].recipe[prev],
                                            np.inf),
            **({COUNT: np.full(len(z), -float(np.log(max(op.alpha_for(COUNT, 0.0), 1e-300))))}
               if self.counts is not None else {}),
        }

    def flags(self, trace: HindsightTrace, op: OperatingPoint | None = None):
        """{channel: (T,) bool}: surprise above its threshold."""
        th = self.thresholds(trace, op)
        out = {c: trace.surprise[c] > th[c] for c in self.channels}
        for c in ("s_emit", "s_verb", "s_noun"):
            out[c] = trace.surprise[c] > th[c] + EMISSION_FLAG_ATOL
        return out

    # ---- whole trials

    def detect_many(self, trials: list[Trial]) -> list[Detection]:
        traces = self.traces(trials)
        return [self._detection(tr) for tr in traces]

    def detect_trial(self, trial: Trial) -> Detection:
        return self.detect_many([trial])[0]

    def _detection(self, trace):
        flags, n = self.flags(trace), len(trace.z_star)
        alarms = [Alarm(int(t), c, float(trace.surprise[c][t]), n - 1)
                  for t in range(n) for c in self.channels if flags[c][t]]
        return Detection(alarms, trace=trace)

    def recipes(self, trials, verb, noun, mask):
        """r_hat per trial: the given dish, or argmax_r P(r | o) (plus a log prior if given)."""
        if all(t.recipe is not None and np.ndim(t.recipe) == 0 for t in trials):
            return jnp.asarray([int(t.recipe) for t in trials])
        r_hat, rho, _ = batch.infer_recipe(self.hsmm.tables, verb, noun, mask, self.hsmm.d_max,
                                           self.chunk_size)
        r_hat = np.asarray(r_hat).copy()
        for i, t in enumerate(trials):
            if t.recipe is not None and np.ndim(t.recipe) == 0:
                r_hat[i] = int(t.recipe)
            elif t.recipe is not None:
                r_hat[i] = int(np.argmax(np.log(np.asarray(rho[i])) + np.asarray(t.recipe)))
        return jnp.asarray(r_hat)

    def traces(self, trials: list[Trial]) -> list[HindsightTrace]:
        """Batched over trials, in chunks, padded to the longest trial (hsmm/batch.py)."""
        t, d_max, c = self.hsmm.tables, self.hsmm.d_max, self.chunk_size
        verbs = [tr.verbs for tr in trials]
        nouns = [tr.nouns for tr in trials]
        verb, noun, mask = batch.pad(verbs, nouns)
        r_hat = self.recipes(trials, verb, noun, mask)
        out = []
        for s in range(0, len(trials), c):
            r = r_hat[s : s + c]
            v, w, m = verb[s : s + c], noun[s : s + c], mask[s : s + c]
            pi = np.asarray(batch.occupancy(v, w, m, t.log_init[r], t.log_A[r], t.log_emit_v,
                                            t.log_emit_n, t.log_pmf[r], t.log_surv[r], d_max))
            segs = batch.viterbi(t, r, v, w, m, d_max)
            for i, seg in enumerate(segs):
                n = len(verbs[s + i])
                out.append(self.assemble(int(r[i]), seg, pi[i][:n], np.asarray(verbs[s + i]),
                                         np.asarray(nouns[s + i])))
        return out
