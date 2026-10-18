"""Empirical left-tail duration channel for the hindsight HSMM (docs/drift_synth.md §9.7).

The negative binomial fitted per (recipe, state) has the right spread but a fat left tail: it
puts a median 9.8% of mass below a cell's empirical 1% point (§9.5), so a run cut to a fraction
of its length rarely reaches a small p-value. This channel uses the training durations
themselves: for each closed segment of a trial, p = (#training segments of that (recipe, state)
no longer than it + 0.5) / (n + 1), flagged at the segment's last tick. Only the short side is
scored; the stall and two-sided channels keep the long side. Cells with fewer than `min_n`
training segments are not scored.
"""
from collections import defaultdict

import numpy as np


class EmpiricalShortDuration:
    def __init__(self, ignore=(), min_n=10):
        self.ignore, self.min_n = set(ignore), min_n
        self.d = defaultdict(list)            # (recipe, state) -> sorted training durations

    @classmethod
    def fit(cls, traces, ignore=(), min_n=10):
        ch = cls(ignore, min_n)
        for tr in traces:
            r = int(tr.r_hat)
            for k, d in tr.segments[:-1]:      # the last segment is cut by the trial's end
                if int(k) not in ch.ignore:
                    ch.d[(r, int(k))].append(int(d))
        ch.d = {k: np.sort(v) for k, v in ch.d.items()}
        return ch

    def pvalues(self, trace):
        r = int(trace.r_hat)
        out, t = [], 0
        for k, d in trace.segments[:-1]:
            t += d
            ref = self.d.get((r, int(k)))
            if int(k) in self.ignore or ref is None or len(ref) < self.min_n:
                continue
            le = int(np.searchsorted(ref, d, side="right"))
            out.append((t - 1, (le + 0.5) / (len(ref) + 1.0)))
        return out

    def flags(self, trace, levels):
        T = int(sum(d for _, d in trace.segments))
        f = np.zeros((T, len(levels)), dtype=bool)
        lv = np.asarray(levels, dtype=float)
        for t, p in self.pvalues(trace):
            f[t] |= p <= lv
        return f
