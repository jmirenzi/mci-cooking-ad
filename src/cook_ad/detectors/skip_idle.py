"""Transitions between consecutive NON-idle states, for the hindsight HSMM (docs/drift_synth.md
§9.7).

In the synthetic streams two actions out of three are separated by an idle gap, and the HSMM's
transitions go A -> idle -> B. Every action follows idle somewhere, so idle -> B is never
surprising, and a repeated, moved or skipped step leaves no unusual transition. Skipping the idle
segments restores the context: on the step-level pools a bigram over non-idle runs sees 91-95% of
omissions, all transpositions and 87-91% of repetitions, at 3-8% of healthy sessions (§9.7).

Fitted from the model's own Viterbi decodes of the training sessions, per recipe:
P(next non-idle state i | previous non-idle state k), with start and end markers, smoothed as
(n_ki + beta) / (n_k + beta * S_r), S_r = the recipe's distinct states + 1. A state the recipe
never visited in training is left to the emission channels. `flags(trace, levels)` -> (T, L)
bool at each non-idle segment start (the end marker at the last tick).
"""
from collections import Counter, defaultdict

import numpy as np

BOS, EOS = -1, -2


class SkipIdleTransitions:
    def __init__(self, ignore=(), beta=0.5):
        self.ignore, self.beta = set(ignore), beta
        self.counts = defaultdict(Counter)    # (recipe, k) -> {i: n}
        self.states = defaultdict(set)        # recipe -> states visited in training

    @classmethod
    def fit(cls, traces, ignore=(), beta=0.5):
        ch = cls(ignore, beta)
        for tr in traces:
            r = int(tr.r_hat)
            ks = [BOS] + [int(k) for k, _ in tr.segments if int(k) not in ch.ignore] + [EOS]
            ch.states[r].update(ks[1:-1])
            for a, b in zip(ks, ks[1:]):
                ch.counts[(r, a)][b] += 1
        return ch

    def pvalues(self, trace):
        r = int(trace.r_hat)
        if r not in self.states:
            return []
        s_r = len(self.states[r]) + 1
        starts = np.cumsum([0] + [d for _, d in trace.segments[:-1]])
        T = int(sum(d for _, d in trace.segments))
        seq = [(int(t), int(k)) for t, (k, _) in zip(starts, trace.segments) if int(k) not in self.ignore]
        out, prev = [], BOS
        for t, k in seq + [(T - 1, EOS)]:
            row = self.counts.get((r, prev))
            known = k == EOS or k in self.states[r]
            if row and known and (prev == BOS or prev in self.states[r]):
                out.append((t, (row[k] + self.beta) / (sum(row.values()) + self.beta * s_r)))
            prev = k
        return out

    def flags(self, trace, levels):
        T = int(sum(d for _, d in trace.segments))
        f = np.zeros((T, len(levels)), dtype=bool)
        lv = np.asarray(levels, dtype=float)
        for t, p in self.pvalues(trace):
            f[t] |= p <= lv
        return f
