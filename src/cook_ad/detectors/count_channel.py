"""A visit-count channel for the hindsight HSMM (docs/drift_synth.md §9.7).

A first-order HSMM sees only adjacent pairs, so a recipe step that is simply skipped is
invisible when the step before and the step after are allowed neighbours, which is the rule in a
partially ordered recipe (§9.5: 74% of omitted recipe actions are visible only by counting). This
channel counts, per recipe, how often each state is visited in a training session (from the
model's own Viterbi decodes) and scores a trial's decoded path against it:

  missing  a state the training sessions of this recipe almost always visit is never visited;
           p = P(count = 0), flagged at the start of the first segment whose state usually
           follows the missing one (where an omission shows), else at the last segment;
  extra    a state's m-th visit, where P(count >= m) is small; flagged at that visit's start.

p-values are smoothed frequencies, (k + 0.5) / (N + 1) over the N training sessions of the
recipe, so the smallest reachable p is 0.5 / (N + 1): the channel says nothing about a recipe
with a handful of sessions. `flags(trace, levels)` -> (T, L) bool, a flag where p <= level.
"""
from collections import Counter, defaultdict

import numpy as np


class CountChannel:
    def __init__(self, ignore=()):
        self.ignore = set(ignore)         # states not counted (the idle gap state)
        self.n = Counter()                # recipe -> training sessions
        self.ge = defaultdict(Counter)    # (recipe, state) -> {m: sessions with count >= m}
        self.zero = defaultdict(Counter)  # recipe -> {state: sessions with count 0}
        self.states = defaultdict(set)    # recipe -> states seen
        self.pred = defaultdict(Counter)  # (recipe, state j) -> {predecessor state: n}

    @classmethod
    def fit(cls, traces, ignore=()):
        ch = cls(ignore)
        for tr in traces:
            r = int(tr.r_hat)
            ks = [int(k) for k, _ in tr.segments if int(k) not in ch.ignore]
            ch.n[r] += 1
            c = Counter(ks)
            ch.states[r].update(c)
            for k, m in c.items():
                for j in range(1, m + 1):
                    ch.ge[(r, k)][j] += 1
            for a, b in zip(ks, ks[1:]):
                ch.pred[(r, b)][a] += 1
        for r in ch.n:
            for k in ch.states[r]:
                ch.zero[r][k] = ch.n[r] - ch.ge[(r, k)][1]
        return ch

    def _p(self, k, n):
        return (k + 0.5) / (n + 1.0)

    def pvalues(self, trace):
        """[(tick, p)] for every missing state and every excess visit of this trial."""
        r = int(trace.r_hat)
        n = self.n.get(r, 0)
        if n == 0:
            return []
        starts = np.cumsum([0] + [d for _, d in trace.segments[:-1]])
        ks = [int(k) for k, _ in trace.segments]
        out = []
        seen = Counter()
        for s, k in zip(starts, ks):
            if k in self.ignore:
                continue
            seen[k] += 1
            out.append((int(s), self._p(self.ge[(r, k)][seen[k]], n)))   # P(count >= this visit)
        for k in self.states[r] - set(seen):
            p = self._p(self.zero[r][k], n)
            # where the missing state would have been: before a state it usually precedes
            at = next((int(s) for s, j in zip(starts, ks)
                       if self.pred[(r, j)] and self.pred[(r, j)].most_common(1)[0][0] == k),
                      int(starts[-1]))
            out.append((at, p))
        return out

    def flags(self, trace, levels):
        T = int(sum(d for _, d in trace.segments))
        f = np.zeros((T, len(levels)), dtype=bool)
        lv = np.asarray(levels, dtype=float)
        for t, p in self.pvalues(trace):
            f[t] |= p <= lv
        return f
