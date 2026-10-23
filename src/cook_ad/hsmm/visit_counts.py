"""Per-recipe precedence statistics for the s_count channel (docs/detectors.md §3.6a, §5.6).

A first-order HSMM sees adjacent pairs only, so a skipped step whose neighbours may follow each
other is invisible. s_count compares the session so far with the person's training sessions,
decoded by the same model (scripts/train/fit_visit_counts.py). At the first visit of j it asks
whether some k that has not happened yet (almost) always came before j's first visit in
training: P(k not before j | r, j) small means k was skipped, or j moved ahead of it. A subtask
the recipe never had in training is the same test with P(j visited | r).

The rule reads only the session so far, so hindsight (on z*) and live (on expected visits) apply
the same test. It replaced "k's usual successor began while k had not happened", which on
flexible recipes fired on 77-91% of healthy sessions live. An extra-visit half (P(C_k >= c | r)
at the c-th visit) was dropped after the 2026-10-20 comparison (scripts/eval/compare_s_count.py):
it cost Breakfast 0.012 G-F1 and bought synthetic nothing.

Probabilities are smoothed frequencies, (n + 0.5) / (N + 1), over the training sessions of the
recipe or those that contain j, so the smallest is 0.5 / (N + 1). Stored as JSON next to model.npz.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

MAX_P = 0.5      # precedence pairs kept: only those that could ever be flagged


class VisitCounts:
    def __init__(self, n_sessions, n_with, before, ignore=()):
        self.n = {int(r): int(v) for r, v in n_sessions.items()}             # recipe -> N_r
        self.n_with = {int(r): {int(j): int(m) for j, m in d.items()} for r, d in n_with.items()}
        self.before = {int(r): {int(j): {int(k): int(m) for k, m in row.items()} for j, row in d.items()}
                       for r, d in before.items()}                              # r -> j -> {k: #(k before j)}
        self.ignore = set(int(k) for k in ignore)
        self._prec = {}

    # ---- fit / io

    @classmethod
    def fit(cls, decodes, ignore=()):
        """decodes: [(recipe, [subtask per segment])] of the training sessions."""
        ignore = set(int(k) for k in ignore)
        n = Counter()
        n_with = defaultdict(Counter)
        before = defaultdict(lambda: defaultdict(Counter))
        for r, ks in decodes:
            r = int(r)
            n[r] += 1
            seen = set()
            for j in (int(k) for k in ks if int(k) not in ignore):
                if j not in seen:                      # j's first visit: what came before it
                    n_with[r][j] += 1
                    for k in seen:
                        before[r][j][k] += 1
                    seen.add(j)
        return cls(n, n_with, before, ignore)

    def save(self, path):
        Path(path).write_text(json.dumps({"n_sessions": self.n, "n_with": self.n_with,
                                          "before": self.before, "ignore": sorted(self.ignore)}))

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        return cls(d["n_sessions"], d["n_with"], d["before"], d["ignore"])

    # ---- probabilities

    def known(self, r):
        return r in self.n

    def p_seen(self, r, j):
        """P(j is visited at all | r)."""
        return (self.n_with.get(r, {}).get(j, 0) + 0.5) / (self.n[r] + 1.0)

    def precedence(self, r, j):
        """[(k, P(k not before j's first visit | r, j))], only the pairs with P <= MAX_P."""
        key = (r, j)
        if key not in self._prec:
            nj = self.n_with.get(r, {}).get(j, 0)
            row = self.before.get(r, {}).get(j, {})
            self._prec[key] = sorted(((k, (nj - m + 0.5) / (nj + 1.0)) for k, m in row.items()
                                      if (nj - m + 0.5) / (nj + 1.0) <= MAX_P), key=lambda x: x[1])
        return self._prec[key]

    def p_first(self, r, j, not_yet):
        """The p-value of j's first visit, given `not_yet(k)` (k has not happened in the session
        so far): the smallest of P(j visited | r) and P(k not before j) over the k not yet."""
        p = self.p_seen(r, j)
        for k, pk in self.precedence(r, j):
            if pk >= p:
                break
            if k not in self.ignore and not_yet(k):
                return pk
        return p

    # ---- hindsight: on a decoded path

    def path_pvalues(self, r, segments):
        """[(tick, p)] at every first visit of a non-idle subtask on the decoded path."""
        if not self.known(r):
            return []
        starts = np.cumsum([0] + [d for _, d in segments[:-1]])
        seen, out = set(), []
        for s, (k, _) in zip(starts, segments):
            k = int(k)
            if k in self.ignore or k in seen:
                continue
            out.append((int(s), self.p_first(r, k, lambda kk: kk not in seen)))
            seen.add(k)
        return out
