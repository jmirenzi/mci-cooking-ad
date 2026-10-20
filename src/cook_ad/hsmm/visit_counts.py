"""Per-recipe visit and precedence statistics for the s_count channel (docs/detectors.md §3.6a, §5.6).

A first-order HSMM sees adjacent pairs only, so a skipped step whose neighbours may follow each
other is invisible, and so is a step done twice with a legal step between. s_count compares the
session so far with the person's training sessions, decoded by the same model
(scripts/train/fit_visit_counts.py):

  extra      the c-th visit of k (c >= 2), where P(C_k >= c | r) is small;
  precedence the first visit of j while some k has not happened yet, where in training k had
             (almost) always happened before j's first visit: P(k not before j | r, j) is small.
             That is an omission (k skipped) or a transposition (j moved before k).

The precedence rule replaced "k's usual successor began while k had not happened", which on
flexible recipes fired on 77-91% of healthy sessions live (the usual successor is often not the
only one, so k legitimately comes later; validate_s_count.py). Both rules read only the session
so far, so hindsight (on z*) and live (on expected visits) apply the same test.

Probabilities are smoothed frequencies, (n + 0.5) / (N + 1), over the training sessions of the
recipe (extra) or those that contain j (precedence), so the smallest is 0.5 / (N + 1). Stored as
JSON next to model.npz.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

MAX_P = 0.5      # precedence pairs kept: only those that could ever be flagged


class VisitCounts:
    def __init__(self, n_sessions, ge, n_with, before, ignore=()):
        self.n = {int(r): int(v) for r, v in n_sessions.items()}             # recipe -> N_r
        self.ge = {int(r): {int(k): {int(c): int(m) for c, m in row.items()} for k, row in d.items()}
                   for r, d in ge.items()}                                     # r -> k -> {c: #(C_k >= c)}
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
        ge = defaultdict(lambda: defaultdict(Counter))
        n_with = defaultdict(Counter)
        before = defaultdict(lambda: defaultdict(Counter))
        for r, ks in decodes:
            r = int(r)
            ks = [int(k) for k in ks if int(k) not in ignore]
            n[r] += 1
            for k, m in Counter(ks).items():
                for c in range(1, m + 1):
                    ge[r][k][c] += 1
            seen = set()
            for j in ks:
                if j not in seen:                      # j's first visit: what came before it
                    n_with[r][j] += 1
                    for k in seen:
                        before[r][j][k] += 1
                    seen.add(j)
        return cls(n, ge, n_with, before, ignore)

    def save(self, path):
        Path(path).write_text(json.dumps({"n_sessions": self.n, "ge": self.ge, "n_with": self.n_with,
                                          "before": self.before, "ignore": sorted(self.ignore)}))

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        return cls(d["n_sessions"], d["ge"], d["n_with"], d["before"], d["ignore"])

    # ---- probabilities

    def known(self, r):
        return r in self.n

    def p_at_least(self, r, k, c):
        """P(C_k >= c | r), c >= 1."""
        return (self.ge.get(r, {}).get(k, {}).get(c, 0) + 0.5) / (self.n[r] + 1.0)

    def precedence(self, r, j):
        """[(k, P(k not before j's first visit | r, j))], only the pairs with P <= MAX_P."""
        key = (r, j)
        if key not in self._prec:
            nj = self.n_with.get(r, {}).get(j, 0)
            row = self.before.get(r, {}).get(j, {})
            self._prec[key] = sorted(((k, (nj - m + 0.5) / (nj + 1.0)) for k, m in row.items()
                                      if (nj - m + 0.5) / (nj + 1.0) <= MAX_P), key=lambda x: x[1])
        return self._prec[key]

    def p_begin(self, r, j, visit, not_yet):
        """The p-value of j beginning as its `visit`-th visit, given `not_yet(k)` (k has not
        happened in the session so far): P(C_j >= visit) for a repeat, else the smallest
        P(k not before j) over the k that have not happened."""
        if visit >= 2:
            return self.p_at_least(r, j, visit)
        p = self.p_at_least(r, j, 1)                    # a subtask this recipe never had in training
        for k, pk in self.precedence(r, j):
            if pk >= p:
                break
            if k not in self.ignore and not_yet(k):
                p = pk
                break
        return p

    # ---- hindsight: on a decoded path

    def path_pvalues(self, r, segments):
        """[(tick, p)] at every non-idle segment start of the decoded path."""
        if not self.known(r):
            return []
        starts = np.cumsum([0] + [d for _, d in segments[:-1]])
        seen, out = Counter(), []
        for s, (k, _) in zip(starts, segments):
            k = int(k)
            if k in self.ignore:
                continue
            seen[k] += 1
            out.append((int(s), self.p_begin(r, k, seen[k], lambda kk: seen[kk] == 0)))
        return out
