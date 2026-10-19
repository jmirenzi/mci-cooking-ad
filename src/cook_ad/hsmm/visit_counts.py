"""Per-recipe visit-count statistics for the s_count channel (docs/detectors.md §5.7).

A first-order HSMM sees adjacent pairs only, so a skipped step whose neighbours may follow each
other is invisible, and so is a step done twice with a legal step between. s_count compares how
often each subtask is visited in a session with the person's training sessions, decoded by the
same model (scripts/train/fit_visit_counts.py):

  extra    the c-th visit of k, where P(C_k >= c | r) is small;
  missing  k never visited, where P(C_k = 0 | r) is small, reported where k's usual successor
           begins (where a skipped step shows).

Probabilities are smoothed frequencies over the N_r training sessions of the recipe,
(n + 0.5) / (N_r + 1), so the smallest is 0.5 / (N_r + 1). Stored as JSON next to model.npz.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


class VisitCounts:
    def __init__(self, n_sessions, ge, successor, ignore=()):
        self.n = {int(r): int(v) for r, v in n_sessions.items()}             # recipe -> N_r
        self.ge = {int(r): {int(k): {int(c): int(m) for c, m in row.items()} for k, row in d.items()}
                   for r, d in ge.items()}                                     # r -> k -> {c: #(C_k >= c)}
        self.successor = {int(r): {int(k): int(j) for k, j in d.items()} for r, d in successor.items()}
        self.ignore = set(int(k) for k in ignore)

    # ---- fit / io

    @classmethod
    def fit(cls, decodes, ignore=()):
        """decodes: [(recipe, [subtask per segment])] of the training sessions."""
        ignore = set(int(k) for k in ignore)
        n, ge, follow = Counter(), defaultdict(lambda: defaultdict(Counter)), defaultdict(lambda: defaultdict(Counter))
        for r, ks in decodes:
            r = int(r)
            ks = [int(k) for k in ks if int(k) not in ignore]
            n[r] += 1
            for k, m in Counter(ks).items():
                for c in range(1, m + 1):
                    ge[r][k][c] += 1
            for a, b in zip(ks, ks[1:]):
                follow[r][a][b] += 1
        successor = {r: {k: row.most_common(1)[0][0] for k, row in d.items()} for r, d in follow.items()}
        return cls(n, ge, successor, ignore)

    def save(self, path):
        Path(path).write_text(json.dumps({"n_sessions": self.n, "ge": self.ge,
                                          "successor": self.successor,
                                          "ignore": sorted(self.ignore)}))

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        return cls(d["n_sessions"], d["ge"], d["successor"], d["ignore"])

    # ---- probabilities

    def _p(self, m, r):
        return (m + 0.5) / (self.n[r] + 1.0)

    def p_at_least(self, r, k, c):
        """P(C_k >= c | r), c >= 1."""
        return self._p(self.ge.get(r, {}).get(k, {}).get(c, 0), r)

    def p_zero(self, r, k):
        return self._p(self.n[r] - self.ge.get(r, {}).get(k, {}).get(1, 0), r)

    def known(self, r):
        return r in self.n

    def states(self, r):
        return set(self.ge.get(r, {}))

    def predecessors(self, r):
        """j -> [k whose usual successor is j]."""
        out = defaultdict(list)
        for k, j in self.successor.get(r, {}).items():
            out[j].append(k)
        return out

    # ---- hindsight: on a decoded path

    def path_pvalues(self, r, segments):
        """[(tick, p)] for every visit (P(C_k >= its number)) and every missing state
        (P(C_k = 0), at its usual successor's start, else at the last segment)."""
        if not self.known(r):
            return []
        starts = np.cumsum([0] + [d for _, d in segments[:-1]])
        seen, out = Counter(), []
        for s, (k, _) in zip(starts, segments):
            k = int(k)
            if k in self.ignore:
                continue
            seen[k] += 1
            out.append((int(s), self.p_at_least(r, k, seen[k])))
        ks = [int(k) for k, _ in segments]
        for k in self.states(r) - set(seen):
            j = self.successor.get(r, {}).get(k)
            at = next((int(s) for s, kk in zip(starts, ks) if kk == j), int(starts[-1]))
            out.append((at, self.p_zero(r, k)))
        return out
