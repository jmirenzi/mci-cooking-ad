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

`prior` ({recipe: {(k, i): probability}}, rows summing to 1, e.g. a population's non-idle
bigrams mapped onto this model's states by `population_prior`) adds `prior_weight`
pseudo-observations per row: (n_ki + w p_ki + beta) / (n_k + w + beta S_r). It matters at cold
start, when a person's own rows hold a handful of transitions.
"""
from collections import Counter, defaultdict

import numpy as np

BOS, EOS = -1, -2


class SkipIdleTransitions:
    def __init__(self, ignore=(), beta=0.5, prior=None, prior_weight=0.0):
        self.ignore, self.beta = set(ignore), beta
        self.prior, self.w = prior or {}, float(prior_weight) if prior else 0.0
        self.counts = defaultdict(Counter)    # (recipe, k) -> {i: n}
        self.states = defaultdict(set)        # recipe -> states visited in training

    @classmethod
    def fit(cls, traces, ignore=(), beta=0.5, prior=None, prior_weight=0.0):
        ch = cls(ignore, beta, prior, prior_weight)
        for tr in traces:
            r = int(tr.r_hat)
            ks = [BOS] + [int(k) for k, _ in tr.segments if int(k) not in ch.ignore] + [EOS]
            ch.states[r].update(ks[1:-1])
            for a, b in zip(ks, ks[1:]):
                ch.counts[(r, a)][b] += 1
        for r, rows in ch.prior.items():          # states the population visits are scored too
            if r in ch.states or ch.w:
                ch.states[r].update(k for pair in rows for k in pair if k >= 0)
        return ch

    def pvalues(self, trace):
        r = int(trace.r_hat)
        if r not in self.states:
            return []
        s_r = len(self.states[r]) + 1
        starts = np.cumsum([0] + [d for _, d in trace.segments[:-1]])
        T = int(sum(d for _, d in trace.segments))
        seq = [(int(t), int(k)) for t, (k, _) in zip(starts, trace.segments) if int(k) not in self.ignore]
        pr = self.prior.get(r, {})
        out, prev = [], BOS
        for t, k in seq + [(T - 1, EOS)]:
            row = self.counts.get((r, prev), Counter())
            known = k == EOS or k in self.states[r]
            n_row = sum(row.values()) + (self.w if any(a == prev for a, _ in pr) else 0.0)
            if n_row and known and (prev == BOS or prev in self.states[r]):
                num = row[k] + self.w * pr.get((prev, k), 0.0) + self.beta
                out.append((t, num / (n_row + self.beta * s_r)))
            prev = k
        return out

    def flags(self, trace, levels):
        T = int(sum(d for _, d in trace.segments))
        f = np.zeros((T, len(levels)), dtype=bool)
        lv = np.asarray(levels, dtype=float)
        for t, p in self.pvalues(trace):
            f[t] |= p <= lv
        return f


def population_prior(sequences, recipes, pairs, recipe_names, idle_pair):
    """{recipe index: {(k, i): P}}: non-idle run bigrams (with start / end markers) of a
    population's tick streams, per recipe name, mapped onto a model's states by their (verb, noun)
    (`pairs[k]`, as hsmm/train/global_prior.state_pairs; pairs without a state are dropped)."""
    state_of = {}
    for k, pr in enumerate(pairs):
        if pr is not None:
            state_of.setdefault(tuple(pr), k)
    by_name = defaultdict(Counter)
    for seq, name in zip(sequences, recipes):
        toks, last = [BOS], None
        for v, n in zip(seq["verb_ids"], seq["noun_ids"]):
            t = (int(v), int(n))
            if t != last and t != tuple(idle_pair):
                if t in state_of:
                    toks.append(state_of[t])
            last = t
        toks.append(EOS)
        by_name[name].update(zip(toks, toks[1:]))
    out = {}
    for r, name in enumerate(recipe_names):
        c = by_name.get(name)
        if not c:
            continue
        rows = defaultdict(float)
        for (a, _), n in c.items():
            rows[a] += n
        out[r] = {(a, b): n / rows[a] for (a, b), n in c.items()}
    return out
