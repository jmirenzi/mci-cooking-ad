"""Is an injected synthetic trial actually anomalous? (docs/drift_synth.md §9.5)

The reference is a large healthy sample of the SAME simulated person, on sessions no pool uses, in
the tick representation the detector sees. Three checks on the trial's runs (maximal stretches of
constant (verb, noun)):
  bigram    a run bigram (with start / end markers) that never occurs in the healthy sample;
  count     a non-idle run count, per recipe, outside its healthy [min, max];
  duration  a run length outside its token's healthy [0.2%, 99.8%] quantiles.
A trial none of them flags is within what the person does locally: scoring a detector on it
measures nothing. A fourth, non-local check is reported apart (`order`): two actions whose first
occurrences come in an order no healthy session of the recipe has. It catches precedence
violations the three local checks cannot (a step moved before one it depends on, whose own
neighbours are ordinary).
"""
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

DUR_Q = (0.002, 0.998)


def _first_order(rr, idle):
    """{(x, y)}: non-idle tokens x, y with x's first run before y's."""
    first = []
    for t, _ in rr:
        if t != idle and t not in first:
            first.append(t)
    return {(x, y) for i, x in enumerate(first) for y in first[i + 1:]}


def runs(verb_ids, noun_ids):
    """[((verb, noun), length)] of a tick stream."""
    out = []
    for a, b in zip(verb_ids, noun_ids):
        if out and out[-1][0] == (int(a), int(b)):
            out[-1][1] += 1
        else:
            out.append([(int(a), int(b)), 1])
    return [(t, k) for t, k in out]


VERSION = 3          # 3: session time frame, CC4D step totals (2026-10-20)


class HealthySupport:
    def __init__(self, run_seqs_by_recipe, idle):
        self.version = VERSION
        self.idle = idle
        self.before = defaultdict(set)     # recipe -> {(x, y)}: x first seen before y first
        self.bigram = defaultdict(set)
        cnt = defaultdict(lambda: defaultdict(list))
        durs = defaultdict(list)
        self.n = {}
        for r, seqs in run_seqs_by_recipe.items():
            self.n[r] = len(seqs)
            for rr in seqs:
                toks = ["BOS"] + [t for t, _ in rr] + ["EOS"]
                self.bigram[r].update(zip(toks, toks[1:]))
                c = Counter(t for t, _ in rr if t != idle)
                for t in set(c) | set(cnt[r]):
                    cnt[r][t].append(c.get(t, 0))
                for t, k in rr:
                    durs[t].append(k)
                self.before[r].update(_first_order(rr, idle))
        # a token first seen late in the sample was 0 in the sessions before it
        self.count_range = {r: {t: (0 if len(v) < self.n[r] else min(v), max(v)) for t, v in d.items()}
                            for r, d in cnt.items()}
        self.dur_range = {t: tuple(np.quantile(x, DUR_Q)) for t, x in durs.items()}
        self.bigram = dict(self.bigram)
        self.before = dict(self.before)

    def check(self, recipe, verb_ids, noun_ids):
        rr = runs(verb_ids, noun_ids)
        toks = ["BOS"] + [t for t, _ in rr] + ["EOS"]
        bi = any(b not in self.bigram[recipe] for b in zip(toks, toks[1:]))
        c = Counter(t for t, _ in rr if t != self.idle)
        rng = self.count_range[recipe]
        cnt = any(not (rng.get(t, (0, 0))[0] <= c.get(t, 0) <= rng.get(t, (0, 0))[1])
                  for t in set(c) | set(rng))
        dur = any(t in self.dur_range and not (self.dur_range[t][0] <= k <= self.dur_range[t][1])
                  for t, k in rr)
        order = bool(_first_order(rr, self.idle) - self.before.get(recipe, set()))
        return {"bigram": bi, "count": cnt, "duration": dur, "anomalous": bi or cnt or dur,
                "order": order, "anomalous_or_order": bi or cnt or dur or order}

    def save(self, path):
        Path(path).write_bytes(pickle.dumps(self))

    @staticmethod
    def load(path):
        return pickle.loads(Path(path).read_bytes())


def person_support(person_idx, seed, assets, vocab, first, n_sessions, tick_seconds, cache=None):
    """HealthySupport from sessions [first, n_sessions) of one person (generated, noise as `assets`)."""
    if cache and Path(cache).exists():
        sup = HealthySupport.load(cache)
        if getattr(sup, "version", 1) == VERSION:
            return sup
    from cook_ad.drift_synth.export import to_ticks
    from cook_ad.drift_synth.sampler import ConditionCfg, generate_person
    _, sessions, _ = generate_person(person_idx, ConditionCfg("null", n_persons=1, n_sessions=n_sessions),
                                     seed, assets)
    ex = assets.gen["export"]
    by = defaultdict(list)
    for s in sessions:
        if s.index >= first:
            v, n = to_ticks(s.events, vocab, ex["sil"], ex["fps"], tick_seconds)
            by[s.recipe_id].append(runs(v, n))
    sup = HealthySupport(dict(by), (vocab["verbs"][ex["sil"][0]], vocab["nouns"][ex["sil"][1]]))
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        sup.save(cache)
    return sup
