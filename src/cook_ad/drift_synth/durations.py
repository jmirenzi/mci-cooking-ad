"""Stage 5: action durations and inter-action gaps.

Log-normal per verb, with a (verb, noun) row where that pair has at least `min_pair` samples, and
a global row behind both. Gaps are a point mass at zero (narrations that touch or overlap) plus a
log-normal on the positive gaps. Source: EPIC-100 narration start/stop times (CaptainCook4D's
fine-grained actions are not public; see docs/drift_synth.md §3).
"""
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class DurationModel:
    source: str
    verb: dict = field(default_factory=dict)        # verb -> [mu, sigma, n] of log seconds
    pair: dict = field(default_factory=dict)        # "verb|noun" -> [mu, sigma, n]
    overall: list = field(default_factory=list)     # [mu, sigma, n]
    gap_zero: float = 0.0                           # P(gap == 0)
    gap: list = field(default_factory=list)         # [mu, sigma, n] of log positive gaps
    clip: tuple = (0.2, 120.0)                      # seconds, duration samples
    gap_clip: float = 30.0                          # seconds, gap samples

    def params(self, token):
        verb, noun = token
        row = self.pair.get(f"{verb}|{noun}") or self.verb.get(verb) or self.overall
        return row[0], row[1]

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), sort_keys=True))

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        d["clip"] = tuple(d["clip"])
        return cls(**d)


def _lognormal(xs):
    logs = np.log(np.asarray(xs, dtype=float))
    return [float(logs.mean()), float(max(logs.std(ddof=1) if len(logs) > 1 else 0.0, 0.05)), len(logs)]


def fit_from_intervals(videos, min_verb=20, min_pair=20, min_duration=0.05, source="epic"):
    """videos: {video: [(t0, t1, verb, noun)] sorted by t0}, names not ids."""
    by_verb, by_pair, allx, gaps = {}, {}, [], []
    for narr in videos.values():
        end = -math.inf
        for t0, t1, verb, noun in narr:
            d = t1 - t0
            if d >= min_duration:
                by_verb.setdefault(verb, []).append(d)
                by_pair.setdefault(f"{verb}|{noun}", []).append(d)
                allx.append(d)
            if end > -math.inf:
                gaps.append(max(t0 - end, 0.0))
            end = max(end, t1)
    gaps = np.asarray(gaps)
    pos = gaps[gaps > 0.05]
    return DurationModel(
        source=source,
        verb={v: _lognormal(x) for v, x in by_verb.items() if len(x) >= min_verb},
        pair={k: _lognormal(x) for k, x in by_pair.items() if len(x) >= min_pair},
        overall=_lognormal(allx),
        gap_zero=float(np.mean(gaps <= 0.05)) if len(gaps) else 0.0,
        gap=_lognormal(pos) if len(pos) else [0.0, 0.05, 0],
    )


def fit_duration_priors(source, root, min_verb=20, min_pair=20):
    """source: "epic" (EPIC-100 annotations dir). "cc4d_fine" and "cc4d_step" are the spec's
    other sources; the first is not public and the second is not built."""
    if source != "epic":
        raise NotImplementedError(f"duration source {source!r} is not available (docs/drift_synth.md §3)")
    from cook_ad.data.parse_epic import load_classes, load_narrations
    verb_keys, noun_keys, _ = load_classes(root)
    videos = {vid: [(t0, t1, verb_keys[v], noun_keys[n]) for t0, t1, v, n in e["narrations"]]
              for vid, e in load_narrations(root).items()}
    return fit_from_intervals(videos, min_verb, min_pair, source=source)


def narrowed(mu, sigma, k):
    """Log-normal (mu, sigma) with its spread times k and the same mean. EPIC's spread mixes many
    people, objects and kitchens; one person repeating an action varies less (k < 1)."""
    return mu + 0.5 * sigma ** 2 * (1.0 - k ** 2), k * sigma


def sample_duration(model, token, scale, rng, sigma_scale=1.0):
    mu, sigma = narrowed(*model.params(token), sigma_scale)
    lo, hi = model.clip
    return float(np.clip(math.exp(mu + sigma * rng.standard_normal()) * scale, lo * scale, hi * scale))


def sample_gap(model, scale, rng, sigma_scale=1.0):
    zero = rng.random() < model.gap_zero
    z = rng.standard_normal()   # drawn either way: a fixed number of draws per gap
    if zero:
        return 0.0
    mu, sigma = narrowed(model.gap[0], model.gap[1], sigma_scale)
    return float(min(math.exp(mu + sigma * z), model.gap_clip) * scale)
