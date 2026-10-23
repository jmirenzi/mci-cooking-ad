"""Stage 5: action durations. One source per quantity (docs/drift_synth.md §2, Durations):

  step totals   CaptainCook4D: real people's time for each step of these recipes (StepTimes, a
                log-normal per step: the median of its normal recordings, one pooled spread);
  logistics     EPIC-100 narration times for take / put / wash (DurationModel, per verb);
  within a step EPIC sets only the split: each core action's share of the step total is its
                EPIC log-normal draw over the step's expected EPIC total (as written).

A CC4D step interval holds the step's own actions: fetching and putting away fit in the time
between steps (12% of a recording, 15-190 s per recipe against 10-25 s of simulated logistics),
so logistics are added on top, not subtracted. There are no gaps between actions: the model
reads idle-free streams (data/scrub.py), so a gap would be drawn only to be deleted.

DurationModel: log-normal per verb, with a (verb, noun) row where that pair has at least
`min_pair` samples, and a global row behind both; fitted from EPIC-100 start/stop times.
"""
import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np


@dataclass
class DurationModel:
    source: str
    verb: dict = field(default_factory=dict)        # verb -> [mu, sigma, n] of log seconds
    pair: dict = field(default_factory=dict)        # "verb|noun" -> [mu, sigma, n]
    overall: list = field(default_factory=list)     # [mu, sigma, n]
    clip: tuple = (0.2, 120.0)                      # seconds, duration samples

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
        keep = {f.name for f in fields(cls)}               # caches from before 2026-10-21 hold gap fits
        return cls(**{k: v for k, v in d.items() if k in keep})


def _lognormal(xs):
    logs = np.log(np.asarray(xs, dtype=float))
    return [float(logs.mean()), float(max(logs.std(ddof=1) if len(logs) > 1 else 0.0, 0.05)), len(logs)]


def fit_from_intervals(videos, min_verb=20, min_pair=20, min_duration=0.05, source="epic"):
    """videos: {video: [(t0, t1, verb, noun)] sorted by t0}, names not ids."""
    by_verb, by_pair, allx = {}, {}, []
    for narr in videos.values():
        for t0, t1, verb, noun in narr:
            d = t1 - t0
            if d >= min_duration:
                by_verb.setdefault(verb, []).append(d)
                by_pair.setdefault(f"{verb}|{noun}", []).append(d)
                allx.append(d)
    return DurationModel(
        source=source,
        verb={v: _lognormal(x) for v, x in by_verb.items() if len(x) >= min_verb},
        pair={k: _lognormal(x) for k, x in by_pair.items() if len(x) >= min_pair},
        overall=_lognormal(allx),
    )


def fit_duration_priors(source, root, min_verb=20, min_pair=20):
    """source: "epic" (EPIC-100 annotations dir). CC4D's fine-grained actions are not public; its
    step totals are StepTimes (fit_step_times)."""
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


def mean_duration(model, token):
    """E[duration] of a token's log-normal (seconds, before scaling and clipping)."""
    mu, sigma = model.params(token)
    return math.exp(mu + 0.5 * sigma ** 2)


@dataclass
class StepTimes:
    """CC4D step totals: recipe -> step -> [mu, n] of log seconds over its normal recordings,
    with one log-sd `sigma` pooled over every step (3-11 recordings a step is too few for its
    own)."""
    step: dict = field(default_factory=dict)
    sigma: float = 0.5
    n_recordings: dict = field(default_factory=dict)

    def params(self, recipe, step_id):
        row = self.step.get(recipe, {}).get(str(step_id))
        return (row[0], self.sigma) if row else None

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), sort_keys=True, indent=1) + "\n")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))


def fit_step_times(recordings):
    """recordings: {recipe: [{step: seconds}]} (normal CC4D recordings) -> StepTimes. A step's
    mu is the log of its median; sigma pools the within-step log deviations from those medians."""
    step, dev, n_rec = {}, [], {}
    for r, recs in recordings.items():
        n_rec[r] = len(recs)
        by = {}
        for durs in recs:
            for k, d in durs.items():
                if d > 0:
                    by.setdefault(str(k), []).append(math.log(d))
        for k, logs in by.items():
            mu = float(np.median(logs))
            step.setdefault(r, {})[k] = [mu, len(logs)]
            dev += [x - mu for x in logs]
    sigma = float(np.sqrt(np.mean(np.square(dev)))) if dev else 0.5
    return StepTimes(step, sigma, n_rec)


def sample_step_total(times, recipe, step_id, rng, sigma_scale=1.0):
    """One step's total time in seconds, or None if CC4D has no recording of the step."""
    p = times.params(recipe, step_id) if times is not None else None
    if p is None:
        return None
    mu, sigma = narrowed(p[0], p[1], sigma_scale)
    return math.exp(mu + sigma * rng.standard_normal())
