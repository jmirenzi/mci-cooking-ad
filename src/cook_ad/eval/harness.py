"""Run detectors over pools, sweep and pick operating points on train, print labelled tables
(docs/eval.md §3). Corpus-specific orchestration lives in scripts/eval/.
"""
import json
import math
import os
from pathlib import Path

import numpy as np

from cook_ad.core import Detection, OperatingPoint, Trial
from cook_ad.eval import metric_g, metric_h

# COOK_METRIC=H scores with metric H (eval/metric_h.py) instead of G; COOK_DELAY is its slider (G's
# deadline is fixed at 10). Read once at import: every table in a run uses one rule.
METRIC = os.environ.get("COOK_METRIC", "G").upper()
SCORER = metric_h if METRIC == "H" else metric_g
DELAY = int(os.environ.get("COOK_DELAY", 0 if METRIC == "H" else metric_g.DEADLINE))
LABEL = f"H, delay {DELAY}" if METRIC == "H" else "G"

CHANNELS = ("s_emit", "s_verb", "s_noun", "s_temporal", "s_dur_two", "s_transition",
            "s_recipe_transition")

# The sweep grids. ALPHAS is dense between 0.1 and 1e-3, where the F1 / false-alarm trade turns
# over; the live event channels are swept over KAPPAS as well.
ALPHAS = [0.5, 0.35, 0.2, 0.15, 0.1, 0.07, 0.05, 0.035, 0.02, 0.015, 0.01, 0.007, 0.005,
          0.003, 0.002, 1e-3, 5e-4, 1e-4, 1e-5, 1e-7, 1e-10]
KAPPAS = [0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02]


# ---- pools as Trials ------------------------------------------------------------------------


def trials_from_streams(streams, recipes=None):
    """{group: [trajectory]} -> [(group, Trial)] in group order; `recipes`[i] is trial i's known
    dish (the same for every group's copy of it), or None."""
    out = []
    for group, items in streams.items():
        for i, t in enumerate(items):
            spans = tuple(tuple(map(int, s)) for s in t.get("spans", [])) if group != "healthy" else ()
            out.append((group, Trial(f"{group}/{i}", np.asarray(t["verb_ids"]),
                                     np.asarray(t["noun_ids"]),
                                     None if recipes is None else int(recipes[i]), spans, group)))
    return out


def trials_from_groups(groups):
    """{group: [(trajectory, recipe or None)]} -> [(group, Trial)]: as trials_from_streams, for
    pools whose groups hold different trials (a filtered or regenerated pool)."""
    out = []
    for group, items in groups.items():
        for i, (t, r) in enumerate(items):
            spans = tuple(tuple(map(int, s)) for s in t.get("spans", [])) if group != "healthy" else ()
            out.append((group, Trial(f"{group}/{i}", np.asarray(t["verb_ids"]), np.asarray(t["noun_ids"]),
                                     None if r is None else int(r), spans, group)))
    return out


def run(detectors, trials):
    """{name: [Detection]} -- each detector over every Trial (detect_many where batched)."""
    return {name: det.detect_many([t for _, t in trials]) for name, det in detectors.items()}


def alarm_mask(detection: Detection, n_ticks):
    """(T,) bool: ticks carrying any alarm."""
    m = np.zeros(n_ticks, dtype=bool)
    m[[a.tick for a in detection.alarms if 0 <= a.tick < n_ticks]] = True
    return m


# ---- operating-point grids ------------------------------------------------------------------


def hindsight_grid(det, trace, alphas):
    """(T, A) bool: any channel flags, at each alpha."""
    cols = []
    for a in alphas:
        f = det.flags(trace, OperatingPoint(a))
        cols.append(np.any(np.stack(list(f.values())), axis=0))   # every channel the detector has
    return np.stack(cols, axis=1)


# The structural group: the transition channels and, when the detector has it, s_count.
TRANSITION_CHANNELS = ("s_transition", "s_recipe_transition", "s_count")


def hindsight_split_grid(det, trace, alphas, split=TRANSITION_CHANNELS):
    """(T, A, A) bool: the `split` channels at alphas[i] OR every other channel at alphas[j].
    For priors that make a transition rare rather than impossible, which needs a looser alpha
    on the transition channels than the emission and duration channels can bear."""
    per = [det.flags(trace, OperatingPoint(a)) for a in alphas]
    sp = np.stack([np.any(np.stack([f[c] for c in split if c in f]), axis=0) for f in per], axis=1)
    rest = np.stack([np.any(np.stack([f[c] for c in f if c not in split]), axis=0)
                     for f in per], axis=1)
    return sp[:, :, None] | rest[:, None, :]


DURATION_CHANNELS = ("s_temporal", "s_dur_two")
DUR_ALPHAS = [0.1, 0.05, 0.02, 0.01, 5e-3, 2e-3, 1e-3, 1e-4, 1e-7]


def hindsight_split3_grid(det, trace, alphas, dur_alphas=DUR_ALPHAS):
    """(T, A, A, D) bool: transition channels at alphas[i] OR emission channels at alphas[j] OR
    the duration channels at dur_alphas[k]. A truncated or stretched run reaches only moderate
    duration p-values, which the emission channels' strict alpha would never let through."""
    per = {a: det.flags(trace, OperatingPoint(a)) for a in sorted(set(alphas) | set(dur_alphas))}
    have = list(next(iter(per.values())))
    em = [c for c in have if c not in TRANSITION_CHANNELS + DURATION_CHANNELS]

    def stack(chans, grid):
        chans = [c for c in chans if c in have]
        return np.stack([np.any(np.stack([per[a][c] for c in chans]), axis=0) for a in grid], axis=1)
    tr, emi, du = stack(TRANSITION_CHANNELS, alphas), stack(em, alphas), stack(DURATION_CHANNELS, dur_alphas)
    return tr[:, :, None, None] | emi[:, None, :, None] | du[:, None, None, :]


def split_op(alphas, idx, split=TRANSITION_CHANNELS):
    """The per-channel OperatingPoint at a hindsight_split_grid index."""
    a_split, a_rest = alphas[idx[0]], alphas[idx[1]]
    return OperatingPoint({c: (a_split if c in split else a_rest) for c in CHANNELS + ("s_count",)})


def live_grid(det, trace, alphas, kappas):
    """(T, A, K) bool: any channel flags, at each (alpha, kappa); emission channels ignore kappa."""
    alphas, kap = np.asarray(alphas), np.asarray(kappas)
    em = np.zeros((len(trace.log_norm), len(alphas)), dtype=bool)
    for c in ("s_emit", "s_verb", "s_noun"):
        em |= trace.p[c][:, None] <= alphas
    out = np.broadcast_to(em[:, :, None], em.shape + (len(kap),)).copy()
    idx = [det.bad.alpha_index(a) for a in alphas]
    for c in ("s_temporal", "s_dur_two", "s_transition", "s_recipe_transition"):
        out |= trace.events[c][:, idx][:, :, None] > kap[None, None, :]
    return out


def sweep(masks_by_trial, trials, tick_seconds=1.0):
    """masks_by_trial[i]: (T, ...) bool grid for trials[i] = (group, Trial) -> metric G grid."""
    score = SCORER.GridScore(tick_seconds, DELAY)
    for m, (group, t) in zip(masks_by_trial, trials):
        score.add(group, m, list(t.spans), t.verbs, t.nouns)
    return score.result()


def pick(result, rule="f1", max_alarms_per_10min=None):
    """Index of the chosen operating point: best F1, or best recall within an alarm budget."""
    if rule == "f1":
        return metric_g.best_index(result["f1"])
    if rule == "recall_under_budget":
        ok = np.asarray(result["alarms_per_10min"]) <= max_alarms_per_10min
        rec = np.where(ok, result["recall"], -1.0)
        return np.unravel_index(int(np.argmax(rec)), np.shape(rec))
    raise ValueError(f"unknown rule {rule!r}")


def at(result, idx):
    """The scalar row of a grid result at one index."""
    get = lambda x: float(np.asarray(x)[idx])  # noqa: E731
    return {**{k: get(result[k]) for k in ("precision", "recall", "f1", "alarms_per_10min",
                                           "run_fpr", "healthy_trial_fpr")},
            "per_type": {k: get(v) for k, v in result["per_type_recall"].items()}}


def score_masks(masks, trials, tick_seconds=1.0):
    """One operating point: [(T,) bool] -> the scalar metric G row."""
    return at(sweep([np.asarray(m)[:, None] for m in masks], trials, tick_seconds), (0,))


# ---- nulls ----------------------------------------------------------------------------------


def null_masks(trials):
    """The two nulls (identical under metric G; both printed): flag every run, flag every tick."""
    return {
        "null: flag every run": [np.isin(np.arange(len(t.verbs)),
                                         metric_g.run_starts(t.verbs, t.nouns)) for _, t in trials],
        "null: flag every tick": [np.ones(len(t.verbs), dtype=bool) for _, t in trials],
    }


# ---- output ---------------------------------------------------------------------------------


def table(rows, title, metric=None, types=None):
    """Print one metric-labelled markdown table: precision, recall, F1, alarms/10 min, run FPR,
    healthy-trial FPR, then recall per type."""
    metric = metric or LABEL
    print(f"\n### {title} -- metric {metric}\n")
    print("| arm | precision | recall | F1 | alarms / 10 min | run FPR | healthy-trial FPR |")
    print("|---|---|---|---|---|---|---|")
    for name, r in rows.items():
        print(f"| {name} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | "
              f"{r['alarms_per_10min']:.2f} | {r['run_fpr']:.3f} | {r['healthy_trial_fpr']:.3f} |")
    types = types or list(next(iter(rows.values()))["per_type"])
    print(f"\nRecall per type (metric {metric}):\n")
    print("| arm | " + " | ".join(types) + " |\n|---|" + "---|" * len(types))
    for name, r in rows.items():
        print(f"| {name} | " + " | ".join(f"{r['per_type'].get(t, 0):.2f}" for t in types) + " |")


def write_operating_points(path, entries):
    """Merge {corpus: {model: {detector: {alpha, kappa}}}} into the YAML at `path`."""
    import yaml
    p = Path(path)
    data = yaml.safe_load(p.read_text()) if p.exists() else {}
    data = data or {}
    for corpus, models in entries.items():
        for model, dets in models.items():
            data.setdefault(corpus, {}).setdefault(model, {}).update(dets)
    header = ("# Operating points picked on each corpus's TRAIN split (metric G, best F1) by\n"
              "# scripts/eval/eval_*.py. Never tuned on test. alpha is the bad-set tail mass,\n"
              "# kappa the posterior cutoff of the live/lag event channels.\n")
    p.write_text(header + yaml.safe_dump(data, sort_keys=False))


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, float) and math.isinf(x):
        return "inf"
    return x


def dump(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(jsonable(obj), fh, indent=1)
