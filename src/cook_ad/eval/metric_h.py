"""Metric H, the headline: hit = any flag in a span's window; miss = one per run with a flag
outside every window (docs/eval.md §1). A stray flag early in a run, followed by a perfect flag on
the span, is one TP and one FP here; metric G dates the run's alarm by the stray and scores one FP
and one FN.

  TP = a span whose window [start, end + delay] contains ANY flag (at most one per span)
  FN = a span with none
  FP = a run holding a flag outside every span's window (one per run, however many flags)
  TN = undefined

`delay` is a slider (ticks after a span's end); the headline uses DEFAULT_DELAY = 2. Flag-every-tick is the null: it hits every
span and misses every run that is not wholly inside a window. Alarms per 10 minutes count
contiguous flagged stretches (a stretch breaks at a gap or a run start), so flicker is charged.
Same inputs and same result keys as metric_g, so the harness and eval scripts swap one for the other.
"""
import numpy as np

from cook_ad.eval.metric_g import (STRENGTH_THRESHOLD, GridScore as _GridScoreG, _kept_ticks,
                                   best_index, correct_runs, run_starts)  # noqa: F401

DEFAULT_DELAY = 2      # the headline's delay, in ticks (docs/eval.md §1)
METRIC_H = {"onset": False, "hit": "any flag in [start, end + delay]",
            "miss": "one per run with a flag outside every window", "delay": DEFAULT_DELAY}


def _run_ids(starts, n):
    return np.searchsorted(np.asarray(starts), np.arange(n), side="right") - 1


def _stretch_onsets(f, starts):
    """(T, P) bool -> True where a contiguous flagged stretch begins (a run start breaks it)."""
    prev = np.zeros_like(f)
    prev[1:] = f[:-1]
    is_start = np.zeros(f.shape[0], dtype=bool)
    is_start[np.asarray(starts)] = True
    return f & ~(prev & ~is_start.reshape((-1,) + (1,) * (f.ndim - 1)))


def score_dense(flag_matrix, spans, correct=(), starts=None, delay=DEFAULT_DELAY):
    """Metric H counts for ONE trial at P operating points; same keys as metric_g.score_dense."""
    f = np.asarray(flag_matrix, dtype=bool)
    t_len = f.shape[0]
    covered = np.zeros(t_len, dtype=bool)
    hits = []
    for s0, s1 in spans:
        a, b = max(int(s0), 0), min(int(s1) + delay, t_len - 1)
        covered[a : b + 1] = True
        hits.append(f[a : b + 1].any(axis=0))
    hits = np.stack(hits) if hits else np.zeros((0,) + f.shape[1:], dtype=bool)
    starts = np.asarray(starts)
    outside = np.concatenate([np.zeros((1,) + f.shape[1:], dtype=np.int64),
                              np.cumsum(f & ~covered.reshape((-1,) + (1,) * (f.ndim - 1)), axis=0)])
    ends = np.r_[starts[1:], t_len]
    fp = ((outside[ends] - outside[starts]) > 0).sum(axis=0)
    runs_flagged = np.zeros(f.shape[1:], dtype=np.int64)
    for a, b in correct:
        runs_flagged += f[a : b + 1].any(axis=0)
    return {"hits": hits, "fp": fp, "onsets": _stretch_onsets(f, starts).sum(axis=0),
            "any_flag": f.any(axis=0), "runs_flagged": runs_flagged,
            "n_correct_runs": len(correct)}


class GridScore(_GridScoreG):
    """metric_g.GridScore with metric H's counting rule and a `delay`."""
    scorer = staticmethod(score_dense)

    def __init__(self, tick_seconds=1.0, delay=DEFAULT_DELAY):
        super().__init__(tick_seconds, delay)


def stretches(ticks, starts):
    """Sorted flag ticks -> the first tick of each contiguous stretch (broken at a run start)."""
    starts = set(int(s) for s in starts)
    out, prev = [], None
    for t in sorted(ticks):
        if prev is None or t != prev + 1 or t in starts:
            out.append(t)
        prev = t
    return out


def score_trial(flags, spans, starts, delay=DEFAULT_DELAY, threshold=STRENGTH_THRESHOLD):
    """One trial under metric H -> per-span rows (hit, latency, width) and the FP runs."""
    ticks = _kept_ticks(flags, threshold)
    inside = lambda t, s: s[0] <= t <= s[1] + delay  # noqa: E731
    per_span = []
    for span in spans:
        found = [t for t in ticks if inside(t, span)]
        per_span.append({"span": tuple(span), "hit": bool(found),
                         "latency": int(min(found) - span[0]) if found else None,
                         "width": int(span[1] - span[0] + 1)})
    outside = [t for t in ticks if not any(inside(t, s) for s in spans)]
    fp_runs = sorted({int(np.searchsorted(np.asarray(starts), t, side="right")) - 1 for t in outside})
    return {"per_span": per_span, "fp_merged": fp_runs, "fp_raw": outside, "extra_in_span": 0,
            "n_flags": len(ticks)}


def evaluate(healthy, trials, healthy_starts, trial_starts, error_types=None, delay=DEFAULT_DELAY,
             threshold=STRENGTH_THRESHOLD):
    """Metric H over a pool; the result has metric_g.evaluate's shape (flags_merged = stretches)."""
    import statistics
    types = list(error_types) if error_types else list(dict.fromkeys(
        t for _, spans in trials for _, t in spans))
    acc = {t: {"tp": 0, "fn": 0, "widths": [], "lat": [], "n_trials": 0} for t in types}
    tp = fn = fp = fp_raw = 0
    latencies = []
    for (flags, spans), rs in zip(trials, trial_starts):
        kinds = [t for _, t in spans]
        for t in set(kinds):
            if t in acc:
                acc[t]["n_trials"] += 1
        r = score_trial(flags, [s for s, _ in spans], rs, delay, threshold)
        for row, t in zip(r["per_span"], kinds):
            if t not in acc:
                continue
            acc[t]["widths"].append(row["width"])
            if row["hit"]:
                acc[t]["tp"] += 1
                acc[t]["lat"].append(row["latency"])
                latencies.append(row["latency"])
                tp += 1
            else:
                acc[t]["fn"] += 1
                fn += 1
        fp += len(r["fp_merged"])
        fp_raw += len(r["fp_raw"])
    med = lambda x: float(statistics.median(x)) if x else float("nan")  # noqa: E731
    per_type = {t: {"n_trials": a["n_trials"], "n_spans": a["tp"] + a["fn"],
                    "recall": a["tp"] / (a["tp"] + a["fn"]) if a["tp"] + a["fn"] else 0.0,
                    "median_span_width": med(a["widths"]), "median_latency": med(a["lat"]),
                    "tp": a["tp"], "fn": a["fn"]} for t, a in acc.items()}
    n_h = len(healthy)
    h_fp_trials = sum(1 for f in healthy if _kept_ticks(f, threshold))
    h_raw = sum(len(_kept_ticks(f, threshold)) for f in healthy)
    h_alarms = sum(len(stretches(_kept_ticks(f, threshold), rs))
                   for f, rs in zip(healthy, healthy_starts))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "pooled": {"precision": precision, "recall": recall,
                   "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                   "tp": tp, "fn": fn, "fp": fp},
        "per_type": per_type,
        "healthy": {"n": n_h, "false_positive_trials": h_fp_trials,
                    "false_positive_rate": h_fp_trials / n_h if n_h else 0.0,
                    "flags_raw": h_raw, "flags_merged": h_alarms},
        "latency": {"median": med(latencies),
                    "p90": float(np.percentile(latencies, 90)) if latencies else float("nan"),
                    "n": len(latencies)},
        "false_positives": {"merged": fp, "raw": fp_raw},
        "extra_in_span": 0, "deadline": delay, "strength_threshold": threshold, "metric": "H",
    }
