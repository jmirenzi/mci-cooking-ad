"""Metric G: the headline metric (docs/eval.md §1). Every table states its letter.

A RUN is a maximal stretch of ticks with one constant (verb, noun) observation -- observed by
every detector, and no claim about the user's steps. An ALARM is the first flag inside a run;
later flags in the same run belong to it. Only alarm onsets are scored, on both sides:

  TP = a span whose [start, end + DEADLINE] contains an alarm onset (at most one per span)
  FN = a span with none          FP = an onset inside no span's window       TN = undefined

Reported with two false-alarm numbers: alarms per 10 minutes of error-free activity, and run FPR
(the share of runs that overlap no mistake but contain any flag). Flagging more ticks can never
beat flagging every run, so "flag every tick" IS the "flag every run" null.

Two shapes of input:
  * dense: (T, P) bool flag matrices, P operating points at once (threshold sweeps);
  * sparse: lists of Flag(tick, strength, channel) (detectors that are not swept).
"""
import statistics
from typing import NamedTuple

import numpy as np

# How long after a span's END an alarm onset still counts as having found it. A deadline, not a
# window: it exists so structurally retrospective channels (an abandonment is only visible once
# the run closes) stay admissible.
DEADLINE = 10

# Flags carry a float strength; a flag counts when strength >= this. Every producer emits 1.0.
STRENGTH_THRESHOLD = 0.5

METRIC_G = {"onset": True, "merge": "run", "deadline": DEADLINE}


class Flag(NamedTuple):
    tick: int
    strength: float = 1.0
    channel: str = ""


def flags_from_channels(flags_by_channel, channels=None):
    """{channel: (T,) bool} -> sorted Flags, one per (tick, channel) that fired."""
    chans = list(channels) if channels is not None else list(flags_by_channel)
    out = []
    for ch in chans:
        out.extend(Flag(int(t), 1.0, ch) for t in np.flatnonzero(np.asarray(flags_by_channel[ch])))
    return sorted(out)


def flags_from_alarms(alarms):
    """core.Alarm list -> sorted Flags."""
    return sorted(Flag(int(a.tick), float(a.score) if a.score == a.score else 1.0, a.channel)
                  for a in alarms)


def run_starts(verb_ids, noun_ids):
    """First tick of every maximal run of constant (verb, noun)."""
    v, n = np.asarray(verb_ids), np.asarray(noun_ids)
    return np.flatnonzero(np.r_[True, (v[1:] != v[:-1]) | (n[1:] != n[:-1])])


def correct_runs(verb_ids, noun_ids, spans):
    """Inclusive (a, b) tick extents of the runs that overlap no span."""
    starts = run_starts(verb_ids, noun_ids)
    ends = np.r_[starts[1:] - 1, len(verb_ids) - 1]
    return [(int(a), int(b)) for a, b in zip(starts, ends)
            if not any(a <= s1 and b >= s0 for s0, s1 in spans)]


# ---- dense: one trial, P operating points --------------------------------------------------


def alarm_onsets(flag_matrix, starts):
    """(T, P) bool -> (T, P) bool, True at the first flag of each run."""
    f = np.asarray(flag_matrix, dtype=bool)
    c = np.concatenate([np.zeros((1,) + f.shape[1:], dtype=np.int64), np.cumsum(f, axis=0)])
    starts = np.asarray(starts)
    run_of = np.searchsorted(starts, np.arange(f.shape[0]), side="right") - 1
    return f & ((c[1:] - c[starts[run_of]]) == 1)


def score_dense(flag_matrix, spans, correct=(), starts=None, deadline=DEADLINE):
    """Metric G counts for ONE trial at P operating points. spans: [(s0, s1), ...] inclusive.
    -> hits (n_spans, P), fp onsets (P,), onsets (P,), any_flag (P,), runs_flagged (P,)."""
    f = np.asarray(flag_matrix, dtype=bool)
    t_len = f.shape[0]
    on = alarm_onsets(f, starts)
    covered = np.zeros(t_len, dtype=bool)
    hits = []
    for s0, s1 in spans:
        a, b = max(int(s0), 0), min(int(s1) + deadline, t_len - 1)
        covered[a : b + 1] = True
        hits.append(on[a : b + 1].any(axis=0))
    hits = np.stack(hits) if hits else np.zeros((0,) + f.shape[1:], dtype=bool)
    runs_flagged = np.zeros(f.shape[1:], dtype=np.int64)
    for a, b in correct:
        runs_flagged += f[a : b + 1].any(axis=0)
    return {"hits": hits, "fp": (on & ~covered.reshape((-1,) + (1,) * (f.ndim - 1))).sum(axis=0),
            "onsets": on.sum(axis=0), "any_flag": f.any(axis=0), "runs_flagged": runs_flagged,
            "n_correct_runs": len(correct)}


class GridScore:
    """Accumulates metric G over trials for a whole operating-point grid.

    add(group, flags (T, ...) bool, spans, verbs, nouns); `group` "healthy" for error-free
    trials (they supply alarms/10 min and healthy-trial FPR), else the error type.
    `tick_seconds` converts ticks to time for alarms/10 min (1.0 on Breakfast and EPIC-Tent;
    0.5 on EPIC-KITCHENS and the synthetic drift data)."""

    def __init__(self, tick_seconds=1.0):
        self.groups = {}
        self.tick_seconds = tick_seconds

    def add(self, group, flags, spans, verbs, nouns):
        flags = np.asarray(flags, dtype=bool)
        shape = flags.shape[1:]
        r = score_dense(flags.reshape(flags.shape[0], -1), spans,
                        correct_runs(verbs, nouns, spans), run_starts(verbs, nouns))
        st = self.groups.setdefault(group, {"hits": 0, "n_spans": 0, "fp": 0, "onsets": 0,
                                            "any_flag": 0, "runs_flagged": 0, "n_correct": 0,
                                            "ticks": 0, "n_trials": 0})
        st["hits"] = st["hits"] + r["hits"].sum(axis=0).reshape(shape)
        st["n_spans"] += len(spans)
        st["fp"] = st["fp"] + r["fp"].reshape(shape)
        st["onsets"] = st["onsets"] + r["onsets"].reshape(shape)
        st["any_flag"] = st["any_flag"] + r["any_flag"].reshape(shape)
        st["runs_flagged"] = st["runs_flagged"] + r["runs_flagged"].reshape(shape)
        st["n_correct"] += r["n_correct_runs"]
        st["ticks"] += flags.shape[0]
        st["n_trials"] += 1

    def result(self):
        """precision, recall, f1, alarms_per_10min, healthy_trial_fpr, run_fpr, per_type_recall;
        each an array over the grid."""
        g = self.groups
        injected = [st for k, st in g.items() if k != "healthy"]
        tp = sum(st["hits"] for st in injected)
        n_spans = sum(st["n_spans"] for st in injected)
        fp = sum(st["fp"] for st in injected)
        h = g["healthy"]
        runs_flagged = sum(st["runs_flagged"] for st in g.values())
        n_correct = sum(st["n_correct"] for st in g.values())
        with np.errstate(invalid="ignore", divide="ignore"):
            prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
            rec = tp / max(n_spans, 1)
            f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
        return {"precision": prec, "recall": rec, "f1": f1,
                "alarms_per_10min": 600.0 * h["onsets"] / (h["ticks"] * self.tick_seconds),
                "healthy_trial_fpr": h["any_flag"] / h["n_trials"],
                "run_fpr": runs_flagged / max(n_correct, 1),
                "per_type_recall": {k: st["hits"] / max(st["n_spans"], 1)
                                    for k, st in g.items() if k != "healthy"}}


def best_index(f1):
    """Index of the highest F1 over a grid (first in C order on ties)."""
    return np.unravel_index(int(np.argmax(f1)), np.shape(f1))


# ---- sparse: Flag lists --------------------------------------------------------------------


def _kept_ticks(flags, threshold=STRENGTH_THRESHOLD):
    """Flags (or bare ints) -> sorted ticks that clear `threshold`."""
    ticks = []
    for f in flags:
        if isinstance(f, (int, np.integer)):
            ticks.append(int(f))
        elif float(f.strength) >= threshold:
            ticks.append(int(f.tick))
    return sorted(ticks)


def alarm_ticks(ticks, starts):
    """Sorted flag ticks -> the onset tick of each alarm (first flag per run)."""
    starts = np.asarray(starts)
    onsets, last_run = [], None
    for t in sorted(ticks):
        run = int(np.searchsorted(starts, t, side="right")) - 1
        if run != last_run:
            onsets.append(t)
            last_run = run
    return onsets


def score_trial(flags, spans, starts, deadline=DEADLINE, threshold=STRENGTH_THRESHOLD):
    """One trial under metric G -> per-span rows (hit, latency, width) and the FP onsets."""
    ticks = _kept_ticks(flags, threshold)
    onsets = alarm_ticks(ticks, starts)
    inside = lambda t, s: s[0] <= t <= s[1] + deadline  # noqa: E731
    per_span, extra = [], 0
    for span in spans:
        found = [t for t in onsets if inside(t, span)]
        per_span.append({"span": tuple(span), "hit": bool(found),
                         "latency": int(min(found) - span[0]) if found else None,
                         "width": int(span[1] - span[0] + 1)})
        extra += max(len(found) - 1, 0)
    fp = [t for t in onsets if not any(inside(t, s) for s in spans)]
    fp_raw = [t for t in ticks if not any(inside(t, s) for s in spans)]
    return {"per_span": per_span, "fp_merged": fp, "fp_raw": fp_raw, "extra_in_span": extra,
            "n_flags": len(ticks)}


def evaluate(healthy, trials, healthy_starts, trial_starts, error_types=None,
             deadline=DEADLINE, threshold=STRENGTH_THRESHOLD):
    """Metric G over a pool.

    healthy: [flags of one error-free trial or fragment]; trials: [(flags, [(span, type), ...])];
    *_starts: the run starts of the stream each entry was cut from. Precision pools the
    injected/annotated trials' false positives only; the healthy population reports alarms
    separately and is never folded into F1."""
    types = list(error_types) if error_types else list(dict.fromkeys(
        t for _, spans in trials for _, t in spans))
    acc = {t: {"tp": 0, "fn": 0, "widths": [], "lat": [], "n_trials": 0} for t in types}
    tp = fn = fp = fp_raw = extra = 0
    latencies = []
    for (flags, spans), rs in zip(trials, trial_starts):
        kinds = [t for _, t in spans]
        for t in set(kinds):
            if t in acc:
                acc[t]["n_trials"] += 1
        r = score_trial(flags, [s for s, _ in spans], rs, deadline, threshold)
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
        extra += r["extra_in_span"]
    med = lambda x: float(statistics.median(x)) if x else float("nan")  # noqa: E731
    per_type = {t: {"n_trials": a["n_trials"], "n_spans": a["tp"] + a["fn"],
                    "recall": a["tp"] / (a["tp"] + a["fn"]) if a["tp"] + a["fn"] else 0.0,
                    "median_span_width": med(a["widths"]), "median_latency": med(a["lat"]),
                    "tp": a["tp"], "fn": a["fn"]} for t, a in acc.items()}
    n_h = len(healthy)
    h_fp_trials = sum(1 for f in healthy if _kept_ticks(f, threshold))
    h_raw = sum(len(_kept_ticks(f, threshold)) for f in healthy)
    h_onsets = sum(len(alarm_ticks(_kept_ticks(f, threshold), rs))
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
                    "flags_raw": h_raw, "flags_merged": h_onsets},
        "latency": {"median": med(latencies),
                    "p90": float(np.percentile(latencies, 90)) if latencies else float("nan"),
                    "n": len(latencies)},
        "false_positives": {"merged": fp, "raw": fp_raw},
        "extra_in_span": extra,
        "deadline": deadline,
        "strength_threshold": threshold,
        "metric": "G",
    }
