"""Span-existence scoring: the THIRD metric layer, beside the tick layer (eval/metrics.py) and
the element layer (eval/element_metrics.py). Neither of those changes; this sits alongside them,
so every published number in docs/eval.md and docs/llm.md still describes the detector it always
did.

Why a third layer. Per-tick recall measures flag DENSITY, not detection ability: a 12-tick
substitution flagged once at its onset scores 1/12, while a detector that flags all 12 ticks
scores 12/12 and is not twelve times better. Span widths also differ by roughly an order of
magnitude across the five injectors -- substitution's altered segment is ~11 ticks, abandonment's
onset is one -- so any tick-SUMMING metric penalises the narrow types structurally, before the
detector has done anything.

Here, ground truth is defined on ticks and matching is by EXISTENCE. A span is either found or it
is not. The unit a detector happens to emit -- a tick, an RLE step, a decoded subtask -- never
enters the recall denominator, so an oscillating observation stream cannot change recall.

Two constraints this layer is written to honour:
  * scoring is NEVER on decoded subtask assignments. The LLM arm has no access to the latent
    state, so a z_star grid would be defined by one detector's own decoding and the cascade's
    residual segmentation error would silently become the other arm's scoring error.
  * NO point adjustment, in any form.
"""
import statistics
from typing import NamedTuple

import numpy as np

# Flags separated by no more than this collapse to a single alarm. Rationale, and the reason it is
# 5 rather than tuned: an oscillating region would otherwise be charged four false positives while
# a clean region is charged one. Same reasoning as NAB penalising clustered false positives less
# than dispersed ones. tools_run_lengths.py is the precondition check -- if Breakfast's RLE runs
# show no spike at 1-2 ticks, oscillation is an EPIC-Kitchens problem rather than a current one
# and this default needs no tuning.
MERGE_TOL = 5

# How long after a span's END a flag still counts as having found it. A DEADLINE, not a window:
# the span already covers every tick the injection touched, so this only governs lateness. It
# exists so structurally retrospective channels stay admissible -- an abandonment is only visible
# once the step closes. 10 ticks, matching element_metrics.DEFAULT_TICK_TOL.
DEADLINE = 10

# Flags carry a float strength so this layer is threshold-sweepable. Today every producer emits
# 1.0 for "fired" and the reducers cast a boolean, so 0.5 is a formality; once the predictive arm
# samples n=K in one request, the symbol score becomes 1 - P_hat(observed) and the duration score
# an empirical tail probability, and sweeping this removes the current asymmetry where only the
# HSMM has a threshold sweep.
STRENGTH_THRESHOLD = 0.5


class Flag(NamedTuple):
    tick: int
    strength: float = 1.0
    channel: str = ""


# ---- producing flags from each arm ----------------------------------------------------------

def flags_from_channels(flags_by_channel, channels=None):
    """anomaly.surprise.flag* output -- {channel: (T,) bool} -- as Flags at the tick itself.

    One Flag per (tick, channel) that fired, so the channel survives into the diagnostics; the
    scoring below only ever reads `.tick` and `.strength`, and duplicate ticks from two channels
    firing together are harmless (a span is satisfied at most once either way, and merging
    collapses them).
    """
    chans = list(channels) if channels is not None else list(flags_by_channel)
    out = []
    for ch in chans:
        mask = np.asarray(flags_by_channel[ch])
        out.extend(Flag(int(t), 1.0, ch) for t in np.flatnonzero(mask))
    return sorted(out)


def flags_from_verdicts(verdicts, elements):
    """element_metrics.StepVerdict list -> Flags at the FIRST TICK of the element judged.

    This is the bridge that makes a step-unit verdict arm comparable to a tick-unit one: the LLM
    answers about element i, and the moment it is answering about is elements[i].tick_start. At
    unit='tick' an element's index IS its tick, so this is the identity.
    """
    by_index = {e.index: e for e in elements}
    return sorted(
        Flag(int(by_index[v.step_index].tick_start), 1.0, v.error_type or "")
        for v in verdicts if v.is_anomaly and v.step_index in by_index
    )


def _kept_ticks(flags, threshold=STRENGTH_THRESHOLD):
    """Any flag shape -> the sorted tick indices that clear `threshold`. Accepts bare ints so a
    caller holding a plain tick list does not have to wrap it."""
    ticks = []
    for f in flags:
        if isinstance(f, (int, np.integer)):
            ticks.append(int(f))
        elif float(f.strength) >= threshold:
            ticks.append(int(f.tick))
    return sorted(ticks)


# ---- metric G: the headline definition (chosen 2026-10-14) ----------------------------------
#
# REVISED 2026-10-15 -- alarms are merged per RUN, not by a 5-tick gap. A run is a maximal stretch
# of ticks with one constant (verb, noun) observation: directly observed, available to every
# detector live, and no claim about the user's steps. An alarm is the first flag inside a run;
# later flags in the same run belong to it; a new run is a new alarm. Under the gap rule an alarm
# could last forever and count once (flag-every-tick: one alarm per trial, "4.8 per 10 min" yet
# alarming 100% of the time). Under runs, flagging more ticks can never beat flagging every run --
# flag-every-tick IS the flag-every-run null -- and there is no free parameter. The 5-tick-gap
# version is kept as "G-gap" (run_starts=None) so 2026-10-14 numbers reproduce.
#
# D (this module's original rule) lets ANY flag inside [start, end + DEADLINE] find a span, while
# false positives are counted on merged alarms. That asymmetry is exploitable: an alarm that never
# stops is ONE false positive yet finds every span -- flag-every-tick scores precision 0.85 on
# EPIC-Tent. G scores ALARM ONSETS on both sides: flags merge into alarms exactly as before, and
# only an alarm's first tick can find a span or be charged as a false positive. That is also what
# a user experiences -- the assistant speaks when an alarm starts, not on every tick it stays on.
#
#   TP = a span whose [start, end + DEADLINE] contains an alarm onset (at most one per span)
#   FN = a span with none          FP = an onset inside no span's window       TN = undefined
#
# Always reported with two false-alarm numbers: alarms per 10 minutes of error-free activity, and
# run FPR (share of runs that overlap no mistake but contain ANY flag). Every table states its
# metric letter.
METRIC_G = {"onset": True, "merge": "run", "deadline": DEADLINE}


def run_starts(verb_ids, noun_ids):
    """First tick of every maximal run of constant (verb, noun) -- the unit metric G merges by."""
    v, n = np.asarray(verb_ids), np.asarray(noun_ids)
    return np.flatnonzero(np.r_[True, (v[1:] != v[:-1]) | (n[1:] != n[:-1])])


def alarm_onsets(flag_matrix, merge_tol=MERGE_TOL, run_starts=None):
    """flag_matrix: (T, P) bool, one column per operating point. -> (T, P) bool, True where an
    alarm STARTS. With run_starts (metric G): the first flag of each run. Without (G-gap): a flag
    with no flag in the merge_tol ticks before it. Identical to merge_flags, vectorised."""
    f = np.asarray(flag_matrix, dtype=bool)
    if run_starts is not None:
        c = np.concatenate([np.zeros((1,) + f.shape[1:], dtype=np.int64), np.cumsum(f, axis=0)])
        starts = np.asarray(run_starts)
        run_of = np.searchsorted(starts, np.arange(f.shape[0]), side="right") - 1
        before = c[starts[run_of]]                       # flags in this run before tick t
        return f & ((c[1:] - before) == 1)
    c = np.concatenate([np.zeros((1,) + f.shape[1:], dtype=np.int64), np.cumsum(f, axis=0)])
    t = np.arange(f.shape[0])
    lo = np.maximum(t - merge_tol, 0)
    prior = (c[t] - c[lo]) > 0
    return f & ~prior.reshape(f.shape)


def score_onsets_dense(flag_matrix, spans, correct_steps=(), deadline=DEADLINE,
                       merge_tol=MERGE_TOL, run_starts=None):
    """Metric G for ONE trial at P operating points at once (sweeps). flag_matrix (T, P) bool;
    spans [((s0, s1), etype), ...]; correct_steps [(a, b), ...] inclusive tick extents of steps
    that touch no span. Returns per-column counts: hits per span (n_spans, P), fp onsets (P,),
    onsets (P,), any_flag (P,), steps_flagged (P,)."""
    f = np.asarray(flag_matrix, dtype=bool)
    t_len = f.shape[0]
    on = alarm_onsets(f, merge_tol, run_starts)
    covered = np.zeros(t_len, dtype=bool)
    hits = []
    for (s0, s1), _ in spans:
        a, b = max(int(s0), 0), min(int(s1) + deadline, t_len - 1)
        covered[a : b + 1] = True
        hits.append(on[a : b + 1].any(axis=0))
    hits = np.stack(hits) if hits else np.zeros((0,) + f.shape[1:], dtype=bool)
    steps_flagged = np.zeros(f.shape[1:], dtype=np.int64)
    for a, b in correct_steps:
        steps_flagged += f[a : b + 1].any(axis=0)
    return {
        "hits": hits,
        "fp": (on & ~covered.reshape((-1,) + (1,) * (f.ndim - 1))).sum(axis=0),
        "onsets": on.sum(axis=0),
        "any_flag": f.any(axis=0),
        "steps_flagged": steps_flagged,
        "n_correct_steps": len(correct_steps),
    }


# ---- merging --------------------------------------------------------------------------------

def merge_flags(ticks, merge_tol=MERGE_TOL, run_starts=None):
    """Runs of flags separated by <= merge_tol collapse to one alarm, represented by its EARLIEST
    member -- earliest rather than median, so latency is preserved.

    Merging applies to FALSE-POSITIVE counting only. It does not apply to the healthy-trial
    false-positive rate, which is binary per trial and therefore already immune to clustering.
    """
    if run_starts is not None:
        # Metric G: one alarm per RUN of constant (verb, noun) -- see METRIC_G.
        starts = np.asarray(run_starts)
        onsets, last_run = [], None
        for t in sorted(ticks):
            run = int(np.searchsorted(starts, t, side="right")) - 1
            if run != last_run:
                onsets.append(t)
                last_run = run
        return onsets
    merged = []
    for t in sorted(ticks):
        if merged and t - merged[-1][1] <= merge_tol:
            merged[-1][1] = t
        else:
            merged.append([t, t])
    return [a for a, _ in merged]


# ---- scoring one trial ----------------------------------------------------------------------

def _satisfies(tick, span, deadline):
    return span[0] <= tick <= span[1] + deadline


def score_trial(flags, spans, deadline=DEADLINE, merge_tol=MERGE_TOL,
                threshold=STRENGTH_THRESHOLD, onset=False, run_starts=None):
    """One trial's spans against one detector's flags.

    Counting, exactly as specified:
      * true positive  -- a span with at least one satisfying flag. AT MOST ONE per span.
      * false negative -- a span with no satisfying flag.
      * false positive -- a MERGED flag that satisfies no span. Counted individually.
      * true negative  -- not computed and not reported. Undefined under existence matching.

    Extra flags inside a satisfied span are neither additional true positives nor false positives.
    They are returned as `extra_in_span`, a diagnostic, and never scored.

    True positives are counted on RAW flags and false positives on MERGED ones. That asymmetry is
    the point of merging: a detector that oscillates through one anomalous region should not be
    credited twice for it, and one that oscillates through a clean region should not be charged
    four times for a single sustained mistake.
    """
    ticks = _kept_ticks(flags, threshold)
    # Metric G (onset=True): only an alarm's FIRST tick can find a span -- see METRIC_G.
    candidates = merge_flags(ticks, merge_tol, run_starts) if onset else ticks
    per_span = []
    extra = 0
    for span in spans:
        satisfying = [t for t in candidates if _satisfies(t, span, deadline)]
        if satisfying:
            per_span.append({"span": tuple(span), "hit": True,
                             "latency": int(min(satisfying) - span[0]),
                             "width": int(span[1] - span[0] + 1)})
            extra += len(satisfying) - 1
        else:
            per_span.append({"span": tuple(span), "hit": False, "latency": None,
                             "width": int(span[1] - span[0] + 1)})

    merged = merge_flags(ticks, merge_tol, run_starts)
    fp_merged = [t for t in merged if not any(_satisfies(t, s, deadline) for s in spans)]
    fp_raw = [t for t in ticks if not any(_satisfies(t, s, deadline) for s in spans)]
    return {
        "per_span": per_span,
        "fp_merged": fp_merged,
        "fp_raw": fp_raw,
        "extra_in_span": extra,
        "n_flags": len(ticks),
    }


# ---- pooling over a sweep -------------------------------------------------------------------

def evaluate_spans(healthy_flags, degraded_by_type, deadline=DEADLINE, merge_tol=MERGE_TOL,
                   threshold=STRENGTH_THRESHOLD, error_types=None, onset=False, run_starts=None):
    """healthy_flags: [flags_for_one_control_trial, ...].
    degraded_by_type: {error_type: [(flags, spans), ...]}.

    Precision pools only the DEGRADED trials' false positives. The healthy-trial false-alarm rate
    is reported separately and is never folded into F1 -- it is a different claim (alarm volume in
    deployment) measured on a different denominator, and no span-level metric can recover it.
    Pooling it in would also put the same constant into every error type's denominator, the same
    self-deception element_metrics.precision_excl_healthy exists to avoid.
    """
    types = list(error_types or degraded_by_type.keys())

    tp = fn = 0
    fp_merged = fp_raw = 0
    extra_in_span = 0
    latencies = []
    per_type = {}

    for etype in types:
        trials = degraded_by_type.get(etype, [])
        et_tp = et_fn = 0
        widths, et_lat = [], []
        et_runs = (run_starts or {}).get(etype) or [None] * len(trials)
        for (flags, spans), rs in zip(trials, et_runs):
            r = score_trial(flags, spans, deadline, merge_tol, threshold, onset=onset, run_starts=rs)
            for row in r["per_span"]:
                widths.append(row["width"])
                if row["hit"]:
                    et_tp += 1
                    et_lat.append(row["latency"])
                else:
                    et_fn += 1
            fp_merged += len(r["fp_merged"])
            fp_raw += len(r["fp_raw"])
            extra_in_span += r["extra_in_span"]
        tp += et_tp
        fn += et_fn
        latencies.extend(et_lat)
        per_type[etype] = {
            "n_trials": len(trials),
            "n_spans": et_tp + et_fn,
            "recall": et_tp / (et_tp + et_fn) if (et_tp + et_fn) else 0.0,
            # The width column is what lets a reader see the asymmetry this metric controls for:
            # abandonment's span is one tick and substitution's is a whole segment, and under
            # existence matching they are worth the same.
            "median_span_width": float(statistics.median(widths)) if widths else float("nan"),
            "median_latency": float(statistics.median(et_lat)) if et_lat else float("nan"),
            "tp": et_tp, "fn": et_fn,
        }

    # Deployability. Binary per trial and computed on UNMERGED flags, so clustering cannot change
    # it. Never folded into F1.
    healthy_fp_trials = sum(1 for f in healthy_flags if _kept_ticks(f, threshold))
    n_healthy = len(healthy_flags)
    healthy_raw = sum(len(_kept_ticks(f, threshold)) for f in healthy_flags)
    h_runs = (run_starts or {}).get("healthy") or [None] * len(healthy_flags)
    healthy_merged = sum(len(merge_flags(_kept_ticks(f, threshold), merge_tol, rs))
                         for f, rs in zip(healthy_flags, h_runs))

    precision = tp / (tp + fp_merged) if (tp + fp_merged) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "pooled": {
            "precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fn": fn, "fp": fp_merged,
        },
        "per_type": per_type,
        "healthy": {
            "n": n_healthy,
            "false_positive_trials": healthy_fp_trials,
            "false_positive_rate": healthy_fp_trials / n_healthy if n_healthy else 0.0,
            "flags_raw": healthy_raw, "flags_merged": healthy_merged,
        },
        # The one place tick resolution genuinely matters, and where the predictive arm's duration
        # channel earns its place. Over hits only -- a latency for a span nobody found is not a
        # number.
        "latency": {
            "median": float(statistics.median(latencies)) if latencies else float("nan"),
            "p90": float(np.percentile(latencies, 90)) if latencies else float("nan"),
            "n": len(latencies),
        },
        # Noise diagnostic. The gap between the two measures how much of the false-alarm load is
        # observation oscillation rather than detector behaviour.
        "false_positives": {"merged": fp_merged, "raw": fp_raw},
        "extra_in_span": extra_in_span,
        "merge_tol": merge_tol,
        "deadline": deadline,
        "strength_threshold": threshold,
        "metric": ("G" if run_starts is not None else "G-gap") if onset else "D",
    }


def evaluate_spans_multi(healthy_flags, trials, deadline=DEADLINE, merge_tol=MERGE_TOL,
                         threshold=STRENGTH_THRESHOLD, error_types=None, onset=False,
                         run_starts=None):
    """The same report as evaluate_spans, for trials carrying SEVERAL anomalies of MIXED type.

    healthy_flags: [flags_for_one_control_trial, ...].
    trials: [(flags, [(span, error_type), ...]), ...] -- one entry per trial, not per type.

    A sibling rather than a generalisation, so every number ever measured through
    evaluate_spans keeps the code path it was measured on.

    The reason it cannot be expressed as evaluate_spans({etype: [(flags, spans_of_that_type)]}):
    that would pass one trial's flag list in once per type, so its false positives would be
    counted once per type, and a flag that correctly hit a `transposition` span would ALSO be
    charged as a false positive in the `substitution` entry. Pooled precision needs every span
    of the trial visible at once. Recall does not, which is why evaluate_spans gets away with
    its shape on an injected corpus -- there, each trial carries exactly one anomaly.

    Per-type recall is still exact here: score_trial returns per_span rows in input order, so
    each row is attributed to the type of the span that produced it.
    """
    tp = fn = 0
    fp_merged = fp_raw = 0
    extra_in_span = 0
    latencies = []

    types = list(error_types) if error_types else []
    if not types:
        seen = {}
        for _, spans in trials:
            for _, etype in spans:
                seen[etype] = True
        types = list(seen)
    acc = {etype: {"tp": 0, "fn": 0, "widths": [], "lat": [], "n_trials": 0} for etype in types}

    t_runs = (run_starts or {}).get("trials") or [None] * len(trials)
    for (flags, spans), rs in zip(trials, t_runs):
        ordered = [span for span, _ in spans]
        etypes = [etype for _, etype in spans]
        for etype in set(etypes):
            if etype in acc:
                acc[etype]["n_trials"] += 1
        result = score_trial(flags, ordered, deadline, merge_tol, threshold, onset=onset,
                             run_starts=rs)
        for row, etype in zip(result["per_span"], etypes):
            bucket = acc.get(etype)
            if bucket is None:
                continue
            bucket["widths"].append(row["width"])
            if row["hit"]:
                bucket["tp"] += 1
                bucket["lat"].append(row["latency"])
                latencies.append(row["latency"])
                tp += 1
            else:
                bucket["fn"] += 1
                fn += 1
        fp_merged += len(result["fp_merged"])
        fp_raw += len(result["fp_raw"])
        extra_in_span += result["extra_in_span"]

    per_type = {
        etype: {
            "n_trials": bucket["n_trials"],
            "n_spans": bucket["tp"] + bucket["fn"],
            "recall": (bucket["tp"] / (bucket["tp"] + bucket["fn"])
                       if (bucket["tp"] + bucket["fn"]) else 0.0),
            "median_span_width": (float(statistics.median(bucket["widths"]))
                                  if bucket["widths"] else float("nan")),
            "median_latency": (float(statistics.median(bucket["lat"]))
                               if bucket["lat"] else float("nan")),
            "tp": bucket["tp"], "fn": bucket["fn"],
        }
        for etype, bucket in acc.items()
    }

    healthy_fp_trials = sum(1 for f in healthy_flags if _kept_ticks(f, threshold))
    n_healthy = len(healthy_flags)
    healthy_raw = sum(len(_kept_ticks(f, threshold)) for f in healthy_flags)
    h_runs = (run_starts or {}).get("healthy") or [None] * len(healthy_flags)
    healthy_merged = sum(len(merge_flags(_kept_ticks(f, threshold), merge_tol, rs))
                         for f, rs in zip(healthy_flags, h_runs))

    precision = tp / (tp + fp_merged) if (tp + fp_merged) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "pooled": {"precision": precision, "recall": recall, "f1": f1,
                   "tp": tp, "fn": fn, "fp": fp_merged},
        "per_type": per_type,
        "healthy": {"n": n_healthy, "false_positive_trials": healthy_fp_trials,
                    "false_positive_rate": healthy_fp_trials / n_healthy if n_healthy else 0.0,
                    "flags_raw": healthy_raw, "flags_merged": healthy_merged},
        "latency": {"median": float(statistics.median(latencies)) if latencies else float("nan"),
                    "p90": float(np.percentile(latencies, 90)) if latencies else float("nan"),
                    "n": len(latencies)},
        "false_positives": {"merged": fp_merged, "raw": fp_raw},
        "extra_in_span": extra_in_span,
        "merge_tol": merge_tol,
        "deadline": deadline,
        "strength_threshold": threshold,
        "metric": ("G" if run_starts is not None else "G-gap") if onset else "D",
    }
