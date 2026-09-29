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


# ---- merging --------------------------------------------------------------------------------

def merge_flags(ticks, merge_tol=MERGE_TOL):
    """Runs of flags separated by <= merge_tol collapse to one alarm, represented by its EARLIEST
    member -- earliest rather than median, so latency is preserved.

    Merging applies to FALSE-POSITIVE counting only. It does not apply to the healthy-trial
    false-positive rate, which is binary per trial and therefore already immune to clustering.
    """
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
                threshold=STRENGTH_THRESHOLD):
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
    per_span = []
    extra = 0
    for span in spans:
        satisfying = [t for t in ticks if _satisfies(t, span, deadline)]
        if satisfying:
            per_span.append({"span": tuple(span), "hit": True,
                             "latency": int(min(satisfying) - span[0]),
                             "width": int(span[1] - span[0] + 1)})
            extra += len(satisfying) - 1
        else:
            per_span.append({"span": tuple(span), "hit": False, "latency": None,
                             "width": int(span[1] - span[0] + 1)})

    merged = merge_flags(ticks, merge_tol)
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
                   threshold=STRENGTH_THRESHOLD, error_types=None):
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
        for flags, spans in trials:
            r = score_trial(flags, spans, deadline, merge_tol, threshold)
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
    healthy_merged = sum(len(merge_flags(_kept_ticks(f, threshold), merge_tol))
                         for f in healthy_flags)

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
    }


def evaluate_spans_multi(healthy_flags, trials, deadline=DEADLINE, merge_tol=MERGE_TOL,
                         threshold=STRENGTH_THRESHOLD, error_types=None):
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

    for flags, spans in trials:
        ordered = [span for span, _ in spans]
        etypes = [etype for _, etype in spans]
        for etype in set(etypes):
            if etype in acc:
                acc[etype]["n_trials"] += 1
        result = score_trial(flags, ordered, deadline, merge_tol, threshold)
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
    healthy_merged = sum(len(merge_flags(_kept_ticks(f, threshold), merge_tol))
                         for f in healthy_flags)

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
    }
