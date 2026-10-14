"""Metric E: PREGO's trimmed, step-level scorer, kept only to compare with PREGO (docs/eval.md §2).

Reproduces step_anticipation/src/models/llm_hf.py:
    correct = matches[:-1]; mistake = matches[-1]
    tn += sum(correct); fp += sum(~correct); tp += int(not mistake); fn += int(mistake)
Each session is trimmed after its first in-group mistake; the collapsed action sequence up to
there is scored step by step; the last step is THE mistake (one positive per session) and every
earlier step a negative. Sessions with no mistake keep every step, all negatives.
"""
import numpy as np

from cook_ad.llm import predict, textify


def step_extents(trajectory, lexicon, sil):
    """PREGO's steps: the action sequence with idle removed and consecutive repeats merged
    (`insert stake, idle, insert stake` is ONE step). Each step owns the ticks from its start to
    the next step's start, so a flag during an idle gap is charged to the step it follows
    (leading idle goes to the first). -> [(a, b)] inclusive."""
    steps = []
    for st in textify.steps_from_trajectory(trajectory, lexicon):
        if (st.verb, st.noun) == sil:
            continue
        if steps and (steps[-1].verb, steps[-1].noun) == (st.verb, st.noun):
            continue            # same action resuming after idle: still the same step
        steps.append(st)
    n_ticks = len(trajectory["verb_ids"])
    if not steps:
        return [(0, n_ticks - 1)]
    starts = [0] + [st.tick_start for st in steps[1:]]
    ends = [st.tick_start - 1 for st in steps[1:]] + [n_ticks - 1]
    return list(zip(starts, ends))


def drop_truncated_under_run(flags, trajectory, lexicon):
    """The predictive LLM's under-run channel fires at a run's closing tick if it ran short.
    Trimming cuts the final run, so an under-run there would be a manufactured true positive."""
    steps = textify.steps_from_trajectory(trajectory, lexicon)
    last_tick = steps[-1].tick_end - 1 if steps else -1
    return [f for f in flags if not (f.channel == predict.DUR_SHORT and f.tick == last_tick)]


def score(flags_by_trial, windows, extents_by_trial, first_types, group):
    """Step-level TP/FN/FP/TN as PREGO counts them, plus recall by type of the FIRST mistake.

    windows[i]: (t0, t1) of trial i's first in-group mistake, or None; extents_by_trial[i]: its
    step_extents; first_types[i]: that mistake's type (ignored when the window is None). A mistake
    covering several steps is still ONE positive, hit if any of them is flagged."""
    tp = fn = fp = tn = 0
    per_type = {t: {"tp": 0, "fn": 0} for t in group}
    for flags, window, extents, first in zip(flags_by_trial, windows, extents_by_trial,
                                             first_types):
        ticks = sorted({f if isinstance(f, (int, np.integer)) else f.tick for f in flags})
        flagged = [any(a <= t <= b for t in ticks) for a, b in extents]
        if window is None:
            fp += sum(flagged)
            tn += len(flagged) - sum(flagged)
            continue
        t0, t1 = window
        positives = [i for i, (a, b) in enumerate(extents) if a <= t1 and b >= t0]
        negatives = [i for i, (_, b) in enumerate(extents) if b < t0]
        hit = any(flagged[i] for i in positives)
        tp += hit
        fn += not hit
        fp += sum(flagged[i] for i in negatives)
        tn += sum(not flagged[i] for i in negatives)
        per_type[first]["tp" if hit else "fn"] += 1
    return _report(tp, fn, fp, tn, per_type)


def _report(tp, fn, fp, tn, per_type):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    # TI-PREGO's BALANCED precision, TP / (TP + (N+/N-) FP): chance scores 0.5 and flagging every
    # step 0.5 / F1_b 0.667. Reported to read their numbers; never compare across the two.
    n_pos, n_neg = tp + fn, fp + tn
    precision_b = tp / (tp + (n_pos / n_neg) * fp) if n_neg and (tp or fp) else 0.0
    return {
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "precision_b": precision_b,
        "f1_b": (2 * precision_b * recall / (precision_b + recall)
                 if precision_b + recall else 0.0),
        "accuracy": (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn else 0.0,
        "false_alarm_rate": fp / (fp + tn) if fp + tn else 0.0,
        "per_type": {k: {**v, "recall": v["tp"] / (v["tp"] + v["fn"]) if v["tp"] + v["fn"]
                         else float("nan")} for k, v in per_type.items()},
        "metric": "E",
    }
