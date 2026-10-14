"""Score the HSMM and the predictive LLM arm against EPIC-Tent's REAL anomaly annotations.

Separate from run_llm_eval.py, which it imports for the pieces that are genuinely shared (the
flagger, the injected-anomaly pool, the span printer). The split is not cosmetic -- four things
about a real-anomaly corpus do not fit that script's shape:

1. **Several anomalies per trial, of mixed type.** run_llm_eval loops over
   error_injection.ERROR_TYPES with one degraded copy of each trial per type. EPIC-Tent sessions
   carry a median of 10 annotated errors spanning up to five types at once, so the unit of
   iteration is the trial, not (trial, type). Scored through span_metrics.evaluate_spans_multi
   -- see its docstring for why the per-type shape would double-count false positives.

2. **No element/step layer.** element_metrics.score_trial_steps breaks at the first hit and
   returns one verdict per trial, and relabel_with_sequence's scope="trial" default is
   documented as unsound for multi-anomaly trials (docs/anomaly.md 6). Those metrics are not
   reported here rather than reported wrongly.

3. **No healthy trial.** Under the mapped-6 taxonomy every one of the 29 sessions carries
   something; p08 is free of PROCEDURAL error but has misuse. The healthy population is
   therefore the error-free FRAGMENTS of each trial, with flags sliced out of the full trial's
   decode -- never re-decoded, because the belief state part-way through an assembly is not the
   start-of-trial prior.

4. **No counterfactual layer.** eval/counterfactual.py needs the same trial with and without
   the anomaly. A real error has no healthy twin, and pairing across trials would destroy the
   matched-null property that is the layer's whole justification (docs/eval.md 5).

**`--n-shot` defaults to 0 here, against run_llm_eval.py's 3.** Two measured reasons, both
consequences of long sessions rather than preferences:

  * `prompts.fewshot_block` opens "Here are some complete, correctly-executed trials". On this
    corpus no whole session is correctly executed -- 28 of 29 carry a procedural error -- so the
    sentence would be false. When exemplars ARE requested they are drawn from the error-free
    fragments, which the claim does fit.
  * Size. EPIC-Tent sessions run 42-192 steps against Breakfast's median 6, so three exemplars
    render as **191 lines of step list** behind a 34-line response grammar. Measured with
    gemma3:27b: it abandoned the DURATION/NEXT grammar on every single request and continued the
    step list instead ("Okay, here's the continuation of the provided trial data..."), so the arm
    produced zero flags. The 0-shot prompt is 34 lines and the grammar dominates.

Two controls are printed beside every arm, and neither is optional reading:

  * **flag-everything.** 27% of ticks in this corpus sit inside an annotated error span, an
    order of magnitude above anything else measured here, so span recall is cheap. The control
    says what a detector with no model at all scores.
  * **injected anomalies**, on the same trials through the same detector. That separates "this
    corpus is harder" from "real errors are harder than injected ones", which is the comparison
    the whole exercise is for.

    ./py run_epic_tent_eval.py --skip-llm                       # HSMM only, no server needed
    ./py run_epic_tent_eval.py --variant with-recipes           # + arm 3
"""
import argparse
import json
import os

import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from cook_ad.xla_env import disable_gpu_autotuning  # noqa: E402

disable_gpu_autotuning()

import run_llm_eval as R  # noqa: E402  -- imports jax; must follow disable_gpu_autotuning
from cook_ad.anomaly import narrate, surprise  # noqa: E402
from cook_ad.data import split as split_mod  # noqa: E402
from cook_ad.data.config import load_config  # noqa: E402
from cook_ad.data.parse_epic_tent import (  # noqa: E402
    CANONICAL_ORDER,
    ERROR_GROUPS,
    ERROR_TYPE_MAP,
    clean_fragments,
)
from cook_ad.eval import span_metrics  # noqa: E402
from cook_ad.hsmm import joint_params  # noqa: E402
from cook_ad.llm import client as llm_client  # noqa: E402
from cook_ad.llm import predict, prompts, textify  # noqa: E402
from cook_ad.synthetic import generate  # noqa: E402

DOMAIN = "assembly"
INTERLEAVED = (("read", "instruction"),)

# Reported in this order, most instances first, so the rare types cannot be mistaken for the
# headline. Every type the parser can emit appears even when a split contains none of it.
ERROR_TYPES = ("substitution", "transposition", "stall", "omission", "correction", "repetition")
assert set(ERROR_TYPES) == {t for t in ERROR_TYPE_MAP.values() if t}

# Types whose n is too small for a per-type rate to mean anything on a 9-trial split. Printed
# with their counts rather than silently pooled.
RARE = ("correction", "repetition")

# Repo type -> the EPIC-Tent label(s) it came from, for display. Results are read by people who
# think in the annotation's vocabulary, not the injector's.
SOURCE_NAME = {"transposition": "order", "omission": "omit", "correction": "correction",
               "repetition": "repeat", "stall": "search+slow", "substitution": "misuse"}


# --------------------------------------------------------------------------------------------
# ground truth from annotations
# --------------------------------------------------------------------------------------------

def group_truth(trial_truth, group):
    """One trial's truth with every type outside `group` REMOVED -> (typed spans, fragments).

    Removed means relabelled as normal, which is PREGO's treatment of the non-procedural types.
    Their input is a collapsed step sequence -- no slot for a misuse -- and their scorer
    (step_anticipation/src/models/llm_hf.py) counts every step before the first procedural mistake
    as correct: `correct = matches[:-1]; mistake = matches[-1]`. So here an out-of-group span
    drops out of the ground truth, and its ticks rejoin the healthy population: a flag on one is
    a false positive, exactly as a mismatch on one of PREGO's "correct" steps is.

    That is why the healthy fragments are recomputed per group rather than read from
    anomalies.json, whose clean_fragments exclude all six mapped types.
    """
    spans = [(tuple(r["window"]), r["error_type"]) for r in trial_truth["records"]
             if r["error_type"] in group]
    fragments = clean_fragments([span for span, _ in spans], trial_truth["n_ticks"])
    return spans, fragments


def slice_flags(flags, fragment):
    """The flags falling inside a closed inclusive tick interval.

    Ticks are NOT re-based. Nothing downstream needs them to be: the healthy report is a binary
    "did this fragment raise anything" plus a merged count, and merge_flags is
    translation-invariant. Re-basing would only invite the reader to compare a fragment's tick
    indices against the trial's.
    """
    start, end = fragment
    return [f for f in flags
            if start <= (f if isinstance(f, (int, np.integer)) else f.tick) <= end]


def attribution(flags_by_trial, truth, group, deadline=span_metrics.DEADLINE):
    """{error_type: {channel: n}} -- which channel raised the earliest flag that caught a span.

    Counted over HITS only; a span nobody found has no channel. The prediction being checked is
    that order/omit/repeat land on the transition channel and search/slow on the temporal ones.
    """
    out = {etype: {} for etype in group}
    for flags, trial in zip(flags_by_trial, truth):
        spans, _ = group_truth(trial, group)
        for (start, end), etype in spans:
            hits = [f for f in flags
                    if not isinstance(f, (int, np.integer))
                    and start <= f.tick <= end + deadline]
            if not hits:
                continue
            channel = min(hits, key=lambda f: f.tick).channel or "?"
            out[etype][channel] = out[etype].get(channel, 0) + 1
    return out


CONTEXT_HEADROOM = 256


def preflight_context(base_url, model, system_prompt, steps):
    """Ask the server what it would charge for the longest request this arm will send, and
    refuse to start if it would not fit.

    Not optional politeness. ollama truncates from the FRONT, so an over-long request silently
    drops the system block -- the response grammar, the vocabulary, and on the with-recipes arm
    the printed instruction order that is the entire point of the comparison -- and still
    returns a plausible-looking reply. The failure mode is a quiet accuracy loss on the longest
    trials and nothing else. run_tick_test.sh preflights the same way; this runner does it
    in-process because it is invoked directly.
    """
    import urllib.error
    import urllib.request

    messages = predict.predictive_messages(system_prompt, steps)[-1]
    body = json.dumps({"model": model, "messages": messages,
                       "temperature": 0.0, "max_tokens": 1}).encode()
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions", data=body,
        headers={"Authorization": "Bearer local", "Content-Type": "application/json"},
    )
    try:
        used = json.load(urllib.request.urlopen(request, timeout=900))["usage"]["prompt_tokens"]
        ps = json.load(urllib.request.urlopen(
            base_url.rstrip("/").rsplit("/v1", 1)[0] + "/api/ps", timeout=60))
    except (urllib.error.URLError, OSError, KeyError) as exc:
        print(f"  preflight skipped: {type(exc).__name__}: {exc}")
        return
    served = next((m.get("context_length", 0) for m in ps.get("models", [])
                   if m.get("name") == model), 0)
    print(f"  preflight: longest request {used} tokens over {len(steps)} steps; "
          f"served context {served}; headroom {served - used}")
    for entry in ps.get("models", []):
        if entry.get("name") == model and entry.get("size"):
            fraction = entry["size_vram"] / entry["size"]
            print(f"  GPU placement: {fraction:.1%} of {model} in VRAM")
            if fraction < 0.99:
                print("  WARNING: ollama fixes layer placement at LOAD time, so a model loaded "
                      "while JAX held the card sits mostly on CPU with no error. Unload and "
                      "reload it.")
    if served and served - used < CONTEXT_HEADROOM:
        raise SystemExit(
            f"error: served context {served} leaves {served - used} tokens of headroom for a "
            f"{used}-token request.\n"
            f"  Prompts would be truncated FROM THE FRONT, silently dropping the system block.\n"
            f"  Restart the server with a larger context, e.g.\n"
            f"    OLLAMA_CONTEXT_LENGTH={1 << (used + CONTEXT_HEADROOM - 1).bit_length()} "
            f"ollama serve\n"
            f"  and bound OLLAMA_NUM_PARALLEL alongside it -- the per-slot KV cache is "
            f"context x parallel,\n"
            f"  and overcommitting it pushes most of the model onto the CPU.\n"
            f"  Pass --skip-preflight to proceed anyway (the numbers will not be trustworthy)."
        )


def flag_everything(n_ticks):
    """The control for a tick-dense detector: no model, one flag per tick."""
    return [span_metrics.Flag(t, 1.0, "all") for t in range(n_ticks)]


def flag_every_boundary(steps):
    """The control for a BOUNDARY-SPARSE detector, which is what arm 3 is.

    Needed separately because MERGE_TOL structurally favours tick-dense detectors: a run of
    consecutive ticks collapses into one alarm, while one flag per step boundary is too far
    apart to merge. Measured on Breakfast, merge ratios were 2.1x for the HSMM, 18-24x for the
    tick-unit verdict arms and 1.3x for the predictive arm -- so scoring arm 3 against
    flag-everything alone flatters it, and its own null is this one.
    """
    return [span_metrics.Flag(step.tick_start, 1.0, "all") for step in steps]


# --------------------------------------------------------------------------------------------
# the arms
# --------------------------------------------------------------------------------------------

def hsmm_flags_per_trial(trajectories, model, chunk_size, alpha, channel_alpha):
    flags, _ = R._hsmm_flags(trajectories, model, chunk_size, alpha, channel_alpha)
    return [span_metrics.flags_from_channels(f, surprise.CHANNELS) for f in flags]


def live_flags_per_trial(trajectories, hsmm, bad_sets, lags, alpha, channel_alpha, kappa):
    """{lag: per-trial span Flags} from the causal detectors (detectors/hsmm_live.py,
    hsmm_lag.py). Each channel uses the same alpha the hindsight arm does (channel_alpha
    overrides, e.g. s_temporal=1e-5, else alpha), so the two arms differ in what they condition
    on, not in how strict they were set."""
    import math
    from cook_ad.core import OperatingPoint, Trial
    from cook_ad.detectors.hsmm_lag import LagHSMMDetector
    from cook_ad.detectors.hsmm_live import LiveHSMMDetector
    op = OperatingPoint({c: channel_alpha.get(c, alpha) for c in surprise.CHANNELS}, kappa)
    dets = {lag: LiveHSMMDetector(hsmm, bad_sets, op) if lag == 0 else
            LagHSMMDetector(hsmm, bad_sets, op, math.inf if lag == "inf" else lag) for lag in lags}
    out = {lag: [] for lag in lags}
    for traj in trajectories:
        trial = Trial("t", np.asarray(traj["verb_ids"]), np.asarray(traj["noun_ids"]))
        for lag, det in dets.items():
            flags = det.flags(det.trace(trial))
            out[lag].append(span_metrics.flags_from_channels(flags, surprise.CHANNELS))
    return out


def predictive_flags_per_trial(trajectories, det, tag):
    """Arm 3 over whole trials, through detectors.LLMPredictiveDetector. One request per step
    boundary -- ~57 per session here, against the ~2124 a tick-unit verdict arm would send on the
    longest one."""
    from cook_ad.core import Trial
    out = []
    totals = {"n_boundaries": 0, "duration_parse_failures": 0, "next_parse_failures": 0}
    for i, traj in enumerate(trajectories):
        d = det.detect_trial(Trial("t", np.asarray(traj["verb_ids"]), np.asarray(traj["noun_ids"])))
        flags = sorted(span_metrics.Flag(a.tick, a.score, a.channel) for a in d.alarms)
        for key in totals:
            totals[key] += d.trace[key]
        out.append(flags)
        print(f"  [{tag}] trial {i + 1}/{len(trajectories)}: {d.trace['n_boundaries']} steps, "
              f"{len(flags)} flags ({det.client.n_would_request} uncached requests so far)",
              flush=True)
    boundaries = max(totals["n_boundaries"], 1)
    return out, {
        "n_boundaries": totals["n_boundaries"],
        "duration_parse_failure_rate": totals["duration_parse_failures"] / boundaries,
        "next_parse_failure_rate": totals["next_parse_failures"] / boundaries,
    }


def step_false_alarms(flags, trajectory, lexicon, sil, spans):
    """-> (flagged correct steps, correct steps) over a WHOLE session: the trimmed protocol's
    false-alarm rate, on untrimmed sessions. Steps are PREGO's (step_extents: idle charged to the
    step it follows, repeats across idle merged); a step is correct if its extent overlaps no
    in-group span, and a false alarm if any flag lands in it. Steps touching a mistake are neither
    -- they are what recall scores."""
    ticks = sorted({f if isinstance(f, (int, np.integer)) else f.tick for f in flags})
    flagged = n = 0
    for a, b in step_extents(trajectory, lexicon, sil):
        if any(t0 <= b and t1 >= a for (t0, t1), _ in spans):
            continue
        n += 1
        flagged += any(a <= t <= b for t in ticks)
    return flagged, n


def score(flags_by_trial, truth, group, merge_tol, trajectories=None, lexicon=None, sil=None,
          onset=True):
    """One arm's per-trial flags, scored against `group` with every other type removed. With
    `trajectories`, also the step-level false-alarm rate (see step_false_alarms)."""
    trials, healthy, healthy_ticks, alarmed_ticks = [], [], 0, 0
    steps_flagged = steps_correct = 0
    runs_flagged = runs_correct = 0
    # Metric G merges one alarm per run of constant (verb, noun); needs the observation stream.
    use_runs = onset and trajectories is not None
    run_starts = {"trials": [], "healthy": []} if use_runs else None
    for i, (flags, trial) in enumerate(zip(flags_by_trial, truth)):
        spans, fragments = group_truth(trial, group)
        trials.append((flags, spans))
        rs = None
        if trajectories is not None:
            k, n = step_false_alarms(flags, trajectories[i], lexicon, sil, spans)
            steps_flagged += k
            steps_correct += n
            rs = span_metrics.run_starts(trajectories[i]["verb_ids"], trajectories[i]["noun_ids"])
            ends = np.r_[rs[1:] - 1, len(trajectories[i]["verb_ids"]) - 1]
            ticks = {f if isinstance(f, (int, np.integer)) else f.tick for f in flags}
            for a, b in zip(rs, ends):
                if any(t0 <= b and t1 >= a for (t0, t1), _ in spans):
                    continue
                runs_correct += 1
                runs_flagged += any(a <= t <= b for t in ticks)
        if use_runs:
            run_starts["trials"].append(rs)
        for fragment in fragments:
            inside = slice_flags(flags, fragment)
            healthy.append(inside)
            if use_runs:
                run_starts["healthy"].append(rs)
            alarmed_ticks += len({f if isinstance(f, (int, np.integer)) else f.tick
                                  for f in inside})
        healthy_ticks += sum(end - start + 1 for start, end in fragments)
    report = span_metrics.evaluate_spans_multi(
        healthy, trials, merge_tol=merge_tol, error_types=group, onset=onset,
        run_starts=run_starts,
    )
    # The per-fragment false-positive rate is binary per fragment, so it rises with fragment
    # LENGTH, and removing out-of-group types merges fragments into longer ones (48 -> 33 on the
    # procedural group). It is therefore not comparable across groups. Alarms per 10 minutes of
    # healthy activity is, and it is also the number a deployment actually lives with.
    #
    # Onset count alone is not enough either: merging collapses a detector that NEVER stops
    # alarming into one long alarm per fragment, so flag-everything scores ~3 alarms/10min --
    # the same as the HSMM. The share of healthy time spent in alarm (unique flagged ticks, so
    # two channels on one tick count once) separates "rarely alarms" from "never stops".
    report["healthy"]["ticks"] = healthy_ticks
    report["healthy"]["alarms_per_10min"] = (
        600.0 * report["healthy"]["flags_merged"] / healthy_ticks if healthy_ticks else 0.0)
    report["healthy"]["alarm_time_frac"] = alarmed_ticks / healthy_ticks if healthy_ticks else 0.0
    if trajectories is not None:
        # Same unit as the trimmed protocol's false_alarm_rate, so the two blocks can be read
        # side by side. Alarm time cannot: a 1-tick flag on a 30 s step is 1/30 there, 1/1 here.
        report["healthy"]["steps_correct"] = steps_correct
        report["healthy"]["steps_flagged"] = steps_flagged
        report["healthy"]["step_false_alarm_rate"] = (
            steps_flagged / steps_correct if steps_correct else 0.0)
        # Metric G's false-alarm companion: share of runs that overlap no in-group mistake but
        # contain any flag.
        report["healthy"]["runs_correct"] = runs_correct
        report["healthy"]["runs_flagged"] = runs_flagged
        report["healthy"]["run_false_alarm_rate"] = (
            runs_flagged / runs_correct if runs_correct else 0.0)
    return report


# ---- PREGO's trimmed step protocol (--trim) --------------------------------------------------
#
# Reproduces the scorer in PREGO's step_anticipation/src/models/llm_hf.py:
#     correct = matches[:-1]; mistake = matches[-1]
#     tn += sum(correct); fp += sum(~correct); tp += int(not mistake); fn += int(mistake)
# Each test video is trimmed "up to the last frame of the first mistake"; the collapsed action
# sequence up to there is scored step by step; the last step is THE mistake (one positive per
# video) and every earlier step is a negative. Videos with no mistake are "maintained
# unaltered" -- every step a negative.

def trim_at_first_mistake(sequences, truth, group):
    """-> [(sequence, (t0, t1) or None)]: each session cut after the last tick of its FIRST
    in-group mistake (earliest onset). Out-of-group types were already removed by group_truth, so
    an earlier misuse neither triggers the cut nor survives as a positive -- PREGO's treatment."""
    out = []
    for seq, trial in zip(sequences, truth):
        spans, _ = group_truth(trial, group)
        if not spans:
            out.append((seq, None))
            continue
        (t0, t1), _ = min(spans, key=lambda s: s[0])
        end = t1 + 1
        out.append(({"trial_id": seq["trial_id"], "verb_ids": seq["verb_ids"][:end],
                     "noun_ids": seq["noun_ids"][:end]}, (t0, t1)))
    return out


def step_extents(trajectory, lexicon, sil):
    """PREGO's steps are the COLLAPSED ACTION sequence: idle removed, then consecutive repeats
    merged -- their gt never repeats an action, so `insert stake, idle, insert stake` is ONE step.
    Each step owns the ticks from its start to the next step's start, so a flag raised during an
    idle gap is charged to the step it follows (leading idle goes to the first). -> [(a, b)]."""
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
    """Arm 3's under-run channel fires at a step's closing tick if it ran shorter than predicted.
    Trimming cuts the final step artificially, so that step never closed -- and the cut is
    exactly the mistake step, so an under-run there would be a manufactured true positive. The
    HSMM needs no equivalent: temporal.completed_segment_surprise censors the final segment."""
    steps = textify.steps_from_trajectory(trajectory, lexicon)
    last_tick = steps[-1].tick_end - 1 if steps else -1
    return [f for f in flags if not (f.channel == predict.DUR_SHORT and f.tick == last_tick)]


def score_prego(flags_by_trial, trimmed, trajectories, lexicon, sil, truth, group):
    """Step-level TP/FN/FP/TN exactly as PREGO counts them, plus per-type recall of the FIRST
    mistake. A mistake whose span covers several steps is still ONE positive, hit if any of them
    is flagged -- PREGO's sequence has exactly one mistake step per video."""
    tp = fn = fp = tn = 0
    per_type = {etype: {"tp": 0, "fn": 0} for etype in group}
    for flags, (_, window), trajectory, trial in zip(flags_by_trial, trimmed, trajectories,
                                                     truth):
        extents = step_extents(trajectory, lexicon, sil)
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
        spans, _ = group_truth(trial, group)
        first_type = min(spans, key=lambda s: s[0])[1]
        per_type[first_type]["tp" if hit else "fn"] += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    # TI-PREGO (arXiv 2411.02570, eq. 3) switched to BALANCED precision,
    #     Prec_b = TP / (TP + (N+/N-) FP)  ==  TPR / (TPR + FPR),
    # which is why its Epic-tent-O table reads 20-40% where PREGO's paper read <=10.7%: it is a
    # different metric, not a better detector. Chance (TPR == FPR) scores exactly 0.5, and
    # flagging every step scores Prec_b 0.5 / F1_b 0.667 -- above every Epic-tent-O row TI-PREGO
    # reports. Reported so our numbers can be read on their scale; never compare across the two.
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
    }


def print_prego_protocol(name, group, reports):
    labels = ", ".join(SOURCE_NAME[t] for t in group)
    first = next(iter(reports.values()))
    print(f"\n{'=' * 96}\nPREGO PROTOCOL (trimmed at first mistake, step-level): {name}  "
          f"({labels})\n{'=' * 96}")
    print(f"  {first['tp'] + first['fn']} positives (one per session with an in-group mistake), "
          f"{first['fp'] + first['tn']} negative steps before them")
    print(f"\n  {'arm':<38}{'tp':>4}{'fn':>4}{'fp':>6}{'tn':>6}{'prec':>8}{'recall':>8}"
          f"{'F1':>7}{'false-alarm/step':>18}{'prec_b':>9}{'F1_b':>7}")
    for arm, r in reports.items():
        print(f"  {arm:<38}{r['tp']:>4}{r['fn']:>4}{r['fp']:>6}{r['tn']:>6}{r['precision']:>8.3f}"
              f"{r['recall']:>8.3f}{r['f1']:>7.3f}{r['false_alarm_rate']:>18.3f}"
              f"{r['precision_b']:>9.3f}{r['f1_b']:>7.3f}")
    if name == "procedural":
        # Flag-every-step on PREGO's OWN gt sequences (the `gt` lists in their
        # step_anticipation/data/predictions/output_miniROAD_Epic-tent-O.json): 15 mistakes, 153
        # correct steps. The one row here computed on exactly their sequences -- ours reproduce
        # their gt exactly in 6/15 sessions (overlap resolution differs), so it is the reference
        # the published numbers should be read against.
        print(f"  {'null on PREGO own gt (15 pos, 153 neg)':<38}{15:>4}{0:>4}{153:>6}{0:>6}"
              f"{15 / 168:>8.3f}{1.0:>8.3f}{2 * (15 / 168) / (1 + 15 / 168):>7.3f}{1.0:>18.3f}"
              f"{0.5:>9.3f}{2 / 3:>7.3f}")
        # Classical precision, PREGO CVPR'24 Table 2 (oracle steps).
        print(f"  {'PREGO + Llama 2 (CVPR24, classical)':<38}{'':>28}{0.107:>8.3f}{0.867:>8.3f}"
              f"{0.191:>7.3f}")
        # BALANCED precision, TI-PREGO arXiv 2411.02570 Table 2 (oracle steps). Their FP and N-
        # cannot be counted in the same unit -- at recall 1.0, Prec_b 0.407 implies FP > N- --
        # and the scorer is not public, so only the definition is reproduced here, not the count.
        print(f"  {'TI-PREGO + Llama 3.1 (balanced)':<38}{'':>28}{'':>8}{1.000:>8.3f}{'':>7}"
              f"{'':>18}{0.407:>9.3f}{0.578:>7.3f}")
    header = "".join(f"{SOURCE_NAME[t][:12]:>13}" for t in group)
    counts = "".join(f"{'n=' + str(first['per_type'][t]['tp'] + first['per_type'][t]['fn']):>13}"
                     for t in group)
    print(f"\n  recall by type of the FIRST mistake{' ' * 3}{header}\n  {' ' * 38}{counts}")
    for arm, r in reports.items():
        cells = "".join(f"{_fmt(r['per_type'][t]['recall']):>13}" for t in group)
        print(f"  {arm:<38}{cells}")


def _fmt(value, spec=".3f"):
    return "n/a" if value is None or (isinstance(value, float) and np.isnan(value)) \
        else format(value, spec)


def print_group(name, group, reports, truth):
    """One table per group, every arm side by side. The healthy FPR column is not optional
    reading: at this base rate span F1 alone is easy to inflate -- see the controls."""
    n_spans = sum(len(group_truth(t, group)[0]) for t in truth)
    fragments = [f for t in truth for f in group_truth(t, group)[1]]
    labels = ", ".join(SOURCE_NAME[t] for t in group)
    print(f"\n{'=' * 96}\nGROUP: {name}  ({labels})\n{'=' * 96}")
    metric = next(iter(reports.values())).get("metric", "D")
    print(f"  metric {metric}" + {"G": " (one alarm per run of constant (verb, noun); its first flag "
                                       "must land in [start, end+10])",
                                  "G-gap": " (alarm onsets after a 5-tick gap merge; deadline 10)"}.get(
        metric, " (ANY flag finds a span; merged alarms are FPs; deadline 10)"))
    print(f"  {n_spans} spans in {len(truth)} sessions; every other error type REMOVED "
          f"(relabelled normal, as PREGO does)")
    print(f"  healthy population: {len(fragments)} error-free fragments, "
          f"{sum(b - a + 1 for a, b in fragments)} ticks")

    print(f"\n  {'arm':<38}{'prec':>7}{'recall':>8}{'F1':>7}{'alarms/10min':>14}"
          f"{'alarm time':>12}{'run FPR':>10}{'frag FPR':>10}{'lat med':>9}")
    for arm, r in reports.items():
        p, h, lat = r["pooled"], r["healthy"], r["latency"]
        step_fpr = h.get("run_false_alarm_rate")
        print(f"  {arm:<38}{p['precision']:>7.3f}{p['recall']:>8.3f}{p['f1']:>7.3f}"
              f"{h['alarms_per_10min']:>14.2f}{h['alarm_time_frac']:>11.1%}"
              f"{_fmt(step_fpr):>10}{h['false_positive_rate']:>10.3f}"
              f"{_fmt(lat['median'], '.1f'):>9}")
    print("  On error-free activity only: alarms/10min = distinct (merged) alarms; alarm time =\n"
          "  share of seconds spent alarming. Read them together -- flag-everything has few\n"
          "  alarms because it is one alarm that never stops. run FPR = share of runs of constant\n"
          "  (verb, noun) overlapping no in-group error that contain any flag. frag FPR = share\n"
          "  of healthy fragments raising anything; it grows with fragment length, so compare\n"
          "  WITHIN a group.")

    header = "".join(f"{SOURCE_NAME[t][:12]:>13}" for t in group)
    counts = "".join(f"{'n=' + str(reports[next(iter(reports))]['per_type'][t]['n_spans']):>13}"
                     for t in group)
    print(f"\n  per-type recall{' ' * 23}{header}\n  {' ' * 38}{counts}")
    for arm, r in reports.items():
        cells = "".join(f"{r['per_type'][t]['recall']:>13.3f}" for t in group)
        print(f"  {arm:<38}{cells}")
    for etype in RARE:
        n = reports[next(iter(reports))]["per_type"].get(etype, {}).get("n_spans", 0)
        if etype in group and 0 < n <= 5:
            print(f"  NOTE: {SOURCE_NAME[etype]} has n={n} spans -- read the count, not the rate.")

    for arm, r in reports.items():
        attrib = r.get("attribution")
        if not attrib or not any(attrib.values()):
            continue
        print(f"\n  earliest channel to catch each hit span -- {arm}:")
        for etype in group:
            rows = attrib.get(etype) or {}
            if rows:
                ranked = ", ".join(f"{c} {n}" for c, n in sorted(rows.items(), key=lambda kv: -kv[1]))
                print(f"    {SOURCE_NAME[etype]:>12}: {ranked}")


def print_prego_comparison(reports, split_file, split_part):
    """The procedural group beside PREGO's published numbers, with the protocol difference
    spelled out. Same error subset and same removal rule; a different denominator."""
    print(f"\n  {'-' * 92}\n  against the PREGO line (same 4 types, other types removed the same "
          f"way)\n  {'-' * 92}")
    for arm, r in reports.items():
        if arm.startswith("control/flag-everything@"):
            continue
        p = r["pooled"]
        print(f"  {arm:<38}  precision {p['precision']:.3f}  recall {p['recall']:.3f}  "
              f"F1 {p['f1']:.3f}")
    print(f"  {'PREGO + Llama 2 (published)':<38}  precision 0.107  recall 0.867  F1 0.191")
    print(f"  {'TI-PREGO + Llama 3.1 (published)':<38}  BALANCED precision 0.407  recall 1.000  "
          f"F1_b 0.578 -- a different metric; see --trim")
    print("  NOT interchangeable. PREGO scores STEPS of the collapsed action sequence, and trims\n"
          "  each test video at the end of its FIRST procedural mistake: one positive per video\n"
          "  (recall 0.867 = 13/15), every earlier step a negative. This run scores tick SPANS\n"
          "  over whole sessions, with every procedural mistake a positive.")
    if "prego" not in str(split_file):
        print(f"  Split: this is {split_part} of {split_file}. PREGO's exact split is\n"
              "  split_prego.json (--split-file dataset/processed/epic_tent/split_prego.json).")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    D = "dataset/processed/epic_tent"
    ap.add_argument("--config", default="configs/epic_tent.yaml")
    ap.add_argument("--sequences", default=f"{D}/sequences.json")
    ap.add_argument("--labels", default=f"{D}/labels.json")
    ap.add_argument("--vocab", default=f"{D}/vocab.json")
    ap.add_argument("--anomalies", default=f"{D}/anomalies.json")
    # The ERROR-EXCISED fit. Every training session contains annotated errors, so a model
    # fitted on whole sessions partly learns the errors as normal -- measured: it alarms less on
    # both counts (recall 0.197 vs 0.260, healthy FPR 0.146 vs 0.354 on all six types).
    ap.add_argument("--joint-params", default=f"{D}/joint_params_clean.npz")
    ap.add_argument("--trim", action="store_true",
                    help="PREGO's protocol instead of span scoring: trim each session at the end "
                         "of its first in-group mistake and score STEPS, one positive per "
                         "session. Detectors are re-run on the trimmed streams.")
    ap.add_argument("--groups", nargs="+", choices=list(ERROR_GROUPS),
                    default=list(ERROR_GROUPS),
                    help="error groups to score; each is scored with every other type removed")
    ap.add_argument("--split-file", default=f"{D}/split.json")
    ap.add_argument("--split-part", choices=["train", "test"], default="test")
    ap.add_argument("--out", default=f"{D}/epic_tent_report.json")
    ap.add_argument("--cache-dir", default=f"{D}/llm_cache")
    ap.add_argument("--max-real", type=int, default=29)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=surprise.DEFAULT_ALPHA)
    ap.add_argument("--channel-alpha", action="append", default=None,
                    help="CHANNEL=ALPHA, repeatable. Defaults to run_llm_eval's measured "
                         "s_temporal override -- re-measure before trusting it on this corpus.")
    ap.add_argument("--chunk-size", type=int, default=8)
    ap.add_argument("--merge-tol", type=int, default=span_metrics.MERGE_TOL)
    ap.add_argument("--metric", choices=["G", "D"], default="G",
                    help="span scoring rule for the untrimmed tables: G (default, the headline) "
                         "merges flags one alarm per run of constant (verb, noun) and lets only "
                         "the alarm's first flag find a span; D is the original any-flag rule, "
                         "kept so earlier reports reproduce. The trimmed tables are always E.")
    ap.add_argument("--skip-hsmm", action="store_true")
    ap.add_argument("--live", action="store_true",
                    help="add the causal detectors (detectors/hsmm_live.py, hsmm_lag.py) as arm(s) "
                         "hsmm-live-L{lag}, at the same alpha and per-channel alphas as hsmm-joint")
    ap.add_argument("--live-lags", default="0,5",
                    help="comma-separated lags for --live; 'inf' is the smoother")
    ap.add_argument("--live-kappa", type=float, default=0.5,
                    help="posterior cutoff for the hidden-event channels (0.5 = more likely than not)")
    ap.add_argument("--skip-llm", action="store_true")
    ap.add_argument("--skip-injected", action="store_true",
                    help="skip the injected-anomaly control -- see the module docstring for why "
                         "it is not optional reading")
    ap.add_argument("--variant", choices=[*prompts.VARIANTS, "both"], default="both")
    ap.add_argument("--clean-sequences",
                    default=f"{D}/sequences_clean.json",
                    help="error-free fragments, used as few-shot exemplars when --n-shot > 0")
    ap.add_argument("--n-shot", type=int, default=0,
                    help="few-shot exemplars. Default 0 HERE, against run_llm_eval's 3, for two "
                         "measured reasons -- see the module docstring.")
    ap.add_argument("--shot-select", choices=list(prompts.SHOT_SELECT), default="random")
    ap.add_argument("--model", default=llm_client.DEFAULT_MODEL)
    ap.add_argument("--base-url", default=llm_client.DEFAULT_BASE_URL)
    ap.add_argument("--api-key-env", default=llm_client.DEFAULT_API_KEY_ENV)
    ap.add_argument("--env-file", default=llm_client.DEFAULT_ENV_FILE)
    ap.add_argument("--rpm", type=float, default=None)
    ap.add_argument("--concurrency", type=int, default=None)
    ap.add_argument("--max-requests", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the system prompt and the longest user turn, send nothing")
    ap.add_argument("--skip-preflight", action="store_true",
                    help="do not check the served context length against the longest request. "
                         "Only sensible against a server that does not front-truncate.")
    args = ap.parse_args()

    channel_alpha = dict(R.DEFAULT_CHANNEL_ALPHA)
    if args.channel_alpha is not None:
        channel_alpha = {}
        for item in args.channel_alpha:
            if "=" not in item:
                ap.error(f"--channel-alpha expects CHANNEL=ALPHA, got {item!r}")
            channel, _, alpha = item.partition("=")
            channel_alpha[channel.strip()] = float(alpha)
        try:
            surprise.per_channel_alphas(channel_alpha)
        except ValueError as exc:
            ap.error(str(exc))

    llm_client.load_env_file(args.env_file)
    config = load_config(args.config)
    # EPIC-Tent is binned at 1s precisely so this passes -- the rendered text says "seconds".
    textify.assert_tick_seconds(config)
    d_max = config["duration"]["d_max_ticks"]
    sil_verb = config["ambient_gaps"]["sil_verb"]
    sil_noun = config["ambient_gaps"]["sil_noun"]

    vocab = json.loads(open(args.vocab).read())
    labels = json.loads(open(args.labels).read())
    sequences = json.loads(open(args.sequences).read())
    truth_by_id = {t["trial_id"]: t for t in json.loads(open(args.anomalies).read())}

    split = split_mod.load_split(args.split_file) if args.split_file else None
    if split:
        sequences = split_mod.filter_sequences(sequences, split, args.split_part)
    sequences = sequences[: args.max_real]
    truth = [truth_by_id[s["trial_id"]] for s in sequences]

    joint = joint_params.load_params(args.joint_params)
    model = {"kind": "joint", "joint": joint, "d_max": d_max}
    inject_params = joint_params.collapse_to_marginal(joint)
    # SIL tokens from the config, not narrate.Lexicon's Breakfast defaults: 'kitchen' is not
    # this corpus' sentinel, and phrase() would leak 'stall idle' into every query card.
    lexicon = narrate.Lexicon(vocab, inject_params, sil_verb=sil_verb, sil_noun=sil_noun)

    n_ticks = sum(len(sq["verb_ids"]) for sq in sequences)
    print(f"EPIC-Tent, real annotations: {len(sequences)} sessions ({args.split_part} of "
          f"{args.split_file}), {n_ticks} ticks")
    print(f"  joint params: {args.joint_params}")

    trajectories = generate.trajectories_from_real_joint(
        joint, sequences, d_max, chunk_size=args.chunk_size
    )

    live_hsmm, live_bad, live_lags = None, None, []
    if args.live:
        from cook_ad.detectors.thresholds import BadSets
        from cook_ad.hsmm import params as hsmm_params
        from cook_ad.hsmm.model import HSMM
        live_lags = [x if x == "inf" else int(x) for x in args.live_lags.split(",")]
        grid = sorted({args.alpha, *channel_alpha.values()})
        live_hsmm = HSMM(hsmm_params.JointHSMMParams(*joint[:7]), d_max)
        live_bad = BadSets(live_hsmm, grid)
        print(f"  live arm: lags={live_lags} kappa={args.live_kappa} alphas={grid}")

    # ---- every arm's flags -----------------------------------------------------------
    def collect(trajectories, trimmed=False):
        """Every arm's per-trial flags over `trajectories`. Flags do not depend on the error group
        -- only the scoring does -- except under --trim, where the streams themselves differ."""
        # Flags do not depend on the error group -- only the scoring does -- so each arm runs once
        # and is re-scored per group below.
        flags_by_arm, meta = {}, {}

        # The two nulls. flag-everything is the null for a tick-dense detector; flag-every-boundary
        # for a boundary-sparse one, which is what arm 3 is. MERGE_TOL structurally favours tick-dense
        # detectors (a run of ticks collapses to one alarm), so scoring arm 3 against flag-everything
        # alone would flatter it.
        flags_by_arm["control/flag-everything"] = [flag_everything(len(sq["verb_ids"]))
                                                   for sq in sequences]
        flags_by_arm["control/flag-every-boundary"] = [
            flag_every_boundary(textify.steps_from_trajectory(t, lexicon)) for t in trajectories
        ]

        if not args.skip_hsmm:
            flags_by_arm["hsmm-joint"] = hsmm_flags_per_trial(
                trajectories, model, args.chunk_size, args.alpha, channel_alpha)
        if args.live:
            by_lag = live_flags_per_trial(trajectories, live_hsmm, live_bad, live_lags, args.alpha,
                                          channel_alpha, args.live_kappa)
            for lag, fl in by_lag.items():
                flags_by_arm[f"hsmm-live-L{lag}"] = fl

        if not args.skip_llm:
            variants = list(prompts.VARIANTS) if args.variant == "both" else [args.variant]
            # Exemplars come from the error-free FRAGMENTS, not from whole sessions. fewshot_block
            # opens "Here are some complete, correctly-executed trials", and on this corpus no whole
            # session is correctly executed -- 28 of 29 carry a procedural error. The fragments are.
            exemplars = []
            if args.n_shot > 0:
                clean = json.loads(open(args.clean_sequences).read())
                if split:
                    members = set(split["train_trial_ids"])
                    clean = [c for c in clean if c["trial_id"].rsplit("_f", 1)[0] in members]
                exemplars = prompts.select_exemplars(
                    [{"verb_ids": t["verb_ids"], "noun_ids": t["noun_ids"]} for t in clean],
                    lexicon, args.n_shot, args.seed, method=args.shot_select, vocab=vocab,
                )
                steps = sum(len(e) for e in exemplars)
                print(f"  few-shot: {args.n_shot} error-free fragments, {steps} steps total")
                if steps > 60:
                    print("  WARNING: that is a long demonstration block relative to the 34-line "
                          "response grammar it follows. Measured on whole sessions (184 steps): "
                          "gemma3:27b abandoned the DURATION/NEXT grammar entirely and continued "
                          "the step list instead, on every request.")

            for variant in variants:
                from cook_ad.detectors.llm_predictive import LLMPredictiveDetector
                det = LLMPredictiveDetector(
                    None, vocab, lexicon, variant=variant, labels=labels, domain=DOMAIN,
                    sil_verb=sil_verb, sil_noun=sil_noun, canonical_order=CANONICAL_ORDER,
                    interleaved=INTERLEAVED)
                det.exemplars = exemplars
                system_prompt = det.build_prompt()
                arm = f"llm-predictive-{variant}"
                if args.dry_run:
                    # The LONGEST trial's LAST boundary -- that is the request that decides whether
                    # the served context length is enough.
                    steps = max((textify.steps_from_trajectory(t, lexicon) for t in trajectories),
                                key=len)
                    print(f"\n{'=' * 92}\nDRY RUN: {arm}\n{'=' * 92}\n{system_prompt}")
                    print(f"\n---- longest user turn ({len(steps)} steps) ----")
                    last_request = predict.predictive_messages(system_prompt, steps)[-1]
                    print(last_request[-1]["content"])
                    words = len(system_prompt.split()) + len(last_request[-1]["content"].split())
                    print(f"\n---- {len(steps)} requests for this trial; longest is ~{words} words "
                          f"(~{int(1.4 * words)} tokens) ----")
                    continue
                if not args.skip_preflight:
                    preflight_context(
                        args.base_url, args.model, system_prompt,
                        max((textify.steps_from_trajectory(t, lexicon) for t in trajectories),
                            key=len),
                    )
                client = llm_client.ChatClient(
                    model=args.model, base_url=args.base_url, api_key_env=args.api_key_env,
                    cache_dir=args.cache_dir, rpm=args.rpm, concurrency=args.concurrency,
                    max_requests=args.max_requests, temperature=R.PREDICTIVE_TEMPERATURE,
                    max_tokens=R.PREDICTIVE_MAX_TOKENS,
                )
                det.client = client
                flags_by_arm[arm], stats = predictive_flags_per_trial(trajectories, det, arm)
                meta[arm] = {"predictive": stats, "prompt_variant": variant,
                             "client": {"model": args.model, "base_url": args.base_url}}
            if "llm-predictive-with-recipes" in flags_by_arm:
                print("  NOTE: the with-recipes preprompt states the printed instruction order and\n"
                      "  per-step median durations taken from labels.json (all sessions, as on\n"
                      "  Breakfast). Read it against the no-recipes arm, not against the HSMM.")

        if args.dry_run:
            return flags_by_arm, meta

        if trimmed:
            for arm in [a for a in flags_by_arm if a.startswith("llm-")]:
                flags_by_arm[arm] = [drop_truncated_under_run(f, t, lexicon)
                                     for f, t in zip(flags_by_arm[arm], trajectories)]
        return flags_by_arm, meta

    sil = (sil_verb, sil_noun)
    reports = {}

    # ---- PREGO's protocol: trim per group, re-run every arm, score steps -----------------------
    if args.trim:
        for name in args.groups:
            group = ERROR_GROUPS[name]
            trimmed = trim_at_first_mistake(sequences, truth, group)
            trimmed_traj = generate.trajectories_from_real_joint(
                joint, [seq for seq, _ in trimmed], d_max, chunk_size=args.chunk_size)
            print(f"\n[{name}] metric E (PREGO trimmed, step level). Trimmed to "
                  f"{sum(len(seq['verb_ids']) for seq, _ in trimmed)} "
                  f"ticks (from {n_ticks}); re-running every arm on the trimmed streams")
            flags_by_arm, meta = collect(trimmed_traj, trimmed=True)
            if args.dry_run:
                return
            # At step level flag-everything and flag-every-boundary are the same detector (every
            # step flagged), so only the latter is kept.
            flags_by_arm.pop("control/flag-everything", None)
            by_arm = {}
            for arm, flags in flags_by_arm.items():
                by_arm[arm] = score_prego(flags, trimmed, trimmed_traj, lexicon, sil, truth, group)
                by_arm[arm].update(meta.get(arm, {}))
            print_prego_protocol(name, group, by_arm)
            for arm, report in by_arm.items():
                reports[f"trimmed/{name}/{arm}"] = report
    else:
        flags_by_arm, meta = collect(trajectories)
        if args.dry_run:
            return

    # ---- score every arm under every group ---------------------------------------------------
    for name in ([] if args.trim else args.groups):
        group = ERROR_GROUPS[name]
        by_arm = {}
        for arm, flags in flags_by_arm.items():
            report = score(flags, truth, group, args.merge_tol,
                           trajectories=trajectories, lexicon=lexicon, sil=sil,
                           onset=args.metric == "G")
            if not arm.startswith("control/"):
                report["attribution"] = attribution(flags, truth, group)
            report.update(meta.get(arm, {}))
            by_arm[arm] = report
        # The same flag-everything control with merging off. The gap between the two rows is the
        # metric's, not any detector's.
        by_arm["control/flag-everything@merge_tol=0"] = score(
            flags_by_arm["control/flag-everything"], truth, group, 0, onset=args.metric == "G")
        # (no trajectories passed: this row deliberately shows the GAP rule with merging off)
        print_group(name, group, by_arm, truth)
        if name == "procedural":
            print_prego_comparison(by_arm, args.split_file, args.split_part)
        for arm, report in by_arm.items():
            reports[f"{name}/{arm}"] = report

    # ---- the injected control -------------------------------------------------------------
    # Not group-specific: it asks whether the same detector on the same sessions finds INJECTED
    # anomalies more easily than annotated ones, which separates "this corpus is harder" from
    # "real errors are harder".
    if not args.skip_hsmm and not args.skip_injected and not args.trim:
        rng = np.random.default_rng(args.seed)
        pool = R.build_pool(trajectories, rng, inject_params)
        stream = R.empty_stream()
        stream["healthy"] = [span_metrics.flags_from_channels(f, surprise.CHANNELS)
                             for f in R._hsmm_flags([t for t, _ in pool], model,
                                                    args.chunk_size, args.alpha,
                                                    channel_alpha)[0]]
        for etype in R.error_injection.ERROR_TYPES:
            injected, _ = R._hsmm_flags([d[etype] for _, d in pool], model, args.chunk_size,
                                        args.alpha, channel_alpha)
            stream["degraded"][etype] = [span_metrics.flags_from_channels(f, surprise.CHANNELS)
                                         for f in injected]
        reports["control/hsmm-injected"] = R.score_spans(
            stream, R.spans_by_type(pool), merge_tol=args.merge_tol, onset=args.metric == "G")
        p = reports["control/hsmm-injected"]["pooled"]
        print(f"\nCONTROL: the same HSMM on INJECTED anomalies, same sessions: precision "
              f"{p['precision']:.3f}  recall {p['recall']:.3f}  F1 {p['f1']:.3f}")
        print("  Its healthy FPR is NOT a false-alarm rate: run_llm_eval's pool treats the\n"
              "  un-injected sessions as healthy, and here those carry all their real errors.")

    payload = {
        "dataset": "epic_tent",
        "split_file": args.split_file, "split_part": args.split_part,
        "n_sessions": len(sequences), "n_ticks": n_ticks,
        "joint_params": args.joint_params,
        "alpha": args.alpha, "channel_alpha": channel_alpha, "merge_tol": args.merge_tol,
        "groups": {name: list(ERROR_GROUPS[name]) for name in args.groups},
        "removal": "out-of-group types relabelled normal: dropped from ground truth, ticks "
                   "rejoin the healthy population (PREGO's treatment)",
        "metric": "E" if args.trim else args.metric,
        "protocol": ("prego-trimmed: each session cut after its first in-group mistake; "
                     "step-level TP/FN/FP/TN with one positive per session") if args.trim
                    else "spans: whole sessions, every in-group mistake a positive",
        "reports": reports,
    }
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=2, default=float)
    print(f"\nwrote {args.out}")

if __name__ == "__main__":
    main()
