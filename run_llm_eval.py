"""LLM-as-anomaly-detector baseline, scored against the HSMM on a common unit.

Every trial is rendered as a list of steps -- 'pour cereals for 19 seconds' -- and both detectors
answer per step (llm/textify.py, eval/element_metrics.py). Critically, the healthy trajectories and
the five injected errors are built ONCE per source here and handed to both arms, so the LLM and
the HSMM score byte-identical degraded trials. That shared pool is the whole basis of the
comparison; regenerating them per arm would silently compare two different datasets.

Two preprompt variants (llm/prompts.py). The with-recipes variant is built from labels.json, which
this repo otherwise never feeds to anything, so that arm sees ground-truth task structure the HSMM
never had -- an asymmetry this script prints next to the numbers rather than burying.

    # cost a sweep without calling out (do this first)
    python run_llm_eval.py --config configs/breakfast.yaml --dry-run

    # cheap smoke test
    python run_llm_eval.py --config configs/breakfast.yaml --protocol batch --n 2 --max-real 2

    # the real thing
    python run_llm_eval.py --config configs/breakfast.yaml --variant both --n 20 --max-real 20
"""
import argparse
import json
import os
from pathlib import Path

# JAX preallocates ~75% of the GPU on first use -- ~37 GB of a 48 GB card, measured. Harmless when
# JAX is the only tenant, but this script can share the GPU with a local inference server
# (--base-url http://localhost:...), and a model that has not loaded yet will then fail or fall
# back silently to CPU. Allocating on demand instead lets the HSMM arm (a few GB for inference)
# and a 27B model (~17 GB) coexist in 48 GB. Set the variable yourself to override.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import numpy as np

from cook_ad.anomaly import narrate, surprise
from cook_ad.data.config import load_config
from cook_ad.data import split as split_mod
from cook_ad.eval import batch, element_metrics, plotting, span_metrics
from cook_ad.hsmm import joint_params, params
from cook_ad.llm import client as llm_client
from cook_ad.llm import detect, predict, prompts, textify
from cook_ad.recipe import recipe_hmm
from cook_ad.synthetic import error_injection, generate

jax.config.update("jax_enable_x64", True)

# The joint model is the default for the HSMM arm, and this is the checkpoint. The choice is NOT
# made on training likelihood: a random recipe partition has been measured to reach a better
# training objective than the ground-truth one by 175 nats on this model, because ~65k per-recipe
# transition parameters sit against ~10^4 observed segment transitions. Nor on recipe ARI, which
# has been measured not to move detection at all (tripling it, 0.31 -> 0.86, moved mean recall
# 0.823 -> 0.821). Nor on segmentation quality, which is indistinguishable across every available
# checkpoint (boundary-F1 0.987-0.990 against the ground-truth action labels, injections landing
# on a step boundary 95% of the time, for all six).
#
# It is chosen on measured detection through this file's own step layer (40 real trials, 5
# injections each). The only axis that separates the available checkpoints is
# EFFECTIVE K_recipe -- how many of the 16 nominal recipe components the fit actually uses -- and
# more of them is measurably WORSE:
#
#     checkpoint        eff K_R   healthy FPR   step prec   step recall   mean prec
#     joint_params            5         0.050       0.951         0.802       0.947
#     joint_kmeans10          9         0.075       0.936         0.792       0.922
#     joint_noun10           11         0.075       0.938         0.762       0.915
#     joint_noun16           14         0.100       0.893         0.759       0.901
#
# Recall is flat (0.880-0.910, no trend); precision and healthy false-positive rate degrade
# monotonically as effective K_R rises. The mechanism is the same over-parameterisation that makes
# training likelihood useless above: splitting the corpus across more live components thins the
# data behind each per-recipe transition and duration table, so the fitted rows are noisier, the
# quantile thresholds derived from them looser, and spurious flags more frequent. Extra recipe
# capacity costs precision and buys no recall.
#
# Caveat on strength of evidence: at n=40 the individual gaps are small (healthy FPR 0.050 vs
# 0.100 is 2 trials vs 4). The MONOTONE trend across four checkpoints in three separate metrics is
# the evidence, not any one pairwise difference. joint_noun16 does have the best abandonment
# recall (0.90 vs 0.78) if that channel is ever the specific target.
#
# joint_clean.npz and joint_d09.npz are deliberately not candidates: they are the same fit as this
# one at iteration 200 and 320 rather than 100, with the same effective K_R=5, and iterating past
# ~100 has been measured to change downstream metrics by nothing (the objective sits in a period-3
# limit cycle, which is also why every checkpoint here reports converged: false -- that is the
# stopping rule being unable to fire, not an unfinished fit).
DEFAULT_JOINT_PARAMS = "dataset/processed/breakfast/joint_params.npz"


# --------------------------------------------------------------------------------------------
# shared trial construction -- one pool, both detectors
# --------------------------------------------------------------------------------------------

def usable_indices(trajectories):
    """Indices of the trajectories build_pool keeps, in pool order.

    build_pool FILTERS, so pool position i is not trajectory i. Callers that want to name a
    pool entry -- its trial_id, its label row -- must map back through this, or they will
    silently attribute one trial's stream to another trial's id once anything is dropped.
    """
    return [i for i, t in enumerate(trajectories)
            if len(t["segments"]) >= error_injection.MIN_SEGMENTS]


def build_pool(trajectories, rng, inject_params):
    """[(healthy_traj, {error_type: degraded_dict})] for every usable trajectory.

    Usability is error_injection.MIN_SEGMENTS, the same gate run_evaluation.py applies: an
    out-of-order step is only out-of-order relative to context. See usable_indices to map a
    pool position back to the trajectory it came from.
    """
    usable = [trajectories[i] for i in usable_indices(trajectories)]
    pool = []
    for traj in usable:
        degraded = {
            et: error_injection.inject(et, traj, rng, inject_params)
            for et in error_injection.ERROR_TYPES
        }
        pool.append((traj, degraded))
    return pool


def steps_and_truth(traj, degraded, lexicon, unit="step"):
    """Degraded steps, the ground-truth step indices, the ground-truth correction, and the
    debris steps the injection created but which are not themselves ground truth
    (textify.injection_touched_steps) -- excluded from false-positive scoring in
    element_metrics.evaluate_steps rather than counted either way."""
    source_steps = textify.elements_from_trajectory(traj, lexicon, unit)
    steps = textify.elements_from_trajectory(degraded, lexicon, unit)
    # Ground truth is the injector's own anomalous POINTS, not the whole disturbed window: for
    # the structural injectors most of that window is a correctly-executed run in the wrong
    # place. The window is still what decides the debris extent, so the two are used together.
    gt_steps = textify.gt_steps_for_ticks(steps, degraded["anomaly_ticks"])
    source = textify.step_covering_tick(source_steps, degraded["window"][0])
    # The ground-truth correction is always the pre-injection STEP -- verb, noun, and the
    # duration that step should have run for -- even at unit="tick", because that is what the
    # prompt asks the detector to name and what correction_accuracy scores. Taking it from the
    # tick element instead would make every duration truth 1 second and the metric meaningless,
    # so the source trial is re-encoded as steps here regardless of the scoring unit.
    if unit != "step":
        source = textify.step_covering_tick(
            textify.steps_from_trajectory(traj, lexicon), degraded["window"][0]
        )
    correction = (source.verb, source.noun, source.duration) if source else None
    debris = textify.injection_touched_steps(
        steps, degraded["tick_map"], degraded["edited_ticks"], gt_steps,
        window=degraded["window"],
    )
    return steps, gt_steps, correction, debris


def spans_by_type(pool):
    """{error_type: [spans_for_one_trial, ...]} in pool order -- the span layer's ground truth.

    Read straight off the injection records, so every arm is scored against byte-identical spans
    for the same reason build_pool exists: ground truth is a property of the pool, not of the
    detector reading it. See synthetic/error_injection._result for what a span is and why it is
    not the same object as `window` or `anomaly_ticks`.
    """
    return {
        error_type: [degraded[error_type]["spans"] for _, degraded in pool]
        for error_type in error_injection.ERROR_TYPES
    }


def empty_stream():
    """The {healthy, degraded} shape every arm fills in, so score_spans has one input shape."""
    return {"healthy": [], "degraded": {et: [] for et in error_injection.ERROR_TYPES}}


def score_spans(stream, spans, merge_tol=span_metrics.MERGE_TOL, onset=False):
    """One arm's tick-flag stream + the pool's spans -> the span-existence report.

    `merge_tol=0` disables merging (identical ticks from two channels still collapse, which is
    deduplication rather than merging). Worth running: merging was insurance against an
    oscillating observation stream, and tools_run_lengths.py measures no such spike on Breakfast,
    while merging demonstrably distorts the comparison in two directions -- see docs/eval.md 7b.
    """
    return span_metrics.evaluate_spans(
        stream["healthy"],
        {et: list(zip(stream["degraded"][et], spans[et])) for et in spans},
        merge_tol=merge_tol, onset=onset,
    )


# --------------------------------------------------------------------------------------------
# the HSMM arm
# --------------------------------------------------------------------------------------------

# Per-channel alpha overrides applied on top of --alpha for the joint HSMM arm.
#
# s_temporal at 1e-5 rather than the shared 5e-3 is a measured result, not a guess. Swept per
# channel against the TICK objective on the TRAIN split and confirmed on held-out test
# (train-fit model, train-selected alpha, test trials):
#
#     config                 tick P   tick R   fp    trial-loc P   healthy FPR
#     all channels 5e-3       0.351    0.495   444        0.664         0.196
#     s_temporal 1e-5         0.460    0.489   278        0.660         0.196
#
# 37% of all false-positive ticks removed for 0.006 recall, with trial-located precision and the
# healthy false-alarm rate both unmoved -- the other six channels already cover what s_temporal
# finds. It is not entirely free: repetition recall falls 0.196 -> 0.175 and transposition
# 0.526 -> 0.515, which is the s_temporal contribution docs/eval.md describes.
#
# Confined to THIS script deliberately. surprise.DEFAULT_ALPHA is untouched, so run_evaluation.py
# and every tick-level number in docs/eval.md still describe the same detector they always did.
# Pass --channel-alpha s_temporal=5e-3 to reproduce the old behaviour here.
DEFAULT_CHANNEL_ALPHA = {"s_temporal": 1e-5}


def _hsmm_flags(trials, model, chunk_size, alpha=surprise.DEFAULT_ALPHA, channel_alpha=None):
    """(flags, traces) for a list of trajectory-shaped dicts, cascade or joint."""
    if model["kind"] == "cascade":
        traces, log_probs, recipe_log_trans = batch.compute_traces(
            model["hsmm"], model["recipe"], trials, model["d_max"], chunk_size=chunk_size
        )
        if channel_alpha:
            # The per-channel flagger is joint-only: it builds its tables from
            # JointHSMMLogProbs. Refusing beats silently ignoring the overrides and reporting
            # numbers that look tuned but are not.
            raise SystemExit("error: --channel-alpha is only supported for the joint model; "
                             "drop --cascade or drop the overrides")
        return [surprise.flag(t, log_probs, recipe_log_trans, alpha=alpha) for t in traces], traces
    return hindsight_flags(trials, model, chunk_size, alpha, channel_alpha)


def hindsight_flags(trials, model, chunk_size, alpha=surprise.DEFAULT_ALPHA, channel_alpha=None):
    """(flags, traces) from detectors.HindsightHSMMDetector over trajectory-shaped dicts."""
    from cook_ad.core import OperatingPoint, Trial
    from cook_ad.detectors.hsmm_hindsight import HindsightHSMMDetector
    from cook_ad.detectors.thresholds import BadSets
    from cook_ad.hsmm import params as hsmm_params
    from cook_ad.hsmm.model import HSMM

    if "hsmm_model" not in model:
        model["hsmm_model"] = HSMM(hsmm_params.JointHSMMParams(*model["joint"][:7]), model["d_max"])
    op = OperatingPoint(surprise.per_channel_alphas(channel_alpha, alpha) if channel_alpha
                        else alpha)
    alphas = sorted(set(op.alpha.values()) if isinstance(op.alpha, dict) else {op.alpha})
    hsmm = model["hsmm_model"]
    det = HindsightHSMMDetector(hsmm, BadSets(hsmm, alphas), op, chunk_size)
    traces = det.traces([Trial(str(i), np.asarray(t["verb_ids"]), np.asarray(t["noun_ids"]))
                         for i, t in enumerate(trials)])
    return [det.flags(tr) for tr in traces], traces


def hsmm_arm(pool, model, lexicon, chunk_size, alpha=surprise.DEFAULT_ALPHA,
             unit="step", tol=None, channel_alpha=None):
    """Score the HSMM through the step layer on the shared pool.

    Returns `(report, stream)`. The stream is the arm's raw TICK flags, which the step layer
    collapses to one verdict per element and therefore throws away; eval/span_metrics.py needs
    them ungrouped, and re-deriving them from the verdicts would silently pass the step layer's
    collapse on to a layer whose whole purpose is not to have one.
    """
    healthy = [t for t, _ in pool]
    healthy_flags, healthy_traces = _hsmm_flags(healthy, model, chunk_size, alpha, channel_alpha)
    healthy_verdicts = [
        element_metrics.step_verdicts_from_flags(
            f, textify.elements_from_trajectory(t, lexicon, unit), trace, lexicon
        )
        for f, trace, t in zip(healthy_flags, healthy_traces, healthy)
    ]
    stream = empty_stream()
    stream["healthy"] = [span_metrics.flags_from_channels(f, element_metrics.ALL_CHANNELS)
                         for f in healthy_flags]

    degraded_by_type = {}
    artifact_steps = {}
    for error_type in error_injection.ERROR_TYPES:
        trials = [d[error_type] for _, d in pool]
        flags, traces = _hsmm_flags(trials, model, chunk_size, alpha, channel_alpha)
        rows = []
        debris_rows = []
        for (traj, degraded), f, trace in zip(pool, flags, traces):
            steps, gt_steps, correction, debris = steps_and_truth(
                traj, degraded[error_type], lexicon, unit)
            verdicts = element_metrics.step_verdicts_from_flags(
                f, steps, trace, lexicon
            )
            rows.append((verdicts, gt_steps, correction))
            debris_rows.append(debris)
            stream["degraded"][error_type].append(
                span_metrics.flags_from_channels(f, element_metrics.ALL_CHANNELS))
        degraded_by_type[error_type] = rows
        artifact_steps[error_type] = debris_rows
        print(f"  [hsmm/{model['kind']}] {error_type}: {len(rows)} degraded trials", flush=True)

    report = element_metrics.evaluate_steps(
        healthy_verdicts, degraded_by_type, artifact_steps=artifact_steps, unit=unit,
        tol_steps=element_metrics.DEFAULT_TOL[unit] if tol is None else tol,
    )
    return report, stream


# --------------------------------------------------------------------------------------------
# the LLM arm
# --------------------------------------------------------------------------------------------

def make_llm_detector(predictive, client, vocab, lexicon, variant, labels, protocol, unit,
                      exemplars, rationales):
    """The detectors/llm_*.py detector for one arm, its prompt built from the given shots."""
    from cook_ad.detectors.llm_predictive import LLMPredictiveDetector
    from cook_ad.detectors.llm_verdict import LLMVerdictDetector
    if predictive:
        det = LLMPredictiveDetector(client, vocab, lexicon, variant=variant, labels=labels,
                                    prompt_unit=unit)
    else:
        det = LLMVerdictDetector(client, vocab, lexicon, protocol=protocol,
                                 unit="run" if unit == "step" else unit, variant=variant,
                                 labels=labels)
    det.exemplars, det.rationales = exemplars, rationales
    det.build_prompt()
    return det


def _detect(det, traj):
    """(detection, span Flags) for one trajectory-shaped dict."""
    from cook_ad.core import Trial
    d = det.detect_trial(Trial("t", np.asarray(traj["verb_ids"]), np.asarray(traj["noun_ids"])))
    return d, sorted(span_metrics.Flag(a.tick, a.score, a.channel) for a in d.alarms)


def llm_arm(pool, lexicon, det, tag, unit="step", tol=None):
    """The two verdict arms. Returns `(report, stream)` -- see hsmm_arm.

    A step verdict becomes a flag at the FIRST TICK of the element it judges, which is the moment
    the model was answering about. At unit='tick' that is the identity.
    """
    healthy_verdicts = []
    stream = empty_stream()
    client = det.client
    for traj, _ in pool:
        d, flags = _detect(det, traj)
        healthy_verdicts.append(element_metrics.from_llm_verdicts(d.trace))
        stream["healthy"].append(flags)
    print(f"  [{tag}] healthy: {len(healthy_verdicts)} trials "
          f"({client.n_would_request} uncached requests so far)", flush=True)

    degraded_by_type = {}
    artifact_steps = {}
    for error_type in error_injection.ERROR_TYPES:
        rows = []
        debris_rows = []
        for traj, degraded in pool:
            steps, gt_steps, correction, debris = steps_and_truth(
                traj, degraded[error_type], lexicon, unit)
            d, flags = _detect(det, degraded[error_type])
            rows.append((element_metrics.from_llm_verdicts(d.trace), gt_steps, correction))
            debris_rows.append(debris)
            stream["degraded"][error_type].append(flags)
        degraded_by_type[error_type] = rows
        artifact_steps[error_type] = debris_rows
        print(f"  [{tag}] {error_type}: {len(rows)} degraded trials "
              f"({client.n_would_request} uncached requests so far)", flush=True)

    report = element_metrics.evaluate_steps(
        healthy_verdicts, degraded_by_type, artifact_steps=artifact_steps, unit=unit,
        tol_steps=element_metrics.DEFAULT_TOL[unit] if tol is None else tol,
    )
    return report, stream


# --------------------------------------------------------------------------------------------
# the predictive arm (arm 3)
# --------------------------------------------------------------------------------------------

def predictive_arm(pool, lexicon, det, tag, unit="step", tol=None):
    """The PREGO-style arm: the model predicts, llm/predict.py decides. Returns (report, stream).

    Runs at STEP BOUNDARIES whatever `unit` the report is scored in -- a prediction is something
    you make once per step, not once per tick -- and emits TICK-indexed flags, which is what the
    span layer wants natively. `unit` is still honoured for the step-layer verdicts, so this arm
    appears in the same tables as the other three: the flags are mapped onto whichever elements
    the run is scored on.

    Per-line parse failures are counted here and reported under `predictive` rather than folded
    into evaluate_steps' parse_failure_rate; see predict.verdicts_from_flags for why.
    """
    stream = empty_stream()
    healthy_verdicts = []
    n_boundaries = n_dur_fail = n_next_fail = 0

    client = det.client

    def _run(traj_like, elements_unit):
        nonlocal n_boundaries, n_dur_fail, n_next_fail
        d, flags = _detect(det, traj_like)
        st = d.trace
        n_boundaries += st["n_boundaries"]
        n_dur_fail += st["duration_parse_failures"]
        n_next_fail += st["next_parse_failures"]
        return flags, predict.verdicts_from_flags(flags, elements_unit)

    for traj, _ in pool:
        elements = textify.elements_from_trajectory(traj, lexicon, unit)
        flags, verdicts = _run(traj, elements)
        healthy_verdicts.append(verdicts)
        stream["healthy"].append(flags)
    print(f"  [{tag}] healthy: {len(healthy_verdicts)} trials "
          f"({client.n_would_request} uncached requests so far)", flush=True)

    degraded_by_type = {}
    artifact_steps = {}
    for error_type in error_injection.ERROR_TYPES:
        rows, debris_rows = [], []
        for traj, degraded in pool:
            steps, gt_steps, correction, debris = steps_and_truth(
                traj, degraded[error_type], lexicon, unit)
            flags, verdicts = _run(degraded[error_type], steps)
            rows.append((verdicts, gt_steps, correction))
            debris_rows.append(debris)
            stream["degraded"][error_type].append(flags)
        degraded_by_type[error_type] = rows
        artifact_steps[error_type] = debris_rows
        print(f"  [{tag}] {error_type}: {len(rows)} degraded trials "
              f"({client.n_would_request} uncached requests so far)", flush=True)

    report = element_metrics.evaluate_steps(
        healthy_verdicts, degraded_by_type, artifact_steps=artifact_steps, unit=unit,
        tol_steps=element_metrics.DEFAULT_TOL[unit] if tol is None else tol,
    )
    report["predictive"] = {
        "n_boundaries": n_boundaries,
        "duration_parse_failure_rate": n_dur_fail / n_boundaries if n_boundaries else 0.0,
        "next_parse_failure_rate": n_next_fail / n_boundaries if n_boundaries else 0.0,
    }
    return report, stream


def pool_request_cost(pool, lexicon, protocol, unit="step"):
    """UPPER BOUND on the requests a sweep over this pool costs, for --dry-run and the budget line.

    A bound rather than an estimate: prefix-only requests cache on the whole prompt, and the six
    conditions share prefixes (a substitution rewrites one tick mid-trial, so every earlier step
    renders identically to the healthy trial's). Measured on 10 real trials: 398 predicted, 251
    actually sent. Budget against this and expect to spend less.
    """
    total = 0
    for traj, degraded in pool:
        # The predictive arm is defined at step boundaries whatever the scoring unit is, so its
        # cost is counted in steps rather than in `unit` elements -- at --unit tick that is the
        # difference between ~6.5 and ~121 requests per trial.
        cost_unit = "step" if protocol == predict.PROTOCOL else unit
        total += _element_cost(traj, lexicon, protocol, cost_unit)
        for error_type in error_injection.ERROR_TYPES:
            total += _element_cost(degraded[error_type], lexicon, protocol, cost_unit)
    return total


def _element_cost(traj_like, lexicon, protocol, cost_unit):
    elements = textify.elements_from_trajectory(traj_like, lexicon, cost_unit)
    if protocol == predict.PROTOCOL:
        return predict.request_cost(elements)
    return detect.request_cost(elements, protocol)


# --------------------------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------------------------

def print_report(report, tag, note=None):
    unit = report.get("unit", "step")
    print(f"\n===== {tag} (unit: {unit}) =====")
    if note:
        print(note)
    h = report["healthy"]
    print(f"healthy false-positive rate: {h['false_positive_rate']:.3f} "
          f"({h['false_positive_trials']}/{h['n']} control trials flagged)")
    sl = report["step_level"]
    print(f"pooled {unit}-level: precision {sl['precision']:.3f}  recall {sl['recall']:.3f}  "
          f"(tp {sl['tp']}, fp {sl['fp']}, fn {sl['fn']})")
    tl = report.get("trial_located")
    if tl:
        f1 = (2 * tl["precision"] * tl["recall"] / (tl["precision"] + tl["recall"])
              if (tl["precision"] + tl["recall"]) else 0.0)
        print(f"TRIAL-LOCATED:     precision {tl['precision']:.3f}  recall {tl['recall']:.3f}  "
              f"F1 {f1:.3f}   (stray {tl['stray_rate']:.3f}/degraded, "
              f"healthy {tl['healthy_fpr']:.3f})")
    print(f"parse failure rate: {report['parse_failure_rate']:.3f}")
    print(f"\n{'error type':>14}  {'n':>4}  {'recall':>7}  {'precision':>9}  {'prec_excl_hlt':>13}  "
          f"{('lat(%ss)' % unit):>10}  {'top pred type':>15}  {'corr v/n':>9}  {'corr dur':>9}")
    for error_type, m in report["per_type"].items():
        conf = report["type_confusion"][error_type]
        top = max(conf, key=conf.get) if any(conf.values()) else "-"
        lat = "n/a" if np.isnan(m["mean_latency"]) else f"{m['mean_latency']:.1f}"
        ca = report["correction_accuracy"][error_type]
        vn = "n/a" if np.isnan(ca["verb_noun_accuracy"]) else f"{ca['verb_noun_accuracy']:.3f}"
        du = "n/a" if np.isnan(ca["duration_accuracy"]) else f"{ca['duration_accuracy']:.3f}"
        print(f"{error_type:>14}  {m['n']:>4}  {m['recall']:>7.3f}  {m['precision']:>9.3f}  "
              f"{m['precision_excl_healthy']:>13.3f}  {lat:>10}  {top:>15}  {vn:>9}  {du:>9}")
    pv = report.get("predictive")
    if pv:
        print(f"\npredictive: {pv['n_boundaries']} boundaries; per-line parse failures "
              f"DURATION {pv['duration_parse_failure_rate']:.3f}  "
              f"NEXT {pv['next_parse_failure_rate']:.3f}")
    if report.get("spans"):
        print_span_report(report["spans"])


def print_span_report(sp):
    """The span-existence layer (eval/span_metrics.py). Printed beside the element layer above,
    never instead of it -- the two answer different questions and both numbers are published."""
    pooled, fp = sp["pooled"], sp["false_positives"]
    print(f"\nSPAN-EXISTENCE (merge_tol {sp['merge_tol']}, deadline {sp['deadline']}):")
    print(f"  pooled: precision {pooled['precision']:.3f}  recall {pooled['recall']:.3f}  "
          f"F1 {pooled['f1']:.3f}   (tp {pooled['tp']}, fn {pooled['fn']}, fp {pooled['fp']})")
    lt, h = sp["latency"], sp["healthy"]
    med = "n/a" if np.isnan(lt["median"]) else f"{lt['median']:.1f}"
    p90 = "n/a" if np.isnan(lt["p90"]) else f"{lt['p90']:.1f}"
    print(f"  latency (ticks, hits only): median {med}  p90 {p90}  (n={lt['n']})")
    print(f"  healthy FP rate: {h['false_positive_rate']:.3f} "
          f"({h['false_positive_trials']}/{h['n']} control trials flagged, unmerged)")
    print(f"  false positives: {fp['merged']} merged / {fp['raw']} raw "
          f"(the gap is observation oscillation, not detector behaviour); "
          f"{sp['extra_in_span']} extra in-span flags, unscored")
    print(f"\n  {'error type':>14}  {'spans':>6}  {'recall':>7}  {'med width':>10}  {'med lat':>8}")
    for etype, m in sp["per_type"].items():
        w = "n/a" if np.isnan(m["median_span_width"]) else f"{m['median_span_width']:.1f}"
        la = "n/a" if np.isnan(m["median_latency"]) else f"{m['median_latency']:.1f}"
        print(f"  {etype:>14}  {m['n_spans']:>6}  {m['recall']:>7.3f}  {w:>10}  {la:>8}")


# The predictive arm's client settings. Temperature 0.6 rather than 0.0 because the arm's future
# is n=K sampling -- the point estimate is element 0 of a list today, and a distribution tomorrow
# -- and a deterministic decode would make that a different experiment. Both values are in the
# cache key (max_tokens conditionally, see client._cache_key), so this arm namespaces itself and
# no already-collected response is invalidated.
PREDICTIVE_TEMPERATURE = 0.6
# Two short labelled lines is the entire expected reply. Small enough to be cheap, wide enough
# that a model which prepends a sentence of chatter still gets both lines out.
PREDICTIVE_MAX_TOKENS = 48
# With Auto-CoT demonstrations the model imitates them and reasons before answering, so the cap
# has to cover a sentence or two plus the two lines. Too tight and a reply is truncated mid-
# reasoning and never reaches DURATION:/NEXT: -- which the parser scores as a double failure.
PREDICTIVE_COT_MAX_TOKENS = 256
RATIONALE_MAX_TOKENS = 200

PREDICTIVE_RECIPES_CAVEAT = (
    "NOTE: for the PREDICTIVE arm the with-recipes preprompt supplies duration priors as well as\n"
    "ordering -- recipe_block prints each step's median length -- so this variant is given the\n"
    "very quantity the duration channel is being tested on. Read its duration-channel numbers\n"
    "against the no-recipes arm, not as a property of the model."
)

WITH_RECIPES_CAVEAT = (
    "NOTE: this arm's preprompt is built from labels.json, which is validation-only everywhere\n"
    "else in this repo and is never fed to training. It therefore sees ground-truth task\n"
    "structure the HSMM never had. Compare it to the no-recipes arm, NOT to the HSMM."
)

BATCH_CAVEAT = (
    "NOTE: protocol=batch is NON-CAUSAL -- the model saw every step of the trial before judging\n"
    "any of them. Its latency column is not comparable to the incremental arm or to the HSMM."
)


# --------------------------------------------------------------------------------------------

# Arguments that must match for two runs to describe the same experiment: everything the shared
# trial pool depends on. --model/--variant/--protocol are deliberately absent -- comparing two
# models or two prompt variants inside one report is the whole point of merging.
# --unit and --tol are pool-defining even though they do not change which trajectories are built:
# they change the ELEMENTS both arms answer about and the deadline they are scored to, so a
# step-unit arm and a tick-unit arm in one report would be a table that looks like a comparison
# and is not one (~18x more elements per trial, hence a different chance precision).
POOL_DEFINING_ARGS = ("seed", "source", "n", "max_ticks", "max_real", "config", "sequences",
                      "joint_params", "cascade", "params", "recipe_params",
                      "split_file", "split_part", "unit", "tol")


def _write_report(args, reports, incomplete=False):
    """Merge into an existing report at --out rather than overwriting it.

    This is what makes `--skip-llm` then `--skip-hsmm` compose into one comparable report, which
    matters because the two arms want the GPU on very different terms: the HSMM arm wants JAX, the
    LLM arm wants a local server holding 17+ GB of weights. Running them as separate invocations
    is always safe, and the shared --seed guarantees both scored the identical trial pool.

    The merge is guarded on POOL_DEFINING_ARGS, because two runs only describe the same experiment
    if they built the same pool. Silently merging arms scored on different data would produce a
    table that looks like a comparison and is not one.
    """
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged = {}
    key = {k: getattr(args, k) for k in POOL_DEFINING_ARGS}

    if out.exists():
        try:
            prior = json.loads(out.read_text())
        except (json.JSONDecodeError, OSError):
            prior = None
        if prior:
            prior_cfg = prior.get("config", {})
            differing = [k for k in POOL_DEFINING_ARGS
                         if str(prior_cfg.get(k)) != str(key.get(k))]
            if not differing:
                merged = prior.get("reports", {})
                print(f"merging into {out} ({len(merged)} prior arm(s): "
                      f"{', '.join(merged) or 'none'})")
            else:
                print(f"NOT merging with the existing {out}: produced with different "
                      f"{', '.join(differing)}, so its arms were scored on a different trial "
                      f"pool. Overwriting -- use a different --out to keep both.")

    merged.update(reports)
    out.write_text(json.dumps({
        "config": dict(vars(args)),
        "incomplete": incomplete,
        "reports": merged,
    }, indent=2, default=str))
    done = [k for k, v in merged.items() if not v.get("incomplete")]
    print(f"\n{'PARTIAL report' if incomplete else 'report'} written to {out} "
          f"({len(done)} arm(s) complete: {', '.join(done) or 'none'})")
    print(f"figures written to {args.figures_dir}/")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/breakfast.yaml")
    parser.add_argument("--params", default="dataset/processed/breakfast/hsmm_params.npz")
    parser.add_argument("--recipe-params", default="dataset/processed/breakfast/recipe_params.npz")
    parser.add_argument("--joint-params", default=DEFAULT_JOINT_PARAMS,
                        help="joint checkpoint for the HSMM arm (the default; see "
                             "DEFAULT_JOINT_PARAMS for why this one)")
    parser.add_argument("--cascade", action="store_true",
                        help="use the 2-stage cascade for the HSMM arm instead of the joint "
                             "model. Note this changes BOTH arms: the model Viterbi-segments the "
                             "healthy trials, which decides where error_injection places an "
                             "error and therefore what text the LLM reads.")
    parser.add_argument("--sequences", default="dataset/processed/breakfast/sequences.json")
    parser.add_argument("--labels", default="dataset/processed/breakfast/labels.json")
    parser.add_argument("--vocab", default="dataset/processed/breakfast/vocab.json")
    parser.add_argument("--figures-dir", default="dataset/processed/breakfast/figures")
    parser.add_argument("--out", default="dataset/processed/breakfast/llm_report.json")
    parser.add_argument("--cache-dir", default="dataset/processed/breakfast/llm_cache")

    parser.add_argument("--source", choices=("synthetic", "real", "both"), default="real")
    parser.add_argument("--variant", choices=(*prompts.VARIANTS, "both"), default="both")
    # `predictive` is not in detect.PROTOCOLS: everything in there returns Verdicts and this one
    # returns tick flags (llm/predict.py). Listed here so it is still a --protocol choice.
    parser.add_argument("--protocol", choices=(*detect.PROTOCOLS, predict.PROTOCOL),
                        default="incremental")
    parser.add_argument("--shot-select", choices=prompts.SHOT_SELECT, default="random",
                        help="how the predictive arm's few-shot exemplars are chosen. 'random' is "
                             "a uniform draw; 'autocot' is Auto-CoT's clustering step (Zhang et "
                             "al., ICLR 2023) -- k-means over the train split, one representative "
                             "per cluster -- which buys DIVERSITY. Measured on Breakfast: 3 random "
                             "trials cover 2 of the 10 recipes, 3 clustered cover 3; at n=10 the "
                             "split is 5 against all 10")
    parser.add_argument("--shot-cot", action="store_true",
                        help="Auto-CoT proper: additionally auto-generate one rationale per "
                             "exemplar with Zero-Shot-CoT and show prefix -> reasoning -> answer "
                             "demonstrations. Costs k extra requests ONCE (not per query), but "
                             "the model then emits reasoning before its two lines, so it needs a "
                             "bigger token budget and generates more slowly")
    parser.add_argument("--n-shot", type=int, default=3,
                        help="few-shot exemplar trials in the predictive arm's system prompt, "
                             "drawn from the TRAIN split with a fixed seed so the system block "
                             "stays a constant shared cache prefix. 0 disables them. Ignored by "
                             "every other protocol")
    parser.add_argument("--unit", choices=textify.UNITS, default="step",
                        help="the element both arms answer about and are scored on. 'step' is one "
                             "run-length-encoded step (~6.5 per trial, the historical default); "
                             "'tick' is one second (~121 per trial), which removes the step "
                             "layer's handicap on the HSMM but costs ~18x the requests and is NOT "
                             "precision-comparable to a step-unit run -- read chance_precision")
    parser.add_argument("--merge-tol", type=int, default=span_metrics.MERGE_TOL,
                        help="span layer only: flags this many ticks apart or closer collapse to "
                             "one alarm for FALSE-POSITIVE counting. 0 disables merging. It is "
                             "not neutral -- merging rewards a saturated detector (one alarm per "
                             "trial) and penalises one that emits a single flag per step, so run "
                             "both before quoting a span F1")
    parser.add_argument("--tol", type=int, default=None,
                        help="detection deadline in elements past the ground-truth window "
                             "(default: 1 step / 10 ticks, element_metrics.DEFAULT_TOL)")
    parser.add_argument("--skip-hsmm", action="store_true",
                        help="skip the HSMM arm (it needs no API budget, so this is only for "
                             "iterating on the LLM side)")
    parser.add_argument("--skip-llm", action="store_true",
                        help="skip the LLM arms entirely -- scores only the HSMM through the step "
                             "layer, needs no API key and no request budget")

    parser.add_argument("--model", default=llm_client.DEFAULT_MODEL)
    parser.add_argument("--base-url", default=llm_client.DEFAULT_BASE_URL,
                        help="any OpenAI-compatible /chat/completions host, e.g. "
                             "https://generativelanguage.googleapis.com/v1beta/openai/")
    parser.add_argument("--api-key-env", default=llm_client.DEFAULT_API_KEY_ENV)
    parser.add_argument("--env-file", default=llm_client.DEFAULT_ENV_FILE,
                        help="untracked KEY=VALUE file to load the API key from (see "
                             ".env.example). Missing file is fine; already-set environment "
                             "variables win over it.")
    parser.add_argument("--rpm", type=int, default=None,
                        help="requests/minute pacing. Default depends on destination: 0 (no "
                             "pacing) for a localhost base URL, 15 for anything else. Pass 0 to "
                             "disable explicitly.")
    parser.add_argument("--concurrency", type=int, default=None,
                        help="requests in flight at once. Default 8 for a localhost base URL, 1 "
                             "otherwise. Only safe because the prefix-only protocol makes every "
                             "request independent. For ollama, also raise OLLAMA_NUM_PARALLEL to "
                             "at least this, or the server will queue them anyway.")
    parser.add_argument("--max-requests", type=int, default=None,
                        help="hard budget: raise rather than silently truncate a sweep")
    parser.add_argument("--dry-run", action="store_true",
                        help="render every prompt and report exactly what a sweep would cost; "
                             "makes no network calls and needs no API key")

    parser.add_argument("--n", type=int, default=20, help="synthetic healthy trials to generate")
    parser.add_argument("--max-ticks", type=int, default=100)
    parser.add_argument("--max-real", type=int, default=20, help="real trials to use")
    parser.add_argument("--split-file", default=None,
                        help="path to a split.json from split_dataset.py; if given, the real "
                             "trial pool is restricted to --split-part before the --max-real cap")
    parser.add_argument("--split-part", choices=["train", "test"], default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=surprise.DEFAULT_ALPHA,
                        help="HSMM arm's flag threshold. The LLM arm has no threshold to sweep, "
                             "so comparing the two at one alpha compares them at whatever point "
                             "that alpha happens to put the HSMM on ITS OWN curve -- match a "
                             "recall or a false-alarm rate before reading a precision gap")
    parser.add_argument("--channel-alpha", action="append", default=None, metavar="CHANNEL=ALPHA",
                        help="per-channel alpha override for the joint HSMM arm, repeatable "
                             f"(default: {DEFAULT_CHANNEL_ALPHA}). Pass 's_temporal=5e-3' to "
                             "restore the single shared alpha. See DEFAULT_CHANNEL_ALPHA for the "
                             "measurement behind the default")
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--dump-streams", default=None, metavar="PATH",
                        help="also pickle the pool (every trial's verb/noun ids and spans) and every "
                             "arm's per-trial span flags, so other detectors and other metrics can "
                             "be scored on EXACTLY these streams (tools_breakfast_g_table.py)")
    parser.add_argument("--min-run", type=int, default=10,
                        help="tick-level persistence requirement applied BEFORE collapsing HSMM "
                             "flags to steps; see element_metrics.step_verdicts_from_flags for "
                             "why a plain per-step OR without it is not a fair reading of the "
                             "HSMM")
    args = parser.parse_args()

    if args.channel_alpha is None:
        args.channel_alpha = dict(DEFAULT_CHANNEL_ALPHA)
    else:
        parsed = {}
        for item in args.channel_alpha:
            if "=" not in item:
                parser.error(f"--channel-alpha expects CHANNEL=ALPHA, got {item!r}")
            ch, _, a = item.partition("=")
            parsed[ch.strip()] = float(a)
        try:
            surprise.per_channel_alphas(parsed)      # validates the channel names
        except ValueError as e:
            parser.error(str(e))
        args.channel_alpha = parsed

    llm_client.load_env_file(args.env_file)

    config = load_config(args.config)
    textify.assert_tick_seconds(config)
    d_max = config["duration"]["d_max_ticks"]

    with open(args.vocab) as f:
        vocab = json.load(f)
    labels = None
    if args.variant in ("with-recipes", "both"):
        with open(args.labels) as f:
            labels = json.load(f)

    if not args.cascade:
        joint = joint_params.load_params(args.joint_params)
        model = {"kind": "joint", "joint": joint, "d_max": d_max}
        inject_params = joint_params.collapse_to_marginal(joint)
        lexicon = narrate.Lexicon(vocab, inject_params)
    else:
        hsmm = params.load_params(args.params)
        model = {"kind": "cascade", "hsmm": hsmm,
                 "recipe": recipe_hmm.load_params(args.recipe_params), "d_max": d_max}
        inject_params = hsmm
        lexicon = narrate.Lexicon(vocab, hsmm)

    # Fail before building pools and running the HSMM arm: discovering a missing key several
    # minutes into a run, after the expensive part, is a waste of the user's time. A missing key
    # is a configuration mistake, not a bug, so report it as one line rather than a traceback.
    if not args.skip_llm and not args.dry_run:
        try:
            llm_client.ChatClient(api_key_env=args.api_key_env)._api_key()
        except llm_client.LLMError as e:
            raise SystemExit(f"error: {e}") from None

    print(f"LLM-vs-HSMM {args.unit}-level evaluation")
    print(f"model={args.model}  base_url={args.base_url}  protocol={args.protocol}  "
          f"hsmm={model['kind']}")

    # ---- shared pools --------------------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    pools = {}
    if args.source in ("synthetic", "both"):
        traj = (generate.generate_healthy_joint(model["joint"], args.n, rng, args.max_ticks, d_max)
                if model["kind"] == "joint"
                else generate.generate_healthy(inject_params, args.n, rng, args.max_ticks, d_max))
        pools["synthetic"] = build_pool(traj, rng, inject_params)
    if args.source in ("real", "both"):
        with open(args.sequences) as f:
            sequences = json.load(f)
        if args.split_file:
            if args.split_part is None:
                parser.error("--split-part is required when --split-file is given")
            split = split_mod.load_split(args.split_file)
            sequences = split_mod.filter_sequences(sequences, split, args.split_part)
        chosen = sequences[: args.max_real]
        if model["kind"] == "joint":
            # batched: the per-trial adapter is a static-shape JAX call, so a loop recompiles
            # the recipe-inference and Viterbi kernels once per distinct trial length
            traj = generate.trajectories_from_real_joint(
                model["joint"], chosen, d_max, chunk_size=args.chunk_size
            )
        else:
            traj = [generate.trajectory_from_real(inject_params, s["verb_ids"], s["noun_ids"], d_max)
                    for s in chosen]
        pools["real"] = build_pool(traj, rng, inject_params)

    # ---- few-shot exemplars (predictive arm only) -----------------------------------------
    # From the TRAIN split, never from the pool this run scores: a demonstration of a trial the
    # model is later asked to predict is not a demonstration. Seeded and sorted inside
    # select_exemplars, so the system prompt is identical on every request of the sweep and the
    # whole system block stays one shared prefix for the server's cache.
    exemplars, rationales = [], None
    if args.protocol == predict.PROTOCOL and args.n_shot > 0:
        with open(args.sequences) as f:
            all_sequences = json.load(f)
        if args.split_file:
            train = split_mod.filter_sequences(
                all_sequences, split_mod.load_split(args.split_file), "train")
        else:
            train = all_sequences
        shots = [{"verb_ids": t["verb_ids"], "noun_ids": t["noun_ids"]} for t in train]
        exemplars = prompts.select_exemplars(shots, lexicon, args.n_shot, args.seed,
                                             method=args.shot_select, vocab=vocab)
        print(f"few-shot: {len(exemplars)} exemplar trial(s) ({args.shot_select}) from "
              f"{'train split' if args.split_file else 'all sequences'} "
              f"({sum(len(e) for e in exemplars)} steps total)")
        if args.shot_cot:
            # k requests, once, before the sweep -- not per query. Its own client so the rationale
            # generation (long, free-form) does not inherit the arm's tight token cap.
            rationale_client = llm_client.ChatClient(
                model=args.model, base_url=args.base_url, api_key_env=args.api_key_env,
                cache_dir=Path(args.cache_dir) / args.model.replace("/", "_"),
                rpm=args.rpm, max_requests=None, concurrency=args.concurrency,
                temperature=0.0, max_tokens=RATIONALE_MAX_TOKENS)
            rationales = predict.build_rationales(rationale_client, exemplars,
                                                  prompts.exemplar_boundary)
            print(f"auto-cot: {len(rationales)} rationale(s) generated "
                  f"({rationale_client.n_requests} new, {rationale_client.n_cache_hits} cached)")
            for i, r in enumerate(rationales):
                print(f"    [{i + 1}] {r[:150]}{'...' if len(r) > 150 else ''}")

    # Report cache state before spending anything: the difference between "this costs 400
    # requests" and "this is mostly cached and costs 40" decides whether a run is worth starting.
    # The model id namespaces the cache, so this also makes an accidental --model switch visible
    # as a suddenly-empty cache rather than as a surprise part-way through.
    cache_dir = Path(args.cache_dir) / args.model.replace("/", "_")
    n_cached = len(list(cache_dir.glob("*.json"))) if cache_dir.exists() else 0
    print(f"cache: {n_cached} response(s) for {args.model} in {cache_dir}")
    if n_cached == 0 and Path(args.cache_dir).exists():
        for d in sorted(Path(args.cache_dir).glob("*")):
            n = len(list(d.glob("*.json"))) if d.is_dir() else 0
            if n and d.name != cache_dir.name:
                print(f"  (note: {n} cached response(s) exist for {d.name}; NOT reused for "
                      f"{args.model}, and results across models are not comparable)")

    for name, pool in pools.items():
        cost = pool_request_cost(pool, lexicon, args.protocol, args.unit)
        n_variants = 0 if args.skip_llm else (
            len(prompts.VARIANTS) if args.variant == "both" else 1)
        print(f"\n[{name}] {len(pool)} usable trials; at most {cost} requests per variant, "
              f"{cost * n_variants} for this run ({args.protocol} protocol). Shared prefixes "
              f"between conditions cache, so the real cost runs ~35% lower.")

    # The tick unit's prefixes are ~18x longer than the step unit's, which is the one way this
    # can go silently wrong: ollama truncates a prompt that exceeds the served context window
    # from the FRONT, dropping the system prompt -- the response grammar, the vocabulary, the
    # anomaly definitions -- while still returning a plausible-looking reply. That shows up as a
    # parse-failure cliff on long trials and nothing else, so the size is stated up front rather
    # than discovered afterwards. ~1.4 tokens per word is a deliberate over-estimate.
    if args.unit == "tick":
        longest = max(
            (len(textify.elements_from_trajectory(t, lexicon, "tick"))
             for pool in pools.values() for t, _ in pool), default=0
        )
        # The LONGEST system prompt of those that will run, since this is a headroom check.
        sys_words = max(
            len(prompts.build_variant(v, vocab, labels, args.protocol, args.unit,
                                      exemplars=exemplars, rationales=rationales).split())
            for v in (prompts.VARIANTS if args.variant == "both" else [args.variant])
        )
        # Calibrated against the served model, not guessed: the with-recipes tick prompt for a
        # 466-tick trial measures 7611 prompt_tokens, of which 1540 is the system block (928
        # words). That is 1.66 tokens/word and 13.0 tokens per rendered line -- twice the 6.5 a
        # bare `pour milk` line cost, because `43. pour milk (5s)` carries a number, a
        # parenthesised counter and a newline. An earlier estimate of 4 tokens/line under-reported
        # this by 1.85x, which is the difference between "fits in 8192" and "silently truncated".
        est = int(1.7 * sys_words + 13 * longest)
        print(f"\ncontext: longest trial is {longest} ticks; final request ~{est} tokens "
              f"(system prompt + {longest} numbered lines).")
        print(f"  ollama truncates from the FRONT and drops the system prompt silently. Serve "
              f"with OLLAMA_CONTEXT_LENGTH >= {2 ** (est - 1).bit_length()} and keep "
              f"OLLAMA_NUM_PARALLEL * context within VRAM, or the model spills to CPU.")

    if args.dry_run:
        name, pool = next(iter(pools.items()))
        traj, _ = pool[0]
        variant = prompts.VARIANTS[0] if args.variant == "both" else args.variant
        print(f"\n----- system prompt ({variant}, {args.protocol}) -----")
        print(prompts.build_variant(variant, vocab, labels, args.protocol, args.unit,
                                    exemplars=exemplars, rationales=rationales))
        if args.protocol == predict.PROTOCOL:
            # The predictive arm asks about a step BEFORE it ends, so its user turn is not the
            # rendered trial -- show the request it would actually issue.
            steps = textify.steps_from_trajectory(traj, lexicon)
            print(f"\n----- first trial, last boundary's user turn ({name}) -----")
            print(predict.predictive_prefix(steps, len(steps) - 1))
        else:
            print(f"\n----- first trial as {args.unit}s ({name}) -----")
            for line in textify.render_trial(
                    textify.elements_from_trajectory(traj, lexicon, args.unit), args.unit):
                print("   ", line)
        print("\ndry run: no requests made.")
        return

    def _dump():
        if not args.dump_streams:
            return
        import pickle
        ids = lambda d: {"verb_ids": np.asarray(d["verb_ids"]), "noun_ids": np.asarray(d["noun_ids"])}
        payload = {
            "pools": {name: [(ids(t), {et: {**ids(d), "spans": d["spans"]} for et, d in deg.items()})
                             for t, deg in pool] for name, pool in pools.items()},
            "streams": dumped,
            "args": vars(args),
        }
        with open(args.dump_streams, "wb") as fh:
            pickle.dump(payload, fh)
        print(f"streams dumped to {args.dump_streams}")

    # ---- score ----------------------------------------------------------------------------
    reports = {}
    dumped = {}
    variants = [] if args.skip_llm else (
        list(prompts.VARIANTS) if args.variant == "both" else [args.variant]
    )

    for name, pool in pools.items():
        # Ground truth for the span layer, read straight off the injection records so every arm
        # scores byte-identical spans.
        spans = spans_by_type(pool)
        if not args.skip_hsmm:
            # The arm KEY carries the non-default alphas, so a tuned run lands beside the
            # baseline one in the same report instead of overwriting it -- these are two
            # operating points of one detector on one pool, and the comparison is the point.
            # (--alpha is not in POOL_DEFINING_ARGS, so without this a re-run at a different
            # threshold would silently replace the arm it should be compared against.)
            suffix = "".join(f"@{ch}={a:g}" for ch, a in sorted(args.channel_alpha.items()))
            arm = f"{name}/hsmm-{model['kind']}{suffix}"
            print(f"\n[{name}] HSMM arm ({model['kind']}) "
                  f"alpha={args.alpha:g} channel_alpha={args.channel_alpha or 'none'}")
            report, stream = hsmm_arm(pool, model, lexicon, args.chunk_size, args.alpha,
                                      args.unit, args.tol, args.channel_alpha)
            report["alpha"] = args.alpha
            report["channel_alpha"] = args.channel_alpha
            report["spans"] = score_spans(stream, spans, args.merge_tol)
            reports[arm] = report
            dumped[arm] = stream
            print_report(report, arm)
            plotting.save_step_figures(report, args.figures_dir,
                                       f"{name}_hsmm_{model['kind']}{suffix}".replace("@", "_"))

        predictive = args.protocol == predict.PROTOCOL
        for variant in variants:
            # A distinct arm key, so a predictive run merges ALONGSIDE the verdict arms in one
            # report rather than overwriting them (--protocol is deliberately not pool-defining).
            # The prompting configuration goes IN THE KEY, the same way the HSMM arm carries its
            # per-channel alphas: two prompt configurations are two operating points of one arm on
            # one pool and the comparison is the point, so a re-run under Auto-CoT must land
            # BESIDE the random-exemplar run rather than silently replacing it.
            shot_suffix = ""
            if predictive:
                if args.shot_select != "random":
                    shot_suffix += f"+{args.shot_select}"
                if args.shot_cot:
                    shot_suffix += "+cot"
            tag = (f"{name}/llm-predictive-{variant}{shot_suffix}" if predictive
                   else f"{name}/llm-{variant}")
            print(f"\n[{name}] LLM arm ({variant}, {args.protocol})")
            client = llm_client.ChatClient(
                model=args.model, base_url=args.base_url, api_key_env=args.api_key_env,
                cache_dir=Path(args.cache_dir) / args.model.replace("/", "_"),
                rpm=args.rpm, max_requests=args.max_requests, dry_run=False,
                concurrency=args.concurrency,
                # The predictive arm samples rather than judging, and needs only two short lines
                # back. Both are in the cache key, so it gets its own namespace for free and no
                # existing response is invalidated.
                temperature=PREDICTIVE_TEMPERATURE if predictive else 0.0,
                max_tokens=((PREDICTIVE_COT_MAX_TOKENS if args.shot_cot
                             else PREDICTIVE_MAX_TOKENS) if predictive else None),
            )
            print(f"  client: rpm={client.rpm} concurrency={client.concurrency} "
                  f"temperature={client.temperature}", flush=True)
            det = make_llm_detector(predictive, client, vocab, lexicon, variant, labels,
                                    args.protocol, args.unit, exemplars, rationales)
            try:
                if predictive:
                    report, stream = predictive_arm(pool, lexicon, det, tag, args.unit, args.tol)
                else:
                    report, stream = llm_arm(pool, lexicon, det, tag, args.unit, args.tol)
            except (llm_client.LLMError, llm_client.BudgetExceeded) as e:
                # Running out of quota part-way through is an expected operating condition on a
                # free tier, not a crash. Write the arms that DID finish rather than discarding an
                # hour of HSMM work because the last arm ran out of requests.
                print(f"\n[{tag}] stopped early: {e}", flush=True)
                reports[tag] = {"incomplete": True, "error": str(e), "client": client.stats(),
                                "prompt_variant": variant, "protocol": args.protocol}
                _write_report(args, reports, incomplete=True)
                raise SystemExit(1) from None
            report["client"] = client.stats()
            report["prompt_variant"] = variant
            report["protocol"] = args.protocol
            report["unit"] = args.unit
            report["spans"] = score_spans(stream, spans, args.merge_tol)
            dumped[tag] = stream
            if predictive:
                report["n_shot"] = len(exemplars)
                report["shot_select"] = args.shot_select
                report["shot_cot"] = bool(args.shot_cot)
            reports[tag] = report

            note = "\n".join(filter(None, [
                WITH_RECIPES_CAVEAT if variant == "with-recipes" else None,
                PREDICTIVE_RECIPES_CAVEAT if (predictive and variant == "with-recipes") else None,
                BATCH_CAVEAT if args.protocol == "batch" else None,
            ]))
            print_report(report, tag, note or None)
            plotting.save_step_figures(
                report, args.figures_dir,
                f"{name}_llm_{'predictive_' if predictive else ''}{variant}"
                f"{shot_suffix.replace('+', '_') if predictive else ''}")

    _write_report(args, reports)
    _dump()


if __name__ == "__main__":
    main()
