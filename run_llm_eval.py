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
from cook_ad.eval import batch, element_metrics, plotting
from cook_ad.hsmm import joint_params, params
from cook_ad.llm import client as llm_client
from cook_ad.llm import detect, prompts, textify
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


# --------------------------------------------------------------------------------------------
# the HSMM arm
# --------------------------------------------------------------------------------------------

def _hsmm_flags(trials, model, chunk_size, alpha=surprise.DEFAULT_ALPHA):
    """(flags, traces) for a list of trajectory-shaped dicts, cascade or joint."""
    if model["kind"] == "cascade":
        traces, log_probs, recipe_log_trans = batch.compute_traces(
            model["hsmm"], model["recipe"], trials, model["d_max"], chunk_size=chunk_size
        )
        return [surprise.flag(t, log_probs, recipe_log_trans, alpha=alpha) for t in traces], traces
    traces, log_probs, r_hat, log_trans_marginal = batch.compute_traces_joint(
        model["joint"], trials, model["d_max"], chunk_size=chunk_size
    )
    flags = [surprise.flag_joint(t, log_probs, int(r_hat[i]), log_trans_marginal, alpha=alpha)
             for i, t in enumerate(traces)]
    return flags, traces


def hsmm_arm(pool, model, lexicon, chunk_size, alpha=surprise.DEFAULT_ALPHA,
             unit="step", tol=None):
    """Score the HSMM through the step layer on the shared pool."""
    healthy = [t for t, _ in pool]
    healthy_flags, healthy_traces = _hsmm_flags(healthy, model, chunk_size, alpha)
    healthy_verdicts = [
        element_metrics.step_verdicts_from_flags(
            f, textify.elements_from_trajectory(t, lexicon, unit), trace, lexicon
        )
        for f, trace, t in zip(healthy_flags, healthy_traces, healthy)
    ]

    degraded_by_type = {}
    artifact_steps = {}
    for error_type in error_injection.ERROR_TYPES:
        trials = [d[error_type] for _, d in pool]
        flags, traces = _hsmm_flags(trials, model, chunk_size, alpha)
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
        degraded_by_type[error_type] = rows
        artifact_steps[error_type] = debris_rows
        print(f"  [hsmm/{model['kind']}] {error_type}: {len(rows)} degraded trials", flush=True)

    return element_metrics.evaluate_steps(
        healthy_verdicts, degraded_by_type, artifact_steps=artifact_steps, unit=unit,
        tol_steps=element_metrics.DEFAULT_TOL[unit] if tol is None else tol,
    )


# --------------------------------------------------------------------------------------------
# the LLM arm
# --------------------------------------------------------------------------------------------

def llm_arm(pool, lexicon, client, system_prompt, vocab, protocol, tag, unit="step", tol=None):
    healthy_verdicts = []
    for traj, _ in pool:
        steps = textify.elements_from_trajectory(traj, lexicon, unit)
        verdicts = detect.run_trial(client, system_prompt, steps, vocab, protocol, unit)
        healthy_verdicts.append(element_metrics.from_llm_verdicts(verdicts))
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
            verdicts = detect.run_trial(client, system_prompt, steps, vocab, protocol, unit)
            rows.append((element_metrics.from_llm_verdicts(verdicts), gt_steps, correction))
            debris_rows.append(debris)
        degraded_by_type[error_type] = rows
        artifact_steps[error_type] = debris_rows
        print(f"  [{tag}] {error_type}: {len(rows)} degraded trials "
              f"({client.n_would_request} uncached requests so far)", flush=True)

    return element_metrics.evaluate_steps(
        healthy_verdicts, degraded_by_type, artifact_steps=artifact_steps, unit=unit,
        tol_steps=element_metrics.DEFAULT_TOL[unit] if tol is None else tol,
    )


def pool_request_cost(pool, lexicon, protocol, unit="step"):
    """UPPER BOUND on the requests a sweep over this pool costs, for --dry-run and the budget line.

    A bound rather than an estimate: prefix-only requests cache on the whole prompt, and the six
    conditions share prefixes (a substitution rewrites one tick mid-trial, so every earlier step
    renders identically to the healthy trial's). Measured on 10 real trials: 398 predicted, 251
    actually sent. Budget against this and expect to spend less.
    """
    total = 0
    for traj, degraded in pool:
        total += detect.request_cost(
            textify.elements_from_trajectory(traj, lexicon, unit), protocol)
        for error_type in error_injection.ERROR_TYPES:
            steps = textify.elements_from_trajectory(degraded[error_type], lexicon, unit)
            total += detect.request_cost(steps, protocol)
    return total


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
    parser.add_argument("--protocol", choices=tuple(detect.PROTOCOLS), default="incremental")
    parser.add_argument("--unit", choices=textify.UNITS, default="step",
                        help="the element both arms answer about and are scored on. 'step' is one "
                             "run-length-encoded step (~6.5 per trial, the historical default); "
                             "'tick' is one second (~121 per trial), which removes the step "
                             "layer's handicap on the HSMM but costs ~18x the requests and is NOT "
                             "precision-comparable to a step-unit run -- read chance_precision")
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
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--min-run", type=int, default=10,
                        help="tick-level persistence requirement applied BEFORE collapsing HSMM "
                             "flags to steps; see element_metrics.step_verdicts_from_flags for "
                             "why a plain per-step OR without it is not a fair reading of the "
                             "HSMM")
    args = parser.parse_args()

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
        n_variants = 0 if args.skip_llm else (2 if args.variant == "both" else 1)
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
        variant = prompts.VARIANTS[-1] if args.variant == "both" else args.variant
        sys_words = len(prompts.build_variant(variant, vocab, labels, args.protocol,
                                              args.unit).split())
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
        print(f"\n----- system prompt ({variant}) -----")
        print(prompts.build_variant(variant, vocab, labels, args.protocol, args.unit))
        print(f"\n----- first trial as {args.unit}s ({name}) -----")
        for line in textify.render_trial(
                textify.elements_from_trajectory(traj, lexicon, args.unit), args.unit):
            print("   ", line)
        print("\ndry run: no requests made.")
        return

    # ---- score ----------------------------------------------------------------------------
    reports = {}
    variants = [] if args.skip_llm else (
        list(prompts.VARIANTS) if args.variant == "both" else [args.variant]
    )

    for name, pool in pools.items():
        if not args.skip_hsmm:
            print(f"\n[{name}] HSMM arm ({model['kind']})")
            report = hsmm_arm(pool, model, lexicon, args.chunk_size, args.alpha,
                              args.unit, args.tol)
            reports[f"{name}/hsmm-{model['kind']}"] = report
            print_report(report, f"{name} / hsmm-{model['kind']}")
            plotting.save_step_figures(report, args.figures_dir, f"{name}_hsmm_{model['kind']}")

        for variant in variants:
            tag = f"{name}/llm-{variant}"
            print(f"\n[{name}] LLM arm ({variant})")
            client = llm_client.ChatClient(
                model=args.model, base_url=args.base_url, api_key_env=args.api_key_env,
                cache_dir=Path(args.cache_dir) / args.model.replace("/", "_"),
                rpm=args.rpm, max_requests=args.max_requests, dry_run=False,
                concurrency=args.concurrency,
            )
            print(f"  client: rpm={client.rpm} concurrency={client.concurrency}", flush=True)
            system_prompt = prompts.build_variant(variant, vocab, labels, args.protocol,
                                                  args.unit)
            try:
                report = llm_arm(pool, lexicon, client, system_prompt, vocab, args.protocol, tag,
                                 args.unit, args.tol)
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
            reports[tag] = report

            note = "\n".join(filter(None, [
                WITH_RECIPES_CAVEAT if variant == "with-recipes" else None,
                BATCH_CAVEAT if args.protocol == "batch" else None,
            ]))
            print_report(report, tag, note or None)
            plotting.save_step_figures(report, args.figures_dir, f"{name}_llm_{variant}")

    _write_report(args, reports)


if __name__ == "__main__":
    main()
