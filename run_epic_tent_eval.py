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


def predictive_flags_per_trial(trajectories, lexicon, client, system_prompt, vocab, tag):
    """Arm 3 over whole trials. One request per step boundary -- ~57 per session here, against
    the ~2124 a tick-unit verdict arm would send on the longest one."""
    out = []
    totals = {"n_boundaries": 0, "duration_parse_failures": 0, "next_parse_failures": 0}
    for i, traj in enumerate(trajectories):
        steps = textify.steps_from_trajectory(traj, lexicon)
        flags, stats = predict.run_predictive(client, system_prompt, steps, vocab)
        for key in totals:
            totals[key] += stats[key]
        out.append(flags)
        print(f"  [{tag}] trial {i + 1}/{len(trajectories)}: {len(steps)} steps, "
              f"{len(flags)} flags ({client.n_would_request} uncached requests so far)",
              flush=True)
    boundaries = max(totals["n_boundaries"], 1)
    return out, {
        "n_boundaries": totals["n_boundaries"],
        "duration_parse_failure_rate": totals["duration_parse_failures"] / boundaries,
        "next_parse_failure_rate": totals["next_parse_failures"] / boundaries,
    }


def score(flags_by_trial, truth, group, merge_tol):
    """One arm's per-trial flags, scored against `group` with every other type removed."""
    trials, healthy, healthy_ticks, alarmed_ticks = [], [], 0, 0
    for flags, trial in zip(flags_by_trial, truth):
        spans, fragments = group_truth(trial, group)
        trials.append((flags, spans))
        for fragment in fragments:
            inside = slice_flags(flags, fragment)
            healthy.append(inside)
            alarmed_ticks += len({f if isinstance(f, (int, np.integer)) else f.tick
                                  for f in inside})
        healthy_ticks += sum(end - start + 1 for start, end in fragments)
    report = span_metrics.evaluate_spans_multi(
        healthy, trials, merge_tol=merge_tol, error_types=group,
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
    return report


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
    print(f"  {n_spans} spans in {len(truth)} sessions; every other error type REMOVED "
          f"(relabelled normal, as PREGO does)")
    print(f"  healthy population: {len(fragments)} error-free fragments, "
          f"{sum(b - a + 1 for a, b in fragments)} ticks")

    print(f"\n  {'arm':<38}{'prec':>7}{'recall':>8}{'F1':>7}{'alarms/10min':>14}"
          f"{'alarm time':>12}{'frag FPR':>10}{'lat med':>9}")
    for arm, r in reports.items():
        p, h, lat = r["pooled"], r["healthy"], r["latency"]
        print(f"  {arm:<38}{p['precision']:>7.3f}{p['recall']:>8.3f}{p['f1']:>7.3f}"
              f"{h['alarms_per_10min']:>14.2f}{h['alarm_time_frac']:>11.1%}"
              f"{h['false_positive_rate']:>10.3f}{_fmt(lat['median'], '.1f'):>9}")
    print("  On error-free activity only: alarms/10min = distinct (merged) alarms; alarm time =\n"
          "  share of seconds spent alarming. Read them together -- flag-everything has few\n"
          "  alarms because it is one alarm that never stops. frag FPR = share of healthy\n"
          "  fragments raising anything; it grows with fragment length, so compare WITHIN a group.")

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
    print(f"  {'TI-PREGO + Llama 3.1 (published)':<38}  precision 0.106  recall 0.935  F1 0.224")
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
    ap.add_argument("--skip-hsmm", action="store_true")
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

    # ---- every arm's flags, once -----------------------------------------------------------
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
            system_prompt = prompts.build_variant(
                variant, vocab, labels=labels, protocol=predict.PROTOCOL,
                exemplars=exemplars, domain=DOMAIN, sil_verb=sil_verb, sil_noun=sil_noun,
                canonical_order=CANONICAL_ORDER, interleaved=INTERLEAVED,
            )
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
            flags_by_arm[arm], stats = predictive_flags_per_trial(
                trajectories, lexicon, client, system_prompt, vocab, arm
            )
            meta[arm] = {"predictive": stats, "prompt_variant": variant,
                         "client": {"model": args.model, "base_url": args.base_url}}
        if "llm-predictive-with-recipes" in flags_by_arm:
            print("  NOTE: the with-recipes preprompt states the printed instruction order and\n"
                  "  per-step median durations taken from labels.json (all sessions, as on\n"
                  "  Breakfast). Read it against the no-recipes arm, not against the HSMM.")

    if args.dry_run:
        return

    # ---- score every arm under every group ---------------------------------------------------
    reports = {}
    for name in args.groups:
        group = ERROR_GROUPS[name]
        by_arm = {}
        for arm, flags in flags_by_arm.items():
            report = score(flags, truth, group, args.merge_tol)
            if not arm.startswith("control/"):
                report["attribution"] = attribution(flags, truth, group)
            report.update(meta.get(arm, {}))
            by_arm[arm] = report
        # The same flag-everything control with merging off. The gap between the two rows is the
        # metric's, not any detector's.
        by_arm["control/flag-everything@merge_tol=0"] = score(
            flags_by_arm["control/flag-everything"], truth, group, 0)
        print_group(name, group, by_arm, truth)
        if name == "procedural":
            print_prego_comparison(by_arm, args.split_file, args.split_part)
        for arm, report in by_arm.items():
            reports[f"{name}/{arm}"] = report

    # ---- the injected control -------------------------------------------------------------
    # Not group-specific: it asks whether the same detector on the same sessions finds INJECTED
    # anomalies more easily than annotated ones, which separates "this corpus is harder" from
    # "real errors are harder".
    if not args.skip_hsmm and not args.skip_injected:
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
            stream, R.spans_by_type(pool), merge_tol=args.merge_tol)
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
        "reports": reports,
    }
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=2, default=float)
    print(f"\nwrote {args.out}")

if __name__ == "__main__":
    main()
