"""Per-channel alpha, coordinate-descent-style, measured against the UNIONED detector.

run_threshold_sweep_by_channel's summary (render_threshold_sweep_by_channel.py) scores each
surprise channel ALONE -- useful for ranking channels by standalone PR-AUC, but it can't say
what a per-channel alpha would do to the real detector, which unions all seven. A channel's
marginal value depends on how much it overlaps with what the other six already catch, which an
isolated sweep cannot see.

This holds six channels at surprise.DEFAULT_ALPHA and sweeps the seventh's own alpha, re-scoring
the FULL UNION (all seven channels, one varied) at each point. One coordinate-descent pass --
each channel swept independently against the all-default baseline, not a joint 7-dimensional
search, which would be intractable and almost certainly overfit at this sample size. Traces are
computed ONCE per source group (the expensive JAX part); the sweep is cheap re-flagging, built
directly from cook_ad.anomaly.quantile's per-channel threshold functions rather than
surprise.flag_joint's single shared alpha.

    python run_threshold_sweep_coordinate.py --split-file dataset/processed/breakfast/split.json \
        --split-part train --joint-params dataset/processed/breakfast/joint_params_train.npz

WHICH GRANULARITY THIS OPTIMISES, because the answer used to be implicit and it matters. Four are
computed and stored -- tick, step, trial, trial_loc -- but only one is printed, and that is the
one a human reads the alpha off. It is now `tick`.

The earlier version printed `trial_loc`, and a `trial_loc` sweep is blind to exactly the quantity
the tick-unit evaluation reports: one trial is one test, so a trial with 1 stray flag and a trial
with 60 score identically, and alpha comes under no pressure at all to reduce alarm VOLUME
(tools_alarm_load.py exists because of that blindness). Calibrating on trial_loc and then quoting
per-tick precision -- which is what the first tick-unit run did -- reports a detector at an
operating point nothing selected for it.

GROUND TRUTH here is the points-and-debris rule (synthetic/error_injection._result,
llm/textify.gt_steps_for_ticks), NOT the range rule this script used to build with
gt_steps_for_window. Under the range rule a transposition's positives spanned ~30% of a trial, so
a sweep against it was calibrating toward a target that no longer exists.

The tick and step rows use eval/element_metrics.evaluate_steps' accounting exactly -- one test
per ground-truth window for recall, per-element false positives outside the injection-touched
extent, debris excluded from both -- so the alpha this sweep selects is optimal for the metric
run_llm_eval.py actually reports, rather than for a differently-shaped proxy.
"""
import argparse
import json
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from cook_ad.anomaly import narrate, quantile, surprise
from cook_ad.data.config import load_config
from cook_ad.data import split as split_mod
from cook_ad.eval import batch
from cook_ad.eval.element_metrics import DEFAULT_TICK_TOL
from cook_ad.hsmm import joint_params
from cook_ad.llm import textify
from cook_ad.synthetic import error_injection, generate

CHANNELS = surprise.CHANNELS
DEFAULT_ALPHA = surprise.DEFAULT_ALPHA

ALPHAS = sorted(
    {round(a, 12) for a in
     [0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001,
      5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]},
    reverse=True,
)


# The per-channel flagger lives in anomaly/surprise.py now, so this sweep and run_llm_eval.py
# cannot drift apart on what a per-channel alpha means.
_mixed_tables = surprise.threshold_tables_joint_per_channel
flag_joint_mixed = surprise.flag_joint_per_channel


def _union_mask(flags):
    mask = None
    for ch in CHANNELS:
        mask = flags[ch].copy() if mask is None else (mask | flags[ch])
    return mask


def _hit_mask(n_ticks, anomaly_ticks, tol):
    """Ticks where a flag counts as having FOUND the anomaly: each contiguous run of ground-truth
    points, extended by `tol` past its end.

    The same admissible region element_metrics._match_run defines, drawn as a mask. The tolerance
    is a detection DEADLINE, not a claim that all those ticks are anomalous -- which is why recall
    below is one test per ground-truth window rather than a per-tick count. Scoring each of the
    ~10 tolerance ticks as a positive that must be flagged would penalise a detector for firing
    once instead of ten times.
    """
    mask = np.zeros(n_ticks, dtype=bool)
    pts = sorted({int(t) for t in anomaly_ticks})
    if not pts:
        return mask
    start = prev = pts[0]
    for t in pts[1:] + [None]:
        if t is None or t != prev + 1:
            mask[start : min(prev + tol + 1, n_ticks)] = True
            if t is not None:
                start = t
        if t is not None:
            prev = t
    return mask


def _prepare_trial(traj, degraded, lexicon, is_degraded, tol=DEFAULT_TICK_TOL):
    """Per-trial masks, in TICK space, under the points-and-debris ground truth.

    Three regions rather than the old two, which is the whole change:
      hit      a flag here found the anomaly            (ground-truth runs + tol)
      excluded the injection disturbed it but it is not the anomaly -- a transposition's
               correctly-executed-but-misplaced runs. Charged as neither hit nor false alarm.
      stray    everything else. A flag here is a false positive.
    """
    v_ids = degraded["verb_ids"] if is_degraded else traj["verb_ids"]
    n_ids = degraded["noun_ids"] if is_degraded else traj["noun_ids"]
    n_ticks = len(v_ids)
    steps = textify.steps_from_ids(v_ids, n_ids, lexicon)

    ticks = textify.ticks_from_ids(v_ids, n_ids, lexicon)

    def _scoreable(elements):
        """element_metrics.evaluate_steps' `scoreable` set as a mask: elements that are neither
        ground truth nor debris, and so are the only ones a flag can be charged a false positive
        on. Built with the real functions rather than approximated as "outside the window" --
        substitution's debris includes the two elements BORDERING the edited segment, which lie
        outside the window, and treating those as scoreable put this 4 false positives adrift of
        the metric it is supposed to be sweeping.
        """
        mask = np.ones(len(elements), dtype=bool)
        if not is_degraded:
            return mask
        gt = textify.gt_steps_for_ticks(elements, degraded["anomaly_ticks"])
        debris = textify.injection_touched_steps(
            elements, degraded["tick_map"], degraded["edited_ticks"], gt, window=degraded["window"]
        )
        for i in set(gt) | set(debris):
            mask[i] = False
        return mask

    hit = (_hit_mask(n_ticks, degraded["anomaly_ticks"], tol) if is_degraded
           else np.zeros(n_ticks, dtype=bool))
    stray = _scoreable(ticks)                  # tick element i IS tick i, so this is a tick mask
    step_stray = _scoreable(steps)
    step_hit = np.array([bool(hit[s.tick_start : s.tick_end].any()) for s in steps])

    return {"n_ticks": n_ticks, "steps": steps, "hit": hit, "stray": stray,
            "step_hit": step_hit, "step_stray": step_stray, "is_degraded": is_degraded}


def _accumulate(counts, mask, static, gt_trial_positive):
    """element_metrics.evaluate_steps' step_level accounting, in numpy.

    Recall is ONE test per ground-truth window: a degraded trial contributes exactly one tp or one
    fn, whichever way its detection went. False positives are counted per ELEMENT, over the
    elements that are neither the anomaly nor debris. `tn` is the remaining scoreable elements,
    carried only so accuracy and fpr stay defined; precision and recall never use it.
    """
    hit, stray = static["hit"], static["stray"]

    c = counts["tick"]
    if gt_trial_positive:
        detected = bool((hit & mask).any())
        c[0] += int(detected)
        c[3] += int(not detected)
    flagged_strays = int((stray & mask).sum())
    c[2] += flagged_strays
    c[1] += int(stray.sum()) - flagged_strays

    step_hit, step_stray = static["step_hit"], static["step_stray"]
    steps = static["steps"]
    step_pred = np.array([bool(mask[s.tick_start : s.tick_end].any()) for s in steps])
    c = counts["step"]
    if gt_trial_positive:
        detected = bool((step_hit & step_pred).any())
        c[0] += int(detected)
        c[3] += int(not detected)
    flagged_strays = int((step_stray & step_pred).sum())
    c[2] += flagged_strays
    c[1] += int(step_stray.sum()) - flagged_strays

    any_flag = bool(mask.any())
    c = counts["trial"]
    if gt_trial_positive:
        c[0 if any_flag else 3] += 1
    else:
        c[2 if any_flag else 1] += 1

    c = counts["trial_loc"]
    if gt_trial_positive:
        found = bool((hit & mask).any())
        strayed = bool((stray & mask).any())
        c[0 if found else 3] += 1
        if strayed:
            c[2] += 1
    else:
        c[2 if any_flag else 1] += 1


def _acc(c):
    tp, tn, fp, fn = c
    tot = tp + tn + fp + fn
    return {
        # The base rate of anomalous elements. Precision at this granularity has to be read
        # against it -- a detector that flags everything scores it for free -- and it is the same
        # for every alpha and every channel, so it is the fixed yardstick for the whole sweep.
        "chance_precision": (tp + fn) / tot if tot else float("nan"),
        "accuracy": (tp + tn) / tot if tot else float("nan"),
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def score_union(groups, joint_log_probs, log_trans_marginal, alphas):
    counts = {level: [0, 0, 0, 0] for level in ("tick", "step", "trial", "trial_loc")}
    for group_name, trials in groups.items():
        gt_trial_positive = group_name != "healthy"
        for trace, rh, static in trials:
            flags = flag_joint_mixed(trace, joint_log_probs, rh, log_trans_marginal, alphas)
            mask = _union_mask(flags)
            _accumulate(counts, mask, static, gt_trial_positive)
    return {level: _acc(counts[level]) for level in ("tick", "step", "trial", "trial_loc")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/breakfast.yaml")
    ap.add_argument("--joint-params", default="dataset/processed/breakfast/joint_params_train.npz")
    ap.add_argument("--sequences", default="dataset/processed/breakfast/sequences.json")
    ap.add_argument("--vocab", default="dataset/processed/breakfast/vocab.json")
    ap.add_argument("--max-real", type=int, default=402)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk-size", type=int, default=8)
    ap.add_argument("--split-file", default="dataset/processed/breakfast/split.json")
    ap.add_argument("--split-part", default="train", choices=["train", "test"])
    ap.add_argument("--tol", type=int, default=DEFAULT_TICK_TOL,
                    help="detection deadline in ticks past each ground-truth run "
                         "(element_metrics.DEFAULT_TICK_TOL)")
    ap.add_argument("--out", default="dataset/processed/breakfast/threshold_sweep_coordinate.json")
    args = ap.parse_args()

    d_max = load_config(args.config)["duration"]["d_max_ticks"]
    vocab = json.load(open(args.vocab))
    jp = joint_params.load_params(args.joint_params)
    marg = joint_params.collapse_to_marginal(jp)
    lexicon = narrate.Lexicon(vocab, marg)

    seqs = json.load(open(args.sequences))
    if args.split_file:
        split = split_mod.load_split(args.split_file)
        seqs = split_mod.filter_sequences(seqs, split, args.split_part)
    seqs = seqs[: args.max_real]

    traj = generate.trajectories_from_real_joint(jp, seqs, d_max, chunk_size=8)
    usable = [t for t in traj if len(t["segments"]) >= error_injection.MIN_SEGMENTS]
    print(f"{len(usable)} usable real trials ({args.split_part} split)", flush=True)

    rng = np.random.default_rng(args.seed)
    degraded_by_type = {
        et: [error_injection.inject(et, t, rng, marg) for t in usable]
        for et in error_injection.ERROR_TYPES
    }

    # ---- trace computation: ONCE per group (the expensive JAX part) --------------------------
    groups = {}
    print("computing traces: healthy", flush=True)
    traces, joint_log_probs, r_hat, log_trans_marginal = batch.compute_traces_joint(
        jp, usable, d_max, chunk_size=args.chunk_size
    )
    statics = [_prepare_trial(t, None, lexicon, is_degraded=False, tol=args.tol) for t in usable]
    groups["healthy"] = list(zip(traces, [int(x) for x in r_hat], statics))

    for et in error_injection.ERROR_TYPES:
        print(f"computing traces: {et}", flush=True)
        deg_trials = degraded_by_type[et]
        traces, joint_log_probs, r_hat, log_trans_marginal = batch.compute_traces_joint(
            jp, deg_trials, d_max, chunk_size=args.chunk_size
        )
        statics = [_prepare_trial(t, d, lexicon, is_degraded=True, tol=args.tol)
                   for t, d in zip(usable, deg_trials)]
        groups[et] = list(zip(traces, [int(x) for x in r_hat], statics))
    print("traces done; sweeping alpha per channel against the union (cheap re-flagging only)", flush=True)

    # ---- baseline: every channel at DEFAULT_ALPHA --------------------------------------------
    default_alphas = {ch: DEFAULT_ALPHA for ch in CHANNELS}
    baseline = score_union(groups, joint_log_probs, log_trans_marginal, default_alphas)
    bt = baseline["tick"]
    print(f"baseline (all channels at alpha={DEFAULT_ALPHA:g}): "
          f"TICK precision={bt['precision']:.4f} recall={bt['recall']:.3f} "
          f"(chance {bt['chance_precision']:.4f}) | "
          f"trial_loc precision={baseline['trial_loc']['precision']:.3f} "
          f"recall={baseline['trial_loc']['recall']:.3f} "
          f"healthy_fpr={baseline['trial']['fpr']:.3f}", flush=True)

    # ---- coordinate sweep: one channel varied, other six held at default ---------------------
    results = {"baseline": baseline, "per_channel": {}}
    for varied in CHANNELS:
        print(f"\nsweeping {varied} (others fixed at alpha={DEFAULT_ALPHA:g})", flush=True)
        rows = []
        for a in ALPHAS:
            alphas = {**default_alphas, varied: a}
            scored = score_union(groups, joint_log_probs, log_trans_marginal, alphas)
            rows.append({"alpha": a, **scored})
            tk, tl = scored["tick"], scored["trial_loc"]
            f1 = (2 * tk["precision"] * tk["recall"] / (tk["precision"] + tk["recall"])
                  if (tk["precision"] + tk["recall"]) else 0.0)
            print(f"  alpha={a:.2e}  TICK prec={tk['precision']:.4f} rec={tk['recall']:.3f} "
                  f"F1={f1:.4f} | trial_loc prec={tl['precision']:.3f} rec={tl['recall']:.3f} "
                  f"healthy_fpr={scored['trial']['fpr']:.3f}", flush=True)
        results["per_channel"][varied] = rows

    # ---- do the per-channel wins COMPOSE? ---------------------------------------------------
    # A coordinate pass measures each channel against the all-default baseline and nothing else,
    # so seven individually-better alphas are seven separate claims, not one joint result. Two
    # channels whose gains come from suppressing the SAME false positives would double-count.
    # Scoring the argmax combination is what turns the pass into a proposal.
    def _f1(row):
        p_, r_ = row["precision"], row["recall"]
        return 2 * p_ * r_ / (p_ + r_) if (p_ + r_) else 0.0

    combined = {ch: max(results["per_channel"][ch], key=lambda r: _f1(r["tick"]))["alpha"]
                for ch in CHANNELS}
    combined_scored = score_union(groups, joint_log_probs, log_trans_marginal, combined)
    results["combined"] = {"alphas": combined, **combined_scored}
    ct, bt = combined_scored["tick"], baseline["tick"]
    print(f"\ncombined (per-channel argmax on tick F1): "
          f"{ {k: f'{v:.0e}' for k, v in combined.items()} }")
    print(f"  tick  P={ct['precision']:.4f} R={ct['recall']:.3f} F1={_f1(ct):.4f}   "
          f"(baseline F1={_f1(bt):.4f}, sum of individual gains would be "
          f"{sum(_f1(max(results['per_channel'][ch], key=lambda r: _f1(r['tick']))['tick']) - _f1(bt) for ch in CHANNELS):+.4f})")
    print(f"  trial_loc P={combined_scored['trial_loc']['precision']:.3f} "
          f"R={combined_scored['trial_loc']['recall']:.3f}  "
          f"healthy_fpr={combined_scored['trial']['fpr']:.3f}", flush=True)

    with open(args.out, "w") as f:
        json.dump({"config": vars(args), **results}, f, indent=2)
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
