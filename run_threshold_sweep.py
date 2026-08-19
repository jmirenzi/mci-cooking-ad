"""Detection accuracy as a function of the surprise threshold alpha.

Traces (the expensive JAX part: predictive occupancy + Viterbi segmentation) are computed ONCE
per source group (healthy + each of the 5 injected error types). `surprise.flag_joint`'s alpha
argument only changes cheap post-hoc thresholding, so the whole alpha grid is swept by
re-flagging the SAME traces -- no JAX recompute per alpha.

Reports accuracy, precision, recall and false-positive rate at four granularities:

    tick        every tick is a test
    step        every llm/textify.Step is a test
    trial       every trial is a test, on "did it flag anywhere at all"
    trial_loc   every trial is a test, on "did it flag in the RIGHT PLACE"  <- the useful one

`trial_loc` is the deployment-shaped question and the one to read. A trial's positive range is
the whole injection-touched extent (ground-truth window UNION debris). A flag inside it is a hit;
a flag outside it is a stray, charged as a false positive independently of whether the trial was
also hit -- so finding the anomaly does not buy absolution for firing five more times elsewhere.
A degraded trial with no in-range flag is a miss. Healthy trials have no range: any flag is a
false positive. TP + FN is exactly the degraded-trial count, so recall stays clean.

`trial` is kept alongside it only as the degenerate comparison -- it scores a detector that flags
the wrong place identically to one that flags the right place, which is why it reads far more
favourably than `trial_loc` and should not be quoted on its own.

Ground truth treats injection DEBRIS AS ANOMALOUS (the opposite of eval/element_metrics.py's
scoring, which excludes it): every tick/step touched by the injection -- the ground-truth window
AND any debris step the injection created (textify.injection_touched_steps) -- counts as a
positive. Debris is genuinely something the injection moved, so flagging it is detecting the
injection's effect; excluding it, as the detection metrics do, answers the narrower question of
whether the detector found the anomaly the injector was specifically testing for.

A correctly calibrated channel's false-positive rate falls as alpha tightens; one whose FPR is
flat across orders of magnitude of alpha is not being gated by its threshold at all, which
localises the problem to that channel.
"""
import argparse
import json
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from cook_ad.anomaly import narrate, surprise
from cook_ad.data.config import load_config
from cook_ad.eval import batch
from cook_ad.hsmm import joint_params
from cook_ad.llm import textify
from cook_ad.synthetic import error_injection, generate

CHANNELS = surprise.CHANNELS


def _union_mask(flags, channels):
    mask = None
    for ch in channels:
        mask = flags[ch].copy() if mask is None else (mask | flags[ch])
    return mask


def _prepare_trial(traj, degraded, lexicon, is_degraded):
    """One trial's static (alpha-independent) bookkeeping: tick count, textify steps, and (for
    degraded trials) the positive tick/step masks -- ground-truth window UNION debris."""
    v_ids = degraded["verb_ids"] if is_degraded else traj["verb_ids"]
    n_ticks = len(v_ids)
    steps = textify.steps_from_ids(v_ids, degraded["noun_ids"] if is_degraded else traj["noun_ids"], lexicon)

    pos_ticks = np.zeros(n_ticks, dtype=bool)
    pos_steps = np.zeros(len(steps), dtype=bool)
    if is_degraded:
        gt_steps = textify.gt_steps_for_window(steps, degraded["window"])
        debris = textify.injection_touched_steps(
            steps, degraded["tick_map"], degraded["edited_ticks"], gt_steps
        )
        positive_step_idx = set(gt_steps) | debris
        for si in positive_step_idx:
            s = steps[si]
            pos_ticks[s.tick_start : s.tick_end] = True
            pos_steps[si] = True

    return {"n_ticks": n_ticks, "steps": steps, "pos_ticks": pos_ticks, "pos_steps": pos_steps}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/breakfast.yaml")
    ap.add_argument("--joint-params", default="dataset/processed/breakfast/joint_params.npz")
    ap.add_argument("--sequences", default="dataset/processed/breakfast/sequences.json")
    ap.add_argument("--vocab", default="dataset/processed/breakfast/vocab.json")
    ap.add_argument("--max-real", type=int, default=447)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk-size", type=int, default=8)
    ap.add_argument("--out", default="dataset/processed/breakfast/threshold_sweep.json")
    args = ap.parse_args()

    d_max = load_config(args.config)["duration"]["d_max_ticks"]
    vocab = json.load(open(args.vocab))
    jp = joint_params.load_params(args.joint_params)
    marg = joint_params.collapse_to_marginal(jp)
    lexicon = narrate.Lexicon(vocab, marg)

    seqs = json.load(open(args.sequences))[: args.max_real]
    traj = [generate.trajectory_from_real_joint(jp, s["verb_ids"], s["noun_ids"], d_max) for s in seqs]
    usable = [t for t in traj if len(t["segments"]) >= error_injection.MIN_SEGMENTS]
    print(f"{len(usable)} usable real trials", flush=True)

    rng = np.random.default_rng(args.seed)
    degraded_by_type = {
        et: [error_injection.inject(et, t, rng, marg) for t in usable]
        for et in error_injection.ERROR_TYPES
    }

    # ---- trace computation: ONCE per group (the expensive JAX part) --------------------------
    groups = {}
    print("computing traces: healthy", flush=True)
    traces, log_probs, r_hat, log_trans_marginal = batch.compute_traces_joint(
        jp, usable, d_max, chunk_size=args.chunk_size
    )
    statics = [_prepare_trial(t, None, lexicon, is_degraded=False) for t in usable]
    groups["healthy"] = list(zip(traces, [int(x) for x in r_hat], statics))

    for et in error_injection.ERROR_TYPES:
        print(f"computing traces: {et}", flush=True)
        deg_trials = degraded_by_type[et]
        traces, log_probs, r_hat, log_trans_marginal = batch.compute_traces_joint(
            jp, deg_trials, d_max, chunk_size=args.chunk_size
        )
        statics = [
            _prepare_trial(t, d, lexicon, is_degraded=True) for t, d in zip(usable, deg_trials)
        ]
        groups[et] = list(zip(traces, [int(x) for x in r_hat], statics))
    print("traces done; sweeping alpha (cheap re-flagging only)", flush=True)

    # ---- alpha sweep: cheap re-flagging ------------------------------------------------------
    alphas = sorted(
        {round(a, 12) for a in
         [0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001,
          5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]},
        reverse=True,
    )

    results = []
    for alpha in alphas:
        counts = {"raw": {"tick": [0, 0, 0, 0], "step": [0, 0, 0, 0],
                          "trial": [0, 0, 0, 0], "trial_loc": [0, 0, 0, 0]}}
        # each [tp, tn, fp, fn]

        def _accumulate(mask, static, gt_trial_positive, bucket):
            pos_ticks = static["pos_ticks"]
            c = counts[bucket]["tick"]
            c[0] += int((pos_ticks & mask).sum())
            c[3] += int((pos_ticks & ~mask).sum())
            c[2] += int((~pos_ticks & mask).sum())
            c[1] += int((~pos_ticks & ~mask).sum())

            pos_steps = static["pos_steps"]
            steps = static["steps"]
            step_pred = np.array([bool(mask[s.tick_start : s.tick_end].any()) for s in steps])
            c = counts[bucket]["step"]
            c[0] += int((pos_steps & step_pred).sum())
            c[3] += int((pos_steps & ~step_pred).sum())
            c[2] += int((~pos_steps & step_pred).sum())
            c[1] += int((~pos_steps & ~step_pred).sum())

            any_flag = bool(mask.any())
            c = counts[bucket]["trial"]
            if gt_trial_positive:
                c[0 if any_flag else 3] += 1
            else:
                c[2 if any_flag else 1] += 1

            # trial_loc: one verdict per trial, but WHERE the flag landed matters. The positive
            # range is the whole injection-touched extent -- ground-truth window UNION debris --
            # since every tick in it is one the injection actually moved.
            #
            #   degraded: any flag inside the range -> TP, else -> FN
            #             any flag outside the range -> FP, counted INDEPENDENTLY of the TP, so a
            #             detector that finds the anomaly and also fires elsewhere is charged for
            #             the stray rather than having it absorbed by the hit
            #   healthy:  no range exists; any flag -> FP, none -> TN
            #
            # TP + FN is exactly the degraded-trial count, so recall is clean; FP pools strays on
            # degraded trials with flags on healthy ones.
            c = counts[bucket]["trial_loc"]
            if gt_trial_positive:
                hit = bool((pos_ticks & mask).any())
                stray = bool((~pos_ticks & mask).any())
                c[0 if hit else 3] += 1
                if stray:
                    c[2] += 1
            else:
                c[2 if any_flag else 1] += 1

        for group_name, trials in groups.items():
            gt_trial_positive = group_name != "healthy"
            for trace, rh, static in trials:
                flags = surprise.flag_joint(trace, log_probs, rh, log_trans_marginal, alpha=alpha)
                raw_mask = _union_mask(flags, CHANNELS)
                _accumulate(raw_mask, static, gt_trial_positive, "raw")

        def _acc(c):
            tp, tn, fp, fn = c
            tot = tp + tn + fp + fn
            return {
                "accuracy": (tp + tn) / tot if tot else float("nan"),
                "precision": tp / (tp + fp) if (tp + fp) else 0.0,
                "recall": tp / (tp + fn) if (tp + fn) else 0.0,
                "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
                "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            }

        row = {"alpha": alpha}
        for level in ("tick", "step", "trial", "trial_loc"):
            row[f"raw_{level}"] = _acc(counts["raw"][level])
        results.append(row)
        print(
            f"alpha={alpha:.2e}  "
            f"tick acc={row['raw_tick']['accuracy']:.3f} prec={row['raw_tick']['precision']:.3f} "
            f"rec={row['raw_tick']['recall']:.3f}  |  "
            f"step acc={row['raw_step']['accuracy']:.3f} prec={row['raw_step']['precision']:.3f} "
            f"rec={row['raw_step']['recall']:.3f}  |  "
            f"trial_loc prec={row['raw_trial_loc']['precision']:.3f} "
            f"rec={row['raw_trial_loc']['recall']:.3f} fpr={row['raw_trial_loc']['fpr']:.3f}",
            flush=True,
        )

    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "results": results}, f, indent=2)
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
