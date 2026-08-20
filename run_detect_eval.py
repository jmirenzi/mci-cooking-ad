"""One-call detection scorecard for a fitted joint model, at the metric run_threshold_sweep.py
already defines: `trial_loc` (one verdict per trial, and WHERE the flag landed matters).

Same ground-truth convention as run_threshold_sweep.py -- positive range = injection window
UNION debris (textify.injection_touched_steps), stray flags charged independently of the hit --
so numbers from the two are directly comparable. What this adds is (a) the per-error-type and
per-channel breakdown in one pass, and (b) batched trajectory construction, so a full 402-trial
sweep is minutes rather than the better part of an hour.

    ./py run_detect_eval.py --joint-params <npz> --split-part train --tag mymodel
"""
import argparse
import json
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from cook_ad.anomaly import narrate, quantile, surprise
from cook_ad.data import split as split_mod
from cook_ad.data.config import load_config
from cook_ad.eval import batch
from cook_ad.hsmm import joint_params
from cook_ad.llm import textify
from cook_ad.synthetic import error_injection, generate

CHANNELS = surprise.CHANNELS
# Denser than run_threshold_sweep.py's grid between 1e-1 and 1e-3 on purpose: that is where
# trial_loc accuracy turns over, and the coarse grid resolves the turn to within a factor of
# 2.5 in alpha, which is wider than the differences between the models being compared.
ALPHAS = [0.5, 0.35, 0.2, 0.15, 0.1, 0.07, 0.05, 0.035, 0.02, 0.015, 0.01, 0.007, 0.005,
          0.003, 0.002, 1e-3, 5e-4, 1e-4, 1e-5, 1e-7, 1e-10]


def positive_ticks(degraded, lexicon):
    """Injection-touched extent for one degraded trial: ground-truth window UNION debris,
    expanded from textify.Step spans to a (T,) boolean tick mask -- byte-identical to what
    run_threshold_sweep.py's _prepare_trial builds."""
    steps = textify.steps_from_ids(degraded["verb_ids"], degraded["noun_ids"], lexicon)
    gt = textify.gt_steps_for_window(steps, degraded["window"])
    debris = textify.injection_touched_steps(steps, degraded["tick_map"], degraded["edited_ticks"], gt)
    pos = np.zeros(len(degraded["verb_ids"]), dtype=bool)
    for si in set(gt) | debris:
        s = steps[si]
        pos[s.tick_start : s.tick_end] = True
    return pos


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--joint-params", default="dataset/processed/breakfast/joint_params_train.npz")
    ap.add_argument("--config", default="configs/breakfast.yaml")
    ap.add_argument("--split-file", default="dataset/processed/breakfast/split.json")
    ap.add_argument("--split-part", choices=["train", "test"], default="train")
    ap.add_argument("--sequences", default="dataset/processed/breakfast/sequences.json")
    ap.add_argument("--vocab", default="dataset/processed/breakfast/vocab.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk-size", type=int, default=8)
    ap.add_argument("--max", type=int, default=10_000)
    ap.add_argument("--tag", default="eval")
    ap.add_argument("--out-dir", default="runs")
    args = ap.parse_args()

    d_max = load_config(args.config)["duration"]["d_max_ticks"]
    vocab = json.load(open(args.vocab))
    jp = joint_params.load_params(args.joint_params)
    marg = joint_params.collapse_to_marginal(jp)
    lexicon = narrate.Lexicon(vocab, marg)

    seqs = json.load(open(args.sequences))
    if args.split_file:
        seqs = split_mod.filter_sequences(seqs, split_mod.load_split(args.split_file), args.split_part)
    seqs = seqs[: args.max]

    traj = generate.trajectories_from_real_joint(jp, seqs, d_max, chunk_size=args.chunk_size)
    usable = [t for t in traj if len(t["segments"]) >= error_injection.MIN_SEGMENTS]
    print(f"[{args.tag}/{args.split_part}] {len(usable)}/{len(seqs)} usable trials", flush=True)

    rng = np.random.default_rng(args.seed)

    # groups[name] = list of (trace, r_hat, positive_tick_mask)
    groups = {}
    traces, log_probs, r_hat, log_trans_marginal = batch.compute_traces_joint(
        jp, usable, d_max, chunk_size=args.chunk_size
    )
    groups["healthy"] = [
        (t, int(r), np.zeros(len(u["verb_ids"]), dtype=bool)) for t, r, u in zip(traces, r_hat, usable)
    ]
    for et in error_injection.ERROR_TYPES:
        deg = [error_injection.inject(et, t, rng, marg) for t in usable]
        tr, lp2, rh, ltm = batch.compute_traces_joint(jp, deg, d_max, chunk_size=args.chunk_size)
        groups[et] = [(t, int(r), positive_ticks(d, lexicon)) for t, r, d in zip(tr, rh, deg)]
        # every group is scored against the SAME tables it was traced with
        log_probs, log_trans_marginal = lp2, ltm
        print(f"  traced {et}", flush=True)

    # log_probs / log_trans_marginal are model-level (not group-level): identical across groups
    # since the params never change here -- rebinding above is just the last group's copy.

    cache = quantile.JointThresholdCache(log_probs, log_trans_marginal)
    results = []
    for alpha in ALPHAS:
        # flags[group][trial][channel]
        flags = {
            g: [surprise.flag_joint_cached(cache, t, log_probs, r, log_trans_marginal, alpha=alpha)
                for t, r, _ in trials]
            for g, trials in groups.items()
        }

        def score(channels):
            """trial_loc, exactly as run_threshold_sweep.py defines it: a degraded trial scores
            TP if any flag lands inside its injection-touched range and FN otherwise, and is
            charged an FP independently for any flag outside it; a healthy trial has no range,
            so any flag is an FP. TP + FN is therefore the degraded-trial count and recall stays
            clean, while FP pools strays on degraded trials with alarms on healthy ones."""
            tp = tn = fp = fn = 0
            per_type = {}
            for g, trials in groups.items():
                hits = strays = 0
                for (_t, _r, pos), f in zip(trials, flags[g]):
                    m = np.zeros(len(pos), dtype=bool)
                    for c in channels:
                        m |= np.asarray(f[c])
                    if g == "healthy":
                        if m.any():
                            fp += 1
                            strays += 1
                        else:
                            tn += 1
                    else:
                        hit = bool((pos & m).any())
                        stray = bool((~pos & m).any())
                        tp += hit
                        fn += not hit
                        fp += stray
                        hits += hit
                        strays += stray
                per_type[g] = {"recall": hits / len(trials), "stray": strays / len(trials)}
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            return {
                "accuracy": (tp + tn) / (tp + tn + fp + fn),
                "precision": prec,
                "recall": rec,
                "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
                "fpr": fp / (fp + tn) if fp + tn else 0.0,
                "tp": tp, "tn": tn, "fp": fp, "fn": fn,
                "per_type": per_type,
            }

        row = {"alpha": alpha, "raw": score(CHANNELS)}
        for c in CHANNELS:
            row[c] = score([c])
        results.append(row)
        r = row["raw"]
        pt = " ".join(f"{k[:4]}={v['recall']:.2f}" for k, v in r["per_type"].items() if k != "healthy")
        print(
            f"alpha={alpha:.0e} acc={r['accuracy']:.3f} prec={r['precision']:.3f} "
            f"rec={r['recall']:.3f} f1={r['f1']:.3f} healthyFP={r['per_type']['healthy']['stray']:.3f} | {pt}",
            flush=True,
        )

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"detect_{args.tag}_{args.split_part}.json")
    with open(out, "w") as f:
        json.dump({"config": vars(args), "results": results}, f, indent=2)
    best = max(results, key=lambda r: r["raw"]["f1"])
    print(f"\nbest-F1 alpha={best['alpha']:.0e}: acc={best['raw']['accuracy']:.3f} "
          f"prec={best['raw']['precision']:.3f} rec={best['raw']['recall']:.3f} f1={best['raw']['f1']:.3f}")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
