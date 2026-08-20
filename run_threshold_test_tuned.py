"""Score ONE fixed alpha configuration on the held-out test split -- no sweep, no search.

Confirms (or not) that the F1 gain run_threshold_sweep_coordinate.py found on the TRAIN split by
loosening s_dur_two to 5e-2 (everything else left at surprise.DEFAULT_ALPHA) survives on data
that configuration never touched. The tuned alpha values are fixed constants here, chosen on the
train split by a prior run -- nothing in this script searches over alpha or looks at test-split
outcomes to pick anything.

    python run_threshold_test_tuned.py
"""
import argparse
import json
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from cook_ad.anomaly import narrate
from cook_ad.data.config import load_config
from cook_ad.data import split as split_mod
from cook_ad.eval import batch
from cook_ad.hsmm import joint_params
from cook_ad.synthetic import error_injection, generate
from run_threshold_sweep_coordinate import CHANNELS, DEFAULT_ALPHA, _prepare_trial, score_union

# Fixed by the TRAIN-split coordinate-descent run (run_threshold_sweep_coordinate.py), not by
# anything here. s_dur_two loosened from the shared default to where that run's train-split
# sweep found the union's F1 peaked (5e-2); every other channel unchanged.
TUNED_ALPHAS = {ch: DEFAULT_ALPHA for ch in CHANNELS}
TUNED_ALPHAS["s_dur_two"] = 5e-2


def _f1(precision, recall):
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def _print_row(label, scored):
    tl = scored["trial_loc"]
    fpr = scored["trial"]["fpr"]
    f1 = _f1(tl["precision"], tl["recall"])
    print(f"{label:>28}  precision={tl['precision']:.3f}  recall={tl['recall']:.3f}  "
          f"healthy_fpr={fpr:.3f}  F1={f1:.3f}")
    return {"precision": tl["precision"], "recall": tl["recall"], "healthy_fpr": fpr, "f1": f1}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/breakfast.yaml")
    ap.add_argument("--joint-params", default="dataset/processed/breakfast/joint_params_train.npz")
    ap.add_argument("--sequences", default="dataset/processed/breakfast/sequences.json")
    ap.add_argument("--vocab", default="dataset/processed/breakfast/vocab.json")
    ap.add_argument("--split-file", default="dataset/processed/breakfast/split.json")
    ap.add_argument("--split-part", default="test", choices=["train", "test"])
    ap.add_argument("--max-real", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk-size", type=int, default=8)
    ap.add_argument("--out", default="dataset/processed/breakfast/threshold_test_tuned.json")
    args = ap.parse_args()

    d_max = load_config(args.config)["duration"]["d_max_ticks"]
    vocab = json.load(open(args.vocab))
    jp = joint_params.load_params(args.joint_params)
    marg = joint_params.collapse_to_marginal(jp)
    lexicon = narrate.Lexicon(vocab, marg)

    seqs = json.load(open(args.sequences))
    split = split_mod.load_split(args.split_file)
    seqs = split_mod.filter_sequences(seqs, split, args.split_part)
    seqs = seqs[: args.max_real]

    traj = [generate.trajectory_from_real_joint(jp, s["verb_ids"], s["noun_ids"], d_max) for s in seqs]
    usable = [t for t in traj if len(t["segments"]) >= error_injection.MIN_SEGMENTS]
    print(f"{len(usable)} usable real trials ({args.split_part} split, held out from training)", flush=True)

    rng = np.random.default_rng(args.seed)
    degraded_by_type = {
        et: [error_injection.inject(et, t, rng, marg) for t in usable]
        for et in error_injection.ERROR_TYPES
    }

    groups = {}
    traces, joint_log_probs, r_hat, log_trans_marginal = batch.compute_traces_joint(
        jp, usable, d_max, chunk_size=args.chunk_size
    )
    statics = [_prepare_trial(t, None, lexicon, is_degraded=False) for t in usable]
    groups["healthy"] = list(zip(traces, [int(x) for x in r_hat], statics))

    for et in error_injection.ERROR_TYPES:
        deg_trials = degraded_by_type[et]
        traces, joint_log_probs, r_hat, log_trans_marginal = batch.compute_traces_joint(
            jp, deg_trials, d_max, chunk_size=args.chunk_size
        )
        statics = [_prepare_trial(t, d, lexicon, is_degraded=True) for t, d in zip(usable, deg_trials)]
        groups[et] = list(zip(traces, [int(x) for x in r_hat], statics))

    default_alphas = {ch: DEFAULT_ALPHA for ch in CHANNELS}
    baseline = score_union(groups, joint_log_probs, log_trans_marginal, default_alphas)
    tuned = score_union(groups, joint_log_probs, log_trans_marginal, TUNED_ALPHAS)

    print()
    b = _print_row("baseline (all default)", baseline)
    t = _print_row("tuned (s_dur_two=5e-2)", tuned)
    print(f"\ndelta:  precision {t['precision'] - b['precision']:+.3f}  "
          f"recall {t['recall'] - b['recall']:+.3f}  "
          f"healthy_fpr {t['healthy_fpr'] - b['healthy_fpr']:+.3f}  "
          f"F1 {t['f1'] - b['f1']:+.3f}")

    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "tuned_alphas": TUNED_ALPHAS,
                   "baseline": b, "tuned": t}, f, indent=2)
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
