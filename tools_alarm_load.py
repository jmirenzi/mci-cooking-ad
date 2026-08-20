"""How much false-positive mass does `trial_loc` collapse away?

`trial_loc` charges a trial ONE false positive if it flags outside the injection-touched range,
whether it does so once or fifty times, and one for a healthy trial that flags at all. Total FP
is therefore capped at (number of trials), which is exactly the quantity a nagging assistant is
not bounded by.

This counts what the trial-level verdict throws away: contiguous runs of the unioned flag mask,
which is the closest tick-level proxy for "one alarm the user would actually be shown"
(anomaly/narrate.py renders a run of flagged ticks as one Query card, not one per tick).

Reports, per model and per source group: the trial-level stray rate `trial_loc` charges, the
mean number of stray ALARMS behind it, and an event-level precision -- of every alarm this
detector raises anywhere, what fraction lands inside an injection-touched range.
"""
import argparse
import json

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


def runs(mask):
    """Number of maximal contiguous True runs -- one raised alarm each."""
    if not mask.any():
        return 0
    return int(np.count_nonzero(np.diff(np.concatenate([[False], mask, [False]]).astype(np.int8)) == 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--joint-params", required=True)
    ap.add_argument("--label", default=None)
    ap.add_argument("--split-part", default="train")
    ap.add_argument("--alpha", type=float, default=2e-2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    label = args.label or args.joint_params

    d_max = load_config("configs/breakfast.yaml")["duration"]["d_max_ticks"]
    root = "dataset/processed/breakfast"
    vocab = json.load(open(f"{root}/vocab.json"))
    seqs = json.load(open(f"{root}/sequences.json"))
    seqs = split_mod.filter_sequences(seqs, split_mod.load_split(f"{root}/split.json"), args.split_part)

    jp = joint_params.load_params(args.joint_params)
    marg = joint_params.collapse_to_marginal(jp)
    lex = narrate.Lexicon(vocab, marg)
    traj = generate.trajectories_from_real_joint(jp, seqs, d_max, chunk_size=32)
    usable = [t for t in traj if len(t["segments"]) >= error_injection.MIN_SEGMENTS]
    rng = np.random.default_rng(args.seed)

    def positives(deg):
        steps = textify.steps_from_ids(deg["verb_ids"], deg["noun_ids"], lex)
        gt = textify.gt_steps_for_window(steps, deg["window"])
        debris = textify.injection_touched_steps(steps, deg["tick_map"], deg["edited_ticks"], gt)
        pos = np.zeros(len(deg["verb_ids"]), dtype=bool)
        for si in set(gt) | debris:
            pos[steps[si].tick_start : steps[si].tick_end] = True
        return pos

    groups = {"healthy": [(t, np.zeros(len(t["verb_ids"]), dtype=bool)) for t in usable]}
    for et in error_injection.ERROR_TYPES:
        deg = [error_injection.inject(et, t, rng, marg) for t in usable]
        groups[et] = [(d, positives(d)) for d in deg]

    print(f"\n=== {label}  ({args.split_part}, alpha={args.alpha:.0e}, {len(usable)} trials/group)")
    print(f"{'group':14s} {'trial_loc stray':>15s} {'stray alarms/trial':>19s} {'max':>5s} "
          f"{'ticks/trial':>12s} {'in-range alarms':>16s}")
    tot_stray_alarms = tot_hit_alarms = tot_trials = 0
    charged = 0
    for g, items in groups.items():
        traces, lp, r_hat, ltm = batch.compute_traces_joint(jp, [d for d, _ in items], d_max, chunk_size=32)
        cache = quantile.JointThresholdCache(lp, ltm)
        s_alarms, h_alarms, s_ticks, n_stray_trials = [], [], [], 0
        for (_d, pos), tr, rh in zip(items, traces, r_hat):
            f = surprise.flag_joint_cached(cache, tr, lp, int(rh), ltm, alpha=args.alpha)
            m = np.zeros(len(pos), dtype=bool)
            for c in surprise.CHANNELS:
                m |= np.asarray(f[c])
            stray_mask = ~pos & m
            s_alarms.append(runs(stray_mask))
            h_alarms.append(runs(pos & m))
            s_ticks.append(int(stray_mask.sum()))
            n_stray_trials += bool(stray_mask.any())
        n = len(items)
        tot_stray_alarms += sum(s_alarms)
        tot_hit_alarms += sum(h_alarms)
        tot_trials += n
        charged += n_stray_trials
        print(f"{g:14s} {n_stray_trials / n:15.3f} {np.mean(s_alarms):19.2f} {max(s_alarms):5d} "
              f"{np.mean(s_ticks):12.1f} {np.mean(h_alarms):16.2f}")

    print(f"\n  false positives trial_loc charges : {charged}")
    print(f"  stray alarms actually raised      : {tot_stray_alarms}  "
          f"({tot_stray_alarms / max(1, charged):.2f}x)")
    print(f"  event-level precision (alarms in range / all alarms): "
          f"{tot_hit_alarms / max(1, tot_hit_alarms + tot_stray_alarms):.3f}")
    print(f"  alarms per trial, all sources     : "
          f"{(tot_hit_alarms + tot_stray_alarms) / tot_trials:.2f}")


if __name__ == "__main__":
    main()
