"""Is the detector's response to an injection actually caused by the injection?

Every degraded trial is paired with its own unaltered counterpart, scored through the SAME
detector, and the healthy flags are projected into degraded tick space via `tick_map`
(eval/counterfactual.py). Three quantities per error type, all over the injection-touched range
(ground-truth window UNION debris -- the same range run_threshold_sweep.py's trial_loc uses):

    observed      any DEGRADED flag lands in the range        <- what recall normally reports
    chance        any PROJECTED HEALTHY flag lands in there   <- what you would get if the
                                                                 injection changed nothing
    attributable  any flag present in the degraded run but
                  NOT in its healthy counterfactual           <- detection the injection caused

`observed` is the number every other metric in this repo reports. `chance` is its null: the same
detector, the same trial, the same range, with no injection present -- so a flag that lands in the
range purely because the detector is chatty there counts exactly as it would in `observed`. If
observed ~= chance for an error type, that type's apparent detection carries no evidence the
injection was noticed, however high the recall looks.

This is a stronger null than a uniform-random baseline: it is matched per trial and per range, so
it absorbs trial length, step count, range width, and the detector's own per-trial noise level.

`lift = observed - chance` is the honest detection signal. `attributable` is the same idea from
the other side (flags the injection added rather than trials it changed), and the two should
agree; a large gap between them means flags moved rather than appeared.
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
from cook_ad.eval import batch, counterfactual
from cook_ad.hsmm import joint_params
from cook_ad.llm import textify
from cook_ad.synthetic import error_injection, generate

CHANNELS = surprise.CHANNELS


def _union(flags, channels):
    mask = None
    for ch in channels:
        mask = flags[ch].copy() if mask is None else (mask | flags[ch])
    return mask


def _touched_ticks(degraded, lexicon):
    """Injection-touched tick mask over the DEGRADED trial: ground-truth window UNION debris."""
    steps = textify.steps_from_ids(degraded["verb_ids"], degraded["noun_ids"], lexicon)
    gt_steps = textify.gt_steps_for_window(steps, degraded["window"])
    debris = textify.injection_touched_steps(
        steps, degraded["tick_map"], degraded["edited_ticks"], gt_steps
    )
    mask = np.zeros(len(degraded["verb_ids"]), dtype=bool)
    for si in set(gt_steps) | debris:
        s = steps[si]
        mask[s.tick_start : s.tick_end] = True
    return mask


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/breakfast.yaml")
    ap.add_argument("--joint-params", default="dataset/processed/breakfast/joint_params.npz")
    ap.add_argument("--sequences", default="dataset/processed/breakfast/sequences.json")
    ap.add_argument("--vocab", default="dataset/processed/breakfast/vocab.json")
    ap.add_argument("--max-real", type=int, default=447)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk-size", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=surprise.DEFAULT_ALPHA)
    ap.add_argument("--out", default="dataset/processed/breakfast/counterfactual_report.json")
    args = ap.parse_args()

    d_max = load_config(args.config)["duration"]["d_max_ticks"]
    vocab = json.load(open(args.vocab))
    jp = joint_params.load_params(args.joint_params)
    marg = joint_params.collapse_to_marginal(jp)
    lexicon = narrate.Lexicon(vocab, marg)

    seqs = json.load(open(args.sequences))[: args.max_real]
    traj = [generate.trajectory_from_real_joint(jp, s["verb_ids"], s["noun_ids"], d_max)
            for s in seqs]
    usable = [t for t in traj if len(t["segments"]) >= error_injection.MIN_SEGMENTS]
    print(f"{len(usable)} usable real trials, alpha={args.alpha:.1e}", flush=True)

    print("computing traces: healthy", flush=True)
    h_traces, log_probs, h_rhat, ltm = batch.compute_traces_joint(
        jp, usable, d_max, chunk_size=args.chunk_size
    )
    healthy_flags = [
        surprise.flag_joint(t, log_probs, int(h_rhat[i]), ltm, alpha=args.alpha)
        for i, t in enumerate(h_traces)
    ]

    rng = np.random.default_rng(args.seed)
    rows = {}
    for et in error_injection.ERROR_TYPES:
        print(f"computing traces: {et}", flush=True)
        degraded = [error_injection.inject(et, t, rng, marg) for t in usable]
        d_traces, log_probs, d_rhat, ltm = batch.compute_traces_joint(
            jp, degraded, d_max, chunk_size=args.chunk_size
        )

        n = len(degraded)
        obs = chance = attrib = both = neither = 0
        for i, deg in enumerate(degraded):
            rng_mask = _touched_ticks(deg, lexicon)
            d_flags = surprise.flag_joint(d_traces[i], log_probs, int(d_rhat[i]), ltm,
                                          alpha=args.alpha)
            projected = counterfactual.project_flags(healthy_flags[i], deg["tick_map"])
            attributable = counterfactual.attributable(d_flags, projected)

            d_hit = bool((_union(d_flags, CHANNELS) & rng_mask).any())
            p_hit = bool((_union(projected, CHANNELS) & rng_mask).any())
            a_hit = bool((_union(attributable, CHANNELS) & rng_mask).any())

            obs += int(d_hit)
            chance += int(p_hit)
            attrib += int(a_hit)
            both += int(d_hit and p_hit)
            neither += int(not d_hit and not p_hit)

        rows[et] = {
            "n": n,
            "observed": obs / n, "chance": chance / n, "attributable": attrib / n,
            "lift": (obs - chance) / n,
            "obs_n": obs, "chance_n": chance, "attrib_n": attrib,
            "both_n": both, "neither_n": neither,
        }
        r = rows[et]
        print(f"  {et:<14} observed {r['observed']:.3f}  chance {r['chance']:.3f}  "
              f"lift {r['lift']:+.3f}  attributable {r['attributable']:.3f}", flush=True)

    print(f"\n{'error type':<15} {'observed':>9} {'chance':>8} {'lift':>8} {'attrib':>8}  "
          f"{'reading':<30}")
    print("-" * 88)
    for et, r in rows.items():
        # McNemar, because the two outcomes are PAIRED: the same trial, the same range, scored
        # with and without the injection. Only the discordant trials carry information --
        # b = injection made it fire in range when the healthy run did not, c = the reverse.
        # An unpaired two-proportion z would both ignore the pairing and blow up when chance == 0.
        b = r["obs_n"] - r["both_n"]
        c = r["chance_n"] - r["both_n"]
        r["mcnemar_b"], r["mcnemar_c"] = b, c
        if b + c == 0:
            stat, verdict = 0.0, "no discordant trials"
        else:
            stat = (abs(b - c) - 1) ** 2 / (b + c)   # continuity-corrected
            r["mcnemar_chi2"] = stat
            sig = stat > 3.84                          # chi2(1) at p=0.05
            verdict = (("above chance" if b > c else "BELOW chance") if sig
                       else "indistinguishable from chance")
        print(f"{et:<15} {r['observed']:>9.3f} {r['chance']:>8.3f} {r['lift']:>+8.3f} "
              f"{r['attributable']:>8.3f}  {verdict:<30} (b={b} c={c} chi2={stat:.1f})")

    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "alpha": args.alpha, "per_type": rows}, f, indent=2)
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
