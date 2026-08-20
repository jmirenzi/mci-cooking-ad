"""Detection accuracy as a function of alpha at the **step** layer -- the granularity that counts
every alarm instead of collapsing a trial's strays into one.

`run_threshold_sweep.py` sweeps alpha at `trial_loc`, which charges at most one false positive per
trial. That is the right unit for "did it bother the user about the right trial" and the wrong one
for "how often did it bother the user" -- see [docs/eval.md] 7 for the measured case where the two
disagree about which of two fits is better. This sweeps the same alpha grid through
`eval/element_metrics.evaluate_steps`, the scoring the LLM comparison already uses, so an
operating point can be chosen on alarm count rather than trial count.

Same cost structure as `run_threshold_sweep.py`: traces are computed ONCE per source group (the
expensive JAX part) and the whole grid is swept by cheap re-flagging.

Reports per alpha: step precision/recall, flagged steps on healthy trials (unambiguous false
alarms -- a healthy trial has no injection and so no debris convention to argue about), and the
same numbers per channel, which is what localises a false-alarm floor to the channel carrying it.

    ./py run_step_sweep.py --joint-params <npz> --split-part train --tag mymodel
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
from cook_ad.eval import batch, element_metrics
from cook_ad.hsmm import joint_params
from cook_ad.llm import textify
from cook_ad.synthetic import error_injection, generate

ALPHAS = [0.2, 0.1, 0.05, 0.035, 0.02, 0.015, 0.01, 0.007, 0.005, 0.003, 0.002,
          1e-3, 5e-4, 2e-4, 1e-4, 1e-5, 1e-7]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--joint-params", required=True)
    ap.add_argument("--config", default="configs/breakfast.yaml")
    ap.add_argument("--split-file", default="dataset/processed/breakfast/split.json")
    ap.add_argument("--split-part", choices=["train", "test"], default="train")
    ap.add_argument("--sequences", default="dataset/processed/breakfast/sequences.json")
    ap.add_argument("--vocab", default="dataset/processed/breakfast/vocab.json")
    ap.add_argument("--traj-params", default=None, help="see run_detect_eval.py --traj-params")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk-size", type=int, default=32)
    ap.add_argument("--max", type=int, default=10_000)
    ap.add_argument("--tag", default="step")
    ap.add_argument("--out-dir", default="runs")
    args = ap.parse_args()

    d_max = load_config(args.config)["duration"]["d_max_ticks"]
    vocab = json.load(open(args.vocab))
    jp = joint_params.load_params(args.joint_params)
    lexicon = narrate.Lexicon(vocab, joint_params.collapse_to_marginal(jp))

    seqs = split_mod.filter_sequences(
        json.load(open(args.sequences)), split_mod.load_split(args.split_file), args.split_part
    )[: args.max]

    traj_jp = joint_params.load_params(args.traj_params) if args.traj_params else jp
    traj_marg = joint_params.collapse_to_marginal(traj_jp)
    traj = generate.trajectories_from_real_joint(traj_jp, seqs, d_max, chunk_size=args.chunk_size)
    usable = [t for t in traj if len(t["segments"]) >= error_injection.MIN_SEGMENTS]
    print(f"[{args.tag}/{args.split_part}] {len(usable)}/{len(seqs)} usable trials", flush=True)

    rng = np.random.default_rng(args.seed)

    # traces ONCE per group; everything below is post-hoc thresholding
    healthy_traces, log_probs, healthy_r, ltm = batch.compute_traces_joint(
        jp, usable, d_max, chunk_size=args.chunk_size
    )
    healthy_steps = [textify.steps_from_trajectory(t, lexicon) for t in usable]

    degraded = {}
    for et in error_injection.ERROR_TYPES:
        deg = [error_injection.inject(et, t, rng, traj_marg) for t in usable]
        tr, _lp, rh, _ltm = batch.compute_traces_joint(jp, deg, d_max, chunk_size=args.chunk_size)
        truth = []
        for traj_i, d in zip(usable, deg):
            steps = textify.steps_from_trajectory(d, lexicon)
            gt = textify.gt_steps_for_window(steps, d["window"])
            src = textify.step_covering_tick(
                textify.steps_from_trajectory(traj_i, lexicon), d["window"][0]
            )
            correction = (src.verb, src.noun, src.duration) if src else None
            debris = textify.injection_touched_steps(steps, d["tick_map"], d["edited_ticks"], gt)
            truth.append((steps, gt, correction, debris))
        degraded[et] = (tr, rh, truth)
        print(f"  traced {et}", flush=True)

    cache = quantile.JointThresholdCache(log_probs, ltm)

    def report_at(alpha, channels):
        healthy_verdicts = [
            element_metrics.step_verdicts_from_flags(
                surprise.flag_joint_cached(cache, tr, log_probs, int(r), ltm, alpha=alpha),
                st, tr, lexicon, channels=channels)
            for tr, r, st in zip(healthy_traces, healthy_r, healthy_steps)
        ]
        by_type, artifacts = {}, {}
        for et, (tr_list, rh, truth) in degraded.items():
            rows, debris_rows = [], []
            for tr, r, (steps, gt, corr, debris) in zip(tr_list, rh, truth):
                v = element_metrics.step_verdicts_from_flags(
                    surprise.flag_joint_cached(cache, tr, log_probs, int(r), ltm, alpha=alpha),
                    steps, tr, lexicon, channels=channels)
                rows.append((v, gt, corr))
                debris_rows.append(debris)
            by_type[et] = rows
            artifacts[et] = debris_rows
        rep = element_metrics.evaluate_steps(healthy_verdicts, by_type, artifact_steps=artifacts)
        rep["healthy_flagged_steps"] = sum(
            1 for v in healthy_verdicts for x in v if x.is_anomaly
        )
        rep["healthy_total_steps"] = sum(len(v) for v in healthy_verdicts)
        return rep

    results = []
    for alpha in ALPHAS:
        row = {"alpha": alpha, "raw": report_at(alpha, surprise.CHANNELS)}
        for ch in surprise.CHANNELS:
            r = report_at(alpha, (ch,))
            row[ch] = {"step_level": r["step_level"], "healthy_flagged_steps": r["healthy_flagged_steps"]}
        s = row["raw"]["step_level"]
        f1 = 2 * s["precision"] * s["recall"] / max(1e-12, s["precision"] + s["recall"])
        print(f"alpha={alpha:.0e} step P={s['precision']:.3f} R={s['recall']:.3f} F1={f1:.3f} "
              f"TP={s['tp']} FP={s['fp']} | healthy flagged steps="
              f"{row['raw']['healthy_flagged_steps']}/{row['raw']['healthy_total_steps']}", flush=True)
        results.append(row)

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"step_{args.tag}_{args.split_part}.json")
    with open(out, "w") as f:
        json.dump({"config": vars(args), "results": results}, f, indent=2)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
