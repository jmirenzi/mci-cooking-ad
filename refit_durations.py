"""Refit only the duration parameters of a fitted joint model from its own Viterbi decode -- a
hard-assignment duration M-step, sweepable in `kappa` without a full EM run.

`kappa` is the pseudocount budget of the pooled per-state duration shape injected into every
(recipe, state) cell (`durations.fit_durations_shrunk`), and it sets how tight P(D | state,
recipe) is -- which is what both retrospective duration channels' power is a function of.
Sweeping it through EM costs ~45 min per value; here it costs seconds, because durations given a
segmentation depend on nothing else.

The last segment of each trial is right-censored and imputed via
`durations.impute_censored_histogram`, as the real M-step does; dropping it would bias every
duration short.

    ./py refit_durations.py --in runs/joint_sh_s.npz --out runs/joint_final.npz --kappa 0.001
"""
import argparse
import functools
import json

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from cook_ad.data import split as split_mod
from cook_ad.data.config import load_config
from cook_ad.hsmm import durations, joint_params
from cook_ad.synthetic import generate


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kappa", type=float, default=5.0)
    ap.add_argument("--config", default="configs/breakfast.yaml")
    ap.add_argument("--sequences", default="dataset/processed/breakfast/sequences.json")
    ap.add_argument("--split-file", default="dataset/processed/breakfast/split.json")
    ap.add_argument("--split-part", default="train")
    args = ap.parse_args()

    d_max = load_config(args.config)["duration"]["d_max_ticks"]
    seqs = json.load(open(args.sequences))
    seqs = split_mod.filter_sequences(seqs, split_mod.load_split(args.split_file), args.split_part)

    jp = joint_params.load_params(args.inp)
    traj = generate.trajectories_from_real_joint(jp, seqs, d_max, chunk_size=32)

    k_recipe, k_subtask = jp.init_counts.shape
    xi = np.zeros((k_recipe, k_subtask, d_max))
    cens = np.zeros((k_recipe, k_subtask, d_max))
    for t in traj:
        r = t["recipe_id"]
        segs = t["segments"]
        for k, d in segs[:-1]:
            xi[r, k, min(int(d), d_max) - 1] += 1.0
        if segs:
            k, d = segs[-1]
            cens[r, k, min(int(d), d_max) - 1] += 1.0

    dur_r, dur_p, _g_r, _g_p = durations.fit_durations_shrunk(
        jnp.asarray(xi), jnp.asarray(cens), jp.dur_r, jp.dur_p, d_max, args.kappa
    )
    out = jp._replace(dur_r=dur_r, dur_p=dur_p)
    joint_params.save_params(out, args.out)

    mean_new = 1.0 + np.asarray(dur_r) * (1 - np.asarray(dur_p)) / np.asarray(dur_p)
    sd_new = np.sqrt(np.asarray(dur_r) * (1 - np.asarray(dur_p)) / np.asarray(dur_p) ** 2)
    occupied = xi.sum(axis=-1) > 0
    print(f"kappa={args.kappa}: {occupied.sum()} occupied (recipe,state) cells of "
          f"{k_recipe * k_subtask}; median CV on occupied cells = "
          f"{np.median((sd_new / np.maximum(mean_new, 1e-9))[occupied]):.3f}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
