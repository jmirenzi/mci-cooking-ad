"""Stage 4 of training: refit the durations from the model's own Viterbi decode at shrinkage
`kappa` (hsmm/train/refit.py; docs/training.md).

    ./py scripts/train/refit_durations.py --in runs/joint_sh_s.npz --out runs/joint_final.npz --kappa 0.001
"""
import argparse
import json

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from cook_ad.data import split as split_mod  # noqa: E402
from cook_ad.data.config import load_config  # noqa: E402
from cook_ad.hsmm import params  # noqa: E402
from cook_ad.hsmm.train import lexical_init, refit  # noqa: E402
from cook_ad.synthetic import generate  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kappa", type=float, default=5.0)
    ap.add_argument("--config", default="configs/breakfast.yaml")
    ap.add_argument("--sequences", default="dataset/processed/breakfast/sequences.json")
    ap.add_argument("--split-file", default="dataset/processed/breakfast/split.json")
    ap.add_argument("--split-part", default="train")
    ap.add_argument("--labels", default="dataset/processed/breakfast/labels.json")
    ap.add_argument("--recipe-labels", action="store_true",
                    help="decode each trial under its labelled dish instead of the inferred one")
    args = ap.parse_args()

    d_max = load_config(args.config)["duration"]["d_max_ticks"]
    seqs = json.load(open(args.sequences))
    seqs = split_mod.filter_sequences(seqs, split_mod.load_split(args.split_file), args.split_part)

    jp = params.load(args.inp)
    r_given = None
    if args.recipe_labels:
        r_given, _names = lexical_init.recipe_label_assignment(seqs, json.load(open(args.labels)))
    traj = generate.trajectories_from_real_joint(jp, seqs, d_max, chunk_size=32, r_hat=r_given)
    out, xi = refit.refit_durations(jp, traj, d_max, args.kappa)
    params.save(out, args.out)

    dur_r, dur_p = np.asarray(out.dur_r), np.asarray(out.dur_p)
    mean_new = 1.0 + dur_r * (1 - dur_p) / dur_p
    sd_new = np.sqrt(dur_r * (1 - dur_p) / dur_p ** 2)
    occupied = xi.sum(axis=-1) > 0
    print(f"kappa={args.kappa}: {occupied.sum()} occupied (recipe,state) cells of "
          f"{xi.shape[0] * xi.shape[1]}; median CV on occupied cells = "
          f"{np.median((sd_new / np.maximum(mean_new, 1e-9))[occupied]):.3f}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
