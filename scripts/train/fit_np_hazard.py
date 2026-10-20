"""Stage 4b of training: the nonparametric duration hazard (hsmm/train/np_hazard.py; docs/training.md
§4b). The final model decodes its training split, and each (recipe, subtask) cell's hazard is the
Kaplan-Meier one, shrunk toward the pooled-over-recipes hazard and then the negative binomial; cells
with fewer than --min-segments segments keep the negative binomial. Written next to the model as
`<model>.durations.npz`, which HSMM.load picks up.

    ./py scripts/train/fit_np_hazard.py --in runs/joint_final.npz --kappa 5 --bandwidth 1 --min-segments 30
"""
import argparse
import json

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from cook_ad.data import split as split_mod  # noqa: E402
from cook_ad.data.config import load_config  # noqa: E402
from cook_ad.hsmm import params  # noqa: E402
from cook_ad.hsmm.model import HSMM, durations_path  # noqa: E402
from cook_ad.hsmm.train import lexical_init  # noqa: E402
from cook_ad.hsmm.train.np_hazard import at_risk, fit_np_hazard, save_tables  # noqa: E402
from cook_ad.hsmm.train.refit import duration_histograms  # noqa: E402
from cook_ad.synthetic import generate  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default=None, help="default: the model's durations_path")
    ap.add_argument("--kappa", type=float, default=5.0)
    ap.add_argument("--kappa0", type=float, default=2.0)
    ap.add_argument("--bandwidth", type=int, default=1)
    ap.add_argument("--min-segments", type=int, default=30)
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
        r_given, _ = lexical_init.recipe_label_assignment(seqs, json.load(open(args.labels)))
    traj = generate.trajectories_from_real_joint(jp, seqs, d_max, chunk_size=32, r_hat=r_given)
    k_r, k = np.asarray(jp.init_counts).shape
    nb = HSMM(jp, d_max)
    log_pmf, log_surv = fit_np_hazard(traj, k_r, k, d_max, np.asarray(nb.tables.log_h), args.kappa,
                                      args.kappa0, args.bandwidth, args.min_segments,
                                      nb_tables=(nb.tables.log_pmf, nb.tables.log_surv))
    out = args.out or durations_path(args.inp)
    save_tables(out, log_pmf, log_surv, {"kappa": args.kappa, "kappa0": args.kappa0,
                                         "bandwidth": args.bandwidth, "min_segments": args.min_segments,
                                         "n_sessions": len(traj)})
    xi, cens = duration_histograms(traj, k_r, k, d_max)
    n = at_risk(xi, cens)[..., 0]
    occupied = n > 0
    print(f"nonparametric hazard from {len(traj)} sessions: {int((n >= args.min_segments).sum())} of "
          f"{int(occupied.sum())} occupied (recipe, state) cells fitted, the rest keep the NB -> {out}")


if __name__ == "__main__":
    main()
