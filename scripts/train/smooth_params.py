"""Stage 3 of training: transition smoothing (hsmm/train/smoothing.py; docs/training.md).

    ./py scripts/train/smooth_params.py --in runs/joint_sh.npz --out runs/joint_sh_s.npz \
        --strength 0.7 --backoff-tau 30
"""
import argparse

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from cook_ad.hsmm import params  # noqa: E402
from cook_ad.hsmm.train import smoothing  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--backoff-tau", type=float, default=0.0,
                    help="pseudocount budget for shrinking each recipe's transition rows toward "
                         "the pooled-over-recipes row for that state; applied before the "
                         "shift. 0 disables it.")
    args = ap.parse_args()

    jp = params.load(args.inp)
    if args.backoff_tau > 0:
        jp = smoothing.backoff_counts(jp, args.backoff_tau)
    out = smoothing.shifted_mode_counts(jp, args.strength)
    params.save(out, args.out)

    b = np.asarray(params.to_log_tables(jp, 2).log_A)
    a = np.asarray(params.to_log_tables(out, 2).log_A)
    finite = np.isfinite(b) & np.isfinite(a)
    print(f"log_A: min {b[finite].min():.1f} -> {a[finite].min():.1f} nats; "
          f"median {np.median(b[finite]):.1f} -> {np.median(a[finite]):.1f}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
