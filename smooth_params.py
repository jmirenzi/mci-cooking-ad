"""Re-express a fitted joint model's transition counts so the Dirichlet-MAP mode stops erasing
singleton transitions. Writes a new .npz; see docs/hsmm.md 3.

`params._row_normalize` takes numerator max(c - 1, floor), so a bigram observed exactly once
floors -- and at K_R=16 over ~400 trials most of a recipe's legal transitions are singletons.
`--strength s` stores c + s, making the numerator c - 1 + s. Useful range is 0 < s < 1: a
singleton clears the floor, a never-observed cell stays on it. s = 1 (the posterior mean) is
worse -- it also lifts never-seen transitions from ~32 nats to ~9.

`--backoff-tau` mixes each recipe's rows toward the pooled-over-recipes row first. Both are
regularisation: select them on held-out data (make_dev_split.py), not in-sample.

Emissions are untouched -- their prior is already 1 per category, so the mode is the plain data
frequency, and shifting it would give unobserved tokens the mass the substitution channel needs.
Post-fit only: the M-step rebuilds trans_counts every iteration, so this cannot perturb a fit.

    ./py smooth_params.py --in runs/joint_sh.npz --out runs/joint_sh_s.npz \
        --strength 0.7 --backoff-tau 30
"""
import argparse

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from cook_ad.hsmm import joint_params


def backoff_counts(jp, tau):
    """Shrink each recipe's transition rows toward the pooled-over-recipes row for that state --
    the transition analogue of `durations.fit_durations_shrunk`'s `kappa`, and the only pooling
    the per-recipe rows otherwise get. A transition unseen everywhere stays maximally surprising;
    one merely unseen in THIS recipe inherits the pooled row's mass and becomes mildly so."""
    trans = np.asarray(jp.trans_counts)
    k = trans.shape[-1]
    off_diagonal = 1.0 - np.eye(k)
    pooled = trans.sum(axis=0) * off_diagonal          # (K,K)
    pooled = pooled / np.maximum(pooled.sum(axis=-1, keepdims=True), 1e-12)
    return jp._replace(trans_counts=trans + tau * pooled[None, :, :])


def shifted_mode_counts(jp, strength=1.0):
    """Add `strength` to every off-diagonal transition cell and to init, so
    `params._row_normalize`'s numerator becomes c - 1 + strength."""
    trans = np.asarray(jp.trans_counts)
    init = np.asarray(jp.init_counts)
    k = trans.shape[-1]
    # off-diagonal only: self-transitions are structurally banned and must stay at zero count
    off_diagonal = (1.0 - np.eye(k))[None, :, :]
    return jp._replace(trans_counts=trans + strength * off_diagonal, init_counts=init + strength)


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

    jp = joint_params.load_params(args.inp)
    if args.backoff_tau > 0:
        jp = backoff_counts(jp, args.backoff_tau)
    out = shifted_mode_counts(jp, args.strength)
    joint_params.save_params(out, args.out)

    lp_before = joint_params.to_log_probs_joint(jp, 2)
    lp_after = joint_params.to_log_probs_joint(out, 2)
    for name, before, after in (("log_trans", lp_before.log_trans, lp_after.log_trans),):
        b, a = np.asarray(before), np.asarray(after)
        finite = np.isfinite(b) & np.isfinite(a)
        print(f"{name}: min {b[finite].min():.1f} -> {a[finite].min():.1f} nats; "
              f"median {np.median(b[finite]):.1f} -> {np.median(a[finite]):.1f}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
