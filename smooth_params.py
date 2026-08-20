"""Re-express a fitted joint model's transition counts so the Dirichlet-MAP mode stops erasing
singleton transitions, and write the result as a new .npz.

`params._row_normalize` takes numerator max(c - 1, floor), which is a rounding error when a cell
holds hundreds of counts and a total erasure when it holds one. At K_R = 16 there are only ~150
observed transitions per recipe over a 64x63 grid, so a large share of the model's LEGAL
transitions are singletons and the mode files them next to ones that never happened.

`--strength s` stores c + s in place of c, making the numerator c - 1 + s. The useful range is
0 < s < 1: a singleton lands at s, safely off the floor, while a cell never observed anywhere
still floors and stays maximally surprising. s = 1 is the full posterior mean and is measurably
worse -- it lifts never-seen transitions from ~32 nats to ~9 and the structural channels lose
their top end. `--backoff-tau` additionally mixes each recipe's rows toward the pooled-over-
recipes row for that state, the transition analogue of `kappa` for durations. Both are
regularisation, so select them on held-out data -- picking them in-sample systematically picks
too little of each (docs/eval.md 7).

Emissions are deliberately not touched: their prior is alpha_emit = vocab width, i.e. exactly 1
per category, so the mode is already the plain data frequency, and adding another count would
hand every unobserved token real probability mass -- the signal the substitution channel lives
on.

Post-fit only. EM's M-step rebuilds `trans_counts` from scratch every iteration (alpha/K +
expected counts), so the shift cannot accumulate or perturb a fit; it changes what the detector
reads, nothing else. Full treatment in docs/hsmm.md 3.

    ./py smooth_params.py --in runs/joint_sh.npz --out runs/joint_sh_s.npz \
        --strength 0.7 --backoff-tau 30
"""
import argparse

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from cook_ad.hsmm import joint_params


def backoff_counts(jp, tau):
    """Shrink each recipe's transition rows toward the pooled-over-recipes row for that state,
    with pseudocount budget `tau` -- the transition analogue of what
    `durations.fit_durations_shrunk`'s `kappa` already does for the duration cells.

    The joint model estimates K_R separate (K,K) transition matrices from one dataset. At
    K_R=16 over 400 trials that is ~150 observed transitions per recipe spread over a 64x63
    grid, so a recipe's own row for a state is thin: a transition that is ordinary *for that
    action in general* can easily have zero count *in this particular cluster*, and the model
    then calls it impossible. On healthy trials that is a false alarm, and it is the dominant
    one -- the per-recipe rows are the only part of this model fit with no pooling of any kind,
    while durations (kappa) and emissions (shared across recipes outright) both have it.

    Backing off to the pooled row keeps the two things the detector needs distinguishable:
    a transition unseen EVERYWHERE stays unseen after backoff and remains maximally surprising,
    while one merely unseen HERE inherits the global row's mass and becomes mildly surprising --
    which is precisely the gradation `s_recipe_transition` (recipe-conditioned minus marginal)
    is supposed to read.
    """
    trans = np.asarray(jp.trans_counts)
    k = trans.shape[-1]
    off_diagonal = 1.0 - np.eye(k)
    pooled = trans.sum(axis=0) * off_diagonal          # (K,K)
    pooled = pooled / np.maximum(pooled.sum(axis=-1, keepdims=True), 1e-12)
    return jp._replace(trans_counts=trans + tau * pooled[None, :, :])


def shifted_mode_counts(jp, strength=1.0, smooth_trans=True, smooth_init=True):
    """Add `strength` to every off-diagonal transition cell (and to init), so
    `params._row_normalize`'s numerator becomes c - 1 + strength. See the module docstring for
    why 0 < strength < 1 is the useful range and why strength = 1 (the posterior mean) is not."""
    trans = np.asarray(jp.trans_counts)
    init = np.asarray(jp.init_counts)
    k = trans.shape[-1]
    if smooth_trans:
        off_diagonal = (1.0 - np.eye(k))[None, :, :]
        # Self-transitions are structurally banned and must stay at zero count; adding mass
        # there would give the diagonal probability the model's whole segmentation story denies.
        trans = trans + strength * off_diagonal
    if smooth_init:
        init = init + strength
    return jp._replace(trans_counts=trans, init_counts=init)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--no-trans", action="store_true")
    ap.add_argument("--no-init", action="store_true")
    ap.add_argument("--backoff-tau", type=float, default=0.0,
                    help="pseudocount budget for shrinking each recipe's transition rows toward "
                         "the pooled-over-recipes row for that state; applied BEFORE the "
                         "posterior-mean shift. 0 disables it.")
    args = ap.parse_args()

    jp = joint_params.load_params(args.inp)
    if args.backoff_tau > 0:
        jp = backoff_counts(jp, args.backoff_tau)
    out = shifted_mode_counts(jp, args.strength, not args.no_trans, not args.no_init)
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
