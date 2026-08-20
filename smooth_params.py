"""Restore the transition counts that `params._row_normalize`'s Dirichlet-MAP mode erases, and
write the result as a new .npz.

Why
---
`_row_normalize` implements the textbook MAP plug-in, numerator max(c - 1, floor). Subtracting a
whole count is a rounding error when a cell holds hundreds and a total erasure when it holds one.
With the weak-limit prior alpha/K = 0.5/64 = 0.0078, a bigram observed EXACTLY ONCE has
c = 1.0078, so c - 1 = 0.0078 -- while one observed twice has c - 1 = 1.008, 130x larger. Per
recipe there are only ~150 transitions over a 64x63 grid, so a large share of the model's LEGAL
transitions are singletons, and the mode files them next to the ones that never happened at all.

That is exactly the shape of a badly calibrated `s_transition`: legal-but-rare transitions in
healthy trials score like impossible ones, and because the per-state alpha-quantile threshold is
computed against that same distorted row, the threshold rises to cover them -- which is what then
stops genuinely impossible transitions from clearing it. Neither sensitive nor specific.

`--strength s` adds s to every off-diagonal cell, so `_row_normalize`'s numerator becomes
c - 1 + s. The useful range is **0 < s < 1**, and s ~= 0.5-0.7 measures best:

* a singleton lands at s (plus its prior), safely above the floor -- it reads as rare, not
  impossible;
* a cell that was never observed ANYWHERE still lands at s - 1 < 0 and floors -- it stays
  maximally surprising, which is the signal omission and transposition detection lives on.

s = 1 is the full Dirichlet posterior mean. It is the principled predictive distribution, and it
is measurably WORSE here (train trial_loc accuracy 0.512 against 0.533 at s = 0.7), because it
lifts never-seen transitions from ~32 nats to ~9 and the structural channels lose their top end.
The gradation, not the Bayesian purity, is what the detector needs.

This is a post-fit transform on purpose. EM's M-step rebuilds `trans_counts` from scratch every
iteration (alpha/K + expected counts), so the addition cannot accumulate or perturb the fit; it
changes only what the detector reads.

Emissions are deliberately NOT touched: their prior is alpha_emit = vocab width, i.e. exactly 1
per category, so the mode is already the plain data frequency, and adding another count would
hand every unobserved token real probability mass -- which is precisely the signal the
substitution channel lives on.

    ./py smooth_params.py --in runs/joint_lex_a50.npz --out runs/joint_la_s07t2.npz \
        --strength 0.7 --backoff-tau 2
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


def unigram_backoff_counts(jp, tau):
    """Shrink each recipe's transition rows toward THAT RECIPE'S OWN state-occupancy
    distribution, rather than toward the pooled-over-recipes bigram (`backoff_counts`).

    The two targets separate two things the pooled target conflates. A transition into a state
    the recipe never uses at all stays deep in the tail after this backoff, because that state
    has ~no occupancy mass in q^(r) -- so recipe identity keeps being decided by which actions
    a trial contains. A transition into a state the recipe DOES use, just not from here, gains
    tau * q^(r)_v -- so a reordering costs a few nats rather than ~30.

    That distinction is what the oracle measurement says is worth having. The MAP recipe is
    re-inferred from the degraded stream, and on a transposition it currently flips away from
    the trial's real recipe ~40% of the time -- because one 30-nat impossible transition swamps
    log Z_ir and some sibling cluster wins. Pinning the recipe to the healthy decode's value
    lifts transposition recall 0.566 -> 0.728 and omission 0.370 -> 0.511, so the flip is the
    single largest remaining loss. Backing off to the recipe's own unigram shrinks the ordering
    term's magnitude without touching the inventory term that should be deciding the recipe.
    """
    trans = np.asarray(jp.trans_counts)
    init = np.asarray(jp.init_counts)
    k = trans.shape[-1]
    occupancy = trans.sum(axis=1) + init                       # (K_R,K): mass entering each state
    occupancy = occupancy / np.maximum(occupancy.sum(axis=-1, keepdims=True), 1e-12)
    # broadcast the recipe's unigram across its from-state rows, then re-ban self-transitions
    target = np.repeat(occupancy[:, None, :], k, axis=1) * (1.0 - np.eye(k))[None, :, :]
    target = target / np.maximum(target.sum(axis=-1, keepdims=True), 1e-12)
    return jp._replace(trans_counts=trans + tau * target)


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
    ap.add_argument("--unigram-tau", type=float, default=0.0,
                    help="pseudocount budget for shrinking each recipe's transition rows toward "
                         "that recipe's OWN state-occupancy distribution; keeps 'this recipe "
                         "never does that action' maximally surprising while making 'right "
                         "action, wrong place' merely surprising. Applied after --backoff-tau.")
    args = ap.parse_args()

    jp = joint_params.load_params(args.inp)
    if args.backoff_tau > 0:
        jp = backoff_counts(jp, args.backoff_tau)
    if args.unigram_tau > 0:
        jp = unigram_backoff_counts(jp, args.unigram_tau)
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
