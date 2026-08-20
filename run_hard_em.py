"""Viterbi (hard-assignment) EM for the joint model, as an alternative to `joint_em`'s soft EM.

Why bother, when soft EM is the principled one
----------------------------------------------
Soft EM maximises the marginal likelihood, and on this data that objective is measurably not
the thing the detector needs. Two direct observations on the 402-trial train split:

* the lexical warm start begins at a HIGHER objective (-12948) than the cascade-warm-started fit
  ever CONVERGES to (-14245), so likelihood does not rank the two initialisations the way
  detection does;
* running soft EM from that warm start then *lowers* the objective for a stretch while degrading
  per-tick subtask ARI from 0.999 to 0.940 -- the fit gets blurrier in exactly the way that lets
  Viterbi re-explain an injected anomaly through an alternative state path.

Hard EM optimises the joint likelihood of parameters AND the MAP segmentation instead. It is the
wrong estimator if you want calibrated posterior uncertainty; it is the right one if what you
need is that the decode the detector reads and the counts the model was fit from are the SAME
object. Every surprise channel scores against `z_star`, so the model may as well be fit to
`z_star`.

Each iteration is: recipe-conditioned Viterbi decode of every trial -> hard init/transition
counts and duration histograms per (recipe, state) -> Dirichlet-MAP renormalisation and a
shrunk NB duration fit. The trial's final segment is right-censored and goes into the censoring
histogram, imputed by `durations.impute_censored_histogram` exactly as the soft M-step does;
treating it as exactly observed would bias every duration short.

Emissions stay pinned to the lexical anchor throughout (--anchor), since the (verb,noun) pair
inventory is what defines the states and is not in question.

    ./py run_hard_em.py --split-part train --out runs/joint_hard.npz --iters 8
"""
import argparse
import json
import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from cook_ad.data import split as split_mod
from cook_ad.data.config import load_config
from cook_ad.hsmm import durations, joint_em, joint_params
from cook_ad.recipe import lexical_init, recipe_hmm, segmentize


def hard_counts(seg_results, r_hat, k_recipe, k_subtask, d_max):
    """Per-recipe init/transition counts and per-(recipe,state) duration histograms from a set
    of decoded segmentations. Returns (init, trans, xi_dur, cens) -- `cens` holding only each
    trial's final, still-open segment."""
    init = np.zeros((k_recipe, k_subtask))
    trans = np.zeros((k_recipe, k_subtask, k_subtask))
    xi = np.zeros((k_recipe, k_subtask, d_max))
    cens = np.zeros((k_recipe, k_subtask, d_max))
    for res, r in zip(seg_results, r_hat):
        segs = res["segments"]
        if not segs:
            continue
        r = int(r)
        init[r, segs[0][0]] += 1.0
        for (u, _), (v, _) in zip(segs[:-1], segs[1:]):
            if u != v:
                trans[r, u, v] += 1.0
        for k, d in segs[:-1]:
            xi[r, k, min(int(d), d_max) - 1] += 1.0
        k, d = segs[-1]
        cens[r, k, min(int(d), d_max) - 1] += 1.0
    return init, trans, xi, cens


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/breakfast.yaml")
    ap.add_argument("--sequences", default="dataset/processed/breakfast/sequences.json")
    ap.add_argument("--labels", default="dataset/processed/breakfast/labels.json",
                    help="scoring only -- never fed to the fit")
    ap.add_argument("--split-file", default="dataset/processed/breakfast/split.json")
    ap.add_argument("--split-part", choices=["train", "test"], default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--init-from", default=None,
                    help="start from this .npz instead of a fresh lexical warm start")
    ap.add_argument("--keep-init-emissions", action="store_true",
                    help="with --init-from, keep that model's fitted emissions instead of "
                         "resetting them to the lexical anchor -- the 'soft EM first, then a "
                         "few hard steps to sharpen' recipe")
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--k-subtask", type=int, default=None)
    ap.add_argument("--k-recipe", type=int, default=None)
    ap.add_argument("--anchor", type=float, default=lexical_init.ANCHOR_MASS)
    ap.add_argument("--background", type=float, default=lexical_init.BACKGROUND_MASS)
    ap.add_argument("--alpha-init", type=float, default=0.5)
    ap.add_argument("--alpha-trans", type=float, default=0.5)
    ap.add_argument("--alpha-pi", type=float, default=None)
    ap.add_argument("--kappa", type=float, default=None)
    ap.add_argument("--chunk-size", type=int, default=32)
    ap.add_argument("--snapshot-every", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    d_max = cfg["duration"]["d_max_ticks"]
    k_subtask = args.k_subtask or cfg["k_subtask"]
    k_recipe = args.k_recipe or cfg["k_recipe"]
    kappa = args.kappa if args.kappa is not None else cfg["duration"]["shrinkage_kappa"]
    alpha_pi = args.alpha_pi if args.alpha_pi is not None else cfg["prior"]["alpha_pi"]

    sequences = json.load(open(args.sequences))
    labels = {e["trial_id"]: e for e in json.load(open(args.labels))}
    split = split_mod.load_split(args.split_file)
    sequences = split_mod.filter_sequences(sequences, split, args.split_part)
    joined = [labels[s["trial_id"]] for s in sequences]
    print(f"trials: {len(sequences)}  K={k_subtask} K_R={k_recipe}", flush=True)

    p, info = lexical_init.lexical_to_joint(
        sequences, k_subtask, k_recipe, d_max, cfg["vocab"]["verbs"], cfg["vocab"]["nouns"],
        kappa, anchor=args.anchor, background=args.background,
        alpha_init=args.alpha_init, alpha_trans=args.alpha_trans, alpha_pi=alpha_pi,
    )
    if args.init_from:
        loaded = joint_params.load_params(args.init_from)
        p = loaded if args.keep_init_emissions else loaded._replace(
            verb_counts=p.verb_counts, noun_counts=p.noun_counts
        )
    emit_v, emit_n = p.verb_counts, p.noun_counts

    verb_ids, noun_ids, mask = joint_em.pad_batch(sequences)
    true_recipes = [e["recipe_label"] for e in joined]
    true_tick = np.concatenate([np.array(e["subtask_labels"]) for e in joined])

    for it in range(args.iters):
        t0 = time.time()
        r_hat, _rho, trial_ll = joint_em.infer_recipe(
            p, verb_ids, noun_ids, mask, d_max, chunk_size=args.chunk_size
        )
        lp = joint_params.to_log_probs_joint(p, d_max)
        seg = segmentize.segment_all_conditioned(lp, r_hat, verb_ids, noun_ids, mask, d_max)

        init_c, trans_c, xi, cens = hard_counts(
            seg, np.asarray(r_hat), k_recipe, k_subtask, d_max
        )
        pi_c = np.bincount(np.asarray(r_hat), minlength=k_recipe).astype(float)

        dur_r, dur_p, _gr, _gp = durations.fit_durations_shrunk(
            jnp.asarray(xi), jnp.asarray(cens), p.dur_r, p.dur_p, d_max, kappa
        )
        p = joint_params.JointHSMMParams(
            init_counts=jnp.asarray(init_c + args.alpha_init / k_subtask),
            trans_counts=jnp.asarray(trans_c + args.alpha_trans / k_subtask)
            * (1.0 - jnp.eye(k_subtask))[None, :, :],
            verb_counts=emit_v,
            noun_counts=emit_n,
            dur_r=dur_r,
            dur_p=dur_p,
            pi_counts=jnp.asarray(pi_c + alpha_pi / k_recipe),
        )

        pred_tick = np.concatenate([r["subtask_per_tick"] for r in seg])
        print(f"iter {it}: marginal loglik={float(jnp.sum(trial_ll)):.1f} "
              f"recipe ARI={recipe_hmm.adjusted_rand(np.asarray(r_hat), true_recipes):.4f} "
              f"subtask ARI={recipe_hmm.adjusted_rand(pred_tick, true_tick):.4f} "
              f"({time.time() - t0:.1f}s)", flush=True)
        if args.snapshot_every and (it + 1) % args.snapshot_every == 0:
            joint_params.save_params(p, f"{args.out}.iter{it + 1:02d}.npz")

    joint_params.save_params(p, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
