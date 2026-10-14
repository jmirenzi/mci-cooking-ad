"""Stage 2 of training: a few Viterbi (hard-assignment) EM iterations from a soft-EM fit
(hsmm/train/hard_em.py; docs/training.md).

    ./py scripts/train/fit_hard_em.py --split-part train --out runs/joint_hard.npz --iters 5
"""
import argparse
import json
import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from cook_ad.xla_env import disable_gpu_autotuning  # noqa: E402 -- must precede the jax import

disable_gpu_autotuning()

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from cook_ad.data import split as split_mod
from cook_ad.data.config import load_config
from cook_ad.hsmm import params
from cook_ad.hsmm.train import hard_em, joint_em, lexical_init, metrics


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
    ap.add_argument("--recipe-labels", action="store_true",
                    help="recipe = labelled dish: each training trial is decoded under its own "
                         "dish instead of the inferred one (needs a --init-from fitted the same way)")
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

    if args.init_from:
        # K / K_R come from the checkpoint, not the config: --init-from is usually pointed at a
        # fit whose state count differs from configs/breakfast.yaml's nominal weak limit, and
        # silently mixing the two shapes surfaces much later as an einsum size mismatch inside
        # the duration M-step.
        loaded = params.load(args.init_from)
        k_recipe, k_subtask = loaded.init_counts.shape
        print(f"init-from {args.init_from}: K={k_subtask} K_R={k_recipe}", flush=True)

    label_assign = None
    if args.recipe_labels:
        label_assign, dish_names = lexical_init.recipe_label_assignment(
            sequences, json.load(open(args.labels)))
        if k_recipe != len(dish_names):
            if args.init_from:
                raise SystemExit(f"--init-from has K_R={k_recipe} but there are {len(dish_names)} dishes")
            k_recipe = len(dish_names)
        print(f"recipe = dish label: {dish_names}", flush=True)
    p, info = lexical_init.lexical_to_joint(
        sequences, k_subtask, k_recipe, d_max, cfg["vocab"]["verbs"], cfg["vocab"]["nouns"],
        kappa, anchor=args.anchor, background=args.background,
        alpha_init=args.alpha_init, alpha_trans=args.alpha_trans, alpha_pi=alpha_pi,
        assign=label_assign,
    )
    if args.init_from:
        p = loaded if args.keep_init_emissions else loaded._replace(
            verb_counts=p.verb_counts, noun_counts=p.noun_counts
        )

    verb_ids, noun_ids, mask = joint_em.pad_batch(sequences)
    true_recipes = [e["recipe_label"] for e in joined]
    true_tick = np.concatenate([np.array(e["subtask_labels"]) for e in joined])

    for it in range(args.iters):
        t0 = time.time()
        p, inferred, seg, trial_ll = hard_em.hard_em_step(
            p, verb_ids, noun_ids, mask, d_max, kappa, args.alpha_init, args.alpha_trans, alpha_pi,
            chunk_size=args.chunk_size, r_given=label_assign)
        pred_tick = np.concatenate([np.repeat([k for k, _ in s], [d for _, d in s]) for s in seg])
        print(f"iter {it}: marginal loglik={float(jnp.sum(trial_ll)):.1f} "
              f"recipe ARI (inferred, unanchored)={metrics.adjusted_rand(inferred, true_recipes):.4f} "
              f"subtask ARI={metrics.adjusted_rand(pred_tick, true_tick):.4f} "
              f"({time.time() - t0:.1f}s)", flush=True)

    params.save(p, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
