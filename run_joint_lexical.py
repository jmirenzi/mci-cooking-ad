"""Fit the joint Dual-HSMM from the lexical warm start (recipe/lexical_init.py) instead of the
cascade one -- one subtask state per observed (verb,noun) pair, recipes seeded from a
bag-of-pairs clustering, emissions held by a per-state anchor prior.

Same model, same joint EM, same artifacts as run_joint.py; only iteration 0 and the emission
prior differ. See recipe/lexical_init.py's module docstring for why.

    ./py run_joint_lexical.py --split-file dataset/processed/breakfast/split.json \
        --split-part train --out runs/joint_lex.npz --anchor 50 --max-iters 60
"""
import argparse
import json
import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from cook_ad.xla_env import disable_gpu_autotuning  # noqa: E402 -- must precede the jax import

disable_gpu_autotuning()

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from cook_ad.data import split as split_mod
from cook_ad.data.config import load_config
from cook_ad.hsmm import joint_em, joint_params
from cook_ad.recipe import lexical_init, recipe_hmm, segmentize


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/breakfast.yaml")
    ap.add_argument("--sequences", default="dataset/processed/breakfast/sequences.json")
    ap.add_argument("--labels", default="dataset/processed/breakfast/labels.json",
                    help="scoring only -- never fed to EM (docs/README.md)")
    ap.add_argument("--split-file", default="dataset/processed/breakfast/split.json")
    ap.add_argument("--split-part", choices=["train", "test"], default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k-subtask", type=int, default=None)
    ap.add_argument("--k-recipe", type=int, default=None)
    ap.add_argument("--max-iters", type=int, default=60)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--chunk-size", type=int, default=None)
    ap.add_argument("--anchor", type=float, default=lexical_init.ANCHOR_MASS,
                    help="Dirichlet pseudocount on a state's own (verb,noun) tokens, applied "
                         "both at iteration 0 and as the M-step prior")
    ap.add_argument("--background", type=float, default=lexical_init.BACKGROUND_MASS)
    ap.add_argument("--global-damping", type=float, default=None)
    ap.add_argument("--min-pair-ticks", type=int, default=lexical_init.MIN_PAIR_TICKS)
    ap.add_argument("--alpha-init", type=float, default=0.5, help="joint_em.run_joint_em alpha_init")
    ap.add_argument("--alpha-trans", type=float, default=0.5,
                    help="Dirichlet concentration for the per-recipe transition rows; alpha/K "
                         "per cell, so smaller = sparser A and a sharper s_transition null")
    ap.add_argument("--kappa", type=float, default=None, help="override duration.shrinkage_kappa")
    ap.add_argument("--idf-recipes", action="store_true", default=None,
                    help="TF-IDF weight the bag-of-pairs histograms the recipe clustering runs "
                         "on. Defaults to the config's joint_em.idf_recipes. Off for Breakfast "
                         "(its vocabulary is almost all goal-diagnostic); on for EPIC, where "
                         "equipment and environment nouns dominate and are identical across "
                         "dishes -- see recipe/lexical_init.cluster_recipes.")
    ap.add_argument("--recipe-features", choices=["pairs", "nouns"], default=None,
                    help="histogram the recipe clustering runs on. Defaults to the config's "
                         "joint_em.recipe_features. Pair histograms are too sparse on a corpus "
                         "with thousands of distinct pairs -- see cluster_recipes.")
    ap.add_argument("--init-prior-scale", type=float, default=1.0,
                    help="scale on the Dirichlet prior added to the ITERATION-0 init/trans/pi "
                         "counts (the M-step's own prior is unaffected). 0.0 is what the best "
                         "measured model uses -- see lexical_init.lexical_to_joint's docstring "
                         "for the measurement and why it is not the default")
    args = ap.parse_args()

    cfg = load_config(args.config)
    d_max = cfg["duration"]["d_max_ticks"]
    k_subtask = args.k_subtask or cfg["k_subtask"]
    k_recipe = args.k_recipe or cfg["k_recipe"]
    kappa = args.kappa if args.kappa is not None else cfg["duration"]["shrinkage_kappa"]
    alpha_pi = cfg["prior"]["alpha_pi"]
    jcfg = cfg["joint_em"]
    # joint_em.chunk_size, when the config sets it, wins over the em.chunk_size // k_recipe
    # derivation. That derivation exists because the joint E-step's (chunk,K_R,T,K) gamma costs
    # an extra K_R-fold in memory, and it lands on 1 for every config in the repo (8 // 16 = 0).
    # 1 is fine at Breakfast's K=64 and is actively pathological at larger K: measured on EPIC,
    # K=128 with chunk 1 does not finish XLA compilation in 4 minutes, while the SAME model at
    # chunk 2 compiles in 33s. Compile cost at these shapes is not monotone in chunk size, so
    # the working value has to be settable per corpus rather than derived.
    if cfg.get("joint_em", {}).get("chunk_size") is not None and args.chunk_size is None:
        chunk_size = int(cfg["joint_em"]["chunk_size"])
    else:
        base_chunk = args.chunk_size if args.chunk_size is not None else cfg["em"]["chunk_size"]
        chunk_size = max(1, base_chunk // k_recipe)
    global_damping = args.global_damping if args.global_damping is not None else jcfg.get("global_damping", 0.0)

    sequences = json.load(open(args.sequences))
    labels = json.load(open(args.labels))
    split = split_mod.load_split(args.split_file)
    sequences = split_mod.filter_sequences(sequences, split, args.split_part)
    by_id = {e["trial_id"]: e for e in labels}
    joined = [by_id[s["trial_id"]] for s in sequences]

    print(f"trials: {len(sequences)}  K={k_subtask} K_R={k_recipe} chunk={chunk_size}", flush=True)

    t0 = time.time()
    # CLI wins, then the config, then the Breakfast default. Both must move together to help on
    # EPIC -- see recipe/lexical_init.cluster_recipes.
    idf_recipes = (args.idf_recipes if args.idf_recipes is not None
                   else bool(cfg.get("joint_em", {}).get("idf_recipes", False)))
    recipe_features = (args.recipe_features
                       or cfg.get("joint_em", {}).get("recipe_features", "pairs"))
    print(f"recipe clustering: features={recipe_features} idf={idf_recipes}", flush=True)

    init_params, info = lexical_init.lexical_to_joint(
        sequences, k_subtask, k_recipe, d_max, cfg["vocab"]["verbs"], cfg["vocab"]["nouns"],
        kappa, seed=args.seed, min_ticks=args.min_pair_ticks,
        anchor=args.anchor, background=args.background,
        alpha_init=args.alpha_init, alpha_trans=args.alpha_trans, alpha_pi=alpha_pi,
        init_prior_scale=args.init_prior_scale, idf_recipes=idf_recipes,
        recipe_features=recipe_features,
    )
    print(f"lexical warm start: {time.time() - t0:.1f}s, "
          f"{len(info['pairs'])} (verb,noun) states used of K={k_subtask}", flush=True)
    print(f"  init cluster sizes: {np.bincount(info['assign'], minlength=k_recipe)}", flush=True)
    init_ari = recipe_hmm.adjusted_rand(info["assign"], [e["recipe_label"] for e in joined])
    print(f"  init bag-of-pairs recipe ARI (scoring only): {init_ari:.4f}", flush=True)

    t0 = time.time()
    best, obj, history, converged = joint_em.run_joint_em(
        init_params, sequences, d_max, alpha_pi=alpha_pi, kappa=kappa,
        alpha_init=args.alpha_init, alpha_trans=args.alpha_trans,
        max_iters=args.max_iters, tol=args.tol, chunk_size=chunk_size, progress=False,
        global_damping=global_damping,
        emit_prior_v=info["emit_prior_v"], emit_prior_n=info["emit_prior_n"],
    )
    print(f"joint EM: obj={float(obj):.1f} iters={len(history)} converged={converged} "
          f"elapsed={time.time() - t0:.1f}s", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    joint_params.save_params(best, args.out)
    with open(args.out + ".meta.json", "w") as f:
        json.dump({"iteration": len(history), "history": history, "converged": bool(converged),
                   "args": vars(args), "n_pairs": len(info["pairs"]), "init_recipe_ari": init_ari}, f)
    print(f"saved {args.out}", flush=True)

    verb_ids, noun_ids, mask = joint_em.pad_batch(sequences)
    r_hat, _, _ = joint_em.infer_recipe(best, verb_ids, noun_ids, mask, d_max, chunk_size=chunk_size)
    true_recipes = [e["recipe_label"] for e in joined]
    print(f"\nrecipe ARI: {recipe_hmm.adjusted_rand(np.asarray(r_hat), true_recipes):.4f}")
    acc, _table, _pv, _tv = recipe_hmm.matched_accuracy(np.asarray(r_hat), true_recipes)
    print(f"recipe matched accuracy: {acc:.4f}")
    print(f"effective K_recipe: {recipe_hmm.effective_k(np.asarray(r_hat))} (nominal {k_recipe})")

    lp = joint_params.to_log_probs_joint(best, d_max)
    seg = segmentize.segment_all_conditioned(lp, r_hat, verb_ids, noun_ids, mask, d_max)
    true_tick = np.concatenate([np.array(e["subtask_labels"]) for e in joined])
    pred_tick = np.concatenate([r["subtask_per_tick"] for r in seg])
    print(f"per-tick subtask ARI: {recipe_hmm.adjusted_rand(pred_tick, true_tick):.4f}")


if __name__ == "__main__":
    main()
