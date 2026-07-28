import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from cook_ad.data.config import load_config
from cook_ad.hsmm import joint_em, joint_params, params
from cook_ad.recipe import recipe_hmm, segmentize, warm_start

jax.config.update("jax_enable_x64", True)


def _load_and_join(sequences_path, labels_path):
    with open(sequences_path) as f:
        sequences = json.load(f)
    with open(labels_path) as f:
        labels = json.load(f)

    labels_by_id = {entry["trial_id"]: entry for entry in labels}
    joined_labels = [labels_by_id[seq["trial_id"]] for seq in sequences]
    return sequences, joined_labels


def _random_init(key, k_recipe, k_subtask, vocab_verbs, vocab_nouns, d_max):
    """Fallback iteration-0 state when joint_em.warm_start is disabled: each recipe gets an
    INDEPENDENT random draw (params.init_weak_limit_params with a different key per recipe),
    not the same draw broadcast K_R times -- a symmetric prior plus a symmetric likelihood
    never breaks symmetry on its own (see init_weak_limit_params's own docstring), so without
    independent per-recipe randomness the E-step would see identical recipes at iteration 0
    and never separate them. Emissions are shared, so their per-recipe draws are averaged down
    to one rather than arbitrarily picking a single recipe's draw.
    """
    keys = jax.random.split(key, k_recipe)
    per_recipe = [
        params.init_weak_limit_params(keys[r], k_subtask, vocab_verbs, vocab_nouns, d_max) for r in range(k_recipe)
    ]
    init_counts = jnp.stack([p.init_counts for p in per_recipe])
    trans_counts = jnp.stack([p.trans_counts for p in per_recipe])
    dur_r = jnp.stack([p.dur_r for p in per_recipe])
    dur_p = jnp.stack([p.dur_p for p in per_recipe])
    verb_counts = jnp.mean(jnp.stack([p.verb_counts for p in per_recipe]), axis=0)
    noun_counts = jnp.mean(jnp.stack([p.noun_counts for p in per_recipe]), axis=0)
    pi_counts = jnp.full((k_recipe,), 1.0)
    return joint_params.JointHSMMParams(
        init_counts, trans_counts, verb_counts, noun_counts, dur_r, dur_p, pi_counts
    )


def _print_warm_start_sanity(init_params, d_max, k_recipe):
    """Iteration-0 differentiation check: identical per-recipe init rows would mean EM starts
    with uniform responsibilities and never separates recipes (spec's sharpest warm-start
    failure mode). Prints pairwise L1 distance between recipes' normalized init distributions."""
    log_probs0 = joint_params.to_log_probs_joint(init_params, d_max)
    init_probs = np.exp(np.asarray(log_probs0.log_init))
    diffs = [
        float(np.abs(init_probs[a] - init_probs[b]).sum())
        for a in range(k_recipe) for b in range(a + 1, k_recipe)
    ]
    print(
        f"warm-start init differentiation (pairwise L1 over recipes): "
        f"min={min(diffs):.3f} mean={np.mean(diffs):.3f} max={max(diffs):.3f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/breakfast_mini.yaml")
    parser.add_argument("--sequences", default="dataset/processed/breakfast_mini/sequences.json")
    parser.add_argument("--labels", default="dataset/processed/breakfast_mini/labels.json")
    parser.add_argument("--out", default="dataset/processed/breakfast_mini/joint_params.npz")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--tol", type=float, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    d_max = config["duration"]["d_max_ticks"]
    k_subtask = config["k_subtask"]
    k_recipe = config["k_recipe"]
    kappa = config["duration"]["shrinkage_kappa"]
    alpha_pi = config["prior"]["alpha_pi"]

    jcfg = config["joint_em"]
    max_iters = args.max_iters if args.max_iters is not None else jcfg["max_iters"]
    tol = args.tol if args.tol is not None else jcfg["tol"]
    chunk_size = max(1, config["em"]["chunk_size"] // k_recipe)

    sequences, joined_labels = _load_and_join(args.sequences, args.labels)
    print(f"trials: {len(sequences)}")
    print(f"k_subtask={k_subtask}, k_recipe={k_recipe}, chunk_size={chunk_size}")

    key = jax.random.PRNGKey(args.seed)

    if jcfg["warm_start"]:
        hsmm_params = params.load_params(jcfg["cascade_hsmm_params"])
        recipe_params = recipe_hmm.load_params(jcfg["cascade_recipe_params"])

        start = time.time()
        init_params = warm_start.cascade_to_joint(
            hsmm_params, recipe_params, sequences, d_max, k_recipe, kappa, seed=args.seed
        )
        print(f"cascade warm start: {time.time() - start:.1f}s")
        print(f"warm-start pi_counts (empirical recipe fractions): {np.asarray(init_params.pi_counts)}")
        _print_warm_start_sanity(init_params, d_max, k_recipe)

        start = time.time()
        best_params, best_obj, history = joint_em.run_joint_em(
            init_params, sequences, d_max, alpha_pi=alpha_pi, kappa=kappa,
            max_iters=max_iters, tol=tol, chunk_size=chunk_size, progress=True,
        )
        print(
            f"joint EM (warm start): best objective={float(best_obj):.1f}, "
            f"iters={len(history)}, elapsed={time.time() - start:.1f}s"
        )
    else:
        n_restarts = jcfg["n_restarts"]
        vocab_verbs = config["vocab"]["verbs"]
        vocab_nouns = config["vocab"]["nouns"]
        restart_keys = jax.random.split(key, n_restarts)

        best_params, best_obj, history = None, -jnp.inf, None
        for i, rk in enumerate(restart_keys):
            init_params = _random_init(rk, k_recipe, k_subtask, vocab_verbs, vocab_nouns, d_max)
            p, obj, hist = joint_em.run_joint_em(
                init_params, sequences, d_max, alpha_pi=alpha_pi, kappa=kappa,
                max_iters=max_iters, tol=tol, chunk_size=chunk_size, progress=True,
            )
            print(f"restart {i + 1}/{n_restarts}: objective={float(obj):.1f}")
            if float(obj) > float(best_obj):
                best_params, best_obj, history = p, obj, hist
        print(f"joint EM (random-init fallback): best objective={float(best_obj):.1f}")

    verb_ids, noun_ids, mask = joint_em.pad_batch(sequences)
    r_hat, rho, trial_ll = joint_em.infer_recipe(best_params, verb_ids, noun_ids, mask, d_max, chunk_size=chunk_size)
    pred_recipes = np.asarray(r_hat)
    true_recipes = [entry["recipe_label"] for entry in joined_labels]

    ari = recipe_hmm.adjusted_rand(pred_recipes, true_recipes)
    accuracy, table, pred_vals, true_vals = recipe_hmm.matched_accuracy(pred_recipes, true_recipes)
    eff_k = recipe_hmm.effective_k(pred_recipes)

    print(f"\nrecipe ARI: {ari:.4f}")
    print(f"recipe matched accuracy: {accuracy:.4f}")
    print(f"effective K_recipe: {eff_k} (nominal {k_recipe})")
    print(f"predicted cluster ids: {list(pred_vals)}")
    print(f"true recipe labels:    {list(true_vals)}")
    print("contingency table (rows=predicted cluster, cols=true recipe):")
    print(table)

    log_probs = joint_params.to_log_probs_joint(best_params, d_max)
    seg_results = segmentize.segment_all_conditioned(log_probs, r_hat, verb_ids, noun_ids, mask, d_max)
    true_subtask_per_tick = np.concatenate(
        [np.array(entry["subtask_labels"]) for entry in joined_labels]
    )
    pred_subtask_per_tick = np.concatenate([r["subtask_per_tick"] for r in seg_results])
    subtask_ari = recipe_hmm.adjusted_rand(pred_subtask_per_tick, true_subtask_per_tick)
    print(f"\nper-tick subtask ARI (recipe-conditioned segmentation): {subtask_ari:.4f}")

    joint_params.save_params(best_params, args.out)
    print(f"\nsaved joint params to {args.out}")


if __name__ == "__main__":
    main()
