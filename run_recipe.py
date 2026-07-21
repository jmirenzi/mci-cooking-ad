import argparse
import json
import time

import jax
import numpy as np

from cook_ad.data.config import load_config
from cook_ad.hsmm import em, params
from cook_ad.recipe import recipe_hmm, segmentize


def two_stage_infer(hsmm_params, recipe_params, verb_ids, noun_ids, d_max):
    """Unified two-stage inference for a single trial: (v,n) stream -> (subtask, recipe).

    verb_ids, noun_ids: (T,) int arrays for one trial (no padding). Returns
    (subtask_per_tick: (T,) int64, recipe_id: int).
    """
    import jax.numpy as jnp

    verb_ids = jnp.asarray(verb_ids)[None, :]
    noun_ids = jnp.asarray(noun_ids)[None, :]
    mask = jnp.ones(verb_ids.shape, dtype=bool)

    result = segmentize.segment_all(hsmm_params, verb_ids, noun_ids, mask, d_max)[0]
    subtask_symbols = [state for state, _ in result["segments"]]

    obs_ids, seg_mask = recipe_hmm.pad_segment_batch([subtask_symbols])
    recipe_id = int(recipe_hmm.decode_recipe(recipe_params, obs_ids, seg_mask)[0])

    return result["subtask_per_tick"], recipe_id


def _load_and_join(sequences_path, labels_path):
    with open(sequences_path) as f:
        sequences = json.load(f)
    with open(labels_path) as f:
        labels = json.load(f)

    labels_by_id = {entry["trial_id"]: entry for entry in labels}
    joined_labels = [labels_by_id[seq["trial_id"]] for seq in sequences]
    return sequences, joined_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/breakfast_mini.yaml")
    parser.add_argument("--params", default="dataset/processed/breakfast_mini/hsmm_params.npz")
    parser.add_argument("--sequences", default="dataset/processed/breakfast_mini/sequences.json")
    parser.add_argument("--labels", default="dataset/processed/breakfast_mini/labels.json")
    parser.add_argument("--out", default="dataset/processed/breakfast_mini/recipe_params.npz")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-restarts", type=int, default=10)
    parser.add_argument("--max-iters", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-4)
    args = parser.parse_args()

    config = load_config(args.config)
    d_max = config["duration"]["d_max_ticks"]
    k_subtask = config["k_subtask"]
    k_recipe = config["k_recipe"]

    sequences, joined_labels = _load_and_join(args.sequences, args.labels)
    hsmm_params = params.load_params(args.params)

    print(f"trials: {len(sequences)}")

    start = time.time()
    verb_ids, noun_ids, mask = em.pad_batch(sequences)
    seg_results = segmentize.segment_all(hsmm_params, verb_ids, noun_ids, mask, d_max)
    seg_elapsed = time.time() - start
    print(f"Viterbi segmentation: {seg_elapsed:.1f}s")

    seg_sequences = [[state for state, _ in r["segments"]] for r in seg_results]

    key = jax.random.PRNGKey(args.seed)
    start = time.time()
    best_recipe_params, best_loglik, history = recipe_hmm.run_em(
        key,
        seg_sequences,
        k_recipe=k_recipe,
        k_subtask=k_subtask,
        n_restarts=args.n_restarts,
        max_iters=args.max_iters,
        tol=args.tol,
        progress=True,
    )
    em_elapsed = time.time() - start
    print(f"recipe EM: restarts={args.n_restarts}, best log-likelihood={float(best_loglik):.1f}, "
          f"elapsed={em_elapsed:.1f}s")

    obs_ids, seg_mask = recipe_hmm.pad_segment_batch(seg_sequences)
    pred_recipes = recipe_hmm.decode_recipe(best_recipe_params, obs_ids, seg_mask)
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

    true_subtask_per_tick = np.concatenate(
        [np.array(entry["subtask_labels"]) for entry in joined_labels]
    )
    pred_subtask_per_tick = np.concatenate([r["subtask_per_tick"] for r in seg_results])
    subtask_ari = recipe_hmm.adjusted_rand(pred_subtask_per_tick, true_subtask_per_tick)
    print(f"\nper-tick subtask ARI (segmentation sanity check): {subtask_ari:.4f}")

    recipe_hmm.save_params(best_recipe_params, args.out)
    print(f"\nsaved recipe params to {args.out}")


if __name__ == "__main__":
    main()
