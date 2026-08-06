import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from cook_ad.data.config import load_config
from cook_ad.hsmm import joint_em, joint_params, params
from cook_ad.recipe import recipe_hmm, segmentize, warm_start

jax.config.update("jax_enable_x64", True)


def _checkpoint_meta_path(out_path):
    return out_path + ".meta.json"


def _save_checkpoint(out_path, joint_hsmm_params, iteration, history, complete=False):
    """Write params + a small JSON sidecar (iteration count, objective history, whether EM had
    genuinely CONVERGED -- tol-based stopping, not merely reaching max_iters -- as of this
    save) atomically: write to a temp path then os.replace, so a process killed mid-write
    (closed laptop, Ctrl+C) never leaves a half-written checkpoint that a later --resume would
    load as valid. Both files are replaced only after both temp writes succeed. `complete` is
    named for the JSON field, which callers read as "was this run fully converged"."""
    meta_path = _checkpoint_meta_path(out_path)
    # np.savez (inside joint_params.save_params) silently APPENDS .npz to any path that
    # doesn't already end in it -- a plain "out_path + '.tmp'" tmp name would actually get
    # written to "out_path + '.tmp.npz'", and the os.replace below would then fail to find
    # the file it just wrote. Keep the tmp name ending in .npz so save_params writes exactly
    # where we tell it to.
    tmp_out = out_path + ".tmp.npz"
    tmp_meta = meta_path + ".tmp"
    joint_params.save_params(joint_hsmm_params, tmp_out)
    with open(tmp_meta, "w") as f:
        json.dump({"iteration": iteration, "history": history, "converged": complete}, f)
    os.replace(tmp_out, out_path)
    os.replace(tmp_meta, meta_path)
    print(f"  [checkpoint] saved at iteration {iteration}, obj={history[-1]:.1f}", flush=True)


def _load_checkpoint(out_path):
    """Returns (params, iteration, history, converged) if a valid checkpoint exists at
    out_path, else None. Requires BOTH the params file and its meta sidecar -- a params file
    with no meta (e.g. from a run predating --resume support, or a manually-placed file) is
    not treated as a resumable checkpoint. `converged=True` means EM's tol-based stopping
    criterion had genuinely fired as of this checkpoint, not merely that it hit max_iters --
    see run_joint_em's docstring for why that distinction matters on resume."""
    meta_path = _checkpoint_meta_path(out_path)
    if not (os.path.exists(out_path) and os.path.exists(meta_path)):
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    loaded_params = joint_params.load_params(out_path)
    return loaded_params, meta["iteration"], meta["history"], meta.get("converged", False)


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
    parser.add_argument("--checkpoint-every", type=int, default=None,
                        help="iterations between resumable checkpoints at --out; falls back to "
                             "joint_em.checkpoint_every in the config, or 5 if that's absent too")
    parser.add_argument("--restart", action="store_true",
                        help="ignore any existing checkpoint at --out and start over from the "
                             "cascade warm start (default: auto-resume if --out has a valid "
                             "checkpoint from a previous run). Only applies to the warm_start "
                             "path -- the random-init fallback (joint_em.warm_start: false) is "
                             "never resumable.")
    parser.add_argument("--global-damping", type=float, default=None,
                        help="EMA damping (0-1) for the duration M-step's pooled global "
                             "per-state fit across iterations; falls back to "
                             "joint_em.global_damping in the config, or 0.0 (off) if absent. "
                             "Guards against a near-empty state's global fit swinging by an "
                             "order of magnitude between M-steps and dragging every recipe's "
                             "copy of that state with it (see durations.fit_durations_shrunk).")
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
    checkpoint_every = args.checkpoint_every if args.checkpoint_every is not None else jcfg.get("checkpoint_every", 5)
    global_damping = args.global_damping if args.global_damping is not None else jcfg.get("global_damping", 0.0)

    sequences, joined_labels = _load_and_join(args.sequences, args.labels)
    print(f"trials: {len(sequences)}")
    print(f"k_subtask={k_subtask}, k_recipe={k_recipe}, chunk_size={chunk_size}")

    key = jax.random.PRNGKey(args.seed)

    checkpoint = None if (args.restart or not jcfg["warm_start"]) else _load_checkpoint(args.out)

    already_saved = False  # random-restart fallback (only path with no checkpointing) still needs the final save below

    if jcfg["warm_start"]:
        on_checkpoint = lambda it, p, hist: _save_checkpoint(args.out, p, it, hist)  # noqa: E731
        skip_em = False

        if checkpoint is not None:
            init_params, start_iteration, init_history, was_converged = checkpoint
            if was_converged and start_iteration >= max_iters:
                # NOT just "let run_joint_em no-op": its empty-range return always reports
                # converged=False (it never re-ran the tol check), so re-saving through it
                # here would silently downgrade an already-converged checkpoint's flag back
                # to False on every subsequent low-max-iters invocation. Reuse the loaded
                # state directly instead, preserving the real converged=True faithfully.
                print(f"[checkpoint] {args.out} already converged at iteration {start_iteration} "
                      f"-- nothing to do at max_iters={max_iters}. Pass --restart to refit, or "
                      f"raise --max-iters if you want it to keep going past where it converged.")
                best_params, best_obj, history, converged = init_params, init_history[-1], init_history, True
                skip_em = True
            else:
                status = "was already converged, but --max-iters was raised" if was_converged else "not yet converged"
                print(f"[checkpoint] resuming {args.out} from iteration {start_iteration} "
                      f"({status}, last objective={init_history[-1]:.1f}) -- skipping cascade warm start.")
        else:
            hsmm_params = params.load_params(jcfg["cascade_hsmm_params"])
            recipe_params = recipe_hmm.load_params(jcfg["cascade_recipe_params"])

            start = time.time()
            init_params = warm_start.cascade_to_joint(
                hsmm_params, recipe_params, sequences, d_max, k_recipe, kappa, seed=args.seed
            )
            print(f"cascade warm start: {time.time() - start:.1f}s")
            print(f"warm-start pi_counts (empirical recipe fractions): {np.asarray(init_params.pi_counts)}")
            _print_warm_start_sanity(init_params, d_max, k_recipe)
            start_iteration, init_history = 0, []

        if not skip_em:
            start = time.time()
            best_params, best_obj, history, converged = joint_em.run_joint_em(
                init_params, sequences, d_max, alpha_pi=alpha_pi, kappa=kappa,
                max_iters=max_iters, tol=tol, chunk_size=chunk_size, progress=True,
                start_iteration=start_iteration, init_history=init_history,
                init_prev_obj=(init_history[-1] if init_history else None),
                on_checkpoint=on_checkpoint, checkpoint_every=checkpoint_every,
                global_damping=global_damping,
            )
            print(
                f"joint EM (warm start): best objective={float(best_obj):.1f}, "
                f"iters={len(history)}, converged={converged}, elapsed={time.time() - start:.1f}s"
            )
            _save_checkpoint(args.out, best_params, len(history), history, complete=converged)
        already_saved = True
    else:
        n_restarts = jcfg["n_restarts"]
        vocab_verbs = config["vocab"]["verbs"]
        vocab_nouns = config["vocab"]["nouns"]
        restart_keys = jax.random.split(key, n_restarts)

        best_params, best_obj, history = None, -jnp.inf, None
        for i, rk in enumerate(restart_keys):
            init_params = _random_init(rk, k_recipe, k_subtask, vocab_verbs, vocab_nouns, d_max)
            p, obj, hist, _converged = joint_em.run_joint_em(
                init_params, sequences, d_max, alpha_pi=alpha_pi, kappa=kappa,
                max_iters=max_iters, tol=tol, chunk_size=chunk_size, progress=True,
                global_damping=global_damping,
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

    if not already_saved:
        joint_params.save_params(best_params, args.out)
        print(f"\nsaved joint params to {args.out}")


if __name__ == "__main__":
    main()
