"""Export the same trials under BOTH the cascade and joint models, on breakfast_mini (the
only checkpoint with a fitted joint_params.npz -- cereals, coffee, tea). Written to compare
subtask granularity and recipe stability against the cascade export, not as a replacement for
it -- the joint model can't be evaluated on sandwich/pancake/etc. until it's fit on the full
dataset (see the plan's Deferred section for that cost)."""
import argparse
import json
import os
from collections import defaultdict
from itertools import groupby

import jax
import jax.numpy as jnp
import numpy as np

from cook_ad.anomaly import narrate
from cook_ad.hsmm import joint_em, joint_params, params
from cook_ad.recipe import recipe_hmm, segmentize

jax.config.update("jax_enable_x64", True)

D_MAX = 50  # breakfast_mini.yaml duration.d_max_ticks


def _load(dataset_dir):
    with open(f"{dataset_dir}/sequences.json") as f:
        sequences = json.load(f)
    with open(f"{dataset_dir}/labels.json") as f:
        labels = {entry["trial_id"]: entry for entry in json.load(f)}
    with open(f"{dataset_dir}/vocab.json") as f:
        vocab = json.load(f)
    return sequences, labels, vocab


def _runs(verb_ids, noun_ids, lexicon):
    runs = []
    pos = 0
    for (v, n), group in groupby(zip(verb_ids.tolist(), noun_ids.tolist())):
        length = sum(1 for _ in group)
        runs.append({
            "verb": lexicon.verb(v), "noun": lexicon.noun(n), "phrase": lexicon.phrase(v, n),
            "start": pos, "end": pos + length, "n": length,
        })
        pos += length
    return runs


def _segments(segments, lexicon):
    out = []
    for state, start, end in segments:
        out.append({
            "z": int(state), "start": int(start), "end": int(end),
            "name": lexicon.subtask(state), "expected_duration": lexicon.expected_duration(state),
        })
    return out


def select_trial_per_recipe(sequences, labels):
    by_recipe = defaultdict(list)
    for s in sequences:
        by_recipe[labels[s["trial_id"]]["recipe_label"]].append(s)
    chosen = {}
    for recipe, trials in sorted(by_recipe.items()):
        lengths = sorted(len(t["verb_ids"]) for t in trials)
        median_len = lengths[len(lengths) // 2]
        true_n_segs = {
            t["trial_id"]: sum(1 for _ in groupby(labels[t["trial_id"]]["subtask_labels"]))
            for t in trials
        }
        chosen[recipe] = min(
            trials,
            key=lambda t: abs(len(t["verb_ids"]) - median_len) * 1000 + true_n_segs[t["trial_id"]],
        )
    return chosen


def build_cascade_record(trial, label, hsmm_params, recipe_params, lexicon):
    from cook_ad.anomaly import surprise

    verb_ids = np.asarray(trial["verb_ids"])
    noun_ids = np.asarray(trial["noun_ids"])
    trace, _, _ = surprise.compute_trace(hsmm_params, recipe_params, verb_ids, noun_ids, D_MAX)
    segments = narrate.segments_from_z(trace.z_star)
    sym = np.array([s for s, _, _ in segments]) if segments else np.array([], dtype=int)
    recipe_path = surprise._segment_recipe_path(recipe_params, sym) if len(sym) else []

    return {
        "trial_id": trial["trial_id"], "true_recipe": label["recipe_label"], "T": int(len(verb_ids)),
        "runs": _runs(verb_ids, noun_ids, lexicon),
        "segments": _segments(segments, lexicon),
        "recipe_path": [{"seg": i, "r": int(r)} for i, r in enumerate(recipe_path)],
    }


def build_joint_record(trial, label, jp, log_probs, lexicon):
    verb_ids = np.asarray(trial["verb_ids"])
    noun_ids = np.asarray(trial["noun_ids"])
    T = len(verb_ids)
    mask = jnp.ones((1, T), dtype=bool)
    v = jnp.asarray(verb_ids)[None, :]
    n = jnp.asarray(noun_ids)[None, :]

    r_hat, rho, trial_ll = joint_em.infer_recipe(jp, v, n, mask, D_MAX)
    r_hat = int(r_hat[0])
    confidence = float(rho[0, r_hat])

    seg_result = segmentize.segment_all_conditioned(log_probs, jnp.array([r_hat]), v, n, mask, D_MAX)[0]
    segments = narrate.segments_from_z(seg_result["subtask_per_tick"])

    return {
        "trial_id": trial["trial_id"], "true_recipe": label["recipe_label"], "T": T,
        "runs": _runs(verb_ids, noun_ids, lexicon),
        "segments": _segments(segments, lexicon),
        "r_hat": r_hat, "confidence": confidence,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="dataset/processed/breakfast_mini")
    parser.add_argument("--out", default="dataset/processed/breakfast_mini/flow/flow_compare.json")
    args = parser.parse_args()

    sequences, labels, vocab = _load(args.dataset_dir)
    chosen = select_trial_per_recipe(sequences, labels)
    print("selected trials:")
    for recipe, trial in sorted(chosen.items()):
        print(f"  {recipe:10s} {trial['trial_id']:16s} T={len(trial['verb_ids'])}")

    cascade_hsmm = params.load_params(f"{args.dataset_dir}/hsmm_params.npz")
    cascade_recipe = recipe_hmm.load_params(f"{args.dataset_dir}/recipe_params.npz")
    cascade_lexicon = narrate.Lexicon(vocab, cascade_hsmm)

    jp = joint_params.load_params(f"{args.dataset_dir}/joint_params.npz")
    joint_log_probs = joint_params.to_log_probs_joint(jp, D_MAX)
    joint_marginal_hsmm = joint_params.collapse_to_marginal(jp)
    joint_lexicon = narrate.Lexicon(vocab, joint_marginal_hsmm)

    records = []
    for trial in chosen.values():
        label = labels[trial["trial_id"]]
        cascade_rec = build_cascade_record(trial, label, cascade_hsmm, cascade_recipe, cascade_lexicon)
        joint_rec = build_joint_record(trial, label, jp, joint_log_probs, joint_lexicon)
        records.append({"trial_id": trial["trial_id"], "cascade": cascade_rec, "joint": joint_rec})

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"trials": records}, f, indent=2)
    print(f"\nwrote {len(records)} comparison records to {args.out}")


if __name__ == "__main__":
    main()
