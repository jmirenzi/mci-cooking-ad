import argparse
import json
from collections import defaultdict
from itertools import groupby

import jax
import numpy as np

from cook_ad.anomaly import narrate, surprise
from cook_ad.hsmm import params
from cook_ad.recipe import recipe_hmm

jax.config.update("jax_enable_x64", True)

D_MAX = 200


def _load(dataset_dir):
    with open(f"{dataset_dir}/sequences.json") as f:
        sequences = json.load(f)
    with open(f"{dataset_dir}/labels.json") as f:
        labels = {entry["trial_id"]: entry for entry in json.load(f)}
    with open(f"{dataset_dir}/vocab.json") as f:
        vocab = json.load(f)
    hsmm_params = params.load_params(f"{dataset_dir}/hsmm_params.npz")
    recipe_params = recipe_hmm.load_params(f"{dataset_dir}/recipe_params.npz")
    return sequences, labels, vocab, hsmm_params, recipe_params


def _runs(verb_ids, noun_ids, lexicon):
    """Run-length encode the observed (verb, noun) stream -- tier 1."""
    runs = []
    pos = 0
    for (v, n), group in groupby(zip(verb_ids.tolist(), noun_ids.tolist())):
        length = sum(1 for _ in group)
        runs.append({
            "verb": lexicon.verb(v),
            "noun": lexicon.noun(n),
            "phrase": lexicon.phrase(v, n),
            "start": pos,
            "end": pos + length,
            "n": length,
        })
        pos += length
    return runs


def _segments(segments, lexicon):
    """narrate.segments_from_z output -- [(state, start, end)] -- to tier-2 records. A
    'clipped' segment is one immediately followed by an identical-state segment starting
    exactly where it ends: the d_max=200 right-censoring seam splitting one long action into
    two adjacent Viterbi segments, not two separate visits to the same subtask."""
    out = []
    for i, (state, start, end) in enumerate(segments):
        clipped = (end - start) >= D_MAX and i + 1 < len(segments) and segments[i + 1][0] == state
        out.append({
            "z": int(state),
            "start": int(start),
            "end": int(end),
            "name": lexicon.subtask(state),
            "expected_duration": lexicon.expected_duration(state),
            "clipped": bool(clipped),
        })
    return out


def _recipe_path(segments, recipe_params):
    """Per-segment cascade recipe-HMM state -- NOT a recipe identity, see narrate.py's own
    caveat (K_recipe is a weak-limit nominal with no learned cluster->name map)."""
    if not segments:
        return []
    sym = np.array([s for s, _, _ in segments])
    path = surprise._segment_recipe_path(recipe_params, sym)
    return [{"seg": i, "r": int(r)} for i, r in enumerate(path)]


def build_trial_record(trial, label, hsmm_params, recipe_params, lexicon):
    verb_ids = np.asarray(trial["verb_ids"])
    noun_ids = np.asarray(trial["noun_ids"])
    trace, _, _ = surprise.compute_trace(hsmm_params, recipe_params, verb_ids, noun_ids, D_MAX)
    segments = narrate.segments_from_z(trace.z_star)

    return {
        "trial_id": trial["trial_id"],
        "true_recipe": label["recipe_label"],
        "T": int(len(verb_ids)),
        "runs": _runs(verb_ids, noun_ids, lexicon),
        "segments": _segments(segments, lexicon),
        "recipe_path": _recipe_path(segments, recipe_params),
        "true_subtask_labels": label["subtask_labels"],
    }


def select_trial_per_recipe(sequences, labels):
    """One trial per recipe: near-median length, Viterbi segment count closest to the
    ground-truth segment count (run-length of true subtask_labels) -- the cleanest tier
    1->2 alignment for a display meant to be read, not just technically correct."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="dataset/processed/breakfast")
    parser.add_argument("--out", default="dataset/processed/breakfast/flow/flow.json")
    args = parser.parse_args()

    sequences, labels, vocab, hsmm_params, recipe_params = _load(args.dataset_dir)
    lexicon = narrate.Lexicon(vocab, hsmm_params)

    chosen = select_trial_per_recipe(sequences, labels)
    print("selected trials:")
    for recipe, trial in sorted(chosen.items()):
        print(f"  {recipe:14s} {trial['trial_id']:16s} T={len(trial['verb_ids'])}")

    records = [
        build_trial_record(trial, labels[trial["trial_id"]], hsmm_params, recipe_params, lexicon)
        for trial in chosen.values()
    ]

    out_path = args.out
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"trials": records}, f, indent=2)
    print(f"\nwrote {len(records)} trial records to {out_path}")


if __name__ == "__main__":
    main()
