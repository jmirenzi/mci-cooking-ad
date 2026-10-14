"""Carve a fit/dev split out of an existing split.json's TRAIN ids, written in the same schema.

Smoothing strength and duration pooling are regularisation: their whole job is generalisation, so
selecting them on the data the model was fit to biases toward under-smoothing. This produces a
nested split whose "train" part is a subset of the outer train ids and whose "test" part is the
held-out dev fold, so every runner's existing --split-file/--split-part plumbing works unchanged
and the outer test split is never touched during selection.
"""
import argparse, json, random
from collections import Counter

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--split", default="dataset/processed/breakfast/split.json")
ap.add_argument("--out", default="dataset/processed/breakfast/split_dev.json")
ap.add_argument("--dev-frac", type=float, default=0.2)
ap.add_argument("--seed", type=int, default=17)
a = ap.parse_args()

outer = json.load(open(a.split))
ids = list(outer["train_trial_ids"])
random.Random(a.seed).shuffle(ids)
n_dev = round(len(ids) * a.dev_frac)
dev, fit = sorted(ids[:n_dev]), sorted(ids[n_dev:])
json.dump({"seed": a.seed, "test_frac": a.dev_frac, "derived_from": a.split,
           "train_trial_ids": fit, "test_trial_ids": dev}, open(a.out, "w"), indent=2)
recipe = lambda t: t.rsplit("_", 1)[1]
print(f"fit {len(fit)} / dev {len(dev)} trials, from {len(ids)} outer-train")
print("dev recipe counts:", dict(Counter(recipe(t) for t in dev)))
print(f"saved {a.out}")
