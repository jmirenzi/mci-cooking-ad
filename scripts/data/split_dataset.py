import argparse
import json
import random
from collections import Counter


def _recipe_of(trial_id, labels=None):
    """The trial-level categorical the summary groups by. `labels` (trial_id -> recipe_label)
    wins when given; the trailing-token fallback is a Breakfast filename convention
    (`P03_juice` -> `juice`) and is wrong on any corpus that does not follow it."""
    if labels is not None:
        return labels.get(trial_id, "?")
    return trial_id.rsplit("_", 1)[1]


def build_split(trial_ids, test_frac, seed):
    shuffled = list(trial_ids)
    random.Random(seed).shuffle(shuffled)
    n_test = round(len(shuffled) * test_frac)
    test_ids = sorted(shuffled[:n_test])
    train_ids = sorted(shuffled[n_test:])
    return train_ids, test_ids


def _print_summary(train_ids, test_ids, labels=None):
    train_recipes = Counter(_recipe_of(t, labels) for t in train_ids)
    test_recipes = Counter(_recipe_of(t, labels) for t in test_ids)
    all_recipes = sorted(set(train_recipes) | set(test_recipes))

    print(f"train: {len(train_ids)} trials, test: {len(test_ids)} trials")
    print(f"{'recipe':<12}{'train':>8}{'test':>8}")
    missing = []
    for recipe in all_recipes:
        n_train, n_test = train_recipes[recipe], test_recipes[recipe]
        print(f"{recipe:<12}{n_train:>8}{n_test:>8}")
        if n_train == 0 or n_test == 0:
            missing.append((recipe, n_train, n_test))
    if missing:
        print("\nWARNING: the following recipes are entirely absent from one partition "
              "(split is over trial_ids directly, not grouped by participant, so this can happen):")
        for recipe, n_train, n_test in missing:
            side = "test" if n_train == 0 else "train"
            print(f"  {recipe}: missing from {side} (train={n_train}, test={n_test})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences", default="dataset/processed/breakfast/sequences.json")
    ap.add_argument("--out", default="dataset/processed/breakfast/split.json")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--labels", default=None,
                    help="labels.json for this corpus. Affects only the printed summary, "
                         "which then groups by recipe_label instead of guessing from the id.")
    args = ap.parse_args()

    with open(args.sequences) as f:
        sequences = json.load(f)
    trial_ids = [s["trial_id"] for s in sequences]

    labels = None
    if args.labels:
        with open(args.labels) as f:
            labels = {r["trial_id"]: r["recipe_label"] for r in json.load(f)}

    train_ids, test_ids = build_split(trial_ids, args.test_frac, args.seed)
    _print_summary(train_ids, test_ids, labels)

    with open(args.out, "w") as f:
        json.dump(
            {"seed": args.seed, "test_frac": args.test_frac,
             "train_trial_ids": train_ids, "test_trial_ids": test_ids},
            f, indent=2,
        )
    print(f"\nsaved split to {args.out}")


if __name__ == "__main__":
    main()
