import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default="dataset/processed/breakfast")
    parser.add_argument("--out-dir", default="dataset/processed/breakfast_mini")
    parser.add_argument("--recipes", nargs="+", default=["cereals", "coffee", "tea"])
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    with open(source_dir / "sequences.json") as f:
        sequences = json.load(f)
    with open(source_dir / "labels.json") as f:
        labels = json.load(f)

    recipes = set(args.recipes)
    label_by_trial = {label["trial_id"]: label for label in labels}
    keep_ids = {
        trial_id for trial_id, label in label_by_trial.items() if label["recipe_label"] in recipes
    }

    mini_sequences = [seq for seq in sequences if seq["trial_id"] in keep_ids]
    mini_labels = [label_by_trial[trial_id] for trial_id in keep_ids]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sequences.json").write_text(json.dumps(mini_sequences, indent=2))
    (out_dir / "labels.json").write_text(json.dumps(mini_labels, indent=2))
    # vocab is unchanged (same global verb/noun/recipe id space, just fewer trials use it)
    (out_dir / "vocab.json").write_text((source_dir / "vocab.json").read_text())

    print(f"recipes: {sorted(recipes)}")
    print(f"trials: {len(mini_sequences)} (from {len(sequences)})")
    print(f"T_max: {max(len(s['verb_ids']) for s in mini_sequences)}")
    print(f"wrote {out_dir}/{{sequences,labels,vocab}}.json")


if __name__ == "__main__":
    main()
