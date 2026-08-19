import json
import subprocess
from pathlib import Path

from cook_ad.data import split as split_mod

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_filter_sequences_and_labels():
    sequences = [{"trial_id": "P01_cereals"}, {"trial_id": "P01_coffee"}, {"trial_id": "P02_tea"}]
    labels = [{"trial_id": "P01_cereals", "x": 1}, {"trial_id": "P01_coffee", "x": 2}, {"trial_id": "P02_tea", "x": 3}]
    split = {"train_trial_ids": ["P01_cereals", "P02_tea"], "test_trial_ids": ["P01_coffee"]}

    train_seqs = split_mod.filter_sequences(sequences, split, "train")
    test_seqs = split_mod.filter_sequences(sequences, split, "test")
    assert [s["trial_id"] for s in train_seqs] == ["P01_cereals", "P02_tea"]
    assert [s["trial_id"] for s in test_seqs] == ["P01_coffee"]

    train_labels = split_mod.filter_labels(labels, split, "train")
    assert [entry["x"] for entry in train_labels] == [1, 3]


def _write_fixture_sequences(path):
    trial_ids = [f"P{p:02d}_{recipe}" for p in range(1, 21) for recipe in ("cereals", "coffee")]
    sequences = [{"trial_id": t, "verb_ids": [0], "noun_ids": [0]} for t in trial_ids]
    with open(path, "w") as f:
        json.dump(sequences, f)
    return trial_ids


def _run_split_dataset(sequences_path, out_path, seed=0, test_frac=0.2):
    return subprocess.run(
        ["uv", "run", "python", "split_dataset.py",
         "--sequences", str(sequences_path), "--out", str(out_path),
         "--test-frac", str(test_frac), "--seed", str(seed)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
    )


def test_split_dataset_partitions_disjoint_and_complete(tmp_path):
    sequences_path = tmp_path / "sequences.json"
    out_path = tmp_path / "split.json"
    trial_ids = _write_fixture_sequences(sequences_path)

    result = _run_split_dataset(sequences_path, out_path)
    assert result.returncode == 0, result.stdout + result.stderr

    with open(out_path) as f:
        split = json.load(f)

    train_ids, test_ids = split["train_trial_ids"], split["test_trial_ids"]
    assert set(train_ids).isdisjoint(test_ids)
    assert set(train_ids) | set(test_ids) == set(trial_ids)
    assert len(train_ids) + len(test_ids) == len(trial_ids)
    assert len(test_ids) == round(len(trial_ids) * 0.2)


def test_split_dataset_deterministic_for_fixed_seed(tmp_path):
    sequences_path = tmp_path / "sequences.json"
    _write_fixture_sequences(sequences_path)

    out_a = tmp_path / "split_a.json"
    out_b = tmp_path / "split_b.json"
    assert _run_split_dataset(sequences_path, out_a, seed=7).returncode == 0
    assert _run_split_dataset(sequences_path, out_b, seed=7).returncode == 0

    with open(out_a) as f:
        split_a = json.load(f)
    with open(out_b) as f:
        split_b = json.load(f)

    assert split_a["train_trial_ids"] == split_b["train_trial_ids"]
    assert split_a["test_trial_ids"] == split_b["test_trial_ids"]
