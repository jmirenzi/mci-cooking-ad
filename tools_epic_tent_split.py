"""The two EPIC-Tent split derivations the primary seeded split cannot produce.

**split_prego.json** -- the PREGO-comparable train/test split, from the per-participant
uncertainty ratings.

**split_clean.json** -- the same train/test membership, expressed over the error-excised
fragments in sequences_clean.json. Those carry ids like `p18_f0`, so data/split.filter_sequences
would match none of them against a split listing `p18`. Membership is INHERITED rather than
re-drawn: a fragment must land on the same side as the session it was cut from, or the
contamination ablation would be confounded with a different split.

PREGO's split is EXACT, not reconstructed. Their CVPR'24 supplementary, section C: "The videos
that form the test set have a median confidence score under 0.6, while the others form the train
set. Only 22 videos have the confidence score annotations, while the remaining 7 do not and are
assigned to the test set." The release ships UNCERTAINTY (0 certain .. 1 uncertain), so
confidence is 1 - uncertainty, and the rule is: test = unrated (1-7) + median confidence < 0.6.

That yields 14 train / 15 test, and the 15 test ids match the keys of PREGO's own
step_anticipation/data/predictions/output_miniROAD_Epic-tent-O.json exactly
(1-7, 10, 16, 17, 18, 20, 23, 25, 28). Nearest participants to the boundary sit at 0.583 and
0.627, so the assignment is not a knife-edge. An earlier version of this tool put the 7 unrated
participants in TRAIN and ranked by mean; that was wrong on both counts.

Two caveats, both recorded in the output file:
  - Most uncertainty files carry exactly 2 rows more than the release README's
    total_frame_count, which is a systematic off-by-two and harmless. Six disagree by more than
    1%, and two of those look like mis-assigned copies: participant 14's file has 54418 rows,
    which is participant 9's frame count, and 29's has 43166, which is 8's. PREGO read the same
    files, so reproducing their split means using them as shipped; for the record, neither
    suspect participant is near the 0.6 boundary (14: 0.987, 29: 0.976).
  - The primary split (split_dataset.py, seeded 20/9) is what the headline numbers should use.
    This one exists only for the PREGO comparison.

    ./py tools_epic_tent_split.py
"""
import argparse
import json
import statistics
from pathlib import Path

# From the release README. A disagreement with a file's row count is reported, not corrected.
# Tolerance is relative because nearly every file is +2 rows, uniformly.
ROW_COUNT_TOLERANCE = 0.01
TOTAL_FRAME_COUNT = {
    1: 45650, 2: 41294, 3: 82300, 4: 45985, 5: 40473, 6: 28796, 7: 34647, 8: 43164,
    9: 54416, 10: 41136, 11: 53046, 12: 58017, 13: 43921, 14: 36069, 15: 37302, 16: 62585,
    17: 50455, 18: 78899, 19: 37903, 20: 51690, 21: 33192, 22: 22888, 23: 57554, 24: 89397,
    25: 89944, 26: 47660, 27: 45734, 28: 131588, 29: 31717,
}

# PREGO supplementary C: test = median confidence under this threshold, plus every unrated video.
CONFIDENCE_THRESHOLD = 0.6


def median_confidence(path):
    """-> (median confidence, n_rows). The file holds UNCERTAINTY -- three whitespace-separated
    columns: participant, GoPro timecode, rating in [0, 1] with 1 = most uncertain -- so
    confidence is its complement. MEDIAN, not mean: that is the statistic PREGO names."""
    values = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 3:
            values.append(float(parts[2]))
    return (1.0 - statistics.median(values) if values else None), len(values)


def build(root, trial_ids, threshold=CONFIDENCE_THRESHOLD):
    """-> (train_ids, test_ids, ranking, warnings). PREGO's rule: an unrated participant goes to
    TEST, a rated one goes to test when its median confidence is below `threshold`."""
    scores, rows, warnings = {}, {}, []
    for trial_id in trial_ids:
        subject = int(trial_id.lstrip("p"))
        path = Path(root) / "uncertainty" / f"{subject:02d}.uncertainty.gopro_timecode.txt"
        if not path.exists():
            continue
        confidence, n = median_confidence(path)
        scores[trial_id], rows[trial_id] = confidence, n
        expected = TOTAL_FRAME_COUNT.get(subject)
        if expected is not None and abs(n - expected) > ROW_COUNT_TOLERANCE * expected:
            warnings.append(f"{trial_id}: {n} rating rows, README frame count {expected} "
                            f"({100 * (n - expected) / expected:+.0f}%)")

    train = sorted(t for t in scores if scores[t] >= threshold)
    test = sorted(set(trial_ids) - set(train))
    ranking = [{"trial_id": t, "median_confidence": round(scores[t], 4), "n_rows": rows[t]}
               for t in sorted(scores, key=lambda t: -scores[t])]
    return train, test, ranking, warnings


def expand_to_fragments(split, clean_sequences):
    """A session-level split -> the same split over excised fragments, by id prefix.

    A fragment id is `{trial_id}_f{i}`, so the session is everything before the last `_f`.
    Fragments whose session is in neither partition are dropped rather than guessed at.
    """
    session_of = {}
    for sequence in clean_sequences:
        fragment_id = sequence["trial_id"]
        session_of[fragment_id] = fragment_id.rsplit("_f", 1)[0]

    out = {}
    for part in ("train", "test"):
        members = set(split[f"{part}_trial_ids"])
        out[f"{part}_trial_ids"] = sorted(
            f for f, session in session_of.items() if session in members
        )
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="dataset/epic_tent")
    ap.add_argument("--sequences", default="dataset/processed/epic_tent/sequences.json")
    ap.add_argument("--out", default="dataset/processed/epic_tent/split_prego.json")
    ap.add_argument("--split-file", default="dataset/processed/epic_tent/split.json",
                    help="the session-level split whose membership split_clean.json inherits")
    ap.add_argument("--clean-sequences",
                    default="dataset/processed/epic_tent/sequences_clean.json")
    ap.add_argument("--clean-out", default="dataset/processed/epic_tent/split_clean.json")
    args = ap.parse_args()

    trial_ids = [s["trial_id"] for s in json.loads(Path(args.sequences).read_text())]
    train, test, ranking, warnings = build(args.root, trial_ids)
    if not ranking:
        raise SystemExit(f"no uncertainty files under {args.root}/uncertainty -- run "
                         "tools_fetch_epic_tent.py --uncertainty")

    Path(args.out).write_text(json.dumps({
        "train_trial_ids": train,
        "test_trial_ids": test,
        "rule": "PREGO (CVPR'24) supplementary C: test = unrated participants (1-7) + median "
                f"confidence < {CONFIDENCE_THRESHOLD}; confidence = 1 - shipped uncertainty",
        "verified_against": "PREGO step_anticipation/data/predictions/"
                            "output_miniROAD_Epic-tent-O.json -- same 15 test ids",
        "n_train": len(train), "n_test": len(test),
        "ranking_most_confident_first": ranking,
        "row_count_warnings": warnings,
    }, indent=2))

    print(f"train: {len(train)} trials {train}")
    print(f"test:  {len(test)} trials {test}")
    print(f"\nrated participants, most confident first (threshold {CONFIDENCE_THRESHOLD}):")
    for entry in ranking:
        side = "train" if entry["median_confidence"] >= CONFIDENCE_THRESHOLD else "test"
        print(f"  {entry['trial_id']}  {entry['median_confidence']:.4f}  {side:<5} "
              f"({entry['n_rows']} rows)")
    if warnings:
        print("\nrow counts disagreeing with the release README:")
        for warning in warnings:
            print(f"  {warning}")
    print(f"\nsaved split to {args.out}")

    split = json.loads(Path(args.split_file).read_text())
    clean = json.loads(Path(args.clean_sequences).read_text())
    expanded = expand_to_fragments(split, clean)
    Path(args.clean_out).write_text(json.dumps({
        **expanded,
        "derived_from": args.split_file,
        "rule": "session-level membership inherited by every fragment cut from that session",
    }, indent=2))
    print(f"split_clean: {len(expanded['train_trial_ids'])} train fragments, "
          f"{len(expanded['test_trial_ids'])} test fragments -> {args.clean_out}")


if __name__ == "__main__":
    main()
