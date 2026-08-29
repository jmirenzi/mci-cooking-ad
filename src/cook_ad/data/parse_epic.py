"""EPIC-KITCHENS-100 annotations -> the same sequences/labels/vocab contract parse_breakfast.py
emits, so every runner, config path and eval script works unchanged.

Annotations only -- no video is downloaded or read. Clone
https://github.com/epic-kitchens/epic-kitchens-100-annotations (~89 MB) and point --root at it.

Four things differ from Breakfast, and each is a decision rather than a detail:

1. **A trial is a video** -- EPIC's protocol was to record every kitchen visit, so one video is
   one visit. But a visit is not a GOAL: a long one may be cook-then-wash-up-then-unpack, which
   breaks the joint model's one-recipe-per-trial assumption. The duration filter keeps visits
   short enough to be plausibly single-purpose (2-15 minutes keeps 357 of 633).

2. **Narrations overlap** -- 28% of consecutive pairs. The HSMM emits one state per tick, so the
   ties must be broken; see resolve_overlaps for why first-come-wins is the only option.

3. **There are no recipe labels**, so `recipe_label` carries a derived dish (see
   derive_dish_labels) and `participant_label` is kept alongside as a control. Both are
   validation-only -- nothing trains on labels.json. The dish label is derived from the
   observations, so scoring a clustering built from the same features against it is partly
   circular.

4. **fps varies per video** (29.97 to 90), so timestamps rather than frame indices are the
   source of truth, re-quantised onto one synthetic FPS_REF grid. That lets
   `tick_expansion.expand_verb_noun_to_ticks` be reused rather than reimplemented.

    ./py -m cook_ad.data.parse_epic --root dataset/epic_kitchens/annotations \
        --config configs/epic.yaml --out-dir dataset/processed/epic
"""
import argparse
import csv
import json
import logging
from pathlib import Path

from cook_ad.data.config import load_config
from cook_ad.data.labels import build_trial_labels
from cook_ad.data.parse_breakfast import trim_terminal_idle
from cook_ad.data.tick_expansion import expand_verb_noun_to_ticks

logger = logging.getLogger(__name__)

# Synthetic frame grid: EPIC's real fps varies per video so its frame columns cannot be pooled.
# 20 Hz is an order of magnitude finer than the 0.5s tick, so binning is unaffected, and keeps
# expand_to_ticks' per-frame loop cheap.
FPS_REF = 20

# Annotated splits only. EPIC_100_test_timestamps.csv carries no verb/noun labels (it is the
# held-out challenge set), so there is nothing to build a sequence from.
ANNOTATION_FILES = ("EPIC_100_train.csv", "EPIC_100_validation.csv")


def parse_timestamp(text):
    """'00:01:02.30' -> seconds."""
    hours, minutes, seconds = text.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def load_narrations(root):
    """-> {video_id: {"participant": str, "narrations": [(t0, t1, verb_class, noun_class)]}},
    each narration list sorted by start time."""
    videos = {}
    for name in ANNOTATION_FILES:
        path = Path(root) / name
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found -- clone epic-kitchens/epic-kitchens-100-annotations and "
                f"pass its directory as --root"
            )
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                entry = videos.setdefault(
                    row["video_id"], {"participant": row["participant_id"], "narrations": []}
                )
                entry["narrations"].append((
                    parse_timestamp(row["start_timestamp"]),
                    parse_timestamp(row["stop_timestamp"]),
                    int(row["verb_class"]),
                    int(row["noun_class"]),
                ))
    for entry in videos.values():
        entry["narrations"].sort(key=lambda n: (n[0], n[1]))
    return videos


# EPIC's own noun-class categories that name something edible. The rest -- appliances, crockery,
# cutlery, cleaning equipment -- are 91% of all observations and barely vary with the dish, which
# is why a dish label has to be read off the food nouns alone.
FOOD_CATEGORIES = frozenset({
    "vegetables", "fruits and nuts", "meat and substitute", "dairy and eggs",
    "baked goods and grains", "spices and herbs and sauces", "prepared food", "drinks",
})

NO_DISH = "unknown"


def derive_dish_labels(videos, noun_categories, noun_keys):
    """{video_id: dish} -- the food noun with the highest TF-IDF for that session. TF so a long
    session does not outscore a short one, IDF so a noun everybody touches (water, oil) loses to
    one that distinguishes this session. No food mentioned at all gives NO_DISH."""
    import math

    per_video = {}
    for video_id, entry in videos.items():
        counts = {}
        for _t0, _t1, _verb, noun in entry["narrations"]:
            if noun_categories[noun] in FOOD_CATEGORIES:
                counts[noun] = counts.get(noun, 0) + 1
        per_video[video_id] = counts

    with_food = [v for v, c in per_video.items() if c]
    doc_freq = {}
    for v in with_food:
        for noun in per_video[v]:
            doc_freq[noun] = doc_freq.get(noun, 0) + 1

    labels = {}
    for video_id, counts in per_video.items():
        if not counts:
            labels[video_id] = NO_DISH
            continue
        total = sum(counts.values())
        best = max(counts, key=lambda n: (counts[n] / total) * math.log(len(with_food) / doc_freq[n]))
        labels[video_id] = noun_keys[best]
    return labels


def load_classes(root, verb_file="EPIC_100_verb_classes.csv", noun_file="EPIC_100_noun_classes_v2.csv"):
    """-> (verb_keys, noun_keys, noun_categories) indexed by EPIC's own class id. Published ids
    are kept rather than re-derived from the filtered subset, so changing the duration filter
    cannot silently renumber the vocab. Absent classes get an all-zero emission column."""
    def read(path, col="key"):
        with open(Path(root) / path, newline="") as f:
            rows = {int(r["id"]): r[col] for r in csv.DictReader(f)}
        return [rows[i] for i in range(max(rows) + 1)]

    return read(verb_file), read(noun_file), read(noun_file, "category")


def build_vocab(verb_keys, noun_keys, sil_verb, sil_noun):
    """EPIC class ids verbatim, SIL appended at the end so a real class never changes id. Raises
    on a name collision -- Breakfast's SIL noun is 'kitchen', which IS a real EPIC noun class."""
    for name, keys, what in ((sil_verb, verb_keys, "verb"), (sil_noun, noun_keys, "noun")):
        if name in keys:
            raise ValueError(
                f"SIL {what} {name!r} collides with a real EPIC {what} class (id "
                f"{keys.index(name)}); choose another in the config's ambient_gaps block"
            )
    verb_to_id = {key: i for i, key in enumerate(verb_keys)}
    noun_to_id = {key: i for i, key in enumerate(noun_keys)}
    verb_to_id[sil_verb] = len(verb_keys)
    noun_to_id[sil_noun] = len(noun_keys)
    return verb_to_id, noun_to_id


def resolve_overlaps(narrations):
    """Sorted, possibly-overlapping narrations -> disjoint ones, first-come-wins: a narration's
    start is pushed past whatever already occupies those seconds, and one swallowed entirely by
    its predecessor is dropped. Returns (kept, n_swallowed).

    Letting the later narration interrupt the earlier instead would turn one segment into
    `A B A`, and `params._row_normalize(mask_diag=True)` bans self-transitions structurally, so
    that re-entry is impossible rather than merely unlikely.
    """
    kept = []
    swallowed = 0
    end = float("-inf")
    for t0, t1, verb, noun in narrations:
        start = max(t0, end)
        if start >= t1:
            swallowed += 1
            continue
        kept.append((start, t1, verb, noun))
        end = max(end, t1)
    return kept, swallowed


def to_frame_segments(narrations, sil_ids):
    """Disjoint (t0, t1, verb, noun) in seconds -> contiguous 1-indexed inclusive frame segments
    on the FPS_REF grid, with SIL filling every gap. Contiguity is required: expand_to_ticks
    majority-votes over a dense frame array, so an uncovered frame would vote `None`.
    Returns (segments, n_too_short)."""
    origin = narrations[0][0]
    sil_verb_id, sil_noun_id = sil_ids
    segments, too_short = [], 0
    cursor = 1
    for t0, t1, verb, noun in narrations:
        start = int(round((t0 - origin) * FPS_REF)) + 1
        stop = int(round((t1 - origin) * FPS_REF))
        start = max(start, cursor)
        if stop < start:
            too_short += 1
            continue
        if start > cursor:
            segments.append((sil_verb_id, sil_noun_id, cursor, start - 1))
        segments.append((verb, noun, start, stop))
        cursor = stop + 1
    return segments, too_short


def build_dataset(config, root, trim_terminal=True):
    """-> (sequences, labels, vocab), each ready to serialize as-is -- the same three objects
    parse_breakfast.build_dataset returns, in the same shapes."""
    data_cfg = config["data"]
    tick_seconds = config["tick_seconds"]
    sil_verb = config["ambient_gaps"]["sil_verb"]
    sil_noun = config["ambient_gaps"]["sil_noun"]
    lo, hi = data_cfg["min_session_minutes"] * 60.0, data_cfg["max_session_minutes"] * 60.0

    verb_keys, noun_keys, noun_categories = load_classes(root)
    verb_to_id, noun_to_id = build_vocab(verb_keys, noun_keys, sil_verb, sil_noun)
    sil_ids = (verb_to_id[sil_verb], noun_to_id[sil_noun])

    videos = load_narrations(root)
    dishes = derive_dish_labels(videos, noun_categories, noun_keys)
    sequences, labels = [], []
    stats = dict(seen=len(videos), filtered=0, swallowed=0, too_short=0, narrations=0, trimmed=0)

    for video_id, entry in sorted(videos.items()):
        narrations = entry["narrations"]
        span = narrations[-1][1] - narrations[0][0]
        if not (lo <= span <= hi):
            continue
        stats["filtered"] += 1
        stats["narrations"] += len(narrations)

        kept, swallowed = resolve_overlaps(narrations)
        stats["swallowed"] += swallowed
        segments, too_short = to_frame_segments(kept, sil_ids)
        stats["too_short"] += too_short

        verb_ids, noun_ids = expand_verb_noun_to_ticks(segments, FPS_REF, tick_seconds)
        trimmed = 0
        if trim_terminal:
            verb_ids, noun_ids, trimmed = trim_terminal_idle(verb_ids, noun_ids, *sil_ids)
            stats["trimmed"] += trimmed
        if not verb_ids:
            continue

        seq = {"trial_id": video_id, "verb_ids": verb_ids, "noun_ids": noun_ids}
        if trim_terminal:
            seq["terminal_idle_ticks"] = trimmed
        sequences.append(seq)

        # `{verb}_{noun}` mirrors Breakfast's `pour_milk`. The free-text narration is not used:
        # it is unnormalised ("take carrots and") and would fragment the ground-truth grouping.
        named = [
            (f"{verb_keys[v]}_{noun_keys[n]}" if v < len(verb_keys) else f"{sil_verb}_{sil_noun}",
             verb_keys[v] if v < len(verb_keys) else sil_verb,
             noun_keys[n] if n < len(noun_keys) else sil_noun, start, end)
            for v, n, start, end in segments
        ]
        trial_labels = build_trial_labels(video_id, dishes[video_id], named, FPS_REF, tick_seconds)
        trial_labels["participant_label"] = entry["participant"]   # control, see docstring (3)
        labels.append(trial_labels)

    _check_against_config(config, verb_to_id, noun_to_id)
    vocab = {
        "verbs": verb_to_id,
        "nouns": noun_to_id,
        # Derived dish labels, not an EPIC annotation (see this module's docstring, point 3).
        "recipes": {d: i for i, d in enumerate(sorted(set(dishes.values())))},
    }
    return sequences, labels, vocab, stats


def _check_against_config(config, verb_to_id, noun_to_id):
    for name, actual, expected in (
        ("verb vocab size", len(verb_to_id), config["vocab"]["verbs"]),
        ("noun vocab size", len(noun_to_id), config["vocab"]["nouns"]),
    ):
        if actual != expected:
            raise ValueError(f"{name} {actual} does not match the config value {expected}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="dataset/epic_kitchens/annotations")
    ap.add_argument("--config", default="configs/epic.yaml")
    ap.add_argument("--out-dir", default="dataset/processed/epic")
    ap.add_argument("--keep-terminal-idle", action="store_true",
                    help="do NOT strip each session's trailing SIL run -- see "
                         "parse_breakfast.trim_terminal_idle")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO)
    config = load_config(args.config)
    sequences, labels, vocab, stats = build_dataset(
        config, args.root, trim_terminal=not args.keep_terminal_idle
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "vocab.json").write_text(json.dumps(vocab, indent=2))
    (out_dir / "sequences.json").write_text(json.dumps(sequences, indent=2))
    (out_dir / "labels.json").write_text(json.dumps(labels, indent=2))

    n = max(stats["narrations"], 1)
    logger.info("sessions: %d of %d within the duration band", stats["filtered"], stats["seen"])
    logger.info("narrations: %d kept, %d swallowed by an overlap (%.2f%%), %d shorter than a "
                "frame (%.2f%%)", stats["narrations"] - stats["swallowed"] - stats["too_short"],
                stats["swallowed"], 100 * stats["swallowed"] / n, stats["too_short"],
                100 * stats["too_short"] / n)
    logger.info("wrote %d sequences to %s (%d terminal idle ticks trimmed)",
                len(sequences), out_dir, stats["trimmed"])


if __name__ == "__main__":
    main()
