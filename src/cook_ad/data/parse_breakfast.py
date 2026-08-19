import argparse
import json
import logging
import re
from pathlib import Path

from cook_ad.data.config import load_config
from cook_ad.data.labels import build_trial_labels
from cook_ad.data.tick_expansion import expand_verb_noun_to_ticks

logger = logging.getLogger(__name__)

FILENAME_RE = re.compile(r"^(P\d+)_([a-zA-Z0-9]+)_P\d+_([a-zA-Z0-9]+)\.txt$")
SEGMENT_RE = re.compile(r"(\d+)-(\d+)\s+(\S+)")

# Multiple camera views record the same physical trial; views are byte-identical
# annotation-wise except for a handful of trials. Pick one canonical view per
# trial, in this priority order, so processed output has exactly one sequence
# per (participant, recipe) trial rather than one per file.
CAMERA_PRIORITY = ["cam01", "cam02", "stereo01", "webcam01", "webcam02"]


def find_trial_files(dataset_root):
    """Group annotation files by (participant, recipe) trial; pick one canonical file per trial."""
    trial_views = {}
    for path in sorted(Path(dataset_root).rglob("*.txt")):
        m = FILENAME_RE.match(path.name)
        if not m:
            continue
        participant, camera, recipe = m.groups()
        trial_views.setdefault((participant, recipe), {})[camera] = path

    canonical = {}
    for trial_key, views in trial_views.items():
        for camera in CAMERA_PRIORITY:
            if camera in views:
                canonical[trial_key] = views[camera]
                break
        else:
            raise ValueError(f"No known camera view for trial {trial_key}: {sorted(views)}")

    _warn_on_disagreements(trial_views, canonical)
    return canonical


def _warn_on_disagreements(trial_views, canonical):
    for trial_key, views in trial_views.items():
        if len(views) < 2:
            continue
        contents = {camera: path.read_text() for camera, path in views.items()}
        if len(set(contents.values())) > 1:
            logger.warning(
                "Camera views disagree for trial %s; using %s. Views checked: %s",
                trial_key, canonical[trial_key].name, sorted(views),
            )


def parse_segments(path):
    """Parse one annotation file into (label, start_frame, end_frame) segments."""
    segments = []
    for line in path.read_text().splitlines():
        m = SEGMENT_RE.match(line.strip())
        if not m:
            continue
        start, end, label = m.groups()
        segments.append((label, int(start), int(end)))
    return segments


def label_to_verb_noun(label, sil_verb, sil_noun):
    if label == "SIL":
        return sil_verb, sil_noun
    verb, noun = label.split("_", 1)
    return verb, noun


def build_trials(dataset_root, sil_verb, sil_noun):
    """Returns trial_id -> {"recipe": str, "segments": [(label, verb, noun, start, end)]}."""
    canonical_files = find_trial_files(dataset_root)

    trials = {}
    for (participant, recipe), path in canonical_files.items():
        segments = []
        for label, start, end in parse_segments(path):
            verb, noun = label_to_verb_noun(label, sil_verb, sil_noun)
            segments.append((label, verb, noun, start, end))
        trials[f"{participant}_{recipe}"] = {"recipe": recipe, "segments": segments}

    return trials


def build_vocab(trials):
    verbs, nouns, recipes = set(), set(), set()
    for trial in trials.values():
        recipes.add(trial["recipe"])
        for label, verb, noun, start, end in trial["segments"]:
            verbs.add(verb)
            nouns.add(noun)

    verb_to_id = {v: i for i, v in enumerate(sorted(verbs))}
    noun_to_id = {n: i for i, n in enumerate(sorted(nouns))}
    recipe_to_id = {r: i for i, r in enumerate(sorted(recipes))}
    return verb_to_id, noun_to_id, recipe_to_id


def _check_against_config(config, trials, verb_to_id, noun_to_id, recipe_to_id):
    data_cfg = config["data"]
    vocab_cfg = config["vocab"]
    checks = [
        ("deduped trial count", len(trials), data_cfg["n_unique_trials"]),
        ("verb vocab size", len(verb_to_id), vocab_cfg["verbs"]),
        ("noun vocab size", len(noun_to_id), vocab_cfg["nouns"]),
        ("recipe count", len(recipe_to_id), data_cfg["n_recipes"]),
    ]
    for name, actual, expected in checks:
        if actual != expected:
            raise ValueError(f"{name} {actual} does not match configs/breakfast.yaml value {expected}")


def trim_terminal_idle(verb_ids, noun_ids, sil_verb_id, sil_noun_id):
    """Strip the trailing run of SIL ticks. Returns (verb_ids, noun_ids, n_trimmed).

    The recipe is OVER during that run -- the participant is standing in a kitchen having
    finished. Two things follow, and the second is the reason this exists.

    1. Anomalies after the recipe ends are not interesting, so there is nothing to gain by
       scoring those ticks.
    2. **The trailing idle is not right-censored, but the model has no way to know that.** The
       E-step treats every trial's final segment as still in progress and weights it by the
       survival function P(D >= d) rather than the pmf, and `durations.impute_censored_histogram`
       then redistributes that mass across d >= d_observed *proportional to the current fit*. For
       a state that is ALWAYS the final segment that is a fixed point with nothing to pull it
       back: the fit says "long", the imputation places mass far out in the tail, the refit says
       "longer". Measured on the shipped checkpoint, the terminal-idle state converged to a mean
       duration of 734 ticks against observed runs with a median of 5 -- and it never appears
       anywhere except as a trial's last segment (19/19 occurrences).

    Trimming removes the concentration rather than the censoring. The final segment of a trimmed
    trial is a real action, and which state plays that role now varies from trial to trial, so no
    single state accumulates a runaway estimate. That is the whole intervention; the E-step's
    censoring handling is deliberately left alone.

    Measured cost on the full corpus: 98% of trials end in a SIL run, but the runs are short
    (median 5 ticks, p90 12), so this removes 4.5% of all ticks and no trial becomes empty.
    """
    n = len(verb_ids)
    end = n
    while end > 0 and verb_ids[end - 1] == sil_verb_id and noun_ids[end - 1] == sil_noun_id:
        end -= 1
    return verb_ids[:end], noun_ids[:end], n - end


def build_dataset(config, trim_terminal=True):
    """Parse, dedupe, and tick-expand the full Breakfast corpus per the given config.

    `trim_terminal` strips each trial's trailing SIL run (see trim_terminal_idle). Pass False to
    reproduce the untrimmed dataset.

    Returns (sequences, labels, vocab), each ready to serialize as-is.
    """
    data_cfg = config["data"]
    fps = data_cfg["fps"]
    tick_seconds = config["tick_seconds"]
    sil_verb = config["ambient_gaps"]["sil_verb"]
    sil_noun = config["ambient_gaps"]["sil_noun"]

    trials = build_trials(data_cfg["dataset_root"], sil_verb, sil_noun)
    verb_to_id, noun_to_id, recipe_to_id = build_vocab(trials)
    _check_against_config(config, trials, verb_to_id, noun_to_id, recipe_to_id)

    sil_verb_id, sil_noun_id = verb_to_id[sil_verb], noun_to_id[sil_noun]
    sequences = []
    labels = []
    n_trimmed_total = n_trials_trimmed = 0
    for trial_id, trial in sorted(trials.items()):
        id_segments = [
            (verb_to_id[verb], noun_to_id[noun], start, end)
            for label, verb, noun, start, end in trial["segments"]
        ]
        verb_ids, noun_ids = expand_verb_noun_to_ticks(id_segments, fps, tick_seconds)
        trimmed = 0
        if trim_terminal:
            verb_ids, noun_ids, trimmed = trim_terminal_idle(
                verb_ids, noun_ids, sil_verb_id, sil_noun_id
            )
            n_trimmed_total += trimmed
            n_trials_trimmed += int(trimmed > 0)
        seq = {"trial_id": trial_id, "verb_ids": verb_ids, "noun_ids": noun_ids}
        if trim_terminal:
            # Recorded, not silently discarded: a consumer that wants the original tick count
            # (or wants to know the recipe ended cleanly rather than being cut off) can recover it.
            seq["terminal_idle_ticks"] = trimmed
        sequences.append(seq)
        # labels.json is left at FULL length on purpose: it is validation-only ground truth and
        # nothing trains on it, so truncating it would only make it harder to line up against the
        # untrimmed annotations it came from. Consumers index it by tick and must respect
        # len(verb_ids), which they already do.
        labels.append(build_trial_labels(trial_id, trial["recipe"], trial["segments"], fps, tick_seconds))

    if trim_terminal:
        logger.info("trimmed terminal idle from %d/%d trials, %d ticks total",
                    n_trials_trimmed, len(sequences), n_trimmed_total)

    vocab = {"verbs": verb_to_id, "nouns": noun_to_id, "recipes": recipe_to_id}
    return sequences, labels, vocab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/breakfast.yaml")
    parser.add_argument("--out-dir", default="dataset/processed/breakfast")
    parser.add_argument("--keep-terminal-idle", action="store_true",
                        help="do NOT strip each trial's trailing SIL run. The trailing idle is "
                             "post-recipe and is not genuinely right-censored, but the E-step "
                             "treats every final segment as censored -- for a state that is "
                             "ALWAYS terminal that is a fixed point with no uncensored data to "
                             "pull it back (see trim_terminal_idle). Use this only to reproduce "
                             "the pre-trim dataset.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    config = load_config(args.config)
    sequences, labels, vocab = build_dataset(config, trim_terminal=not args.keep_terminal_idle)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "vocab.json").write_text(json.dumps(vocab, indent=2))
    (out_dir / "sequences.json").write_text(json.dumps(sequences, indent=2))
    (out_dir / "labels.json").write_text(json.dumps(labels, indent=2))

    logger.info("Wrote %d trials to %s", len(sequences), out_dir)


if __name__ == "__main__":
    main()
