"""EPIC-Tent annotations -> the same sequences/labels/vocab contract parse_breakfast.py emits,
plus a fourth file this corpus alone can supply: anomalies.json.

Annotations only -- no video. Run tools_fetch_epic_tent.py and point --root at its output.

EPIC-Tent is the first corpus here that carries REAL anomaly labels: two observers annotated
nine error types with begin and end frames. Everything else in the repo detects anomalies that
synthetic/error_injection.py manufactured, so this is the one place the detector can be scored
against errors nobody placed for it. Five things differ from the other two drivers, and each is
a decision:

1. **A trial is a participant.** 29 sessions, one task, one session each. There is no recipe
   taxonomy to recover, so `recipe_label` is the constant "tent" and k_recipe collapses to 1 --
   the recipe layer becomes a no-op rather than a source of noise. `participant_label` carries
   the subject id, as on EPIC.

2. **The 12 action classes factor into 12 distinct (verb, noun) pairs**, with no collisions (see
   ACTION_TO_VERB_NOUN). That is unusual: Breakfast's labels are already `verb_noun`, EPIC has
   3806 pairs the subtask layer must group, and here the pairs ARE the states. So k_subtask is
   16 against 12 real pairs plus SIL, and recipe/lexical_init hands the state space over almost
   directly instead of discovering it.

3. **Action segments overlap** -- 10.5% of consecutive pairs, because the paper's own design
   note is that instruction reading interrupts other subtasks. resolve_overlaps' first-come-wins
   is reused unchanged, which costs 127 of 1252 segments and 79 of the 304 instruction reads:
   the interrupter is systematically the one that loses. See docs/data.md for why the
   alternative is unrepresentable.

4. **Errors carry their own frame interval**, so localisation never joins to an action segment.
   The `action_label` column on an error row is context, not position. That matters because 35
   of 365 kept error rows lose their host segment to (3) and would otherwise be unplaceable.

5. **Order errors are defined against the printed instructions**, not against what participants
   actually did. CANONICAL_ORDER records that sequence. Measured: the transition INTO an
   order-error segment has median corpus count 23 against a 22 baseline over all 1223
   transitions, so a first-order transition model cannot separate them -- the bigrams are
   estimated from error-laden trials. CANONICAL_ORDER is what the LLM arm is given instead.

    ./py -m cook_ad.data.parse_epic_tent --root dataset/epic_tent \
        --config configs/epic_tent.yaml --out-dir dataset/processed/epic_tent
"""
import argparse
import csv
import json
import logging
from pathlib import Path

from cook_ad.data.config import load_config
from cook_ad.data.labels import build_trial_labels
from cook_ad.data.parse_breakfast import trim_terminal_idle
from cook_ad.data.parse_epic import resolve_overlaps, to_frame_segments
from cook_ad.data.tick_expansion import expand_verb_noun_to_ticks

logger = logging.getLogger(__name__)

# Synthetic frame grid, as on EPIC and for the same reason: expand_to_ticks materialises a dense
# per-frame array, and the source clock is ~60 Hz, so re-quantising to 20 Hz makes that loop 3x
# cheaper while staying 20x finer than the 1s tick.
FPS_REF = 20

# GoPro frame rate. The release README gives one rate for all participants except three, which
# differ in the 5th decimal; at a 1s tick that is a sub-millisecond effect over a 35-minute
# session, but the table is here so the numbers are the published ones rather than a round one.
FPS_SRC = 59.94005994005994
FPS_SRC_OVERRIDES = {3: 59.9392305055482, 18: 59.93919475267356, 24: 59.939296351401424}

ACTION_FILE = "Synchronised_action_label.txt"
ERROR_FILE = "Synchronised_error_label.txt"

# The 12 published action (sub-task) labels, factored into the (verb, noun) pair the HSMM emits.
# The factorisation is injective, which is what makes this corpus' state space essentially given
# rather than learned: `insert support` and `insert support tab` stay distinct on the noun, and
# `pickup/place ventcover` and `place guyline` stay distinct on the noun too. The 38 finer
# sub-tasks in the paper's Table 1 (which distinguish tent corners, `insert stake c1`..`c4`) are
# NOT in the public release -- only the 12-way collapse is.
ACTION_TO_VERB_NOUN = {
    0: ("assemble", "support"),
    1: ("insert", "stake"),
    2: ("insert", "support"),
    3: ("insert", "support_tab"),
    4: ("read", "instruction"),
    5: ("open", "stakebag"),
    6: ("open", "supportbag"),
    7: ("open", "tentbag"),
    8: ("place", "ventcover"),
    9: ("place", "guyline"),
    10: ("spread", "tent"),
    11: ("tie", "top"),
}

ACTION_NAMES = {
    0: "assemble support", 1: "insert stake", 2: "insert support", 3: "insert support tab",
    4: "instruction", 5: "pickup/open stakebag", 6: "pickup/open supportbag",
    7: "pickup/open tentbag", 8: "pickup/place ventcover", 9: "place guyline",
    10: "spread tent", 11: "tie top",
    # Three error rows (all `search`) carry action_label -1: the participant was hunting for a
    # part between subtasks, so there is no host action. Evidence for point 4 of the module
    # docstring -- an error's position comes from its own frames, never from a segment join.
    -1: "between actions",
}

# EPIC-Tent's nine error types -> the repo's anomaly vocabulary. `motor` (219 instances) and
# `failure` (42) map to None: they are physical mishaps -- fumbling a pole, a snapped guyline --
# with no signature in a (verb, noun) tick stream, so scoring against them would only depress
# recall for something the detector cannot see.
#
# `search` and `slow` both map to `stall`, the type run_rollout_demo.py already uses for a
# duration-only anomaly. The other four are the repo's own names.
ERROR_TYPE_MAP = {
    0: None,              # motor
    1: "substitution",    # misuse of equipment -- the wrong object for the step
    2: "transposition",   # steps out of order, relative to the printed instructions
    3: None,              # equipment failure
    4: "omission",        # omission of a step
    5: "stall",           # searching for an item
    6: "correction",      # correction of a prior error
    7: "stall",           # slowness in movement
    8: "repetition",      # repetition
}

ERROR_NAMES = {
    0: "motor", 1: "misuse", 2: "order", 3: "failure", 4: "omit",
    5: "search", 6: "correction", 7: "slow", 8: "repeat",
}

# PREGO and TI-PREGO score only these four, on the argument that the other five leave the
# procedure itself intact. Kept as a named subset so their numbers stay comparable without
# throwing away `misuse`, which is the closest match the corpus has to the repo's substitution.
PROCEDURAL_TYPES = ("transposition", "omission", "correction", "repetition")

# The error groups results are reported over. Each is scored with every OTHER type removed --
# relabelled as normal, so its ticks join the healthy population and a flag on one is a false
# positive. That is how PREGO treats the five non-procedural types: its input is a collapsed step
# sequence, which has no slot for a misuse, and its scorer (step_anticipation/src/models/
# llm_hf.py) counts every step before the first procedural mistake as correct.
#
# `substitution` (misuse) is in neither group, deliberately. It is annotated WITHIN a correctly
# labelled action -- the wrong guyline during a step still called `place guyline` -- so 94% of
# its spans leave the (verb, noun) stream unchanged. No detector reading this representation can
# see it, so including it would measure the representation rather than the detector.
ERROR_GROUPS = {
    # PREGO's subset: order, omit, correction, repeat.
    "procedural": PROCEDURAL_TYPES,
    # + search and slow, which both map to `stall`: the pair and its position are right, the
    # duration is wrong -- the one kind of error the HSMM's duration channels exist to catch.
    "procedural+temporal": (*PROCEDURAL_TYPES, "stall"),
}

# The printed-instruction order, which is the reference `order` errors are annotated against.
# Derived two ways that agree: the median normalised onset of each action over the 29 sessions,
# and the workflow column of the paper's Table 1. `read/instruction` is deliberately absent --
# it interleaves freely (median position 0.36 over 300 segments) and has no place in the
# sequence. This list is the LLM arm's substitute for a transition model (see point 5 above).
CANONICAL_ORDER = (
    ("open", "tentbag"),
    ("spread", "tent"),
    ("open", "supportbag"),
    ("assemble", "support"),
    ("insert", "support"),
    ("insert", "support_tab"),
    ("open", "stakebag"),
    ("insert", "stake"),
    ("tie", "top"),
    ("place", "ventcover"),
    ("place", "guyline"),
)

# Minimum length of an error-free run kept as a healthy fragment. Below this a fragment carries
# no usable duration evidence and its first ticks are still inside the previous anomaly's
# detection deadline (span_metrics.DEADLINE = 10).
MIN_FRAGMENT_TICKS = 10

RECIPE_LABEL = "tent"


def fps_for(subject_id):
    return FPS_SRC_OVERRIDES.get(subject_id, FPS_SRC)


def build_vocab(sil_verb, sil_noun):
    """-> (verb_to_id, noun_to_id) over ACTION_TO_VERB_NOUN, with the SIL pair appended last.

    Raises if a SIL token collides with a real one, for the same reason parse_epic does: reusing
    a real name would silently merge the idle state with a real action or object, and the merge
    is invisible downstream. ('stall' and 'idle' are both safe here; 'kitchen', Breakfast's SIL
    noun, is safe too, but see the note in configs/epic_tent.yaml.)
    """
    verbs = sorted({verb for verb, _ in ACTION_TO_VERB_NOUN.values()})
    nouns = sorted({noun for _, noun in ACTION_TO_VERB_NOUN.values()})
    for token, real, what in ((sil_verb, verbs, "verb"), (sil_noun, nouns, "noun")):
        if token in real:
            raise ValueError(f"SIL {what} {token!r} collides with a real EPIC-Tent {what}")
    return ({verb: i for i, verb in enumerate([*verbs, sil_verb])},
            {noun: i for i, noun in enumerate([*nouns, sil_noun])})


def _read_rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def load_action_labels(root, verb_to_id, noun_to_id):
    """-> {subject_id: {"origin": first_frame, "narrations": [(t0, t1, verb_id, noun_id)]}},
    sorted by t0, times in seconds relative to the subject's own first annotated frame.

    Rows whose GoPro frame index is -1 are dropped: the annotations were made on the SMI video
    and synchronised afterwards, and 9 rows (subjects 1, 6, 15, 21) failed to synchronise. The
    SMI clock is not a fallback -- mixing the two would put one trial's segments on two grids.
    """
    by_subject = {}
    dropped = 0
    for row in _read_rows(Path(root) / ACTION_FILE):
        start, end = int(row["str_GoPro_frame"]), int(row["end_GoPro_frame"])
        if start < 0 or end < 0:
            dropped += 1
            continue
        subject = int(row["subject_id"])
        verb, noun = ACTION_TO_VERB_NOUN[int(row["action_label"])]
        by_subject.setdefault(subject, []).append(
            (start, end, verb_to_id[verb], noun_to_id[noun])
        )

    videos = {}
    for subject, rows in sorted(by_subject.items()):
        rows.sort()
        fps = fps_for(subject)
        origin = rows[0][0]
        videos[subject] = {
            "origin": origin,
            "narrations": [((s - origin) / fps, (e - origin) / fps, v, n) for s, e, v, n in rows],
        }
    return videos, dropped


def load_error_labels(root, origins):
    """-> {subject_id: [(t0, t1, error_type, source_type, action_name)]}, seconds relative to the
    same origin the action segments use, sorted by t0. Rows mapping to None are dropped.

    `error_label_sub` records a concurrent error and is -1 on 624 of 626 rows; the two that are
    set both say `misuse`. It is folded in as a separate record when it maps, so a tick that is
    both out of order and a misuse contributes to both types' recall.

    This stays faithful to the file: the annotation README's rule that omission end times are
    meaningless is applied in build_truth, where the record is shaped, so that one function
    owns it rather than two.
    """
    by_subject = {}
    kept = dropped = 0
    for row in _read_rows(Path(root) / ERROR_FILE):
        subject = int(row["subject_id"])
        if subject not in origins:
            continue
        fps = fps_for(subject)
        origin = origins[subject]
        t0 = (int(row["str_GoPro_frame"]) - origin) / fps
        t1 = (int(row["end_GoPro_frame"]) - origin) / fps
        action = ACTION_NAMES[int(row["action_label"])]
        for column in ("error_label_main", "error_label_sub"):
            raw = int(row[column])
            if raw < 0:
                continue
            error_type = ERROR_TYPE_MAP[raw]
            if error_type is None:
                dropped += 1
                continue
            by_subject.setdefault(subject, []).append(
                (t0, max(t0, t1), error_type, ERROR_NAMES[raw], action)
            )
            kept += 1
    for rows in by_subject.values():
        rows.sort()
    return by_subject, kept, dropped


def _to_ticks(t0, t1, tick_seconds, n_ticks):
    """Seconds since the trial origin -> a closed inclusive tick interval clipped to the trial.

    Floor rather than round on both ends: the tick an event starts in is the first tick that
    contains any of it, which is what a causal detector could flag.
    """
    a = max(0, min(n_ticks - 1, int(t0 // tick_seconds)))
    b = max(a, min(n_ticks - 1, int(t1 // tick_seconds)))
    return a, b


def clean_fragments(spans, n_ticks, min_ticks=MIN_FRAGMENT_TICKS):
    """Closed inclusive error spans -> the maximal error-free tick intervals of length >=
    min_ticks. These are the healthy population: under the mapped taxonomy NO trial is free of
    annotated error (p08 has no procedural error but does have `misuse`), so a healthy
    false-positive rate has to be a fragment-level quantity, not a trial-level one.

    Flags for a fragment must be sliced out of the full trial's decode. Re-decoding a fragment
    on its own would start it from the init distribution, and the belief state part-way through
    a tent assembly is not the start-of-trial prior.
    """
    merged = []
    for a, b in sorted(spans):
        if merged and a <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])

    fragments, cursor = [], 0
    for a, b in merged:
        if a - cursor >= min_ticks:
            fragments.append((cursor, a - 1))
        cursor = max(cursor, b + 1)
    if n_ticks - cursor >= min_ticks:
        fragments.append((cursor, n_ticks - 1))
    return fragments


def build_truth(errors, tick_seconds, n_ticks):
    """Error annotations for one trial -> error_injection-shaped ground-truth records.

    `tick_map` is the identity and `edited_ticks` is empty, and neither is a placeholder:
    nothing was spliced, so llm/textify.injection_touched_steps correctly finds no debris. The
    observation stream is not copied into the record either -- it is in sequences.json, and it
    is the same stream, because a real anomaly has no un-anomalous twin.

    Omission collapses to its onset tick, because the annotation README says so: "End time
    codes for omission should not be used as omission is the absence of an action." The
    annotated end is where the NEXT thing happened, not where a missing action stopped.
    """
    records = []
    for t0, t1, error_type, source_type, action in errors:
        a, b = _to_ticks(t0, t0 if error_type == "omission" else t1, tick_seconds, n_ticks)
        records.append({
            "error_type": error_type,
            "source_type": source_type,
            "window": [a, b],
            "spans": [[a, b]],
            "anomaly_ticks": list(range(a, b + 1)),
            "action_label": action,
            "procedural": error_type in PROCEDURAL_TYPES,
        })
    return records


def excise(verb_ids, noun_ids, fragments, trial_id, trial_labels):
    """-> (sequences, labels), one pair per error-free fragment, ids `{trial_id}_f{i}`.

    This is the error-excised training corpus, the second half of the contamination ablation:
    every training session contains annotated errors, so a model fitted on all of them partly
    models the errors it should flag.

    Excision rather than masking, because hsmm/em.pad_batch returns a PREFIX length mask and
    both _chunk_ticks and _length_order read sum(mask) as the sequence length. A mid-sequence
    False would leave those two disagreeing with the mask and silently corrupt right-censoring.
    Cutting instead costs only the transitions that straddle a cut, and needs no EM change.

    Labels are emitted alongside because docs/data.md's contract is that the two files zip by
    position -- run_joint_lexical joins them on trial_id and would KeyError on a fragment
    otherwise. A fragment's labels are its parent's, sliced; nothing is recomputed.
    """
    sequences, labels = [], []
    for i, (a, b) in enumerate(fragments):
        fragment_id = f"{trial_id}_f{i}"
        sequences.append({"trial_id": fragment_id,
                          "verb_ids": list(verb_ids[a:b + 1]),
                          "noun_ids": list(noun_ids[a:b + 1])})
        labels.append({"trial_id": fragment_id,
                       "recipe_label": trial_labels["recipe_label"],
                       "subtask_labels": list(trial_labels["subtask_labels"][a:b + 1]),
                       "participant_label": trial_labels["participant_label"]})
    return sequences, labels


def build_dataset(config, root, trim_terminal=True):
    """-> (sequences, labels, vocab, truth, clean, clean_labels, stats).

    Seven-tuple, against parse_epic's four and parse_breakfast's three: `truth` is the real
    anomaly labels, and `clean`/`clean_labels` are the error-excised training corpus, neither of
    which the other two corpora can supply. The first three elements are byte-compatible with
    theirs, and clean/clean_labels obey the same zip-by-position contract.
    """
    tick_seconds = config["tick_seconds"]
    sil_verb = config["ambient_gaps"]["sil_verb"]
    sil_noun = config["ambient_gaps"]["sil_noun"]

    verb_to_id, noun_to_id = build_vocab(sil_verb, sil_noun)
    sil_ids = (verb_to_id[sil_verb], noun_to_id[sil_noun])

    videos, dropped_rows = load_action_labels(root, verb_to_id, noun_to_id)
    origins = {subject: entry["origin"] for subject, entry in videos.items()}
    errors_by_subject, errors_kept, errors_dropped = load_error_labels(root, origins)

    sequences, labels, truth, clean, clean_labels = [], [], [], [], []
    stats = dict(seen=len(videos), swallowed=0, too_short=0, segments=0, trimmed=0,
                 dropped_rows=dropped_rows, errors_kept=errors_kept,
                 errors_dropped=errors_dropped, errors_clipped=0, fragments=0, clean_ticks=0)

    verb_name = {i: v for v, i in verb_to_id.items()}
    noun_name = {i: n for n, i in noun_to_id.items()}

    for subject, entry in sorted(videos.items()):
        trial_id = f"p{subject:02d}"
        stats["segments"] += len(entry["narrations"])

        kept, swallowed = resolve_overlaps(entry["narrations"])
        stats["swallowed"] += swallowed
        segments, too_short = to_frame_segments(kept, sil_ids, fps=FPS_REF)
        stats["too_short"] += too_short

        verb_ids, noun_ids = expand_verb_noun_to_ticks(segments, FPS_REF, tick_seconds)
        trimmed = 0
        if trim_terminal:
            verb_ids, noun_ids, trimmed = trim_terminal_idle(verb_ids, noun_ids, *sil_ids)
            stats["trimmed"] += trimmed
        if not verb_ids:
            continue

        sequence = {"trial_id": trial_id, "verb_ids": verb_ids, "noun_ids": noun_ids}
        if trim_terminal:
            sequence["terminal_idle_ticks"] = trimmed
        sequences.append(sequence)

        n_ticks = len(verb_ids)
        records = build_truth(errors_by_subject.get(subject, []), tick_seconds, n_ticks)
        stats["errors_clipped"] += sum(1 for r in records if r["window"][1] == n_ticks - 1)
        fragments = clean_fragments([tuple(r["window"]) for r in records], n_ticks)
        stats["fragments"] += len(fragments)
        stats["clean_ticks"] += sum(b - a + 1 for a, b in fragments)

        truth.append({"trial_id": trial_id, "n_ticks": n_ticks, "records": records,
                      "clean_fragments": [list(f) for f in fragments]})

        # `{verb}_{noun}` mirrors Breakfast's `pour_milk`, which is what prompts._label_to_step_text
        # and the recipe block parse. The published `pickup/open stakebag` spelling would not.
        named = [
            (f"{verb_name[v]}_{noun_name[n]}", verb_name[v], noun_name[n], start, end)
            for v, n, start, end in segments
        ]
        trial_labels = build_trial_labels(trial_id, RECIPE_LABEL, named, FPS_REF, tick_seconds)
        trial_labels["participant_label"] = trial_id
        # build_trial_labels leaves subtask_labels at the UNTRIMMED length on purpose, so the
        # fragment slices have to index the same stream the ticks do.
        trial_labels["subtask_labels"] = trial_labels["subtask_labels"][:n_ticks]
        labels.append(trial_labels)

        fragment_seqs, fragment_labels = excise(
            verb_ids, noun_ids, fragments, trial_id, trial_labels
        )
        clean.extend(fragment_seqs)
        clean_labels.extend(fragment_labels)

    _check_against_config(config, verb_to_id, noun_to_id)
    vocab = {
        "verbs": verb_to_id,
        "nouns": noun_to_id,
        # One task, so this is a real annotation rather than EPIC's derived proxy -- and a
        # degenerate one. k_recipe: 1 in the config is the matching decision.
        "recipes": {RECIPE_LABEL: 0},
    }
    return sequences, labels, vocab, truth, clean, clean_labels, stats


def _check_against_config(config, verb_to_id, noun_to_id):
    for name, actual, expected in (
        ("verb vocab size", len(verb_to_id), config["vocab"]["verbs"]),
        ("noun vocab size", len(noun_to_id), config["vocab"]["nouns"]),
    ):
        if actual != expected:
            raise ValueError(f"{name} {actual} does not match the config value {expected}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="dataset/epic_tent")
    ap.add_argument("--config", default="configs/epic_tent.yaml")
    ap.add_argument("--out-dir", default="dataset/processed/epic_tent")
    ap.add_argument("--keep-terminal-idle", action="store_true",
                    help="do NOT strip each session's trailing SIL run -- see "
                         "parse_breakfast.trim_terminal_idle")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = load_config(args.config)
    sequences, labels, vocab, truth, clean, clean_labels, stats = build_dataset(
        config, args.root, trim_terminal=not args.keep_terminal_idle
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "vocab.json").write_text(json.dumps(vocab, indent=2))
    (out_dir / "sequences.json").write_text(json.dumps(sequences, indent=2))
    (out_dir / "labels.json").write_text(json.dumps(labels, indent=2))
    (out_dir / "anomalies.json").write_text(json.dumps(truth, indent=2))
    (out_dir / "sequences_clean.json").write_text(json.dumps(clean, indent=2))
    (out_dir / "labels_clean.json").write_text(json.dumps(clean_labels, indent=2))

    ticks = sum(len(s["verb_ids"]) for s in sequences)
    by_type = {}
    for trial in truth:
        for record in trial["records"]:
            by_type[record["error_type"]] = by_type.get(record["error_type"], 0) + 1
    logger.info("sessions: %d, %d ticks, %d action segments (%d swallowed by an overlap, "
                "%.1f%%), %d rows dropped for a -1 GoPro frame",
                len(sequences), ticks, stats["segments"], stats["swallowed"],
                100 * stats["swallowed"] / max(stats["segments"], 1), stats["dropped_rows"])
    logger.info("anomalies: %d kept, %d dropped as non-symbolic (motor/failure); %s",
                stats["errors_kept"], stats["errors_dropped"],
                ", ".join(f"{k} {v}" for k, v in sorted(by_type.items(), key=lambda kv: -kv[1])))
    logger.info("healthy: %d clean fragments, %d ticks (%.0f%% of the corpus)",
                stats["fragments"], stats["clean_ticks"],
                100 * stats["clean_ticks"] / max(ticks, 1))
    logger.info("wrote %d sequences and %d excised fragments to %s (%d terminal idle ticks "
                "trimmed)", len(sequences), len(clean), out_dir, stats["trimmed"])


if __name__ == "__main__":
    main()
