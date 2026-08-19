from collections import defaultdict
from pathlib import Path

import pytest

from cook_ad.data.config import load_config
from cook_ad.data.parse_breakfast import (
    FILENAME_RE,
    build_dataset,
    build_trials,
    build_vocab,
    find_trial_files,
)
from cook_ad.data.tick_expansion import expand_verb_noun_to_ticks

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "breakfast.yaml"


@pytest.fixture(scope="module")
def config():
    return load_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def trials(config):
    sil_verb = config["ambient_gaps"]["sil_verb"]
    sil_noun = config["ambient_gaps"]["sil_noun"]
    return build_trials(config["data"]["dataset_root"], sil_verb, sil_noun)


@pytest.fixture(scope="module")
def vocab(trials):
    return build_vocab(trials)


def test_dedup_and_vocab_sizes_match_config(config, trials, vocab):
    verb_to_id, noun_to_id, recipe_to_id = vocab
    assert len(trials) == config["data"]["n_unique_trials"]
    assert len(verb_to_id) == config["vocab"]["verbs"]
    assert len(noun_to_id) == config["vocab"]["nouns"]
    assert len(recipe_to_id) == config["data"]["n_recipes"]


def test_canonical_camera_view_picks_highest_priority(config):
    canonical = find_trial_files(config["data"]["dataset_root"])
    assert canonical[("P03", "cereals")].name == "P03_cam01_P03_cereals.txt"


def test_single_camera_view_trial_still_included(config):
    canonical = find_trial_files(config["data"]["dataset_root"])

    view_counts = defaultdict(int)
    for path in Path(config["data"]["dataset_root"]).rglob("*.txt"):
        m = FILENAME_RE.match(path.name)
        if m:
            participant, camera, recipe = m.groups()
            view_counts[(participant, recipe)] += 1

    single_view_trials = [key for key, count in view_counts.items() if count == 1]
    assert single_view_trials
    for key in single_view_trials:
        assert key in canonical


def test_p03_cereals_segments_and_tick_expansion(config, trials, vocab):
    verb_to_id, noun_to_id, _ = vocab
    trial = trials["P03_cereals"]

    labels = [label for label, verb, noun, start, end in trial["segments"]]
    assert labels == ["SIL", "take_bowl", "pour_cereals", "pour_milk", "stir_cereals", "SIL"]

    id_segments = [
        (verb_to_id[verb], noun_to_id[noun], start, end)
        for label, verb, noun, start, end in trial["segments"]
    ]
    verb_ids, noun_ids = expand_verb_noun_to_ticks(
        id_segments, config["data"]["fps"], config["tick_seconds"]
    )
    assert len(verb_ids) == len(noun_ids)

    # tick 0 is SIL -> the configured stall/kitchen token
    assert verb_ids[0] == verb_to_id["stall"]
    assert noun_ids[0] == noun_to_id["kitchen"]

    # take_bowl (frames 31-150) starts exactly on a tick boundary at 15 fps
    assert verb_ids[2] == verb_to_id["take"]
    assert noun_ids[2] == noun_to_id["bowl"]

    # tick 28 (frames 421-435) straddles pour_cereals (421-428, 8 frames) and
    # pour_milk (429-435, 7 frames); majority vote picks pour_cereals.
    assert verb_ids[28] == verb_to_id["pour"]
    assert noun_ids[28] == noun_to_id["cereals"]

    # tick 29 (frames 436-450) is fully inside pour_milk
    assert verb_ids[29] == verb_to_id["pour"]
    assert noun_ids[29] == noun_to_id["milk"]


def test_short_final_segment_produces_nonempty_ticks(config):
    fps = config["data"]["fps"]
    tick_seconds = config["tick_seconds"]
    # Final segment is only 2 frames, well under one tick (15 frames at 1s ticks).
    segments = [(0, 0, 1, 20), (1, 1, 21, 22)]
    verb_ids, noun_ids = expand_verb_noun_to_ticks(segments, fps, tick_seconds)
    assert len(verb_ids) == len(noun_ids) == 2
    assert all(v is not None for v in verb_ids)
    assert all(n is not None for n in noun_ids)


def test_full_dataset_build_end_to_end(config):
    sequences, labels, vocab = build_dataset(config)
    assert len(sequences) == len(labels) == config["data"]["n_unique_trials"]
    for seq in sequences:
        assert len(seq["verb_ids"]) == len(seq["noun_ids"]) > 0
    assert {s["trial_id"] for s in sequences} == {lab["trial_id"] for lab in labels}


def test_trim_terminal_idle_strips_only_the_trailing_sil_run():
    """The trailing idle is post-recipe, and the E-step would otherwise treat it as
    right-censored -- a fixed point for any state that is ALWAYS terminal (see the function's
    docstring). Interior SIL is real ambient gap between steps and must survive."""
    from cook_ad.data.parse_breakfast import trim_terminal_idle
    SV, SN = 11, 18

    # interior SIL preserved, trailing run removed
    v, n, k = trim_terminal_idle([1, 2, SV, 3, SV, SV], [1, 2, SN, 3, SN, SN], SV, SN)
    assert (v, n, k) == ([1, 2, SV, 3], [1, 2, SN, 3], 2)

    # nothing to strip
    assert trim_terminal_idle([1, 2, 3], [1, 2, 3], SV, SN) == ([1, 2, 3], [1, 2, 3], 0)

    # an all-SIL trial degenerates to empty rather than raising -- the caller decides
    assert trim_terminal_idle([SV, SV], [SN, SN], SV, SN) == ([], [], 2)
    assert trim_terminal_idle([], [], SV, SN) == ([], [], 0)

    # a SIL VERB with a non-SIL noun is not the terminal sentinel and is kept
    v, n, k = trim_terminal_idle([1, SV], [1, 5], SV, SN)
    assert k == 0 and v == [1, SV]
