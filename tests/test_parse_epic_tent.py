"""EPIC-Tent ingest. The hazards this driver introduces, one test each: the action->(verb, noun)
factorisation must stay injective, CANONICAL_ORDER must stay in step with it, the error taxonomy
must not silently gain a type, and omission must lose its end time."""
import pytest

from cook_ad.data import parse_epic_tent as pet


def test_the_action_factorisation_is_injective():
    """This is the corpus' defining property: 12 action classes -> 12 distinct (verb, noun)
    pairs, which is what lets lexical_init anchor one state per pair instead of discovering the
    grouping. A collision would silently merge two subtasks into one state."""
    pairs = list(pet.ACTION_TO_VERB_NOUN.values())
    assert len(set(pairs)) == len(pairs) == 12
    assert ("insert", "support") != ("insert", "support_tab")
    assert ("place", "ventcover") != ("place", "guyline")


def test_action_tables_cover_the_same_classes():
    assert set(pet.ACTION_TO_VERB_NOUN) == set(pet.ACTION_NAMES) - {-1}


def test_canonical_order_covers_every_action_but_instruction_exactly_once():
    """CANONICAL_ORDER is the printed-instruction sequence and the LLM arm's substitute for a
    transition model, so it has to stay a permutation of the real actions. `read/instruction`
    is deliberately absent: it interleaves freely and has no position in the procedure."""
    assert len(pet.CANONICAL_ORDER) == len(set(pet.CANONICAL_ORDER))
    assert set(pet.CANONICAL_ORDER) == set(pet.ACTION_TO_VERB_NOUN.values()) - {
        ("read", "instruction")
    }


def test_the_error_map_keeps_the_non_symbolic_types_out():
    """motor and failure are physical mishaps with no signature in a (verb, noun) stream.
    Mapping them to a type would only depress recall for something invisible."""
    assert pet.ERROR_TYPE_MAP[0] is None and pet.ERROR_TYPE_MAP[3] is None
    assert set(pet.ERROR_TYPE_MAP) == set(pet.ERROR_NAMES)
    mapped = {v for v in pet.ERROR_TYPE_MAP.values() if v}
    assert mapped == {"substitution", "transposition", "stall", "omission", "correction",
                      "repetition"}
    assert set(pet.PROCEDURAL_TYPES) <= mapped


def test_search_and_slow_share_the_stall_type():
    assert pet.ERROR_TYPE_MAP[5] == pet.ERROR_TYPE_MAP[7] == "stall"


def test_the_procedural_group_is_exactly_pregos_subset():
    """order, omit, correction, repeat -- by their EPIC-Tent names, not just by repo names."""
    names = {pet.ERROR_NAMES[k] for k, v in pet.ERROR_TYPE_MAP.items()
             if v in pet.ERROR_GROUPS["procedural"]}
    assert names == {"order", "omit", "correction", "repeat"}


def test_procedural_plus_temporal_adds_exactly_search_and_slow():
    names = {pet.ERROR_NAMES[k] for k, v in pet.ERROR_TYPE_MAP.items()
             if v in pet.ERROR_GROUPS["procedural+temporal"]}
    assert names == {"order", "omit", "correction", "repeat", "search", "slow"}


def test_no_group_contains_the_symbolically_invisible_types():
    """misuse is 94% invisible in a (verb, noun) stream; motor and failure are not mapped at all."""
    for group in pet.ERROR_GROUPS.values():
        assert "substitution" not in group
        assert set(group) <= {v for v in pet.ERROR_TYPE_MAP.values() if v}


def test_sil_token_colliding_with_a_real_token_is_rejected():
    with pytest.raises(ValueError, match="collides"):
        pet.build_vocab("insert", "idle")
    with pytest.raises(ValueError, match="collides"):
        pet.build_vocab("stall", "tent")


def test_vocab_appends_sil_last_and_is_deterministic():
    verbs, nouns = pet.build_vocab("stall", "idle")
    assert len(verbs) == 8 and len(nouns) == 12
    assert verbs["stall"] == 7 and nouns["idle"] == 11
    assert list(verbs) == sorted(list(verbs)[:-1]) + ["stall"]
    assert pet.build_vocab("stall", "idle") == (verbs, nouns)


def test_to_ticks_floors_both_ends_and_clips_to_the_trial():
    """The tick an event starts in is the first tick containing any of it -- that is the tick a
    causal detector could flag. Rounding would place half of them a tick late."""
    assert pet._to_ticks(0.0, 2.9, 1.0, 100) == (0, 2)
    assert pet._to_ticks(1.9, 1.9, 1.0, 100) == (1, 1)
    assert pet._to_ticks(50.0, 200.0, 1.0, 100) == (50, 99), "clipped to the last tick"
    assert pet._to_ticks(-3.0, 1.0, 1.0, 100) == (0, 1), "clipped to the first tick"


def test_omission_records_are_a_single_tick():
    """The annotation README: 'End time codes for omission should not be used as omission is the
    absence of an action.' Everything else keeps its span."""
    errors = [(10.0, 40.0, "omission", "omit", "place guyline"),
              (10.0, 40.0, "substitution", "misuse", "place guyline")]
    omit, sub = pet.build_truth(errors, tick_seconds=1.0, n_ticks=100)
    assert omit["window"] == [10, 10] and omit["spans"] == [[10, 10]]
    assert sub["window"] == [10, 40]


def test_truth_records_carry_identity_semantics_not_an_edited_stream():
    """A real anomaly has no un-anomalous twin, so there is no tick_map and no debris. The
    record must not claim otherwise."""
    record, = pet.build_truth([(1.0, 2.0, "stall", "slow", "tie top")], 1.0, 50)
    assert record["procedural"] is False
    assert record["anomaly_ticks"] == [1, 2]
    assert "tick_map" not in record and "edited_ticks" not in record


def test_procedural_flag_tracks_the_prego_subset():
    errors = [(1.0, 2.0, t, t, "tie top") for t in ("transposition", "substitution")]
    trans, sub = pet.build_truth(errors, 1.0, 50)
    assert trans["procedural"] is True and sub["procedural"] is False


def test_clean_fragments_are_disjoint_from_every_span_and_from_each_other():
    fragments = pet.clean_fragments([(20, 29), (60, 61)], n_ticks=100, min_ticks=10)
    assert fragments == [(0, 19), (30, 59), (62, 99)]
    for (_, a1), (b0, _) in zip(fragments, fragments[1:]):
        assert a1 < b0


def test_clean_fragments_merge_touching_and_nested_spans():
    """Errors overlap: a misuse inside an out-of-order step is two records over one stretch."""
    assert pet.clean_fragments([(10, 20), (21, 30), (12, 15)], 60, min_ticks=5) == [(0, 9), (31, 59)]


def test_clean_fragments_drops_runs_below_the_floor():
    """A 4-tick gap between two anomalies is inside span_metrics' 10-tick detection deadline, so
    scoring a flag there as a false positive would punish a correct late detection."""
    assert pet.clean_fragments([(10, 20), (25, 40)], 45, min_ticks=10) == [(0, 9)]


def test_excise_cuts_only_the_annotated_spans():
    verb_ids, noun_ids = list(range(20)), list(range(100, 120))
    parent = {"recipe_label": "tent", "participant_label": "p07",
              "subtask_labels": [f"L{i}" for i in range(20)]}
    clean, clean_labels = pet.excise(verb_ids, noun_ids, [(0, 4), (10, 19)], "p07", parent)
    assert [c["trial_id"] for c in clean] == ["p07_f0", "p07_f1"]
    assert clean[0]["verb_ids"] == [0, 1, 2, 3, 4]
    assert clean[1]["noun_ids"] == list(range(110, 120))
    assert all(len(c["verb_ids"]) == len(c["noun_ids"]) for c in clean)


def test_excise_emits_label_rows_that_zip_with_its_sequences():
    """docs/data.md's contract: the two files zip by position. run_joint_lexical joins them on
    trial_id, so a fragment with no label row is a KeyError, not a missing summary."""
    parent = {"recipe_label": "tent", "participant_label": "p07",
              "subtask_labels": [f"L{i}" for i in range(20)]}
    clean, clean_labels = pet.excise(list(range(20)), list(range(20)), [(3, 7)], "p07", parent)
    assert [c["trial_id"] for c in clean] == [lab["trial_id"] for lab in clean_labels]
    assert clean_labels[0]["subtask_labels"] == ["L3", "L4", "L5", "L6", "L7"]
    assert len(clean_labels[0]["subtask_labels"]) == len(clean[0]["verb_ids"])
    assert clean_labels[0]["participant_label"] == "p07"


# ---- errors whose host action preprocessing deleted ---------------------------------------
# Segments are (t0, t1, verb_id, noun_id) in seconds; ACTIONS maps an action name to its ids.
ACTIONS = {"tie top": (6, 11), "insert support": (1, 4)}
TIE, SUPPORT = (*ACTIONS["tie top"],), (*ACTIONS["insert support"],)


def test_an_error_on_a_swallowed_action_is_unobservable():
    raw = [(0.0, 60.0, *SUPPORT), (20.0, 30.0, *TIE)]     # tie top interrupts insert support
    kept = [(0.0, 60.0, *SUPPORT)]                        # ... and first-come-wins drops it
    assert pet.host_swallowed(22.0, 28.0, "transposition", "tie top", raw, kept, ACTIONS)


def test_an_error_on_a_surviving_action_is_kept():
    raw = kept = [(0.0, 20.0, *SUPPORT), (20.0, 30.0, *TIE)]
    assert not pet.host_swallowed(22.0, 28.0, "transposition", "tie top", raw, kept, ACTIONS)


def test_a_partly_surviving_action_is_kept():
    """resolve_overlaps pushes a late start forward rather than dropping the segment; if any of
    it still overlaps the error, the stream does show the action there."""
    raw = [(0.0, 25.0, *SUPPORT), (20.0, 30.0, *TIE)]
    kept = [(0.0, 25.0, *SUPPORT), (25.0, 30.0, *TIE)]
    assert not pet.host_swallowed(22.0, 28.0, "transposition", "tie top", raw, kept, ACTIONS)


def test_omissions_and_annotation_inconsistencies_are_never_dropped():
    raw = [(0.0, 60.0, *SUPPORT), (20.0, 30.0, *TIE)]
    kept = [(0.0, 60.0, *SUPPORT)]
    # An omitted action is absent by definition.
    assert not pet.host_swallowed(22.0, 22.0, "omission", "tie top", raw, kept, ACTIONS)
    # No raw tie-top segment overlaps at all: the annotation disagrees with itself, and the
    # stream around the error is intact -- not preprocessing's doing.
    assert not pet.host_swallowed(40.0, 45.0, "transposition", "tie top", raw, kept, ACTIONS)
    # No host action.
    assert not pet.host_swallowed(22.0, 28.0, "stall", "between actions", raw, kept, ACTIONS)


def test_fps_overrides_only_apply_to_the_three_published_exceptions():
    assert pet.fps_for(1) == pet.FPS_SRC
    assert pet.fps_for(3) != pet.FPS_SRC
    assert set(pet.FPS_SRC_OVERRIDES) == {3, 18, 24}


# ---------------------------------------------------------------------------
# End-to-end, against the real annotations. Skipped when they are absent, unlike
# test_parse_breakfast's equivalent: Breakfast is a prerequisite for the whole repo, whereas
# these 16 MB are fetched on demand by scripts/data/fetch_epic_tent.py.
# ---------------------------------------------------------------------------
from pathlib import Path  # noqa: E402

from cook_ad.data.config import load_config  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
ROOT = REPO_ROOT / "dataset" / "epic_tent"
CONFIG_PATH = REPO_ROOT / "configs" / "epic_tent.yaml"

needs_data = pytest.mark.skipif(
    not (ROOT / pet.ACTION_FILE).exists(),
    reason="run scripts/data/fetch_epic_tent.py to fetch the annotations",
)


@pytest.fixture(scope="module")
def built():
    return pet.build_dataset(load_config(CONFIG_PATH), ROOT)


@needs_data
def test_full_dataset_build_end_to_end(built):
    sequences, labels, vocab, truth, clean, clean_labels, stats = built
    assert len(sequences) == len(labels) == len(truth) == 29
    assert {s["trial_id"] for s in sequences} == {lab["trial_id"] for lab in labels}
    assert [s["trial_id"] for s in sequences] == sorted(s["trial_id"] for s in sequences), (
        "docs/data.md: both lists are ordered by sorted trial_id so they zip without a join"
    )
    for seq, lab in zip(sequences, labels):
        assert len(seq["verb_ids"]) == len(seq["noun_ids"]) > 0
        assert len(lab["subtask_labels"]) == len(seq["verb_ids"])
        assert lab["recipe_label"] == pet.RECIPE_LABEL
    assert len(vocab["verbs"]) == 8 and len(vocab["nouns"]) == 12
    assert sum(len(s["verb_ids"]) for s in sequences) == 23226


@needs_data
def test_every_tick_carries_a_real_vocab_id(built):
    """expand_to_ticks votes None for an uncovered frame, which would surface here as a None id
    rather than as an error. The SIL gap fill is what prevents it."""
    sequences, _, vocab, *_ = built
    for seq in sequences:
        assert all(isinstance(v, int) and 0 <= v < len(vocab["verbs"]) for v in seq["verb_ids"])
        assert all(isinstance(n, int) and 0 <= n < len(vocab["nouns"]) for n in seq["noun_ids"])


@needs_data
def test_the_procedural_subset_matches_pregos_published_count(built):
    """PREGO and TI-PREGO both report EPIC-Tent's procedural errors as order + omit +
    correction + repeat. Recovering their 163 is the check that the taxonomy mapping is right."""
    _, _, _, truth, _, _, _ = built
    every = [t["records"] + t["unobservable_records"] for t in truth]
    procedural = [r for records in every for r in records if r["procedural"]]
    assert len(procedural) == 163
    assert sum(1 for records in every if any(r["procedural"] for r in records)) == 28, (
        "28 of 29 participants make at least one procedural error; only p08 does not"
    )


@needs_data
def test_fifteen_errors_lose_their_host_action_to_overlap_resolution(built):
    """Measured 2026-10-06: 15 rows, none an omission. They leave `records` but are still
    excised from the training corpus, so the clean-tick count below does not move."""
    _, _, _, truth, _, _, stats = built
    gone = [r for t in truth for r in t["unobservable_records"]]
    assert len(gone) == stats["errors_unobservable"] == 15
    assert not any(r["error_type"] == "omission" for r in gone)


@needs_data
def test_no_trial_is_free_of_annotated_error(built):
    """The reason the healthy population is fragments, not trials: under the mapped taxonomy
    every single session carries something. p08 has no PROCEDURAL error but does have misuse."""
    _, _, _, truth, _, _, _ = built
    assert all(t["records"] for t in truth)
    p08, = [t for t in truth if t["trial_id"] == "p08"]
    assert not any(r["procedural"] for r in p08["records"])
    assert p08["records"], "p08 is procedurally clean but not error-free"


@needs_data
def test_every_truth_record_lands_inside_its_trial(built):
    sequences, _, _, truth, _, _, _ = built
    for seq, trial in zip(sequences, truth):
        n = len(seq["verb_ids"])
        assert trial["n_ticks"] == n
        for record in trial["records"]:
            start, end = record["window"]
            assert 0 <= start <= end < n
            assert record["spans"] == [[start, end]]


@needs_data
def test_clean_fragments_never_touch_an_error_span(built):
    _, _, _, truth, clean, _, _ = built
    for trial in truth:
        spans = [tuple(r["window"]) for r in trial["records"] + trial["unobservable_records"]]
        for a, b in trial["clean_fragments"]:
            assert b - a + 1 >= pet.MIN_FRAGMENT_TICKS
            assert not any(start <= b and a <= end for start, end in spans)
    assert len(clean) == sum(len(t["clean_fragments"]) for t in truth)
    assert sum(len(c["verb_ids"]) for c in clean) == 17008


# ---------------------------------------------------------------------------
# PREGO's split. Loaded by path because scripts/data/epic_tent_splits.py is a script, not part
# of the package. An earlier version put the unrated participants in TRAIN and ranked by mean,
# which reproduced PREGO's 14/15 counts with the wrong membership.
# ---------------------------------------------------------------------------
import importlib.util  # noqa: E402


def _split_tool():
    spec = importlib.util.spec_from_file_location("tools_epic_tent_split",
                                                  REPO_ROOT / "scripts" / "data" / "epic_tent_splits.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_uncertainty(root, subject, values):
    path = root / "uncertainty" / f"{subject:02d}.uncertainty.gopro_timecode.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{subject}\t00:00:00:{i:02d}\t{v}\n" for i, v in enumerate(values)))


def test_prego_split_sends_unrated_participants_to_test(tmp_path):
    """PREGO supplementary C: 'the remaining 7 do not [have confidence annotations] and are
    assigned to the test set.'"""
    _write_uncertainty(tmp_path, 8, [0.1] * 5)
    train, test, _, _ = _split_tool().build(tmp_path, ["p01", "p08"])
    assert "p01" in test and "p01" not in train


def test_prego_split_thresholds_median_confidence_not_mean_uncertainty(tmp_path):
    """Confidence is 1 - the shipped uncertainty, and the statistic is the MEDIAN: this file's
    mean uncertainty is 0.52 (confidence 0.48, would be test) but its median is 0.1
    (confidence 0.9, train)."""
    _write_uncertainty(tmp_path, 9, [0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0])
    train, test, ranking, _ = _split_tool().build(tmp_path, ["p09"])
    assert train == ["p09"] and test == []
    assert ranking[0]["median_confidence"] == pytest.approx(0.9)


def test_prego_split_boundary_is_strictly_below_point_six(tmp_path):
    _write_uncertainty(tmp_path, 10, [0.4] * 3)      # confidence exactly 0.6 -> train
    _write_uncertainty(tmp_path, 11, [0.41] * 3)     # confidence 0.59 -> test
    train, test, _, _ = _split_tool().build(tmp_path, ["p10", "p11"])
    assert train == ["p10"] and test == ["p11"]


@pytest.mark.skipif(not (ROOT / "uncertainty").exists(),
                    reason="run scripts/data/fetch_epic_tent.py --uncertainty")
def test_prego_split_reproduces_their_published_test_ids():
    """The 15 keys of PREGO's own step_anticipation/data/predictions/
    output_miniROAD_Epic-tent-O.json."""
    ids = [f"p{i:02d}" for i in range(1, 30)]
    train, test, _, _ = _split_tool().build(ROOT, ids)
    assert test == [f"p{i:02d}" for i in (1, 2, 3, 4, 5, 6, 7, 10, 16, 17, 18, 20, 23, 25, 28)]
    assert len(train) == 14
