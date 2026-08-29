"""EPIC-100 ingest. The three EPIC-specific hazards get a test each: overlapping narrations,
gaps that must become SIL, and a SIL token colliding with a real class."""
import pytest

from cook_ad.data import parse_epic


def test_parse_timestamp():
    assert parse_epic.parse_timestamp("00:00:00.14") == pytest.approx(0.14)
    assert parse_epic.parse_timestamp("01:02:03.50") == pytest.approx(3723.5)


def test_resolve_overlaps_truncates_the_later_start():
    """The common case: `take board` 43.2-46.6 overlapping `put down board` 46.4-47.8."""
    kept, swallowed = parse_epic.resolve_overlaps([(43.2, 46.6, 0, 0), (46.4, 47.8, 1, 0)])
    assert swallowed == 0
    assert kept[0] == (43.2, 46.6, 0, 0)
    assert kept[1][0] == pytest.approx(46.6), "later narration must start where the earlier ends"


def test_resolve_overlaps_drops_a_fully_swallowed_narration():
    """A short narration inside a long one is dropped, NOT allowed to interrupt it: splitting
    the outer segment would produce `A B A`, and self-transitions are structurally banned."""
    kept, swallowed = parse_epic.resolve_overlaps([(10.0, 30.0, 0, 0), (15.0, 16.0, 1, 1)])
    assert swallowed == 1
    assert kept == [(10.0, 30.0, 0, 0)]


def test_resolve_overlaps_output_is_disjoint_and_ordered():
    rng = [(0.0, 5.0, 0, 0), (2.0, 9.0, 1, 1), (3.0, 4.0, 2, 2), (8.5, 12.0, 3, 3)]
    kept, _ = parse_epic.resolve_overlaps(rng)
    for (a0, a1, _, _), (b0, _b1, _, _) in zip(kept, kept[1:]):
        assert a0 < a1 <= b0


def test_gaps_become_sil_and_segments_tile_without_holes():
    """tick_expansion.expand_to_ticks majority-votes over a dense frame array, so an uncovered
    frame would vote None. Every frame from 1 to the last must carry a value."""
    segs, _ = parse_epic.to_frame_segments([(0.0, 1.0, 3, 4), (5.0, 6.0, 7, 8)], sil_ids=(97, 305))
    assert segs[0][2] == 1
    for (_, _, _, end), (_, _, start, _) in zip(segs, segs[1:]):
        assert start == end + 1, "segments must be contiguous"
    assert (97, 305) in [(v, n) for v, n, _, _ in segs], "the gap must be filled with SIL"


def test_sil_token_colliding_with_a_real_class_is_rejected():
    """Breakfast's SIL noun is 'kitchen', which IS a real EPIC noun class -- reusing it would
    silently merge the idle state with a real object."""
    with pytest.raises(ValueError, match="collides"):
        parse_epic.build_vocab(["take"], ["kitchen", "pan"], "stall", "kitchen")


def test_vocab_keeps_epic_class_ids_and_appends_sil():
    verbs, nouns = parse_epic.build_vocab(["take", "put"], ["pan", "tap"], "stall", "idle")
    assert verbs == {"take": 0, "put": 1, "stall": 2}
    assert nouns == {"pan": 0, "tap": 1, "idle": 2}
