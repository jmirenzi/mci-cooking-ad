from cook_ad.data.scrub import map_spans, scrub_idle, scrub_sequence

SIL = (9, 9)


def test_scrub_drops_idle_and_merges_a_pause_a():
    v = [9, 1, 1, 9, 9, 1, 2, 9]
    n = [9, 5, 5, 9, 9, 5, 6, 9]
    sv, sn, kept = scrub_idle(v, n, SIL)
    assert sv.tolist() == [1, 1, 1, 2] and sn.tolist() == [5, 5, 5, 6]   # A pause A -> one run
    assert kept.tolist() == [1, 2, 5, 6]


def test_spans_follow_the_kept_ticks():
    kept = [1, 2, 5, 6]
    assert map_spans([(5, 6)], kept) == [(2, 3)]
    assert map_spans([(3, 4)], kept) == [(2, 2)]       # all idle: onto the next kept tick
    assert map_spans([(7, 9)], kept) == []


def test_labels_scrubbed_with_the_stream():
    seq = {"trial_id": "t", "verb_ids": [9, 1, 9, 2], "noun_ids": [9, 5, 9, 6]}
    lab = {"trial_id": "t", "subtask_labels": [0, 3, 0, 4]}
    s, l_, _ = scrub_sequence(seq, SIL, lab)
    assert s["verb_ids"] == [1, 2] and l_["subtask_labels"] == [3, 4]
