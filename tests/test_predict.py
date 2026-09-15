"""The predictive-disagreement arm: parsing, the two channels, and the protocol invariants that
make its latency comparable to the other arms'."""
import hashlib
import json

import pytest

from cook_ad.llm import client as llm_client
from cook_ad.llm import predict, prompts, textify

VOCAB = {"verbs": {"stall": 0, "pour": 1, "stir": 2, "take": 3},
         "nouns": {"kitchen": 0, "milk": 1, "bowl": 2, "cereals": 3}}


class _Lex:
    _V = {0: "stall", 1: "pour", 2: "stir", 3: "take"}
    _N = {0: "kitchen", 1: "milk", 2: "bowl", 3: "cereals"}

    def verb(self, i):
        return self._V[int(i)]

    def noun(self, i):
        return self._N[int(i)]


class StubClient:
    """Records every message list it is handed and replays canned replies in order. Keeps the
    suite offline -- nothing here touches the network."""

    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def complete(self, messages):
        self.calls.append([dict(m) for m in messages])
        i = len(self.calls) - 1
        return self.replies[i] if i < len(self.replies) else "DURATION: 5 seconds\nNEXT: pour milk"

    def complete_many(self, message_lists, concurrency=None):
        return [self.complete(m) for m in message_lists]

    n_would_request = 0


def _steps(runs):
    """runs: [(verb_id, noun_id, n_ticks), ...] -> RLE Steps."""
    v, n = [], []
    for vi, ni, d in runs:
        v += [vi] * d
        n += [ni] * d
    return textify.steps_from_ids(v, n, _Lex())


# ---- the response grammar --------------------------------------------------------------------

@pytest.mark.parametrize("reply,duration", [
    ("DURATION: 12 seconds\nNEXT: pour milk", 12),
    ("DURATION: 12\nNEXT: pour milk", 12),
    ("DURATION: 12s\nNEXT: pour milk", 12),
    ("DURATION: 12 secs\nNEXT: pour milk", 12),
    ("DURATION: for 12 seconds\nNEXT: pour milk", 12),
    ("DURATION: (12s)\nNEXT: pour milk", 12),
    ("Sure, here you go.\nDURATION: 12 seconds\nNEXT: pour milk", 12),
])
def test_the_duration_line_tolerates_the_formats_the_prompts_teach(reply, duration):
    """Same reasoning as detect.py's tolerances: being strict about a surface slip measures
    obedience to a format the prompt itself showed, not prediction."""
    assert predict.parse_predictions(reply, VOCAB)[0].duration == duration


@pytest.mark.parametrize("reply", [
    "DURATION: 12 seconds\nNEXT: pour milk",
    "DURATION: 12 seconds\nNEXT: pour_milk",
    "DURATION: 12 seconds\nNEXT: pour milk (12s)",
    "DURATION: 12 seconds\nNEXT: pour milk for 12 seconds",
])
def test_the_next_line_tolerates_the_formats_the_prompts_teach(reply):
    assert predict.parse_predictions(reply, VOCAB)[0].symbol == ("pour", "milk")


def test_a_malformed_line_degrades_one_channel_and_leaves_the_other_intact():
    """The reason the reply is two labelled lines rather than one structured object."""
    bad_next = predict.parse_predictions("DURATION: 12 seconds\nNEXT: fly rocket", VOCAB)[0]
    assert bad_next.duration == 12 and bad_next.dur_parse_ok
    assert bad_next.symbol is None and not bad_next.next_parse_ok

    bad_dur = predict.parse_predictions("DURATION: soon\nNEXT: pour milk", VOCAB)[0]
    assert bad_dur.symbol == ("pour", "milk") and bad_dur.next_parse_ok
    assert bad_dur.duration is None and not bad_dur.dur_parse_ok


def test_an_out_of_vocabulary_prediction_is_a_parse_failure_not_a_fuzzy_match():
    """A fuzzy threshold inside a metric would make the metric tunable -- detect._canon_correction
    refuses for the same reason."""
    p = predict.parse_predictions("DURATION: 5 seconds\nNEXT: whisk souffle", VOCAB)[0]
    assert p.symbol is None and not p.next_parse_ok


def test_parse_returns_a_list_so_k_sampling_needs_no_interface_change():
    out = predict.parse_predictions("DURATION: 5 seconds\nNEXT: pour milk", VOCAB)
    assert isinstance(out, list) and len(out) == 1


def test_an_unparseable_reply_is_recorded_rather_than_read_as_agreement():
    p = predict.parse_predictions("I'm not sure.", VOCAB)[0]
    assert (p.duration, p.symbol) == (None, None)
    assert not p.dur_parse_ok and not p.next_parse_ok
    # and it produces no flag either way -- a missing prediction is not a disagreement
    assert predict.symbol_disagreement([p], ("pour", "milk")) == (None, 0.0)
    assert predict.duration_disagreement([p], 999) == (None, 0.0)


# ---- the request ------------------------------------------------------------------------------

def test_the_step_being_asked_about_is_shown_without_its_duration():
    """The duration is what is being predicted. render_step would print `pour milk for 10
    seconds` and hand the model the answer."""
    steps = _steps([(1, 1, 10), (2, 2, 4)])
    text = predict.predictive_prefix(steps, 1)
    assert text.splitlines()[-1] == "2. stir bowl (in progress)"
    assert "for 4 seconds" not in text
    assert text.splitlines()[0] == "1. pour milk for 10 seconds"   # completed steps keep theirs


def test_a_request_never_contains_a_step_after_the_one_it_asks_about():
    steps = _steps([(1, 1, 3), (2, 2, 3), (3, 3, 3)])
    for i in range(len(steps)):
        assert len(predict.predictive_prefix(steps, i).splitlines()) == i + 1


def test_requests_share_every_line_but_the_last_which_is_what_prefix_caching_reuses():
    """Not a STRICT prefix -- line i changes from `(in progress)` to a completed step -- which is
    inherent to asking about a step before it ends, so it is pinned rather than assumed away."""
    steps = _steps([(1, 1, 3), (2, 2, 3), (3, 3, 3)])
    a = predict.predictive_prefix(steps, 1).splitlines()
    b = predict.predictive_prefix(steps, 2).splitlines()
    assert a[:-1] == b[:-2]


def test_one_request_per_step_boundary():
    steps = _steps([(1, 1, 3), (2, 2, 3), (3, 3, 3)])
    stub = StubClient([])
    predict.run_predictive(stub, "SYS", steps, VOCAB)
    assert len(stub.calls) == len(steps) == predict.request_cost(steps)
    # two turns, system then one user turn -- the layout the prefix cache needs
    assert [m["role"] for m in stub.calls[0]] == ["system", "user"]


# ---- the symbol channel -------------------------------------------------------------------

def test_the_symbol_channel_cannot_fire_on_the_first_step():
    """No prior prediction exists to disagree with."""
    steps = _steps([(1, 1, 5), (2, 2, 5)])
    flags, _ = predict.run_predictive(
        StubClient(["DURATION: 5 seconds\nNEXT: take cereals",
                    "DURATION: 5 seconds\nNEXT: take cereals"]), "SYS", steps, VOCAB)
    assert all(f.tick != 0 for f in flags if f.channel == predict.SYMBOL)


def test_the_symbol_channel_fires_at_the_first_tick_of_the_step_that_disagrees():
    steps = _steps([(1, 1, 5), (2, 2, 5)])
    flags, _ = predict.run_predictive(
        StubClient(["DURATION: 5 seconds\nNEXT: take cereals",   # predicts take cereals
                    "DURATION: 5 seconds\nNEXT: pour milk"]), "SYS", steps, VOCAB)
    sym = [f for f in flags if f.channel == predict.SYMBOL]
    assert [f.tick for f in sym] == [5]      # step 1 begins at tick 5 and is `stir bowl`


def test_a_correct_prediction_is_silent():
    steps = _steps([(1, 1, 5), (2, 2, 5)])
    flags, _ = predict.run_predictive(
        StubClient(["DURATION: 5 seconds\nNEXT: stir bowl",
                    "DURATION: 5 seconds\nNEXT: pour milk"]), "SYS", steps, VOCAB)
    assert flags == []


# ---- the duration channel, two-sided -------------------------------------------------------

def test_the_over_run_side_fires_once_then_suppresses_until_the_next_boundary():
    """Predicted 3s, tolerance max(2, 1.5) = 2, so the alarm is the first tick past 5s elapsed."""
    steps = _steps([(1, 1, 20)])
    flags, _ = predict.run_predictive(
        StubClient(["DURATION: 3 seconds\nNEXT: stir bowl"]), "SYS", steps, VOCAB)
    over = [f for f in flags if f.channel == predict.DUR_LONG]
    assert [f.tick for f in over] == [5]


def test_the_under_run_side_fires_at_the_tick_that_closes_the_short_step():
    """Retrospective by construction: a step that stopped early is only visible once it stops."""
    steps = _steps([(1, 1, 4), (2, 2, 5)])
    flags, _ = predict.run_predictive(
        StubClient(["DURATION: 20 seconds\nNEXT: stir bowl",
                    "DURATION: 5 seconds\nNEXT: pour milk"]), "SYS", steps, VOCAB)
    under = [f for f in flags if f.channel == predict.DUR_SHORT]
    assert [f.tick for f in under] == [3]     # step 0 spans ticks 0..3


def test_the_under_run_side_is_what_makes_abandonment_reachable_at_all():
    """Over-run alone can never catch a step that stopped EARLY, and the step following a
    truncated segment is the correct next symbol, so the symbol channel is silent too."""
    steps = _steps([(1, 1, 3), (2, 2, 8)])
    replies = ["DURATION: 30 seconds\nNEXT: stir bowl",     # step 0 quits at 3 of a predicted 30
               "DURATION: 8 seconds\nNEXT: pour milk"]      # and step 1 is exactly as expected
    flags, _ = predict.run_predictive(StubClient(replies), "SYS", steps, VOCAB)
    assert [f.channel for f in flags] == [predict.DUR_SHORT]


def test_a_duration_inside_tolerance_is_silent_on_both_sides():
    steps = _steps([(1, 1, 10)])
    flags, _ = predict.run_predictive(
        StubClient(["DURATION: 12 seconds\nNEXT: stir bowl"]), "SYS", steps, VOCAB)
    assert flags == []


def test_the_duration_channel_conditions_on_the_observed_symbol_so_the_two_are_independent():
    """A substitution cannot leak into the temporal channel: the duration being timed is that of
    the step actually seen, whatever it turned out to be."""
    steps = _steps([(1, 1, 5), (3, 3, 5)])
    flags, _ = predict.run_predictive(
        StubClient(["DURATION: 5 seconds\nNEXT: stir bowl",   # wrong symbol...
                    "DURATION: 5 seconds\nNEXT: pour milk"]), "SYS", steps, VOCAB)
    assert {f.channel for f in flags} == {predict.SYMBOL}      # ...but no duration flag


def test_per_line_parse_failures_are_counted_separately_per_channel():
    steps = _steps([(1, 1, 5), (2, 2, 5)])
    _, stats = predict.run_predictive(
        StubClient(["DURATION: 5 seconds\nNEXT: fly rocket", "nothing useful"]),
        "SYS", steps, VOCAB)
    assert stats == {"n_boundaries": 2, "duration_parse_failures": 1, "next_parse_failures": 2}


# ---- the bridge back to the element layer ----------------------------------------------------

def test_flags_become_verdicts_on_whichever_element_contains_them():
    steps = _steps([(1, 1, 10), (2, 2, 10)])
    ticks = textify.ticks_from_ids([1] * 10 + [2] * 10, [1] * 10 + [2] * 10, _Lex())
    flags = [predict.Flag(12, 1.0, predict.SYMBOL)]
    assert [v.is_anomaly for v in predict.verdicts_from_flags(flags, steps)] == [False, True]
    assert [v.step_index for v in predict.verdicts_from_flags(flags, ticks) if v.is_anomaly] == [12]


def test_the_verdict_type_labels_mirror_the_hsmm_channel_map():
    """So this arm's type_confusion is read against the HSMM's on the same axes."""
    assert predict.CHANNEL_TO_TYPE[predict.SYMBOL] == "omission"
    assert predict.CHANNEL_TO_TYPE[predict.DUR_LONG] == "repetition"
    assert predict.CHANNEL_TO_TYPE[predict.DUR_SHORT] == "abandonment"


def test_the_most_specific_channel_wins_when_two_fire_on_one_element():
    steps = _steps([(1, 1, 10)])
    flags = [predict.Flag(3, 1.0, predict.DUR_LONG), predict.Flag(3, 1.0, predict.SYMBOL)]
    v, = predict.verdicts_from_flags(flags, steps)
    assert v.error_type == predict.CHANNEL_TO_TYPE[predict.SYMBOL]


# ---- the response cache ----------------------------------------------------------------------

def test_the_cache_key_is_unchanged_when_max_tokens_is_unset():
    """~30k responses were collected before the field existed; folding a None into the key would
    orphan every one of them."""
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    c = llm_client.ChatClient(dry_run=True)
    legacy = hashlib.sha256(json.dumps(
        {"base_url": c.base_url, "model": c.model, "temperature": c.temperature,
         "messages": msgs}, sort_keys=True).encode()).hexdigest()
    assert c._cache_key(msgs) == legacy
    assert llm_client.ChatClient(dry_run=True, max_tokens=None)._cache_key(msgs) == legacy
    assert llm_client.ChatClient(dry_run=True, max_tokens=48)._cache_key(msgs) != legacy


def test_the_predictive_preprompt_does_not_disturb_the_frozen_verdict_ones():
    vocab = {"verbs": {"pour": 0}, "nouns": {"milk": 0}}
    a = prompts.build_variant("no-recipes", vocab, protocol="incremental", unit="step")
    b = prompts.build_variant("no-recipes", vocab, protocol="predictive", unit="step")
    assert a != b
    assert "DURATION:" in b and "NEXT:" in b
    assert "No Anomaly" in a and "No Anomaly" not in b   # it does not ask for a verdict at all


# ---- Auto-CoT demonstrations -----------------------------------------------------------------

def test_the_answer_line_is_taken_from_the_end_not_the_middle_of_a_reasoning_reply():
    """With Auto-CoT the model reasons first, so a number cited mid-reasoning must not be mistaken
    for the answer."""
    reply = ("Reasoning: pouring milk usually runs about 5 seconds, and DURATION: 99 seconds "
             "would be unusual.\nDURATION: 12 seconds\nNEXT: stir bowl")
    p = predict.parse_predictions(reply, VOCAB)[0]
    assert p.duration == 12 and p.symbol == ("stir", "bowl")


def test_rationales_are_generated_conditioned_on_the_exemplars_true_outcome():
    """Otherwise the demonstration argues for one answer and states another -- measured happening
    on real output, and worse than shipping no rationale at all."""
    steps = _steps([(1, 1, 5), (2, 2, 9), (1, 1, 4), (3, 3, 6)])
    b = prompts.exemplar_boundary(steps)
    stub = StubClient(["because the bowl is already full"])
    predict.build_rationales(stub, [steps], prompts.exemplar_boundary)
    sent = stub.calls[0][0]["content"]
    assert f"ran for {steps[b].duration} seconds" in sent
    assert f"'{steps[b + 1].verb} {steps[b + 1].noun}'" in sent


def test_a_cot_demonstration_states_the_exemplars_own_ground_truth():
    steps = _steps([(1, 1, 5), (2, 2, 9), (1, 1, 4), (3, 3, 6)])
    b = prompts.exemplar_boundary(steps)
    block = prompts.fewshot_block([steps], ["some reasoning"])
    assert f"DURATION: {steps[b].duration} seconds" in block
    assert f"NEXT: {steps[b + 1].verb} {steps[b + 1].noun}" in block
    assert f"{steps[b].verb} {steps[b].noun} (in progress)" in block
    assert "some reasoning" in block


def test_clustered_exemplars_cover_more_of_the_corpus_than_a_uniform_draw():
    """Auto-CoT's selection step. Two synthetic 'dishes', heavily imbalanced: a uniform draw of 2
    is likely to take both from the majority dish, clustering cannot."""
    common = {"verb_ids": [1] * 10 + [2] * 10, "noun_ids": [1] * 10 + [2] * 10}
    rare = {"verb_ids": [3] * 10 + [3] * 10, "noun_ids": [3] * 10 + [3] * 10}
    trajectories = [common] * 30 + [rare] * 3
    vocab = {"verbs": VOCAB["verbs"], "nouns": VOCAB["nouns"]}
    picked = prompts.select_exemplars(trajectories, _Lex(), 2, seed=0,
                                      method="autocot", vocab=vocab)
    kinds = {(s[0].verb, s[0].noun) for s in picked}
    assert len(kinds) == 2          # one representative from each cluster, not two of the same


def test_shot_select_rejects_an_unknown_method():
    with pytest.raises(ValueError):
        prompts.select_exemplars([{"verb_ids": [1], "noun_ids": [1]}], _Lex(), 1, method="nope")
