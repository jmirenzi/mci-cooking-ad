"""Porting the preprompt to a second domain (EPIC-Tent) without disturbing the first.

The load-bearing constraint is that the cooking prompts are frozen response-cache keys, so the
first test here matters more than the rest: adding a domain must not change a single byte of what
the cooking path emits.
"""
import pytest

from cook_ad.llm import prompts

VOCAB = {"verbs": {"stall": 0, "insert": 1, "place": 2, "read": 3},
         "nouns": {"idle": 0, "stake": 1, "guyline": 2, "instruction": 3}}

ORDER = (("insert", "stake"), ("place", "guyline"))
INTERLEAVED = (("read", "instruction"),)


def _labels(n=3):
    """n trials, each: 4 ticks of insert_stake, 2 of place_guyline, 6 of read_instruction."""
    return [{"trial_id": f"p{i:02d}", "recipe_label": "tent", "participant_label": f"p{i:02d}",
             "subtask_labels": ["insert_stake"] * 4 + ["place_guyline"] * 2
                               + ["read_instruction"] * 6}
            for i in range(n)]


# ---------------------------------------------------------------------------
# the frozen cooking path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("protocol", ["incremental", "conversational", "batch", "predictive"])
@pytest.mark.parametrize("unit", ["step", "tick"])
@pytest.mark.parametrize("variant", prompts.VARIANTS)
def test_the_cooking_path_is_byte_identical_to_the_undomained_call(protocol, unit, variant):
    """Every collected response is keyed on this text. The new parameters must be inert."""
    labels = _labels()
    before = prompts.build_variant(variant, VOCAB, labels=labels, protocol=protocol, unit=unit)
    after = prompts.build_variant(variant, VOCAB, labels=labels, protocol=protocol, unit=unit,
                                  domain="cooking", sil_verb="stall", sil_noun="kitchen")
    assert before == after


# ---------------------------------------------------------------------------
# the assembly head
# ---------------------------------------------------------------------------

def test_the_assembly_head_is_a_sibling_not_a_substitution():
    cooking = prompts.predictive_task_block("cooking")
    assembly = prompts.predictive_task_block("assembly")
    assert cooking != assembly
    assert "breakfast" in cooking and "breakfast" not in assembly
    assert "camping tent" in assembly


def test_the_assembly_head_still_asks_for_a_prediction_not_a_verdict():
    """Arm 3's defining property: no verdict, no anomaly vocabulary. If the assembly port
    smuggled either in, its flags would stop coming from predict.py's comparison code."""
    assembly = prompts.predictive_task_block("assembly")
    assert "No Anomaly" not in assembly
    for error_type in prompts.error_injection.ERROR_TYPES:
        assert error_type not in assembly
    assert "DURATION:" in assembly and "NEXT:" in assembly


def test_an_unknown_domain_raises():
    with pytest.raises(ValueError, match="unknown domain"):
        prompts.predictive_task_block("plumbing")
    with pytest.raises(ValueError, match="unknown domain"):
        prompts.build_system_prompt(VOCAB, domain="plumbing")


def test_a_verdict_protocol_on_a_non_cooking_domain_raises():
    """Better to refuse than to hand a tent-assembly trial a preprompt that opens 'as they cook
    a single breakfast recipe' and defines substitution with a mustard example."""
    with pytest.raises(ValueError, match="no 'assembly' head"):
        prompts.build_system_prompt(VOCAB, protocol="incremental", domain="assembly")


# ---------------------------------------------------------------------------
# the SIL sentinel
# ---------------------------------------------------------------------------

def test_sil_tokens_reach_the_vocab_block():
    """The bug this fixes: the block named 'kitchen' as the placeholder on every corpus, and on
    EPIC 'kitchen' is a real noun class while 'idle' is the sentinel."""
    prompt = prompts.build_system_prompt(VOCAB, protocol="predictive", domain="assembly",
                                         sil_verb="stall", sil_noun="idle")
    assert "'stall' is not a real action and 'idle' is not a real object" in prompt
    assert "kitchen" not in prompt


# ---------------------------------------------------------------------------
# the assembly structure block
# ---------------------------------------------------------------------------

def test_median_step_durations_pool_every_run_not_just_the_modal_trial():
    labels = [{"subtask_labels": ["a"] * 3 + ["b"] * 9},
              {"subtask_labels": ["a"] * 5 + ["b"] * 1},
              {"subtask_labels": ["a"] * 100}]
    medians = prompts._median_step_durations(labels)
    assert medians == {"a": 5, "b": 5}


def test_median_step_durations_splits_a_repeated_label_into_runs():
    medians = prompts._median_step_durations(
        [{"subtask_labels": ["a", "a", "b", "a", "a", "a", "a"]}]
    )
    assert medians["a"] == 3, "runs of 2 and 4, not one run of 6"


def test_assembly_block_renders_the_prescribed_order_in_order():
    block = prompts.assembly_block(_labels(), ORDER, INTERLEAVED)
    assert block.index("1. insert stake") < block.index("2. place guyline")
    assert "(~4s)" in block and "(~2s)" in block
    assert "read instruction (~6s)" in block
    assert block.index("2. place guyline") < block.index("read instruction")


def test_assembly_block_states_the_order_is_prescribed_not_empirical():
    """EPIC-Tent's `order` errors are annotated against the printed instructions. Reporting
    empirical order frequencies the way recipe_block does would describe the errors instead."""
    block = prompts.assembly_block(_labels(), ORDER, INTERLEAVED)
    assert "instructions prescribe" in block
    assert "distinct orders observed" not in block, "that is recipe_block's empirical framing"


def test_assembly_block_does_not_leak_how_often_people_deviate():
    """Arm 3 must predict a NORMAL continuation -- the disagreement is the detector. Telling the
    model deviation is common would make it predict deviations and erase the signal."""
    block = prompts.assembly_block(_labels(), ORDER, INTERLEAVED)
    for leak in ("skip", "out of sequence", "repeat", "vary", "mandatory"):
        assert leak not in block.lower()


def test_assembly_block_survives_a_step_nobody_performed():
    """`tie top` is absent from several sessions. A step with no recorded run must still appear
    in the prescribed order, just without a duration."""
    block = prompts.assembly_block(_labels(), (*ORDER, ("tie", "top")))
    assert "3. tie top" in block and "3. tie top (~" not in block


def test_with_recipes_on_assembly_requires_the_canonical_order():
    with pytest.raises(ValueError, match="requires canonical_order"):
        prompts.build_system_prompt(VOCAB, with_recipes=True, labels=_labels(),
                                    protocol="predictive", domain="assembly")


def test_assembly_with_recipes_uses_assembly_block_not_recipe_block():
    prompt = prompts.build_variant("with-recipes", VOCAB, labels=_labels(), protocol="predictive",
                                   domain="assembly", sil_verb="stall", sil_noun="idle",
                                   canonical_order=ORDER, interleaved=INTERLEAVED)
    assert "instructions prescribe" in prompt
    assert "exactly one of the following recipes" not in prompt


def test_the_no_recipes_assembly_arm_carries_no_order_at_all():
    """The comparison the with-recipes arm is read against: same head, no procedure given."""
    prompt = prompts.build_variant("no-recipes", VOCAB, labels=_labels(), protocol="predictive",
                                   domain="assembly", sil_verb="stall", sil_noun="idle",
                                   canonical_order=ORDER)
    assert "instructions prescribe" not in prompt
    assert "camping tent" in prompt
