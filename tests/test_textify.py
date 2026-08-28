import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cook_ad.anomaly import narrate
from cook_ad.hsmm import params
from cook_ad.llm import textify
from cook_ad.synthetic import error_injection, generate

jax.config.update("jax_enable_x64", True)

K, V, N, D_MAX = 5, 6, 8, 30

VOCAB = {
    "verbs": {"stall": 0, "pour": 1, "take": 2, "stir": 3, "cut": 4, "put": 5},
    "nouns": {"kitchen": 0, "bowl": 1, "cereals": 2, "milk": 3,
              "cup": 4, "water": 5, "tea": 6, "sugar": 7},
}


def _peaked_params():
    """Same construction as tests/test_evaluation.py: sharp emissions and moderate durations, so
    sampled trajectories are well-structured and injections are genuinely anomalous."""
    p = params.init_weak_limit_params(jax.random.PRNGKey(0), K, V, N, D_MAX)
    verb = jnp.full((K, V), 0.2).at[jnp.arange(K), jnp.arange(K) % V].set(200.0)
    noun = jnp.full((K, N), 0.2).at[jnp.arange(K), jnp.arange(K) % N].set(200.0)
    return p._replace(verb_counts=verb, noun_counts=noun,
                      dur_r=jnp.full((K,), 8.0), dur_p=jnp.full((K,), 0.5))


def _lexicon():
    return narrate.Lexicon(VOCAB, _peaked_params())


def test_rle_produces_maximal_runs():
    lex = _lexicon()
    verbs = [0, 0, 1, 1, 1, 2]
    nouns = [0, 0, 2, 2, 3, 1]  # verb constant across the 2->3 noun change: still a boundary
    steps = textify.steps_from_ids(verbs, nouns, lex)

    assert [(s.tick_start, s.tick_end, s.duration) for s in steps] == [
        (0, 2, 2), (2, 4, 2), (4, 5, 1), (5, 6, 1)
    ]
    assert [s.index for s in steps] == [0, 1, 2, 3]
    # steps tile the trial exactly, with no gaps and no overlap
    assert steps[0].tick_start == 0
    assert steps[-1].tick_end == len(verbs)
    assert all(a.tick_end == b.tick_start for a, b in zip(steps, steps[1:]))


def test_render_uses_raw_names_and_pluralises():
    lex = _lexicon()
    steps = textify.steps_from_ids([1, 1, 2], [2, 2, 1], lex)
    assert textify.render_trial(steps) == ["pour cereals for 2 seconds", "take bowl for 1 second"]


def test_sil_renders_literally_not_collapsed():
    """lexicon.phrase() collapses SIL to 'idle'/bare-noun, which would break the fixed VERB NOUN
    template the response grammar and parser are built on."""
    lex = _lexicon()
    assert lex.phrase(0, 0) == "idle"                    # what narration would say
    steps = textify.steps_from_ids([0, 0], [0, 0], lex)
    assert textify.render_step(steps[0]) == "stall kitchen for 2 seconds"

    steps = textify.steps_from_ids([0], [1], lex)        # SIL verb, real noun
    assert lex.phrase(0, 1) == "bowl"                    # narration drops the SIL verb
    assert textify.render_step(steps[0]) == "stall bowl for 1 second"   # textify keeps both


def test_step_covering_tick():
    lex = _lexicon()
    steps = textify.steps_from_ids([1, 1, 1, 2, 2], [2, 2, 2, 1, 1], lex)
    assert textify.step_covering_tick(steps, 0).index == 0
    assert textify.step_covering_tick(steps, 2).index == 0
    assert textify.step_covering_tick(steps, 3).index == 1
    assert textify.step_covering_tick(steps, 99) is None


def test_gt_steps_for_window_is_inclusive_of_both_ends():
    lex = _lexicon()
    steps = textify.steps_from_ids([1] * 5 + [2] * 5, [2] * 5 + [1] * 5, lex)
    assert textify.gt_steps_for_window(steps, (0, 0)) == [0]
    assert textify.gt_steps_for_window(steps, (4, 5)) == [0, 1]   # spans the boundary
    assert textify.gt_steps_for_window(steps, (5, 9)) == [1]


@pytest.mark.parametrize("error_type", error_injection.ERROR_TYPES)
def test_injection_window_maps_onto_at_least_one_step(error_type):
    """The tick-space ground truth from error_injection must land on real steps, for every
    injector -- this is the bridge the whole step-level evaluation rests on."""
    p = _peaked_params()
    lex = _lexicon()
    traj = generate.sample_trajectory(p, np.random.default_rng(3), max_ticks=150, d_max=D_MAX)
    assert len(traj["segments"]) >= error_injection.MIN_SEGMENTS

    deg = error_injection.inject(error_type, traj, np.random.default_rng(4), p)
    steps = textify.steps_from_trajectory(deg, lex)
    gt = textify.gt_steps_for_window(steps, deg["window"])

    assert gt, f"{error_type} window {deg['window']} mapped to no steps"
    assert all(0 <= i < len(steps) for i in gt)
    assert steps[-1].tick_end == len(deg["verb_ids"])


@pytest.mark.parametrize("error_type", error_injection.ERROR_TYPES)
def test_source_step_at_window_start_recovers_a_correction(error_type):
    """step_covering_tick on the SOURCE trial at the degraded window's first tick is the
    ground-truth 'correct move' for every injector -- see its docstring for why one rule works
    for all five."""
    p = _peaked_params()
    lex = _lexicon()
    traj = generate.sample_trajectory(p, np.random.default_rng(3), max_ticks=150, d_max=D_MAX)
    deg = error_injection.inject(error_type, traj, np.random.default_rng(4), p)

    source_steps = textify.steps_from_trajectory(traj, lex)
    source = textify.step_covering_tick(source_steps, deg["window"][0])
    assert source is not None
    assert source.duration >= 1


def test_abandonment_correction_recovers_the_full_original_duration():
    """The specific case the generic rule has to get right: abandonment leaves the verb and noun
    alone and only truncates, so the correction IS the original duration."""
    p = _peaked_params()
    lex = _lexicon()
    traj = generate.sample_trajectory(p, np.random.default_rng(11), max_ticks=200, d_max=D_MAX)
    deg = error_injection.inject_abandonment(traj, np.random.default_rng(0))

    steps = textify.steps_from_trajectory(deg, lex)
    gt = textify.gt_steps_for_window(steps, deg["window"])
    truncated = steps[gt[0]]

    source = textify.step_covering_tick(
        textify.steps_from_trajectory(traj, lex), deg["window"][0]
    )
    assert (source.verb, source.noun) == (truncated.verb, truncated.noun)
    assert source.duration > truncated.duration


def test_assert_tick_seconds_rejects_non_one_second_ticks():
    textify.assert_tick_seconds({"tick_seconds": 1.0})
    textify.assert_tick_seconds({})  # default
    with pytest.raises(ValueError, match="seconds"):
        textify.assert_tick_seconds({"tick_seconds": 0.5})


def test_recipe_block_does_not_teach_the_underscore_format():
    """The prompt must name steps the same way the observation stream does. Rendering them as
    `stir_dough` made models copy that into corrections, which the grammar then rejected."""
    from cook_ad.llm import prompts
    assert prompts._label_to_step_text("stir_dough") == "stir dough"
    assert prompts._label_to_step_text("SIL") == "stall kitchen"
    assert prompts._label_to_step_text("put_egg2plate") == "put egg2plate"  # only first _ splits


# --------------------------------------------------------------------------------------------
# the tick unit
# --------------------------------------------------------------------------------------------

def test_tick_elements_are_one_per_tick_with_index_equal_to_tick():
    """The property the whole tick unit rests on: a tick element's index IS its tick, so every
    consumer written against (index, tick_start, tick_end) -- the ground-truth bridge, the
    metrics -- transfers with no translation layer."""
    lex = _lexicon()
    verbs = [0, 0, 1, 1, 1, 2]
    nouns = [0, 0, 2, 2, 3, 1]
    ticks = textify.ticks_from_ids(verbs, nouns, lex)

    assert len(ticks) == len(verbs)
    for t, e in enumerate(ticks):
        assert (e.index, e.tick_start, e.tick_end, e.duration) == (t, t, t + 1, 1)
    assert [e.verb_id for e in ticks] == verbs
    assert [e.noun_id for e in ticks] == nouns


def test_tick_elements_cover_the_same_ticks_as_the_steps_they_replace():
    lex = _lexicon()
    verbs = [0, 0, 1, 1, 1, 2]
    nouns = [0, 0, 2, 2, 3, 1]
    steps = textify.steps_from_ids(verbs, nouns, lex)
    ticks = textify.ticks_from_ids(verbs, nouns, lex)

    assert steps[-1].tick_end == ticks[-1].tick_end
    # every tick element falls inside exactly one step, carrying that step's labels
    for e in ticks:
        owner = textify.step_covering_tick(steps, e.tick_start)
        assert (owner.verb, owner.noun) == (e.verb, e.noun)


def test_tick_rendering_omits_the_duration():
    """One line is one second by construction. Restating 'for 1 second' on every line would both
    waste the prompt and invite the model to read each tick as a completed one-second step."""
    lex = _lexicon()
    ticks = textify.ticks_from_ids([1, 1], [2, 2], lex)
    assert textify.render_tick(ticks[0]) == "pour cereals"
    assert textify.render_element(ticks[0], "tick") == "pour cereals"
    assert textify.render_element(ticks[0], "step") == "pour cereals for 1 second"


def test_gt_window_maps_to_exactly_its_own_ticks_at_the_tick_unit():
    """gt_steps_for_window is reused unchanged for both units. At the tick unit the answer must be
    the window itself -- an off-by-one here would silently shift every ground truth."""
    lex = _lexicon()
    ticks = textify.ticks_from_ids([0] * 10, [0] * 10, lex)
    assert textify.gt_steps_for_window(ticks, (3, 5)) == [3, 4, 5]
    assert textify.gt_steps_for_window(ticks, (0, 0)) == [0]


def test_elements_from_trajectory_dispatches_on_unit():
    lex = _lexicon()
    traj = {"verb_ids": [0, 0, 1], "noun_ids": [0, 0, 2]}
    assert len(textify.elements_from_trajectory(traj, lex, "step")) == 2
    assert len(textify.elements_from_trajectory(traj, lex, "tick")) == 3
    assert len(textify.elements_from_trajectory(traj, lex)) == 2   # step is the default
    with pytest.raises(ValueError, match="unknown unit"):
        textify.elements_from_trajectory(traj, lex, "segment")


def test_debris_is_the_border_rule_and_only_for_substitution_at_both_units():
    """Pins which debris rule actually fires, because the docstring here once claimed the
    opposite of the measurement. Rule 'borders an edited tick' is the live one; the interior
    -splice rule has never fired for a shipped injector at either unit.

    Built by hand rather than through an injector so the assertion does not move if injector
    internals change: a 3-run trial whose MIDDLE run was retagged, which is substitution's shape.
    """
    lex = _lexicon()
    verbs = [1] * 4 + [2] * 4 + [3] * 4     # the middle run is the substituted segment
    tick_map = np.arange(12)                # no splices at all -- substitution reorders nothing
    edited = list(range(4, 8))

    steps = textify.steps_from_ids(verbs, verbs, lex)
    gt_steps = textify.gt_steps_for_window(steps, (4, 7))
    assert gt_steps == [1]
    # both neighbouring RUNS border an edited tick -- ~8 ticks excused from FP scoring
    assert textify.injection_touched_steps(steps, tick_map, edited, gt_steps) == {0, 2}

    ticks = textify.ticks_from_ids(verbs, verbs, lex)
    gt_ticks = textify.gt_steps_for_window(ticks, (4, 7))
    assert gt_ticks == [4, 5, 6, 7]
    # at the tick unit the same rule excuses only the two bordering TICKS: a stricter halo
    assert textify.injection_touched_steps(ticks, tick_map, edited, gt_ticks) == {3, 8}


def test_a_splice_on_an_element_boundary_is_not_debris_at_either_unit():
    """The interior-splice rule is deliberately silent when the splice lands where the RLE
    already breaks -- which is every splice the five shipped injectors produce, and which at the
    tick unit is every splice there can be."""
    lex = _lexicon()
    verbs = [1] * 3 + [3] * 3
    tick_map = np.array([0, 1, 2, 7, 8, 9])   # ticks 3..5 come from elsewhere: splice at 2->3

    steps = textify.steps_from_ids(verbs, verbs, lex)
    assert textify.injection_touched_steps(steps, tick_map, (), gt_steps=[]) == set()
    ticks = textify.ticks_from_ids(verbs, verbs, lex)
    assert textify.injection_touched_steps(ticks, tick_map, (), gt_steps=[]) == set()


# --------------------------------------------------------------------------------------------
# point ground truth
# --------------------------------------------------------------------------------------------

def _traj_from(verbs, nouns, seg_lengths):
    return {"verb_ids": np.array(verbs), "noun_ids": np.array(nouns),
            "segments": [(i, d) for i, d in enumerate(seg_lengths)]}


def test_structural_injectors_mark_junctions_not_whole_ranges():
    """A transposition's two runs are each performed correctly; only their ORDER is wrong, so the
    ground truth is the three junctions and the ~46 ticks between them are debris. Before this,
    30% of a trial counted as transposition ground truth at the tick unit against 0.7% for
    abandonment, and per-type recalls were not comparable to each other."""
    lex = _lexicon()
    verbs = [0] * 5 + [1] * 6 + [2] * 7 + [3] * 5
    traj = _traj_from(verbs, verbs, [5, 6, 7, 5])
    dg = error_injection.inject("transposition", traj, np.random.default_rng(0), _peaked_params())

    ticks = textify.ticks_from_trajectory(dg, lex)
    gt = textify.gt_steps_for_ticks(ticks, dg["anomaly_ticks"])
    assert len(gt) == 3                                    # three junctions, not the whole span
    window = dg["window"]
    assert len(textify.gt_steps_for_window(ticks, window)) == window[1] - window[0] + 1 > 3

    debris = textify.injection_touched_steps(
        ticks, dg["tick_map"], dg["edited_ticks"], gt, window=window)
    # every tick of the disturbed span is now exactly one of: ground truth, or debris
    span = set(range(window[0], window[1] + 1))
    assert span == set(gt) | debris
    assert not (set(gt) & debris)


def test_point_ground_truth_leaves_the_step_unit_bit_for_bit_unchanged():
    """The property that lets the existing step-unit results stand. Both junction points of a
    repetition sit INSIDE the copy, and a transposition's exit junction is the last tick of the
    misplaced run rather than the first tick of its innocent successor -- so at the step unit the
    points select exactly the steps the window did. Measured over 290 real injections; asserted
    here on all five injectors so a future injector cannot silently break it."""
    lex = _lexicon()
    verbs = [0] * 5 + [1] * 6 + [2] * 7 + [3] * 6 + [4] * 5
    traj = _traj_from(verbs, verbs, [5, 6, 7, 6, 5])

    for etype in error_injection.ERROR_TYPES:
        dg = error_injection.inject(etype, traj, np.random.default_rng(1), _peaked_params())
        steps = textify.steps_from_trajectory(dg, lex)
        by_window = set(textify.gt_steps_for_window(steps, dg["window"]))
        by_points = set(textify.gt_steps_for_ticks(steps, dg["anomaly_ticks"]))
        assert by_points == by_window, etype
        assert textify.injection_touched_steps(
            steps, dg["tick_map"], dg["edited_ticks"], by_points, window=dg["window"]
        ) == textify.injection_touched_steps(
            steps, dg["tick_map"], dg["edited_ticks"], by_window
        ), etype


def test_substitution_keeps_its_whole_range_as_ground_truth():
    """Not every injector is a junction. Every second of a step done with the wrong object is
    itself wrong, so substitution's points ARE its window -- at both units."""
    lex = _lexicon()
    verbs = [0] * 5 + [1] * 6 + [2] * 7
    traj = _traj_from(verbs, verbs, [5, 6, 7])
    dg = error_injection.inject("substitution", traj, np.random.default_rng(0), _peaked_params())

    t0, t1 = dg["window"]
    assert list(dg["anomaly_ticks"]) == list(range(t0, t1 + 1))
    ticks = textify.ticks_from_trajectory(dg, lex)
    assert (textify.gt_steps_for_ticks(ticks, dg["anomaly_ticks"])
            == textify.gt_steps_for_window(ticks, dg["window"]))
