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
