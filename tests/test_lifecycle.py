import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cook_ad.hsmm import params
from cook_ad.lifecycle import divergence, state_manager
from cook_ad.lifecycle.online_update import PreferenceEvent

jax.config.update("jax_enable_x64", True)

K, V, N, D_MAX = 4, 6, 6, 20
DOMINANT_STATE = 1
DOMINANT_NOUN = 2
SUB_NOUN = 5  # a near-prior (surprising) noun in DOMINANT_STATE -- the "substitution"


def _toy_frozen():
    p = params.init_weak_limit_params(jax.random.PRNGKey(0), K, V, N, D_MAX)
    verb = p.verb_counts.at[DOMINANT_STATE, 0].set(200.0)
    noun = p.noun_counts.at[DOMINANT_STATE, DOMINANT_NOUN].set(200.0)
    return p._replace(verb_counts=verb, noun_counts=noun)


def _noun_prob(p, state, token):
    _, _, _, log_emit_n = params.normalize_categoricals(p)
    return float(jnp.exp(log_emit_n[state, token]))


def test_preference_bump_is_bounded_and_stays_elevated():
    """Repeatedly confirming a substitution nudges the live model toward it but -- because the
    bump is capped at frozen + max_bump -- it plateaus well below the state's dominant token,
    so the substitution keeps registering as surprising (low probability). One acceptance, or
    even ten, does not blind the detector."""
    frozen = _toy_frozen()
    dual = state_manager.init_dual_model(frozen)
    event = PreferenceEvent("noun", DOMINANT_STATE, SUB_NOUN)
    delta, max_bump = 2.0, 5.0

    frozen_cell = float(frozen.noun_counts[DOMINANT_STATE, SUB_NOUN])
    for _ in range(10):
        dual, rec = state_manager.handle_confirmation(dual, event, "preference", delta, max_bump)
        assert rec["updated"]

    live_cell = float(dual.live.noun_counts[DOMINANT_STATE, SUB_NOUN])
    assert live_cell == pytest.approx(frozen_cell + max_bump)  # capped, not runaway
    assert _noun_prob(dual.live, DOMINANT_STATE, SUB_NOUN) < 0.1  # still elevated surprise
    assert _noun_prob(dual.live, DOMINANT_STATE, SUB_NOUN) < _noun_prob(dual.live, DOMINANT_STATE, DOMINANT_NOUN)


def test_breakdown_updates_nothing():
    frozen = _toy_frozen()
    dual = state_manager.init_dual_model(frozen)
    event = PreferenceEvent("noun", DOMINANT_STATE, SUB_NOUN)

    new_dual, rec = state_manager.handle_confirmation(dual, event, "breakdown")

    assert rec["flagged"] and not rec["updated"]
    for field in params.HSMMParams._fields:
        if getattr(dual.live, field) is None:
            continue   # kernel_v/kernel_n: absent == identity, and never touched by an update
        assert jnp.array_equal(getattr(new_dual.live, field), getattr(dual.live, field))
        assert jnp.array_equal(getattr(new_dual.frozen, field), getattr(dual.frozen, field))


def test_kl_zero_at_init_grows_localizes_then_resets():
    frozen = _toy_frozen()
    dual = state_manager.init_dual_model(frozen)
    event = PreferenceEvent("noun", DOMINANT_STATE, SUB_NOUN)

    assert float(divergence.model_divergence(dual.live, dual.frozen)["total"]) == pytest.approx(0.0, abs=1e-9)

    for _ in range(4):
        dual, _ = state_manager.handle_confirmation(dual, event, "preference")

    div = divergence.model_divergence(dual.live, dual.frozen)
    assert float(div["total"]) > 0.0
    per_state = np.asarray(div["per_state"])
    assert int(np.argmax(per_state)) == DOMINANT_STATE  # drift localizes to the bumped state
    assert float(div["noun"][DOMINANT_STATE]) > 0.0
    assert float(div["verb"][DOMINANT_STATE]) == pytest.approx(0.0, abs=1e-9)  # untouched channel

    consolidated = state_manager.consolidate(dual, approved="all")
    assert float(divergence.model_divergence(consolidated.live, consolidated.frozen)["total"]) == pytest.approx(0.0, abs=1e-9)


def test_consolidation_rebaselines_and_bounds_precision():
    """After consolidation frozen absorbs the approved change and live resets to it; a second
    window of bumps cannot push live precision past frozen + max_bump -- precision does not
    compound across windows (the reset-not-decay guarantee)."""
    frozen = _toy_frozen()
    dual = state_manager.init_dual_model(frozen)
    event = PreferenceEvent("noun", DOMINANT_STATE, SUB_NOUN)
    max_bump = 5.0

    frozen_cell0 = float(frozen.noun_counts[DOMINANT_STATE, SUB_NOUN])
    for _ in range(10):
        dual, _ = state_manager.handle_confirmation(dual, event, "preference", 2.0, max_bump)
    dual = state_manager.consolidate(dual, approved="all")

    frozen_cell1 = float(dual.frozen.noun_counts[DOMINANT_STATE, SUB_NOUN])
    assert frozen_cell1 == pytest.approx(frozen_cell0 + max_bump)  # approved change baked in
    assert jnp.array_equal(dual.live.noun_counts, dual.frozen.noun_counts)  # live reset

    for _ in range(10):
        dual, _ = state_manager.handle_confirmation(dual, event, "preference", 2.0, max_bump)
    live_cell2 = float(dual.live.noun_counts[DOMINANT_STATE, SUB_NOUN])
    assert live_cell2 == pytest.approx(frozen_cell1 + max_bump)  # bounded relative to NEW frozen, no compounding


def test_selective_consolidation_discards_unapproved_drift():
    frozen = _toy_frozen()
    dual = state_manager.init_dual_model(frozen)
    pref = PreferenceEvent("noun", DOMINANT_STATE, SUB_NOUN)
    other = PreferenceEvent("verb", 2, 3)

    dual, _ = state_manager.handle_confirmation(dual, pref, "preference")
    dual, _ = state_manager.handle_confirmation(dual, other, "preference")

    consolidated = state_manager.consolidate(dual, approved=[pref])

    # approved noun cell copied into frozen; unapproved verb drift discarded
    assert float(consolidated.frozen.noun_counts[DOMINANT_STATE, SUB_NOUN]) > float(frozen.noun_counts[DOMINANT_STATE, SUB_NOUN])
    assert jnp.array_equal(consolidated.frozen.verb_counts, frozen.verb_counts)
    assert jnp.array_equal(consolidated.live.verb_counts, frozen.verb_counts)  # live reset drops it too


def test_categorical_kl_matches_manual_and_is_nonnegative():
    log_p = jnp.log(jnp.array([0.7, 0.2, 0.1]))
    log_q = jnp.log(jnp.array([0.5, 0.3, 0.2]))
    manual = sum(pp * (np.log(pp) - np.log(qq)) for pp, qq in [(0.7, 0.5), (0.2, 0.3), (0.1, 0.2)])
    assert float(divergence.categorical_kl(log_p, log_q)) == pytest.approx(manual)
    assert float(divergence.categorical_kl(log_p, log_p)) == pytest.approx(0.0, abs=1e-12)
