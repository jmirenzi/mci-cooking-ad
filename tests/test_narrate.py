import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cook_ad.anomaly import narrate, surprise
from cook_ad.hsmm import params
from cook_ad.lifecycle import state_manager
from cook_ad.recipe import recipe_hmm
from cook_ad.synthetic import error_injection, generate

jax.config.update("jax_enable_x64", True)

K, V, N, D_MAX = 4, 5, 5, 30


def _peaked_params():
    """Sharp verb/noun tables and a moderate duration -- same shape as test_evaluation's
    _peaked_params, so injections are unambiguous and detection is reliable."""
    p = params.init_weak_limit_params(jax.random.PRNGKey(0), K, V, N, D_MAX)
    verb = jnp.full((K, V), 0.2).at[jnp.arange(K), jnp.arange(K) % V].set(200.0)
    noun = jnp.full((K, N), 0.2).at[jnp.arange(K), jnp.arange(K) % N].set(200.0)
    return p._replace(
        verb_counts=verb, noun_counts=noun,
        dur_r=jnp.full((K,), 6.0), dur_p=jnp.full((K,), 0.5),
    )


def _toy_vocab():
    return {
        "verbs": {f"v{i}": i for i in range(V)},
        "nouns": {f"n{i}": i for i in range(N)},
        "recipes": {},
    }


def _recipe_params():
    return recipe_hmm.init_weak_limit_recipe_params(jax.random.PRNGKey(5), k_recipe=2, k_subtask=K)


def _long_trajectory(seed):
    for offset in range(20):
        traj = generate.sample_trajectory(
            _peaked_params(), np.random.default_rng(seed + offset), max_ticks=150, d_max=D_MAX
        )
        if len(traj["segments"]) >= error_injection.MIN_SEGMENTS:
            return traj
    raise RuntimeError("could not find a trajectory with enough segments")


def test_missing_step_finds_bridge():
    log_trans = np.full((4, 4), np.log(0.01))
    log_trans[0, 1] = np.log(0.9)  # a(0) -> b(1) likely
    log_trans[1, 2] = np.log(0.9)  # b(1) -> c(2) likely
    log_trans[0, 2] = np.log(0.01)  # a(0) -> c(2) direct: unlikely
    log_trans[np.arange(4), np.arange(4)] = -np.inf

    bridge, gain = narrate.missing_step(log_trans, a=0, c=2, min_gain=2.0)
    assert bridge == 1
    expected_gain = (log_trans[0, 1] + log_trans[1, 2]) - log_trans[0, 2]
    assert gain == pytest.approx(float(expected_gain))
    assert gain >= 2.0

    log_trans_no_bridge = log_trans.copy()
    log_trans_no_bridge[0, 2] = np.log(0.5)  # direct jump now plausible; no bridge clears min_gain
    bridge2, gain2 = narrate.missing_step(log_trans_no_bridge, a=0, c=2, min_gain=2.0)
    assert bridge2 is None
    assert gain2 < 2.0


def test_narrate_emits_one_query_per_segment_not_per_tick():
    p = _peaked_params()
    recipe_params = _recipe_params()
    traj = _long_trajectory(seed=1)

    verb_ids = list(traj["verb_ids"])
    noun_ids = list(traj["noun_ids"])
    segments = traj["segments"]

    # elongate the second segment far past its expected duration by repeating its last tokens
    pos = sum(d for _, d in segments[:1])
    _, d = segments[1]
    insert_at = pos + d
    extra = 40
    verb_ids = verb_ids[:insert_at] + [verb_ids[insert_at - 1]] * extra + verb_ids[insert_at:]
    noun_ids = noun_ids[:insert_at] + [noun_ids[insert_at - 1]] * extra + noun_ids[insert_at:]

    trace, log_probs, recipe_log_trans = surprise.compute_trace(
        p, recipe_params, verb_ids, noun_ids, D_MAX
    )
    flags = surprise.flag(trace, log_probs, recipe_log_trans)

    # locate the elongated segment in the re-decoded trace (its start tick is unchanged; only
    # its end moved). Other segments elsewhere in the trial may independently stall too, so
    # target this one specifically rather than counting s_temporal queries trace-wide.
    new_segments = narrate.segments_from_z(trace.z_star)
    seg_idx, (state, start, end) = next(
        (i, s) for i, s in enumerate(new_segments) if s[1] <= pos < s[2]
    )
    assert int(np.sum(flags["s_temporal"][start:end])) > 5  # many ticks flagged, one segment

    queries = narrate.narrate(
        trace, flags, _toy_vocab(), p, verb_ids, noun_ids, log_probs, recipe_log_trans
    )
    matching = [q for q in queries if q.channel == "s_temporal" and q.segment_index == seg_idx]
    assert len(matching) == 1


def test_narrate_query_event_routes_to_state_manager():
    p = _peaked_params()
    recipe_params = _recipe_params()
    vocab = _toy_vocab()

    noun_query = None
    for seed in range(10):
        traj = _long_trajectory(seed=100 + seed)
        rng = np.random.default_rng(seed)
        sub = error_injection.inject("substitution", traj, rng, p)
        trace, log_probs, recipe_log_trans = surprise.compute_trace(
            p, recipe_params, sub["verb_ids"], sub["noun_ids"], D_MAX
        )
        flags = surprise.flag(trace, log_probs, recipe_log_trans)
        queries = narrate.narrate(
            trace, flags, vocab, p, sub["verb_ids"], sub["noun_ids"], log_probs, recipe_log_trans
        )
        noun_queries = [q for q in queries if q.channel == "s_noun"]
        if len(noun_queries) == 1:
            noun_query = noun_queries[0]
            break

    assert noun_query is not None, "substitution never produced a clean single s_noun query"
    event = noun_query.event
    assert event is not None and event.channel == "noun"

    dual = state_manager.init_dual_model(p)
    new_dual, rec = state_manager.handle_confirmation(dual, event, "preference")
    assert rec["updated"]

    for field in params.HSMMParams._fields:
        old = np.asarray(getattr(dual.live, field))
        new = np.asarray(getattr(new_dual.live, field))
        if field == "noun_counts":
            changed = np.argwhere(new != old)
            assert changed.shape[0] == 1
            assert tuple(changed[0]) == (event.state, event.token)
        else:
            assert np.array_equal(new, old)
