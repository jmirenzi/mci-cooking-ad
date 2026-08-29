import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cook_ad.anomaly import narrate, quantile, surprise
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

    pi_all = surprise.compute_pi_all(log_probs, verb_ids, noun_ids, D_MAX)
    queries = narrate.narrate(
        trace, flags, _toy_vocab(), p, verb_ids, noun_ids, log_probs, recipe_log_trans, pi_all
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
        pi_all = surprise.compute_pi_all(log_probs, sub["verb_ids"], sub["noun_ids"], D_MAX)
        queries = narrate.narrate(
            trace, flags, vocab, p, sub["verb_ids"], sub["noun_ids"], log_probs, recipe_log_trans, pi_all
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
        if getattr(dual.live, field) is None:
            continue   # kernel_v/kernel_n: absent == identity
        old = np.asarray(getattr(dual.live, field))
        new = np.asarray(getattr(new_dual.live, field))
        if field == "noun_counts":
            changed = np.argwhere(new != old)
            assert changed.shape[0] == 1
            assert tuple(changed[0]) == (event.state, event.token)
        else:
            assert np.array_equal(new, old)


def test_narrate_severity_uses_dilution_corrected_threshold():
    """Regression for the original threshold-mismatch bug: narrate() used to recompute
    quantile.threshold_tables() directly with no correction at all, dividing severity by the
    raw per-state quantile even though flag() scores s_noun against the pi_all MIXTURE, not
    the pure per-state distribution. narrate()'s severity ratio must come from the SAME
    per-tick dilution-corrected threshold flag() used internally (surprise.emission_thresholds:
    raw quantile - log(pi_at_zstar)), not the raw per-state table."""
    p = _peaked_params()
    recipe_params = _recipe_params()
    vocab = _toy_vocab()

    noun_query = trace = log_probs = recipe_log_trans = None
    for seed in range(10):
        traj = _long_trajectory(seed=200 + seed)
        rng = np.random.default_rng(seed)
        sub = error_injection.inject("substitution", traj, rng, p)
        trace, log_probs, recipe_log_trans = surprise.compute_trace(
            p, recipe_params, sub["verb_ids"], sub["noun_ids"], D_MAX
        )
        flags = surprise.flag(trace, log_probs, recipe_log_trans)
        pi_all = surprise.compute_pi_all(log_probs, sub["verb_ids"], sub["noun_ids"], D_MAX)
        queries = narrate.narrate(
            trace, flags, vocab, p, sub["verb_ids"], sub["noun_ids"], log_probs, recipe_log_trans, pi_all
        )
        noun_queries = [q for q in queries if q.channel == "s_noun"]
        if len(noun_queries) == 1:
            noun_query = noun_queries[0]
            break

    assert noun_query is not None, "substitution never produced a clean single s_noun query"

    raw_tables = quantile.threshold_tables(log_probs, recipe_log_trans, surprise.DEFAULT_ALPHA)
    _, _, noun_thresh = surprise.emission_thresholds(trace, raw_tables)
    corrected_threshold = float(noun_thresh[noun_query.tick])

    state = trace.z_star[noun_query.tick]
    raw_threshold = float(raw_tables.noun[state])
    pi_at_tick = float(trace.pi_at_zstar[noun_query.tick])
    assert pi_at_tick < 1.0, "fixture's belief happened to be exactly one-hot at this tick"

    s_noun_value = float(trace.s_noun[noun_query.tick])
    expected_ratio = s_noun_value / corrected_threshold
    uncorrected_ratio = s_noun_value / raw_threshold  # what narrate used to compute, pre-fix

    assert noun_query.ratio == pytest.approx(expected_ratio)
    assert corrected_threshold == pytest.approx(raw_threshold - np.log(pi_at_tick))
    assert noun_query.ratio != pytest.approx(uncorrected_ratio)


_TOY_LOG_EMIT_3STATE = np.log(np.array([
    [0.98, 0.01, 0.01],  # state 0: 'idle'-like
    [0.05, 0.90, 0.05],  # state 1
    [0.05, 0.05, 0.90],  # state 2
]))
_TOY_PI_ALL_3STATE = np.log(np.array([[0.85, 0.10, 0.05]]))  # (T=1, K=3), mostly state 0


_FAKE_VERB_THRESH = np.array([1.0])  # (T=1,) -- per-tick, post-emission_thresholds
_FAKE_NOUN_THRESH = np.array([1.0])


def test_emission_query_uses_conditional_expected_when_only_one_channel_flagged():
    """Regression for the 'pour kitchen' bug: when only the noun channel is flagged (verb is
    presumably fine), the expected noun must be picked CONDITIONED on the held-constant
    observed verb (surprise.conditional_expected), not by marginalizing over all states
    regardless of whether they're even compatible with that verb."""
    p = _peaked_params()
    lexicon = narrate.Lexicon(_toy_vocab(), p)
    log_emit_v = log_emit_n = _TOY_LOG_EMIT_3STATE
    pi_all = _TOY_PI_ALL_3STATE

    observed_verb, observed_noun = 1, 2
    trace = types.SimpleNamespace(
        s_noun=np.array([10.0]),
        s_verb=np.array([0.0]),
        attribution=np.array(["item"], dtype=object),
    )
    flags = {"s_noun": np.array([True]), "s_verb": np.array([False])}
    segments = [(0, 0, 1)]
    verb_ids = np.array([observed_verb])
    noun_ids = np.array([observed_noun])

    queries = narrate._emission_queries(
        trace, flags, segments, verb_ids, noun_ids, lexicon, _FAKE_VERB_THRESH, _FAKE_NOUN_THRESH,
        pi_all, log_emit_v, log_emit_n,
    )
    assert len(queries) == 1
    q = queries[0]

    expected_noun = surprise.conditional_expected(pi_all[0], log_emit_v[:, observed_verb], log_emit_n)
    assert expected_noun == 1  # sanity: conditioning shifts away from the naive (idle) pick

    observed_phrase = lexicon.phrase(observed_verb, observed_noun)
    expected_phrase = lexicon.phrase(observed_verb, expected_noun)  # verb held constant
    assert q.text == (
        f"Wait --, that's {observed_phrase} -- based on what I'd seen up to then, "
        f"I expected {expected_phrase}."
    )


def test_emission_query_uses_joint_expected_when_both_channels_flagged():
    """When BOTH channels are independently flagged at the same tick, neither observed token
    is a trustworthy anchor for the other -- _emission_queries must fall back to
    surprise.joint_expected (letting both verb and noun vary), not conditional_expected's
    held-constant-verb assumption."""
    p = _peaked_params()
    lexicon = narrate.Lexicon(_toy_vocab(), p)
    log_emit_v = log_emit_n = _TOY_LOG_EMIT_3STATE
    pi_all = _TOY_PI_ALL_3STATE

    observed_verb, observed_noun = 1, 2
    trace = types.SimpleNamespace(
        s_noun=np.array([10.0]),
        s_verb=np.array([8.0]),
        attribution=np.array(["item"], dtype=object),  # noun still dominates by margin
    )
    flags = {"s_noun": np.array([True]), "s_verb": np.array([True])}  # both independently flagged
    segments = [(0, 0, 1)]
    verb_ids = np.array([observed_verb])
    noun_ids = np.array([observed_noun])

    queries = narrate._emission_queries(
        trace, flags, segments, verb_ids, noun_ids, lexicon, _FAKE_VERB_THRESH, _FAKE_NOUN_THRESH,
        pi_all, log_emit_v, log_emit_n,
    )
    assert len(queries) == 1
    q = queries[0]

    expected_verb, expected_noun = surprise.joint_expected(pi_all[0], log_emit_v, log_emit_n)
    assert expected_verb != observed_verb  # proves the joint path ran, not conditional's held verb

    observed_phrase = lexicon.phrase(observed_verb, observed_noun)
    expected_phrase = lexicon.phrase(expected_verb, expected_noun)
    assert q.text == (
        f"Wait --, that's {observed_phrase} -- based on what I'd seen up to then, "
        f"I expected {expected_phrase}."
    )
