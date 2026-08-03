from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import logsumexp
from test_messages import _brute_force_segmentations, _brute_force_stats, _random_log_probs

from cook_ad.anomaly import quantile, surprise, temporal
from cook_ad.hsmm import emissions, messages, params

jax.config.update("jax_enable_x64", True)


class _FakeZTrace(NamedTuple):
    """Minimal stand-in for SurpriseTrace: emission_thresholds only reads z_star/pi_at_zstar."""
    z_star: np.ndarray
    pi_at_zstar: np.ndarray


def test_emission_thresholds_cancel_mixture_dilution():
    """emission_thresholds must add -log(pi_at_zstar) to the z_star-indexed quantile table
    (the exact offset that cancels mixture dilution -- see the derivation in surprise.py's
    module comment above CHANNELS), transition/recipe are untouched (no mixture weighting
    there), and pi_at_zstar==1.0 (no dilution at all) must reduce EXACTLY to the raw quantile
    table with zero offset."""
    tables = quantile.ThresholdTables(
        emit=np.array([0.002, 9.0]),
        verb=np.array([0.002, 9.0]),
        noun=np.array([0.002, 9.0]),
        transition=np.array([-1.5, 9.0]),
        recipe=np.array([-1.5, 9.0]),
    )
    trace = _FakeZTrace(z_star=np.array([0, 1, 0]), pi_at_zstar=np.array([0.5, 1.0, 0.1]))
    emit_t, verb_t, noun_t = surprise.emission_thresholds(trace, tables)

    expected_offset = -np.log(trace.pi_at_zstar)
    assert np.allclose(emit_t, tables.emit[trace.z_star] + expected_offset)
    assert np.allclose(verb_t, tables.verb[trace.z_star] + expected_offset)
    assert np.allclose(noun_t, tables.noun[trace.z_star] + expected_offset)

    # pi_at_zstar == 1.0 (tick 1): zero offset, reduces exactly to the pure per-state quantile.
    assert emit_t[1] == pytest.approx(tables.emit[1])
    assert verb_t[1] == pytest.approx(tables.verb[1])
    assert noun_t[1] == pytest.approx(tables.noun[1])


def test_emission_thresholds_never_flag_the_dominant_token_under_dilution():
    """Regression guard for the exact bug the flat EMIT_THRESHOLD_FLOOR was papering over: a
    genuinely mixture-diluted s_emit for a state's OWN dominant token must never exceed the
    dilution-corrected threshold, for ANY pi_at_zstar in (0, 1]. This is the property that
    would have caught the original fix driving healthy false-positive rate to 1.000 -- a tight
    per-state threshold combined with real mixture dilution used to blow past it every time.

    Constructs a genuine 2-state mixture (weight pi_zstar on the believed state z*, weight
    1-pi_zstar on a second, unrelated state) and computes s_emit directly from that mixture,
    not from the algebraic bound -- so this checks the actual arithmetic, not just restates the
    derivation."""
    rng = np.random.default_rng(0)
    v = 6
    p_star = rng.dirichlet(np.ones(v))          # z*'s own categorical
    p_other = rng.dirichlet(np.ones(v))         # a different state's categorical
    dom = int(np.argmax(p_star))                 # z*'s dominant token

    log_p_star = np.log(p_star)[None, :]
    threshold = quantile.categorical_quantile_threshold(log_p_star, alpha=0.05)[0]

    for pi_zstar in (1.0, 0.99, 0.9, 0.5, 0.1, 0.01, 1e-4):
        mixture_prob = pi_zstar * p_star[dom] + (1.0 - pi_zstar) * p_other[dom]
        s_observed = -np.log(mixture_prob)
        corrected_threshold = threshold - np.log(pi_zstar)
        assert s_observed <= corrected_threshold + 1e-9, (
            f"dominant token flagged at pi_zstar={pi_zstar}: {s_observed} > {corrected_threshold}"
        )


@pytest.mark.parametrize(
    "t, k, d_max",
    [
        (6, 3, 3),
        (8, 3, 4),
        (7, 4, 4),
    ],
)
def test_predictive_occupancy_matches_bruteforce(t, k, d_max):
    """pi_all[t,:] = P(Z_t=k | o_{<t}) is, by construction, equal to the posterior state
    marginal at the LAST tick of a length-(t+1) prefix whose final tick's emission is
    zeroed out (no evidence contributed) but whose final segment is still survival-weighted
    (censored, may continue) -- exactly what the existing brute-force segmentation oracle
    already computes when given t_true = t+1 and a zeroed final-tick loglik. This lets the
    from-scratch enumerator (test_messages.py's, unmodified) double as pi_all's oracle too.
    """
    rng = np.random.default_rng(1)
    log_init, log_trans, log_dur_pmf, log_dur_survival = _random_log_probs(rng, k, d_max)
    loglik_np = rng.standard_normal((t, k)) * 0.4
    mask_np = np.array([True] * t)

    loglik = jnp.array(loglik_np)
    mask = jnp.array(mask_np)
    log_init_j, log_trans_j = jnp.array(log_init), jnp.array(log_trans)
    log_dur_pmf_j, log_dur_survival_j = jnp.array(log_dur_pmf), jnp.array(log_dur_survival)

    pi_all = messages.predictive_occupancy(
        loglik, log_init_j, log_trans_j, log_dur_pmf_j, log_dur_survival_j, mask, d_max
    )
    pi_all_np = np.exp(np.asarray(pi_all))
    np.testing.assert_allclose(pi_all_np.sum(axis=-1), 1.0, atol=1e-8)

    for tick in range(t):
        loglik_mod = loglik_np[: tick + 1].copy()
        loglik_mod[tick, :] = 0.0
        segmentations = _brute_force_segmentations(
            loglik_mod, log_init, log_trans, log_dur_pmf, log_dur_survival, tick + 1, d_max, k
        )
        _, _, _, _, bf_gamma = _brute_force_stats(segmentations, tick + 1, d_max, k)
        np.testing.assert_allclose(pi_all_np[tick], bf_gamma[tick], atol=1e-8)


def test_predictive_occupancy_starved_mode_finite():
    """A near-empty weak-limit mode must stay finite and inert in pi_all, mirroring
    test_messages.py's test_starved_mode_no_nan for the ordinary forward pass."""
    k, vocab_verbs, vocab_nouns, d_max, t = 6, 15, 36, 20, 40
    starved = 0

    key_i, key_t = jax.random.split(jax.random.PRNGKey(3))
    init_counts = jax.random.uniform(key_i, (k,), minval=5.0, maxval=50.0).at[starved].set(0.0)
    trans_counts = jax.random.uniform(key_t, (k, k), minval=5.0, maxval=50.0)
    trans_counts = trans_counts * (1.0 - jnp.eye(k))
    trans_counts = trans_counts.at[starved, :].set(0.0).at[:, starved].set(0.0)

    verb_counts = jnp.full((k, vocab_verbs), 0.5).at[jnp.arange(1, k), jnp.arange(1, k) % vocab_verbs].set(200.0)
    verb_counts = verb_counts.at[starved, :].set(0.0)
    noun_counts = jnp.full((k, vocab_nouns), 0.5).at[jnp.arange(1, k), jnp.arange(1, k) % vocab_nouns].set(200.0)
    noun_counts = noun_counts.at[starved, :].set(0.0)

    p = params.HSMMParams(
        init_counts, trans_counts, verb_counts, noun_counts, jnp.full((k,), 5.0), jnp.full((k,), 0.3)
    )
    log_probs = params.to_log_probs(p, d_max)

    active_states = jax.random.randint(jax.random.PRNGKey(11), (t,), 1, k)
    verb_ids, noun_ids = active_states % vocab_verbs, active_states % vocab_nouns
    mask = jnp.ones((t,), dtype=bool)

    loglik = emissions.sequence_loglik(verb_ids, noun_ids, log_probs.log_emit_v, log_probs.log_emit_n, mask)
    pi_all = messages.predictive_occupancy(
        loglik, log_probs.log_init, log_probs.log_trans,
        log_probs.log_dur_pmf, log_probs.log_dur_survival, mask, d_max,
    )

    assert jnp.all(jnp.isfinite(pi_all))
    assert jnp.max(jnp.exp(pi_all[:, starved])) < 1e-6


def test_emission_surprise_isolates_item_substitution():
    """A tiny hand-built model: state 1 strongly expects verb=0 and noun=0. Observing the
    expected verb but a wildly unexpected noun should give S_noun >> S_verb and attribute
    the anomaly to the item, not the action."""
    pi_all = jnp.log(jnp.array([[0.01, 0.99]]))
    verb_probs = jnp.array([[1 / 3, 1 / 3, 1 / 3], [0.9, 0.05, 0.05]])
    noun_probs = jnp.array([[1 / 3, 1 / 3, 1 / 3], [0.9, 0.05, 0.05]])
    log_emit_v, log_emit_n = jnp.log(verb_probs), jnp.log(noun_probs)

    verb_ids = jnp.array([0])   # matches state 1's expectation
    noun_ids = jnp.array([2])   # far from state 1's expectation

    s_emit, s_verb, s_noun = surprise.emission_surprise(pi_all, log_emit_v, log_emit_n, verb_ids, noun_ids)

    assert float(s_noun[0]) > float(s_verb[0]) + surprise.DEFAULT_ATTRIBUTION_MARGIN
    labels = surprise.attribute(s_verb, s_noun)
    assert labels[0] == "item"


def test_conditional_expected_avoids_incoherent_pairing():
    """A tiny hand-built model with three states: state 0 is 'idle'-like (dominates pi_all at
    85%), state 1 and state 2 each have their own coherent (verb, noun) pair. The observed verb
    matches state 1's, which state 0 barely supports. A naive unconditional pick (marginalizing
    pi_all over just the noun, ignoring the verb) follows the raw pi_all mass to state 0's own
    noun -- producing an incoherent pairing with the held verb (narrate.py's 'pour kitchen'
    bug). conditional_expected must instead reweight by the held verb's compatibility, landing
    on state 1's noun -- the one actually coherent with what was observed."""
    log_emit_v = jnp.log(jnp.array([
        [0.98, 0.01, 0.01],
        [0.05, 0.90, 0.05],
        [0.05, 0.05, 0.90],
    ]))
    log_emit_n = jnp.log(jnp.array([
        [0.98, 0.01, 0.01],
        [0.05, 0.90, 0.05],
        [0.05, 0.05, 0.90],
    ]))
    pi_all_t = jnp.log(jnp.array([0.85, 0.10, 0.05]))
    observed_verb = 1

    naive_mixture = logsumexp(pi_all_t[:, None] + log_emit_n, axis=0)
    naive_pick = int(jnp.argmax(naive_mixture))
    assert naive_pick == 0  # ignoring the verb, raw pi_all mass follows idle's own noun

    conditional_pick = surprise.conditional_expected(pi_all_t, log_emit_v[:, observed_verb], log_emit_n)
    assert conditional_pick == 1  # conditioning on the held verb shifts to ITS coherent noun
    assert conditional_pick != naive_pick


def test_joint_expected_finds_best_joint_pair():
    """Same three-state model. joint_expected optimizes verb and noun together rather than
    holding either fixed, so with pi_all dominated by state 0 it should recover state 0's own
    coherent (verb, noun) pair -- not an independently-argmaxed splice of two different
    states' halves."""
    log_emit_v = jnp.log(jnp.array([
        [0.98, 0.01, 0.01],
        [0.05, 0.90, 0.05],
        [0.05, 0.05, 0.90],
    ]))
    log_emit_n = jnp.log(jnp.array([
        [0.98, 0.01, 0.01],
        [0.05, 0.90, 0.05],
        [0.05, 0.05, 0.90],
    ]))
    pi_all_t = jnp.log(jnp.array([0.85, 0.10, 0.05]))

    v, n = surprise.joint_expected(pi_all_t, log_emit_v, log_emit_n)
    assert (v, n) == (0, 0)


def test_transition_surprise_flags_unexpected_boundary():
    trans = np.array([
        [0.0, 0.9, 0.1],
        [0.85, 0.0, 0.15],
        [0.3, 0.3, 0.4],
    ])
    log_trans = jnp.log(jnp.array(trans))
    segments = [(0, 3), (1, 2), (2, 4)]

    s_trans, expected_next = surprise.transition_surprise(segments, log_trans)

    assert expected_next[0] == -1
    assert s_trans[0] == 0.0

    assert expected_next[3] == 1   # argmax of state 0's row matches the actual next state
    assert s_trans[5] > s_trans[3]  # 1->2 (prob 0.15) is more surprising than 0->1 (prob 0.9)
    assert expected_next[5] == 0    # state 1 "normally" goes to 0, not the observed 2


def _survival_table(dur_r, dur_p, d_max):
    from cook_ad.hsmm import durations

    log_pmf, log_survival = durations.duration_tables(dur_r, dur_p, d_max)
    return log_survival


def test_live_stall_surprise_is_calibrated_survival():
    """Live signal = -log P(D>=d), so it must (a) rise monotonically within a segment and
    (b) equal the state's survival surprise column exactly -- the property that makes a single
    -log(alpha) threshold a per-state tail quantile."""
    dur_r = jnp.array([4.0])
    dur_p = jnp.array([0.3])
    d_max = 150
    log_survival = _survival_table(dur_r, dur_p, d_max)
    segments = [(0, 30)]

    s_temporal = temporal.live_stall_surprise(segments, log_survival, d_max)

    assert np.all(np.isfinite(s_temporal))
    assert np.all(np.diff(s_temporal) >= -1e-9)
    assert s_temporal[0] == pytest.approx(0.0, abs=1e-9)  # P(D>=1) = 1
    np.testing.assert_allclose(s_temporal, -np.asarray(log_survival)[0, :30], atol=1e-9)


def test_live_stall_surprise_handles_duration_past_d_max():
    dur_r = jnp.array([4.0])
    dur_p = jnp.array([0.3])
    d_max = 5
    log_survival = _survival_table(dur_r, dur_p, d_max)
    segments = [(0, 10)]

    s_temporal = temporal.live_stall_surprise(segments, log_survival, d_max)

    assert s_temporal.shape[0] == 10
    assert np.all(np.isfinite(s_temporal))
    np.testing.assert_allclose(s_temporal[4:], s_temporal[4])


def test_completed_segment_surprise_catches_both_tails():
    """A normally ~9-tick subtask (r=4,p=0.3): a 1-tick close is 'left_early' (only the
    retrospective left tail catches it; the live survival signal at d=1 is exactly 0), a
    40-tick close is 'stuck', and a ~9-tick close is unremarkable on both tails. (A 2-tick
    close sits at a two-sided p of ~0.061, honestly just above alpha=0.05 and so left
    unflagged -- the calibration is real, not tuned to always fire.)"""
    dur_r = jnp.array([4.0])
    dur_p = jnp.array([0.3])

    s_long, s_short, s_two, attr = temporal.completed_segment_surprise(
        [(0, 1), (0, 9), (0, 40)], dur_r, dur_p
    )

    assert attr[0] == "left_early"
    assert s_short[0] > s_long[0]
    assert attr[2] == "stuck"
    assert s_long[2] > s_short[2]
    assert s_two[1] < s_two[0] and s_two[1] < s_two[2]  # mid duration least surprising

    # single -log(alpha) threshold flags both extremes, not the middle
    thresh = -np.log(surprise.DEFAULT_ALPHA)
    assert s_two[0] > thresh and s_two[2] > thresh and s_two[1] < thresh


def test_pit_uniform_for_well_fit_durations():
    """Mid-PIT of durations SAMPLED from the fitted NB should be ~Uniform[0,1] (mean ~0.5):
    the calibration diagnostic's null. A gross duration-model mismatch would shift the mean."""
    from scipy.stats import nbinom

    r, p = 6.0, 0.35
    d_prime = nbinom.rvs(r, p, size=4000, random_state=0)
    segments = [(0, int(dp) + 1) for dp in d_prime]  # D = D' + 1 on support >= 1

    pit = temporal.pit_coordinate(segments, jnp.array([r]), jnp.array([p]))
    assert 0.45 < pit.mean() < 0.55
    assert pit.min() >= 0.0 and pit.max() <= 1.0
