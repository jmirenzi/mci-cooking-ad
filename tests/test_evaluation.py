import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cook_ad.eval import metrics
from cook_ad.hsmm import params
from cook_ad.synthetic import error_injection, generate

jax.config.update("jax_enable_x64", True)

K, V, N, D_MAX = 5, 6, 8, 30


def _peaked_params():
    """A model where each state has a sharp verb/noun and a moderate duration -- so sampled
    trajectories are well-structured and injections are genuinely anomalous."""
    p = params.init_weak_limit_params(jax.random.PRNGKey(0), K, V, N, D_MAX)
    verb = jnp.full((K, V), 0.2).at[jnp.arange(K), jnp.arange(K) % V].set(200.0)
    noun = jnp.full((K, N), 0.2).at[jnp.arange(K), jnp.arange(K) % N].set(200.0)
    return p._replace(
        verb_counts=verb, noun_counts=noun,
        dur_r=jnp.full((K,), 8.0), dur_p=jnp.full((K,), 0.5),
    )


def test_generator_is_well_formed():
    p = _peaked_params()
    rng = np.random.default_rng(1)
    traj = generate.sample_trajectory(p, rng, max_ticks=120, d_max=D_MAX)

    assert traj["verb_ids"].shape[0] == 120
    assert traj["noun_ids"].shape[0] == 120
    assert sum(d for _, d in traj["segments"]) == 120
    assert traj["subtask_per_tick"].shape[0] == 120
    assert traj["verb_ids"].min() >= 0 and traj["verb_ids"].max() < V
    assert traj["noun_ids"].min() >= 0 and traj["noun_ids"].max() < N
    # no banned self-transitions between consecutive segments
    states = [s for s, _ in traj["segments"]]
    assert all(a != b for a, b in zip(states, states[1:]))


def test_generator_emissions_track_peaked_params():
    p = _peaked_params()
    rng = np.random.default_rng(2)
    traj = generate.sample_trajectory(p, rng, max_ticks=4000, d_max=D_MAX)
    # each state's most common emitted noun should be its peaked noun (state % N)
    per_tick_state = traj["subtask_per_tick"]
    for s in np.unique(per_tick_state):
        nouns_here = traj["noun_ids"][per_tick_state == s]
        if nouns_here.size >= 20:
            assert np.bincount(nouns_here, minlength=N).argmax() == s % N


@pytest.mark.parametrize("error_type", error_injection.ERROR_TYPES)
def test_injection_structurally_correct(error_type):
    p = _peaked_params()
    rng = np.random.default_rng(3)
    traj = generate.sample_trajectory(p, rng, max_ticks=150, d_max=D_MAX)
    orig_len = traj["verb_ids"].shape[0]

    deg = error_injection.inject(error_type, traj, rng, p)
    new_len = deg["verb_ids"].shape[0]
    t0, t1 = deg["window"]

    assert 0 <= t0 <= t1 < new_len  # window lies inside the degraded sequence
    assert deg["error_type"] == error_type

    if error_type in ("abandonment", "omission"):
        assert new_len < orig_len
    elif error_type == "repetition":
        assert new_len > orig_len
    elif error_type == "transposition":
        assert new_len == orig_len
        assert not np.array_equal(deg["noun_ids"], traj["noun_ids"])
    elif error_type == "substitution":
        assert new_len == orig_len
        assert int(np.sum(deg["noun_ids"] != traj["noun_ids"])) == 1  # exactly one tick changed


def test_injection_raises_when_too_few_segments():
    tiny = {"verb_ids": np.zeros(3, np.int64), "noun_ids": np.zeros(3, np.int64),
            "segments": [(0, 3)], "subtask_per_tick": np.zeros(3, np.int64)}
    with pytest.raises(ValueError):
        error_injection.inject("omission", tiny, np.random.default_rng(0), _peaked_params())


def _flags(length, true_ticks, channel="s_noun"):
    d = {ch: np.zeros(length, dtype=bool) for ch in metrics.ALL_CHANNELS}
    for t in true_ticks:
        d[channel][t] = True
    return d


def test_metrics_recall_precision_latency_arithmetic():
    # one degraded trial flagged 2 ticks after onset (window=(10,10), tol default 5) -> detected, latency 2
    degraded = {"substitution": [(_flags(40, [12]), (10, 10))]}
    healthy = [_flags(40, [])]  # a clean healthy control (no flags)
    report = metrics.evaluate(healthy, degraded)
    m = report["per_type"]["substitution"]
    assert m["recall"] == 1.0
    assert m["precision"] == 1.0
    assert m["mean_latency"] == 2.0
    assert report["attribution"]["substitution"]["s_noun"] == 1.0


def test_metrics_healthy_flag_is_false_positive():
    degraded = {"substitution": [(_flags(40, [10]), (10, 10))]}
    healthy = [_flags(40, [3])]  # a healthy control that spuriously flags
    report = metrics.evaluate(healthy, degraded)
    m = report["per_type"]["substitution"]
    assert m["recall"] == 1.0
    assert m["precision"] == 0.5  # 1 TP, 1 FP (healthy)
    assert report["healthy"]["false_positive_rate"] == 1.0


def test_metrics_out_of_window_flag_is_miss_and_fp():
    degraded = {"omission": [(_flags(40, [30]), (10, 10))]}  # flag far outside [10,15]
    healthy = [_flags(40, [])]
    report = metrics.evaluate(healthy, degraded)
    m = report["per_type"]["omission"]
    assert m["recall"] == 0.0
    assert m["precision"] == 0.0  # 0 TP, 1 out-of-window FP


def test_kl_sanity_nonzero_when_distributions_differ():
    healthy = [{"noun_ids": np.zeros(50, np.int64)}]
    degraded = [{"noun_ids": np.full(50, 3, np.int64)}]
    assert metrics.kl_sanity(healthy, degraded, n_nouns=N) > 0.0
    assert metrics.kl_sanity(healthy, healthy, n_nouns=N) == pytest.approx(0.0, abs=1e-9)


def test_end_to_end_channel_isolation_on_generated_trials():
    """The detector's isolation claim on generated+injected trials: a substitution fires the
    noun channel in-window; an abandonment fires the retrospective short-duration channel."""
    from cook_ad.anomaly import surprise
    from cook_ad.recipe import recipe_hmm

    p = _peaked_params()
    # trivial single-recipe HMM over the K subtask symbols (enough for compute_trace's recipe path)
    recipe_params = recipe_hmm.init_weak_limit_recipe_params(jax.random.PRNGKey(5), k_recipe=3, k_subtask=K)
    rng = np.random.default_rng(7)

    sub_hits, aband_hits, left_early_hits = 0, 0, 0
    trials = 0
    for seed in range(6):
        traj = generate.sample_trajectory(p, np.random.default_rng(seed), max_ticks=150, d_max=D_MAX)
        if len(traj["segments"]) < error_injection.MIN_SEGMENTS:
            continue
        trials += 1

        sub = error_injection.inject("substitution", traj, rng, p)
        f = surprise.flag(surprise.compute_trace(p, recipe_params, sub["verb_ids"], sub["noun_ids"], D_MAX))
        _, _, _, hits = metrics.score_trial(f, sub["window"])
        sub_hits += "s_noun" in hits

        ab = error_injection.inject("abandonment", traj, rng, p)
        trace = surprise.compute_trace(p, recipe_params, ab["verb_ids"], ab["noun_ids"], D_MAX)
        _, _, _, hits = metrics.score_trial(surprise.flag(trace), ab["window"], latency_tol=8)
        aband_hits += "s_dur_two" in hits  # the calibrated duration channel actually flagged
        t0, t1 = ab["window"]
        left_early_hits += "left_early" in set(trace.temporal_attribution[t0 : t1 + 9])

    assert trials >= 3
    # substitution predominantly fires the noun channel (not 100% -- the predictive occupancy
    # can hedge across states -- but a clear majority, and when detected it IS the noun channel)
    assert sub_hits >= (trials + 1) // 2
    assert aband_hits >= 1        # abandonment -> the two-sided duration channel flags
    assert left_early_hits >= 1   # ...and its direction is attributed 'left_early'
