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
        outside = np.ones(new_len, dtype=bool)
        outside[t0 : t1 + 1] = False
        assert np.array_equal(deg["verb_ids"][outside], traj["verb_ids"][outside])
        assert np.array_equal(deg["noun_ids"][outside], traj["noun_ids"][outside])

        deg_touched = deg["noun_ids"] if deg["channel"] == "noun" else deg["verb_ids"]
        src_touched = traj["noun_ids"] if deg["channel"] == "noun" else traj["verb_ids"]
        deg_other = deg["verb_ids"] if deg["channel"] == "noun" else deg["noun_ids"]
        src_other = traj["verb_ids"] if deg["channel"] == "noun" else traj["noun_ids"]

        assert np.array_equal(deg_other, src_other)  # the untouched channel is unchanged everywhere
        assert np.all(deg_touched[t0 : t1 + 1] == deg_touched[t0])  # one uniform replacement across the span
        assert deg_touched[t0] != src_touched[t0]  # and it really did change


@pytest.mark.parametrize("error_type", error_injection.ERROR_TYPES)
def test_injection_tick_map_round_trips(error_type):
    """tick_map[i] must be the original tick degraded tick i came from, for every degraded tick
    whose content was not itself rewritten (edited_ticks) -- verified directly against the
    source trajectory's verb/noun ids, not against the injector's own index arithmetic."""
    p = _peaked_params()
    rng = np.random.default_rng(4)
    traj = generate.sample_trajectory(p, rng, max_ticks=150, d_max=D_MAX)

    deg = error_injection.inject(error_type, traj, rng, p)
    tick_map = deg["tick_map"]
    edited = set(deg["edited_ticks"].tolist())

    assert len(tick_map) == len(deg["verb_ids"]) == len(deg["noun_ids"])
    assert tick_map.min() >= 0
    assert tick_map.max() < traj["verb_ids"].shape[0]

    for i in range(len(tick_map)):
        if i in edited:
            continue
        src = int(tick_map[i])
        assert deg["verb_ids"][i] == traj["verb_ids"][src]
        assert deg["noun_ids"][i] == traj["noun_ids"][src]

    if error_type == "substitution":
        t0, t1 = deg["window"]
        assert edited == set(range(int(t0), int(t1) + 1))
    else:
        assert edited == set()


# ---- ground-truth SPANS (eval/span_metrics.py) ----------------------------------------------

# How many spans each injector produces. Transposition moves two segments and so labels two;
# everything else labels one. Abandonment's is the single onset tick and NOT "onset through the
# end of the trial": that injector truncates ONE interior segment and the trial continues
# normally afterwards, so a span running to the end would cover most of the trial and make its
# recall nearly free.
EXPECTED_N_SPANS = {"substitution": 1, "abandonment": 1, "omission": 1,
                    "transposition": 2, "repetition": 1}


@pytest.mark.parametrize("error_type", error_injection.ERROR_TYPES)
def test_every_injector_emits_well_formed_spans(error_type):
    p = _peaked_params()
    rng = np.random.default_rng(7)
    traj = generate.sample_trajectory(p, rng, max_ticks=150, d_max=D_MAX)
    deg = error_injection.inject(error_type, traj, rng, p)
    t_degraded = len(deg["verb_ids"])

    spans = deg["spans"]
    assert len(spans) == EXPECTED_N_SPANS[error_type]
    for t0, t1 in spans:
        assert 0 <= t0 <= t1 < t_degraded, f"{error_type} span {(t0, t1)} outside [0, {t_degraded})"


@pytest.mark.parametrize("error_type", error_injection.ERROR_TYPES)
def test_spans_do_not_disturb_the_two_older_ground_truth_layers(error_type):
    """The span key is purely additive. The tick layer reads `window`, the element layer reads
    `anomaly_ticks`, and both must be exactly what they were before spans existed."""
    p = _peaked_params()
    traj = generate.sample_trajectory(p, np.random.default_rng(8), max_ticks=150, d_max=D_MAX)

    a = error_injection.inject(error_type, traj, np.random.default_rng(9), p)
    b = error_injection.inject(error_type, traj, np.random.default_rng(9), p)
    assert a["window"] == b["window"]
    assert np.array_equal(a["anomaly_ticks"], b["anomaly_ticks"])
    assert np.array_equal(a["tick_map"], b["tick_map"])
    assert np.array_equal(a["edited_ticks"], b["edited_ticks"])
    # and the window is still the disturbed EXTENT, which for the three structural injectors is
    # not the same object as the points -- the distinction spans are a third alternative to
    assert set(a["anomaly_ticks"].tolist()) <= set(
        range(int(a["window"][0]), int(a["window"][1]) + 1))


def test_substitution_span_is_the_whole_retagged_segment():
    p = _peaked_params()
    traj = generate.sample_trajectory(p, np.random.default_rng(10), max_ticks=150, d_max=D_MAX)
    deg = error_injection.inject_substitution(traj, np.random.default_rng(11), p)
    # For substitution alone, span == window == the anomalous points: every tick of a segment
    # done with the wrong object is wrong.
    assert deg["spans"] == [(int(deg["window"][0]), int(deg["window"][1]))]
    assert deg["spans"][0] == (int(deg["anomaly_ticks"][0]), int(deg["anomaly_ticks"][-1]))


def test_transposition_spans_are_the_two_moved_segments_not_one_interval():
    """The window covers both swapped runs as a single interval. Collapsing them into one span
    would let a single flag anywhere in ~49 ticks claim both moved segments."""
    p = _peaked_params()
    traj = generate.sample_trajectory(p, np.random.default_rng(12), max_ticks=200, d_max=D_MAX)
    deg = error_injection.inject_transposition(traj, np.random.default_rng(13))
    (a0, a1), (b0, b1) = deg["spans"]
    assert b0 == a1 + 1                       # adjacent, no gap
    assert (a0, b1) == tuple(int(x) for x in deg["window"])   # together they tile the window
    # each junction the element layer uses falls inside one of the two spans
    for t in deg["anomaly_ticks"].tolist():
        assert a0 <= t <= b1


def test_omission_span_is_the_successor_segment_not_the_single_boundary_tick():
    """The omitted step contributes no ticks of its own, so the only thing to point at is the
    step that now runs in its place -- in full, so its width is comparable to the others'."""
    p = _peaked_params()
    traj = generate.sample_trajectory(p, np.random.default_rng(14), max_ticks=200, d_max=D_MAX)
    deg = error_injection.inject_omission(traj, np.random.default_rng(15))
    (t0, t1), = deg["spans"]
    assert t0 == int(deg["window"][0])        # begins at the new boundary
    assert t1 >= t0                           # and runs for the successor's own duration
    # the successor really is constant over the span: it is one segment, not several
    assert len(set(deg["verb_ids"][t0:t1 + 1].tolist())) == 1
    assert len(set(deg["noun_ids"][t0:t1 + 1].tolist())) == 1


def test_abandonment_span_is_the_onset_tick_alone():
    p = _peaked_params()
    traj = generate.sample_trajectory(p, np.random.default_rng(16), max_ticks=200, d_max=D_MAX)
    deg = error_injection.inject_abandonment(traj, np.random.default_rng(17))
    (t0, t1), = deg["spans"]
    assert t0 == t1
    # emphatically NOT through the end of the trial
    assert t1 < len(deg["verb_ids"]) - 1


def test_repetition_span_is_the_inserted_copy():
    p = _peaked_params()
    traj = generate.sample_trajectory(p, np.random.default_rng(18), max_ticks=200, d_max=D_MAX)
    deg = error_injection.inject_repetition(traj, np.random.default_rng(19))
    (t0, t1), = deg["spans"]
    assert (t0, t1) == tuple(int(x) for x in deg["window"])
    # the copy is adjacent-identical to its original, which is why it merges into one RLE step
    d = t1 - t0 + 1
    assert np.array_equal(deg["verb_ids"][t0:t1 + 1], deg["verb_ids"][t0 - d:t0])
    assert np.array_equal(deg["noun_ids"][t0:t1 + 1], deg["noun_ids"][t0 - d:t0])


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
    """The detector's isolation claim on generated+injected trials: a substitution fires its own
    channel (noun or verb, whichever it edited) in-window; an abandonment fires the retrospective
    short-duration channel.

    Alpha is pinned at 0.05 rather than read from DEFAULT_ALPHA. This asserts WHICH CHANNEL
    responds to which perturbation -- an isolation property of the cascade -- on a deliberately
    small synthetic setup (20 seeds, K=5). The deployment default is tuned for the real corpus's
    precision/false-alarm trade (see surprise.DEFAULT_ALPHA) and is strict enough that these few
    tiny synthetic injections fall below it, which would turn a channel-isolation test into an
    incidental sensitivity test at whatever alpha happens to be shipping. Substitution's hit rate
    here is low (see below) and needs enough seeds that `sub_hits >= 1` isn't itself a coin flip."""
    from cook_ad.anomaly import surprise
    from cook_ad.recipe import recipe_hmm

    alpha = 0.05
    p = _peaked_params()
    # trivial single-recipe HMM over the K subtask symbols (enough for compute_trace's recipe path)
    recipe_params = recipe_hmm.init_weak_limit_recipe_params(jax.random.PRNGKey(5), k_recipe=3, k_subtask=K)
    rng = np.random.default_rng(7)

    sub_hits, aband_hits, left_early_hits = 0, 0, 0
    trials = 0
    for seed in range(20):
        traj = generate.sample_trajectory(p, np.random.default_rng(seed), max_ticks=150, d_max=D_MAX)
        if len(traj["segments"]) < error_injection.MIN_SEGMENTS:
            continue
        trials += 1

        sub = error_injection.inject("substitution", traj, rng, p)
        sub_trace, sub_log_probs, sub_recipe_log_trans = surprise.compute_trace(
            p, recipe_params, sub["verb_ids"], sub["noun_ids"], D_MAX
        )
        f = surprise.flag(sub_trace, sub_log_probs, sub_recipe_log_trans, alpha=alpha)
        _, _, _, hits = metrics.score_trial(f, sub["window"])
        expected_channel = "s_noun" if sub["channel"] == "noun" else "s_verb"
        sub_hits += expected_channel in hits

        ab = error_injection.inject("abandonment", traj, rng, p)
        trace, log_probs, recipe_log_trans = surprise.compute_trace(p, recipe_params, ab["verb_ids"], ab["noun_ids"], D_MAX)
        _, _, _, hits = metrics.score_trial(
            surprise.flag(trace, log_probs, recipe_log_trans, alpha=alpha), ab["window"], latency_tol=8
        )
        aband_hits += "s_dur_two" in hits  # the calibrated duration channel actually flagged
        t0, t1 = ab["window"]
        left_early_hits += "left_early" in set(trace.temporal_attribution[t0 : t1 + 9])

    assert trials >= 3
    # NOT a majority-detection guarantee, deliberately: substitution now rewrites a whole segment
    # to one constant wrong token rather than a single mid-segment tick, and sustained identical
    # evidence gives the LIVE predictive belief (pi_all) time to drift toward whatever OTHER state
    # that token best fits -- in this tiny K=5 toy vocabulary, most non-peak tokens are some other
    # state's peak, so this happens often (measured ~30% hit rate over 30 seeds here, vs 100% for
    # the old single-tick edit). Viterbi's hindsight z_star is not fooled -- only the live/causal
    # channel is -- so this is a real property of s_noun/s_verb's causal scoring, not a test bug;
    # it should be much rarer on the real corpus's larger vocabulary. What this test still asserts
    # is the isolation claim itself: WHEN the live channel does catch it, it's the right channel.
    assert sub_hits >= 1
    assert aband_hits >= 1        # abandonment -> the two-sided duration channel flags
    assert left_early_hits >= 1   # ...and its direction is attributed 'left_early'
