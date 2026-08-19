import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cook_ad.eval import counterfactual as cf
from cook_ad.hsmm import params
from cook_ad.synthetic import error_injection, generate

jax.config.update("jax_enable_x64", True)

K, V, N, D_MAX = 5, 6, 8, 30


def _peaked_params():
    p = params.init_weak_limit_params(jax.random.PRNGKey(0), K, V, N, D_MAX)
    verb = jnp.full((K, V), 0.2).at[jnp.arange(K), jnp.arange(K) % V].set(200.0)
    noun = jnp.full((K, N), 0.2).at[jnp.arange(K), jnp.arange(K) % N].set(200.0)
    return p._replace(
        verb_counts=verb, noun_counts=noun,
        dur_r=jnp.full((K,), 8.0), dur_p=jnp.full((K,), 0.5),
    )


# ---- project_flags -----------------------------------------------------------------------------

@pytest.mark.parametrize("error_type", error_injection.ERROR_TYPES)
def test_project_flags_round_trips_through_each_injectors_tick_map(error_type):
    """project_flags must reproduce exactly the healthy flag value at each degraded tick's
    mapped source tick, for every injector's tick_map -- checked directly against fancy indexing,
    not against the injector's own arithmetic."""
    p = _peaked_params()
    rng = np.random.default_rng(11)
    traj = generate.sample_trajectory(p, rng, max_ticks=150, d_max=D_MAX)
    deg = error_injection.inject(error_type, traj, rng, p)
    tick_map = deg["tick_map"]

    T = traj["verb_ids"].shape[0]
    rng2 = np.random.default_rng(1)
    healthy_flags = {"s_noun": rng2.random(T) < 0.3, "s_verb": rng2.random(T) < 0.1}

    projected = cf.project_flags(healthy_flags, tick_map)
    for ch in healthy_flags:
        assert len(projected[ch]) == len(tick_map)
        assert np.array_equal(projected[ch], healthy_flags[ch][tick_map])


# ---- attributable -------------------------------------------------------------------------------

def test_attributable_excludes_flags_present_in_both():
    """A flag present in both the healthy counterfactual and the degraded run at the same
    underlying tick is baseline noise, not injection-caused -- not attributable."""
    healthy = {"s_noun": np.array([True, False, True, False])}
    tick_map = np.array([0, 1, 2, 3])
    projected = cf.project_flags(healthy, tick_map)
    degraded = {"s_noun": np.array([True, False, True, True])}  # only tick 3 is new
    attrib = cf.attributable(degraded, projected)
    assert list(attrib["s_noun"]) == [False, False, False, True]


def test_attributable_flag_only_in_degraded_run_is_attributable():
    healthy = {"s_noun": np.zeros(5, dtype=bool)}
    tick_map = np.arange(5)
    projected = cf.project_flags(healthy, tick_map)
    degraded = {"s_noun": np.array([False, True, False, False, False])}
    attrib = cf.attributable(degraded, projected)
    assert list(attrib["s_noun"]) == [False, True, False, False, False]


def test_attributable_is_indifferent_to_where_the_healthy_flag_came_from():
    """The healthy flag is compared at the MAPPED tick, not the same raw index -- this is what
    makes the comparison meaningful across injectors that reorder/insert/delete ticks."""
    healthy = {"s_noun": np.array([False, True, False])}  # flagged at healthy tick 1
    tick_map = np.array([1, 0, 2])  # degraded tick 0 maps to healthy tick 1
    projected = cf.project_flags(healthy, tick_map)
    degraded = {"s_noun": np.array([True, False, False])}  # same flag, degraded tick 0
    attrib = cf.attributable(degraded, projected)
    assert list(attrib["s_noun"]) == [False, False, False]  # not attributable: baseline noise


# ---- score_counterfactual -----------------------------------------------------------------------

def test_score_counterfactual_detected_and_localized_in_window():
    healthy = {"s_noun": np.zeros(20, dtype=bool)}
    degraded = {"s_noun": np.zeros(20, dtype=bool)}
    degraded["s_noun"][12] = True
    tick_map = np.arange(20)
    detected, localized, latency, _ = cf.score_counterfactual(
        healthy, degraded, tick_map, window=(10, 10), channels=("s_noun",)
    )
    assert (detected, localized, latency) == (True, True, 2)


def test_score_counterfactual_detected_but_not_localized_downstream():
    """A downstream 'blast radius' flag the injection caused, but outside window + tol, is still
    a detection (the injection changed the output) even though it is not a localisation."""
    healthy = {"s_noun": np.zeros(20, dtype=bool)}
    degraded = {"s_noun": np.zeros(20, dtype=bool)}
    degraded["s_noun"][19] = True  # well past window + tol
    tick_map = np.arange(20)
    detected, localized, _, _ = cf.score_counterfactual(
        healthy, degraded, tick_map, window=(0, 0), channels=("s_noun",), latency_tol=2
    )
    assert detected is True
    assert localized is False


def test_score_counterfactual_baseline_noise_is_not_a_detection():
    """A flag the healthy run also had at the corresponding tick is the detector's baseline
    noise on this trial, not the injection's doing, and must not count as detected."""
    healthy = {"s_noun": np.zeros(20, dtype=bool)}
    healthy["s_noun"][5] = True
    degraded = {"s_noun": np.zeros(20, dtype=bool)}
    degraded["s_noun"][5] = True  # same flag, same underlying tick
    tick_map = np.arange(20)
    detected, *_ = cf.score_counterfactual(
        healthy, degraded, tick_map, window=(10, 10), channels=("s_noun",)
    )
    assert detected is False


def test_score_counterfactual_no_flags_at_all():
    healthy = {"s_noun": np.zeros(10, dtype=bool)}
    degraded = {"s_noun": np.zeros(10, dtype=bool)}
    tick_map = np.arange(10)
    detected, localized, latency, mask = cf.score_counterfactual(
        healthy, degraded, tick_map, window=(3, 3), channels=("s_noun",)
    )
    assert (detected, localized, latency) == (False, False, None)
    assert not mask.any()


# ---- evaluate_counterfactual ---------------------------------------------------------------------

def test_evaluate_counterfactual_report_shape_and_rates():
    healthy_flags_by_trial = [{"s_noun": np.zeros(10, dtype=bool)}]
    degraded_flags = {"s_noun": np.zeros(10, dtype=bool)}
    degraded_flags["s_noun"][5] = True
    report = cf.evaluate_counterfactual(
        healthy_flags_by_trial,
        {"substitution": [(degraded_flags, np.arange(10), (5, 5))]},
        channels=("s_noun",),
    )
    assert report["per_type"]["substitution"]["detection_rate"] == 1.0
    assert report["per_type"]["substitution"]["localisation_rate"] == 1.0
    assert report["per_type"]["substitution"]["n"] == 1
    assert report["healthy"]["n"] == 1
    assert report["healthy"]["false_positive_rate"] == 0.0


def test_evaluate_counterfactual_healthy_fpr_unaffected_by_counterfactual_pairing():
    """Healthy-trial false positives are scored exactly as metrics.evaluate does -- any flag on
    the trial's own output -- with no counterfactual comparison involved."""
    from cook_ad.eval import metrics
    flagged_healthy = {"s_noun": np.zeros(10, dtype=bool)}
    flagged_healthy["s_noun"][3] = True
    report = cf.evaluate_counterfactual(
        [flagged_healthy], {"substitution": []}, channels=("s_noun",),
    )
    assert report["healthy"]["false_positive_rate"] == 1.0
    assert report["healthy"]["false_positive_rate"] == pytest.approx(
        1.0 if metrics._any_flag(flagged_healthy, ("s_noun",)) else 0.0
    )
