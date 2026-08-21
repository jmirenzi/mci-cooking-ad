"""anomaly/temporal.py -- the duration channels and the NB tail functions they score with."""
def test_numpy_nb_tails_match_scipy_reference():
    """durations.nb_*_np are what anomaly/temporal.py scores segments with (the jax versions
    cost ~0.1s per call on a float64 GPU regardless of size). They must agree with
    scipy.stats.nbinom, which is the reference the jax versions' own docstrings cite -- and on
    the representable range they agree to machine precision, tighter than the jax path's
    betainc approximation manages.
    """
    import numpy as np
    from scipy.stats import nbinom

    from cook_ad.hsmm import durations

    rng = np.random.default_rng(1)
    r = rng.uniform(0.1, 500.0, 2000)
    p = rng.uniform(1e-3, 1 - 1e-3, 2000)
    d = rng.integers(1, 200, 2000).astype(float)

    ref_cdf = nbinom.logcdf(d - 1, r, p)
    ref_pmf = nbinom.logpmf(d - 1, r, p)
    got_cdf = durations.nb_log_cdf_np(d, r, p)
    got_pmf = durations.nb_log_pmf_np(d, r, p)

    finite = np.isfinite(ref_cdf) & np.isfinite(got_cdf)
    assert np.max(np.abs(got_cdf[finite] - ref_cdf[finite])) < 1e-9
    assert np.array_equal(np.isfinite(got_cdf), np.isfinite(ref_cdf))
    finite_pmf = np.isfinite(ref_pmf) & np.isfinite(got_pmf)
    assert np.max(np.abs(got_pmf[finite_pmf] - ref_pmf[finite_pmf])) < 1e-9

    # survival: d <= 1 is P(D >= 1) = 1 by definition, not a betainc evaluation
    assert np.all(durations.nb_log_survival_np(np.ones(5), r[:5], p[:5]) == 0.0)


def test_completed_segment_surprise_is_vectorised_but_unchanged():
    """The per-segment loop this replaced dispatched one JAX call per segment; the vectorised
    form must produce the same three channels and the same attribution labels."""
    import numpy as np

    from cook_ad.anomaly import temporal
    from cook_ad.hsmm import durations

    rng = np.random.default_rng(0)
    k = 8
    dur_r = rng.uniform(1.0, 20.0, k)
    dur_p = rng.uniform(0.05, 0.9, k)
    segments = [(int(rng.integers(k)), int(rng.integers(1, 60))) for _ in range(20)]

    s_long, s_short, s_two, attr = temporal.completed_segment_surprise(segments, dur_r, dur_p)

    # last segment is right-censored and left unscored
    assert s_long[-1] == 0.0 and s_short[-1] == 0.0 and s_two[-1] == 0.0 and attr[-1] == "none"
    for i, (state, d) in enumerate(segments[:-1]):
        ls = float(durations.nb_log_survival_np(float(d), dur_r[state], dur_p[state]))
        lc = float(durations.nb_log_cdf_np(float(d), dur_r[state], dur_p[state]))
        assert np.isclose(s_long[i], -ls)
        assert np.isclose(s_short[i], -lc)
        assert attr[i] == ("stuck" if ls < lc else "left_early")
