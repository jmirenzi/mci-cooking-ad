"""s_pair: the verb-noun compatibility channel.

s_emit is a MIXTURE of products, so it does not equal s_verb + s_noun, and the residual is the
pointwise mutual information under the predictive mixture. s_pair exposes it, sign-flipped so
that -- like every other channel -- larger means more surprising.
"""
import numpy as np
import pytest

from cook_ad.anomaly import quantile, surprise


def _emissions(rng, k, v, n):
    return np.log(rng.dirichlet(np.ones(v), k)), np.log(rng.dirichlet(np.ones(n), k))


def test_pair_is_identically_zero_under_a_single_state():
    """The load-bearing property. Within one state the emission IS a product by construction, so
    the PMI is exactly 0 -- which is why s_pair cannot have a per-STATE quantile table and why
    all of its signal is a mixture effect."""
    rng = np.random.default_rng(0)
    lv, ln = _emissions(rng, 3, 4, 5)
    pi = np.full((6, 3), -np.inf)
    pi[:, 1] = 0.0
    s_emit, s_verb, s_noun = surprise.emission_surprise(
        pi, lv, ln, np.array([0, 1, 2, 3, 0, 1]), np.array([0, 1, 2, 3, 4, 0])
    )
    assert np.allclose(np.asarray(s_emit) - np.asarray(s_verb) - np.asarray(s_noun), 0.0, atol=1e-9)


def test_pair_threshold_is_zero_for_a_degenerate_mixture():
    rng = np.random.default_rng(0)
    lv, ln = _emissions(rng, 3, 4, 5)
    pi = np.full((4, 3), -1e9)
    pi[:, 1] = 0.0
    assert np.allclose(quantile.pair_quantile_threshold(pi, lv, ln, 0.05), 0.0, atol=1e-9)


def test_pair_threshold_is_finite_and_nonnegative_under_a_real_mixture():
    rng = np.random.default_rng(1)
    lv, ln = _emissions(rng, 4, 5, 6)
    pi = np.log(rng.dirichlet(np.ones(4), 8))
    t = quantile.pair_quantile_threshold(pi, lv, ln, 0.05)
    assert t.shape == (8,)
    assert np.isfinite(t).all()
    assert (t >= 0).all(), "the null's own mean PMI is >= 0, so an alpha-tail threshold cannot be negative"


def test_pair_threshold_tightens_as_alpha_shrinks():
    rng = np.random.default_rng(2)
    lv, ln = _emissions(rng, 4, 5, 6)
    pi = np.log(rng.dirichlet(np.ones(4), 5))
    loose = quantile.pair_quantile_threshold(pi, lv, ln, 0.2)
    tight = quantile.pair_quantile_threshold(pi, lv, ln, 0.01)
    assert (tight >= loose - 1e-12).all()


def test_a_scrambled_pair_scores_higher_than_a_coherent_one():
    """Two states, each a near-delta on its own (verb, noun) pair. Observing one state's verb
    with the OTHER state's noun is the scramble: both tokens are ordinary under the mixture and
    only the combination is wrong, which is exactly the case s_verb and s_noun cannot see."""
    lv = np.log(np.array([[0.98, 0.02], [0.02, 0.98]]))
    ln = np.log(np.array([[0.98, 0.02], [0.02, 0.98]]))
    pi = np.log(np.array([[0.5, 0.5], [0.5, 0.5]]))
    s_emit, s_verb, s_noun = surprise.emission_surprise(
        pi, lv, ln, np.array([0, 0]), np.array([0, 1])   # coherent (v0,n0), scrambled (v0,n1)
    )
    s_pair = np.asarray(s_emit) - np.asarray(s_verb) - np.asarray(s_noun)
    assert s_pair[1] > s_pair[0]
    assert s_pair[0] < 0 < s_pair[1], "coherent pairs cohere (negative), scrambled ones do not"
    # and the marginals really are blind to it
    assert np.allclose(s_verb[0], s_verb[1])
    assert np.asarray(s_noun)[1] == pytest.approx(np.asarray(s_noun)[0], abs=1e-9)


def test_pair_channel_is_not_in_the_default_channel_tuple():
    """Every scorecard ORs CHANNELS for its headline number; silently adding an eighth channel
    would move every result already recorded in the repo."""
    assert surprise.PAIR_CHANNEL not in surprise.CHANNELS
    assert surprise.CHANNELS_WITH_PAIR == surprise.CHANNELS + (surprise.PAIR_CHANNEL,)
