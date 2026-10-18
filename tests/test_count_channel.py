"""detectors/count_channel.py on hand-made decoded paths."""
from types import SimpleNamespace

import numpy as np

from cook_ad.detectors.count_channel import CountChannel


def _tr(states, r=0, d=3):
    return SimpleNamespace(r_hat=r, segments=[(k, d) for k in states])


def test_missing_state_is_flagged_where_it_belonged_and_an_extra_visit_too():
    train = [_tr([9, 1, 9, 2, 3, 9, 4]) for _ in range(40)] + [_tr([9, 1, 2, 9, 3, 4]) for _ in range(9)]
    ch = CountChannel.fit(train, ignore={9})
    ok = ch.flags(_tr([9, 1, 2, 3, 4]), [0.0, 0.05])
    assert not ok[:, 1].any() and not ok[:, 0].any()
    omit = ch.flags(_tr([9, 1, 2, 4]), [0.0, 0.05])         # 3 skipped; 4 usually follows 3
    assert np.flatnonzero(omit[:, 1]).tolist() == [9]       # the start of the segment of state 4
    rep = ch.flags(_tr([9, 1, 2, 3, 2, 4]), [0.0, 0.05])     # 2 done twice
    assert np.flatnonzero(rep[:, 1]).tolist() == [12]
    assert not rep[:, 0].any()                               # level 0 is off


def test_an_unknown_recipe_says_nothing():
    ch = CountChannel.fit([_tr([1, 2])], ignore=())
    assert not ch.flags(_tr([5, 6], r=3), [0.5]).any()


def test_skip_idle_sees_through_gaps():
    from cook_ad.detectors.skip_idle import SkipIdleTransitions
    train = [_tr([1, 9, 2, 9, 3, 9, 4]) for _ in range(50)]
    ch = SkipIdleTransitions.fit(train, ignore={9})
    assert not ch.flags(_tr([1, 9, 2, 9, 3, 9, 4]), [0.02]).any()
    rep = ch.flags(_tr([1, 9, 2, 9, 3, 9, 2, 9, 4]), [0.02])     # 2 again after 3, through idle
    assert np.flatnonzero(rep[:, 0]).tolist() == [18, 24]        # 3 -> 2 and 2 -> 4


def test_empirical_short_duration():
    from cook_ad.detectors.empirical_duration import EmpiricalShortDuration
    rng = np.random.default_rng(0)
    train = [SimpleNamespace(r_hat=0, segments=[(1, int(rng.integers(18, 23))), (9, 2), (2, 5)])
             for _ in range(100)]
    ch = EmpiricalShortDuration.fit(train, ignore={9})
    ok = ch.flags(SimpleNamespace(r_hat=0, segments=[(1, 20), (9, 2), (2, 5)]), [0.02])
    cut = ch.flags(SimpleNamespace(r_hat=0, segments=[(1, 3), (9, 2), (2, 5)]), [0.02])
    assert not ok.any() and np.flatnonzero(cut[:, 0]).tolist() == [2]


def test_skip_idle_population_prior_softens_a_cold_start():
    from cook_ad.detectors.skip_idle import BOS, EOS, SkipIdleTransitions
    train = [_tr([1, 9, 2, 9, 3])]                                  # one session: 1 2 3
    prior = {0: {(BOS, 1): 1.0, (1, 3): 0.5, (1, 2): 0.5, (3, 2): 1.0, (2, EOS): 0.5, (2, 3): 0.5,
                 (3, EOS): 0.0}}
    bare = SkipIdleTransitions.fit(train, ignore={9})
    warm = SkipIdleTransitions.fit(train, ignore={9}, prior=prior, prior_weight=5.0)
    odd = _tr([1, 9, 3, 9, 2])                                      # legal for the population
    p_bare = dict(bare.pvalues(odd))
    p_warm = dict(warm.pvalues(odd))
    assert p_warm[6] > p_bare[6]                                     # 1 -> 3 is no longer rare
