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
