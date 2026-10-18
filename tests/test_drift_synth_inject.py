"""Step-level injectors (drift_synth/inject.py) on the toy omelet."""
import numpy as np
import pytest

from _drift_fixtures import OMELET, theta, toy_assets
from cook_ad.drift_synth.graphs import parents
from cook_ad.drift_synth.inject import (MIN_KEEP_S, event_spans, inject, items_from_session,
                                        is_action, to_events)
from cook_ad.drift_synth.noise import NoiseCfg
from cook_ad.drift_synth.sampler import rng_for, sample_session

A = toy_assets()


def _session(seed):
    s = sample_session(OMELET, A.expansions["omelet"], theta(), 1.0, A, NoiseCfg().off(), rng_for(seed, 4))
    return {"meta": s.meta, "recipe_id": "omelet"}


def test_round_trip_keeps_times():
    sess = _session(0)
    ev, marked = to_events(items_from_session(sess))
    assert marked == []
    assert [[e.t_start, e.t_end, e.verb, e.noun] for e in ev] == [c[:4] for c in sess["meta"]["clean_events"]]


@pytest.mark.parametrize("seed", range(6))
def test_each_error_is_what_it_says(seed):
    sess = _session(seed)
    items = items_from_session(sess)
    acts = [(it.step, it.verb, it.noun) for it in items if is_action(it)]
    rng = np.random.default_rng(seed)

    ev, marked = inject("omission", sess, OMELET, rng)
    assert len(ev) < len(items) and len(marked) == 1

    ev, marked = inject("repetition", sess, OMELET, rng)
    assert len(ev) > len(items)

    ev, marked = inject("transposition", sess, OMELET, rng)
    assert sorted((e.verb, e.noun) for e in ev) == sorted((it.verb, it.noun) for it in items)
    assert len({m[1] for m in marked}) == 2

    ev, marked = inject("substitution", sess, OMELET, rng, unseen_nouns=["onion"])
    assert sum(e.noun == "onion" for e in ev) == 1

    res = inject("abandonment", sess, OMELET, rng)
    if res is not None:
        ev, marked = res
        cut = ev[marked[0][0]]
        assert cut.t_end - cut.t_start >= MIN_KEEP_S - 1e-6
    assert acts


def test_transposition_breaks_a_precedence():
    from cook_ad.drift_synth.inject import transposition
    par = parents(OMELET)
    for seed in range(10):
        sess = _session(seed)
        out = transposition(items_from_session(sess), sess["meta"]["order"], OMELET,
                            np.random.default_rng(seed))
        b = next(it.step for it in out if it.mark == 1)
        a = next(it.step for it in out if it.mark == 2)
        assert a in par[b]                                     # a is required before b ...
        first = {}
        for k, it in enumerate(out):
            if is_action(it):
                first.setdefault(it.step, k)
        assert first[b] < first[a]                             # ... and b now comes first


def test_spans_are_on_the_tick_grid():
    sess = _session(1)
    ev, marked = inject("repetition", sess, OMELET, np.random.default_rng(0))
    (a, b), = event_spans(ev, marked, 0.5)
    e = ev[marked[0][0]]
    assert a == round((e.t_start - ev[0].t_start) / 0.5) and b >= a
