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
    assert len(ev) < len(items) and {m[1] for m in marked} == {1}
    assert [m[0] for m in marked] == list(range(marked[0][0], marked[-1][0] + 1))   # one stretch

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
        b = next(it.step for it in out if it.mark == 1 and it.step is not None)
        a = next(it.step for it in out if it.mark == 2 and it.step is not None)
        assert all(it.step is None for it in out if it.mark and it.step not in (a, b))
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


def test_injection_does_not_depend_on_the_hash_seed():
    """Step ids are strings: iterating a set of them follows PYTHONHASHSEED, so a pool drawn in
    two processes differed (transposition's candidate pairs came from parents()' sets)."""
    import os
    import subprocess
    import sys
    code = ("import sys; sys.path.insert(0, 'tests'); import numpy as np\n"
            "from test_drift_synth_inject import _session, OMELET; from cook_ad.drift_synth.inject import inject\n"
            "rng = np.random.default_rng(0)\n"
            "print([[(e.verb, e.noun) for e in inject('transposition', _session(s), OMELET, rng)[0]] for s in range(4)])")
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                           env=dict(os.environ, PYTHONHASHSEED=str(h))).stdout for h in (1, 2, 3)}
    assert len(outs) == 1
