"""eval/pools.py: the canonical pools are a pure function of (sequences, placement model, seed),
consumed one trial at a time, and the frozen files say what they were built from."""
import json
from pathlib import Path

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from _fixtures import stack_recipes, weak_limit_params  # noqa: E402
from cook_ad.eval import pools  # noqa: E402
from cook_ad.synthetic import generate  # noqa: E402

D_MAX = 20
REPO_ROOT = Path(__file__).resolve().parent.parent


def _sequences(n=8):
    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    jp = stack_recipes([weak_limit_params(k, 6, 7, 8, D_MAX, dur_mean_range=(3.0, 8.0)) for k in keys])
    trajs = generate.generate_healthy_joint(jp, n, np.random.default_rng(1), 80, D_MAX)
    return jp, [{"trial_id": f"t{i}", "verb_ids": t["verb_ids"].tolist(),
                 "noun_ids": t["noun_ids"].tolist()} for i, t in enumerate(trajs)]


def _same(a, b):
    return (np.array_equal(a["verb_ids"], b["verb_ids"]) and np.array_equal(a["noun_ids"], b["noun_ids"])
            and [tuple(s) for s in a.get("spans", [])] == [tuple(s) for s in b.get("spans", [])])


def test_a_pool_built_on_a_prefix_is_a_prefix_of_the_full_pool():
    """Every type is injected into a trial before the next trial is touched, so the rng state at
    trial i does not depend on later trials: a longer split extends the pool, never reshuffles it."""
    jp, seqs = _sequences()
    full = pools.build_pool(seqs, jp, D_MAX)
    part = pools.build_pool(seqs[:5], jp, D_MAX)
    assert len(part) <= len(full) and part
    for a, b in zip(part, full):
        assert a["trial_id"] == b["trial_id"]
        assert _same(a["healthy"], b["healthy"])
        assert all(_same(a["injected"][t], b["injected"][t]) for t in pools.ERROR_TYPES)


def test_save_load_round_trip_and_streams(tmp_path):
    jp, seqs = _sequences(4)
    entries = pools.build_pool(seqs, jp, D_MAX)
    pools.save_pool(tmp_path / "p.pkl", entries, {"seed": 0})
    back, meta = pools.load_pool(tmp_path / "p.pkl")
    assert meta == {"seed": 0} and [e["trial_id"] for e in back] == [e["trial_id"] for e in entries]
    streams = pools.to_streams(back)
    assert list(streams) == ["healthy", *pools.ERROR_TYPES]
    assert all(len(v) == len(entries) for v in streams.values())


@pytest.mark.skipif(not (REPO_ROOT / pools.pool_path("breakfast", "test")).exists(),
                    reason="run scripts/data/build_pools.py")
def test_the_breakfast_test_pool_is_the_llm_pool():
    entries, meta = pools.load_pool(REPO_ROOT / pools.pool_path("breakfast", "test"))
    assert meta["placement"].endswith("joint_params.npz") and meta["seed"] == 0
    assert meta.get("equals_llm_pool", True)
    assert len(entries) == meta["n_usable"]


@pytest.mark.skipif(not (REPO_ROOT / "dataset/processed/epic_tent/pools/split_test.pkl").exists(),
                    reason="run scripts/data/build_pools.py")
def test_the_frozen_epic_tent_pool_is_its_split():
    from cook_ad.data import split as split_mod
    E = REPO_ROOT / "dataset/processed/epic_tent"
    seqs, truth = pools.load_epic_tent_pool(E / "pools/split_test.pkl")
    want = split_mod.filter_sequences(json.load(open(E / "sequences.json")),
                                      split_mod.load_split(E / "split.json"), "test")
    assert [s["trial_id"] for s in seqs] == [s["trial_id"] for s in want]
    assert [t["trial_id"] for t in truth] == [s["trial_id"] for s in want]
