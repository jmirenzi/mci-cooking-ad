"""Cost check for the restructure (docs/restructure.md §9.3). TEMPORARY: deleted in Phase G.

Times, on the lexhard Breakfast test pool (all six groups, run_live_eval's streams):
  hindsight   batch.compute_traces_joint over every group
  live        live.compute_live_trace(lags=(0,)) over every trial
  epic_tent   `run_epic_tent_eval.py --skip-llm` wall clock (subprocess)
  step        HSMM predict+update per tick on one trial (informational; skipped if unavailable)
Each in-process timing runs twice: `cold` includes jit compilation, `warm` is the second pass.
Run with the GPU otherwise idle.

    ./py scripts/verify/time_golden.py --out runs/golden/timings.json
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import dump_golden as G  # noqa: E402  (sets up XLA flags and x64 on import)
from cook_ad.anomaly import live  # noqa: E402
from cook_ad.data import split as split_mod  # noqa: E402
from cook_ad.data.config import load_config  # noqa: E402
from cook_ad.eval import batch  # noqa: E402
from cook_ad.hsmm import joint_params  # noqa: E402


def timed(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/golden/timings.json")
    args = ap.parse_args()
    d_max = load_config("configs/breakfast.yaml")["duration"]["d_max_ticks"]
    cfg = G.BF_CONFIGS["bf_lexhard"]
    jp = joint_params.load_params(cfg["params"])
    seqs = json.load(open(f"{G.BF}/sequences.json"))
    seqs = split_mod.filter_sequences(seqs, split_mod.load_split(f"{G.BF}/split.json"), "test")
    streams, _ = G.breakfast_pool(jp, seqs, d_max, 8, 0, False, None)
    tb = live.build_tables(jp, d_max, G.BF_ALPHA_GRID)
    out = {}

    def hindsight():
        for items in streams.values():
            batch.compute_traces_joint(jp, items, d_max, chunk_size=8)

    def live_all():
        for items in streams.values():
            for s in items:
                live.compute_live_trace(tb, s["verb_ids"], s["noun_ids"], lags=(0,))

    for name, fn in (("hindsight", hindsight), ("live", live_all)):
        out[name] = {"cold": timed(fn), "warm": timed(fn)}
        print(name, out[name], flush=True)

    try:
        from cook_ad.hsmm.model import HSMM
        hsmm = HSMM.load(cfg["params"], d_max)
        s = streams["healthy"][0]
        hsmm.reset()
        hsmm.update(int(s["verb_ids"][0]), int(s["noun_ids"][0]))  # compile
        hsmm.reset()
        n = len(s["verb_ids"])
        sec = timed(lambda: [hsmm.update(int(v), int(w)) for v, w in zip(s["verb_ids"], s["noun_ids"])])
        out["step_per_tick_ms"] = 1000.0 * sec / n
        print("step", out["step_per_tick_ms"], "ms/tick", flush=True)
    except ImportError:
        pass

    out["epic_tent"] = timed(lambda: subprocess.run(
        ["./py", "run_epic_tent_eval.py", "--skip-llm", "--out", "/dev/null"],
        check=True, stdout=subprocess.DEVNULL))
    print("epic_tent", out["epic_tent"], flush=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
