"""Cost check for the restructure (docs/restructure.md §9.3). TEMPORARY: deleted in Phase G.

Times, on the lexhard Breakfast test pool (all six groups, run_live_eval's streams):
  hindsight_detect_many   HindsightHSMMDetector.detect_many over every group
  live_detect_trial       LiveHSMMDetector.detect_trial over every trial
  epic_tent               `scripts/eval/eval_epic_tent.py --skip-llm` wall clock (subprocess)
  step                    HSMM.update and LiveHSMMDetector.step per tick on one trial
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

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import dump_golden as G  # noqa: E402  (sets up XLA flags and x64 on import)
from cook_ad.core import OperatingPoint, Trial  # noqa: E402
from cook_ad.data import split as split_mod  # noqa: E402
from cook_ad.data.config import load_config  # noqa: E402
from cook_ad.detectors.hsmm_hindsight import HindsightHSMMDetector  # noqa: E402
from cook_ad.detectors.hsmm_live import LiveHSMMDetector  # noqa: E402
from cook_ad.detectors.thresholds import BadSets  # noqa: E402
from cook_ad.hsmm.model import HSMM  # noqa: E402


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
    hsmm = HSMM.load(cfg["params"], d_max)
    seqs = json.load(open(f"{G.BF}/sequences.json"))
    seqs = split_mod.filter_sequences(seqs, split_mod.load_split(f"{G.BF}/split.json"), "test")
    streams, _ = G.breakfast_pool(hsmm.params, seqs, d_max, 8, False, None)
    bad = BadSets(hsmm, G.BF_ALPHA_GRID)
    trials = {g: [Trial(str(i), np.asarray(s["verb_ids"]), np.asarray(s["noun_ids"]))
                  for i, s in enumerate(items)] for g, items in streams.items()}
    hind = HindsightHSMMDetector(hsmm, bad, OperatingPoint(0.005), chunk_size=8)
    livedet = LiveHSMMDetector(hsmm, bad, OperatingPoint(0.02, 0.5))
    out = {}

    def hindsight_det():
        for ts in trials.values():
            hind.detect_many(ts)

    def live_det():
        for ts in trials.values():
            for t in ts:
                livedet.detect_trial(t)

    for name, fn in (("hindsight_detect_many", hindsight_det), ("live_detect_trial", live_det)):
        out[name] = {"cold": timed(fn), "warm": timed(fn)}
        print(name, out[name], flush=True)

    s = streams["healthy"][0]
    n = len(s["verb_ids"])
    hsmm.reset()
    hsmm.update(int(s["verb_ids"][0]), int(s["noun_ids"][0]))  # compile
    hsmm.reset()
    sec = timed(lambda: [hsmm.update(int(v), int(w)) for v, w in zip(s["verb_ids"], s["noun_ids"])])
    out["step_per_tick_ms"] = 1000.0 * sec / n
    livedet.reset()
    livedet.step(int(s["verb_ids"][0]), int(s["noun_ids"][0]))  # compile
    livedet.reset()
    sec = timed(lambda: [livedet.step(int(v), int(w)) for v, w in zip(s["verb_ids"], s["noun_ids"])])
    out["live_step_per_tick_ms"] = 1000.0 * sec / n
    print("step", out["step_per_tick_ms"], "live step", out["live_step_per_tick_ms"], flush=True)

    out["epic_tent"] = timed(lambda: subprocess.run(
        ["./py", "scripts/eval/eval_epic_tent.py", "--skip-llm", "--out", "/dev/null"],
        check=True, stdout=subprocess.DEVNULL))
    print("epic_tent", out["epic_tent"], flush=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
