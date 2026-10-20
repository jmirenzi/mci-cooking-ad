# ruff: noqa: E402
"""The s_count variants of compare_s_count.py on Breakfast: dish model, true dish given, stage-4b
durations (metric G, tick 1 s). Operating points picked on the TRAIN pool, reported on TEST, as in
eval_breakfast.py. The train-pool sessions are the model's training sessions, so their s_count
statistics are cross-fitted: two folds by trial, each scored with the statistics of the other
(about 200 sessions, 20 per dish); the test pool uses all 402.

    ./py scripts/eval/compare_s_count_breakfast.py [--mode hindsight --mode live]
"""
import argparse
import json
import os
import sys
import zlib

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "scripts/eval")
import numpy as np

import compare_s_count as C
from cook_ad.core import OperatingPoint, Trial
from cook_ad.data.config import load_config
from cook_ad.detectors.hsmm_hindsight import HindsightHSMMDetector
from cook_ad.detectors.thresholds import BadSets
from cook_ad.eval import harness, pools
from cook_ad.hsmm.model import HSMM
from cook_ad.hsmm.train.lexical_init import recipe_label_assignment

B = "dataset/processed/breakfast"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", action="append", choices=["hindsight", "live"])
    ap.add_argument("--out", default="runs/drift_synth/scount/gh/compare_s_count_breakfast.json")
    a = ap.parse_args()
    d_max = load_config("configs/breakfast.yaml")["duration"]["d_max_ticks"]
    vocab = json.load(open(f"{B}/vocab.json"))
    labels = json.load(open(f"{B}/labels.json"))
    hsmm = HSMM.load(f"{B}/joint_params_train_dish.npz", d_max, durations="auto")
    bad = BadSets(hsmm, C.ALPHAS)
    k = int(np.asarray(hsmm.tables.log_emit_v).shape[0])
    idle = set(np.flatnonzero(np.argmax(np.asarray(hsmm.tables.log_emit_v), axis=-1)
                              == vocab["verbs"]["stall"]).tolist())

    train = pools.split_sequences(f"{B}/sequences.json", f"{B}/split.json", "train")
    td = recipe_label_assignment(train, labels)[0]
    det = HindsightHSMMDetector(hsmm, bad, OperatingPoint(0.01), chunk_size=8)
    trs = det.traces([Trial(s["trial_id"], np.asarray(s["verb_ids"]), np.asarray(s["noun_ids"]),
                            int(r), (), "healthy") for s, r in zip(train, td)])
    dec = [(t.r_hat, [s for s, _ in t.segments]) for t in trs]
    fold = lambda tid: zlib.crc32(tid.encode()) % 2  # noqa: E731
    folds = [fold(s["trial_id"]) for s in train]
    st_fold = {f: C.Stats([x for x, g in zip(dec, folds) if g != f], k, idle) for f in (0, 1)}
    st_all = C.Stats(dec, k, idle)

    parts, pool_ids = {}, {}
    for part in ("train", "test"):
        entries, _ = pools.load_pool(pools.pool_path("breakfast", part))
        dishes = recipe_label_assignment([{"trial_id": e["trial_id"]} for e in entries], labels)[0]
        parts[part] = harness.trials_from_streams(pools.to_streams(entries), dishes)
        pool_ids[part] = [e["trial_id"] for e in entries]
    ids = {s["trial_id"] for s in train}
    assert all(t in ids for t in pool_ids["train"])

    def stats_val(t):       # Trial ids are "<group>/<pool index>"
        return st_fold[fold(pool_ids["train"][int(t.trial_id.split("/")[1])])]

    out = {"metric": list(C.METRICS), "corpus": "breakfast", "model": "dish-given", "levels": C.LEVELS,
           "variants": C.VARIANTS, "results": {}}
    for mode in a.mode or ("hindsight", "live"):
        out["results"][mode] = C.evaluate("breakfast", len(train), mode, parts["train"], parts["test"],
                                          (hsmm, bad, (stats_val, st_all), k), tick=1.0,
                                          hind_grid=harness.hindsight_grid)
        harness.dump(a.out, out)


if __name__ == "__main__":
    main()
