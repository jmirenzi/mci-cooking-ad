"""Breakfast evaluation under metric G (docs/eval.md): every detector, operating points picked on
TRAIN, the headline table on TEST.

Pools are the canonical files (scripts/data/build_pools.py): every usable trial healthy plus one
injected copy per error type, all placed by one fixed model. The test pool IS the pool the cached
LLM answers were collected on, so every row -- HSMM, LLM and the nulls -- is scored on the same
97 trials per group.

For each HSMM model (--models):
  1. TRAIN pool: sweep hindsight over ALPHAS and live/lag over ALPHAS x KAPPAS; pick each arm's
     best G-F1.
  2. TEST pool: the same sweep, reported as grids (no picking).
  3. The headline table: each arm at its train-picked operating point on TEST, beside the LLM
     arms and the nulls.

LLM arms read the response cache first. Pass --max-requests 0 to replay only: a miss then means a prompt byte
changed. The cache key includes the base URL, and Breakfast's responses were collected through
http://localhost:11435/v1.

    ./py scripts/eval/eval_breakfast.py --out runs/eval_breakfast.json \
        --operating-points configs/operating_points.yaml
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.getcwd())
from cook_ad.xla_env import disable_gpu_autotuning  # noqa: E402

disable_gpu_autotuning()

import jax  # noqa: E402
import numpy as np  # noqa: E402

jax.config.update("jax_enable_x64", True)

from cook_ad.core import OperatingPoint, Trial  # noqa: E402
from cook_ad.data.config import load_config  # noqa: E402
from cook_ad.detectors import llm_predictive as llm_pred  # noqa: E402
from cook_ad.detectors.hsmm_hindsight import HindsightHSMMDetector  # noqa: E402
from cook_ad.detectors.hsmm_lag import LagHSMMDetector  # noqa: E402
from cook_ad.detectors.hsmm_live import LiveHSMMDetector  # noqa: E402
from cook_ad.detectors.llm_verdict import LLMVerdictDetector  # noqa: E402
from cook_ad.detectors.narrate import Lexicon  # noqa: E402
from cook_ad.detectors.thresholds import BadSets  # noqa: E402
from cook_ad.eval import harness, metric_g, pools  # noqa: E402
from cook_ad.hsmm import params as hparams  # noqa: E402
from cook_ad.hsmm.model import HSMM  # noqa: E402
from cook_ad.hsmm.train.lexical_init import recipe_label_assignment  # noqa: E402
from cook_ad.llm import client as llm_client  # noqa: E402

B = "dataset/processed/breakfast"
MODELS = {  # name -> (params, recipe given?)
    "dish-given": (f"{B}/joint_params_train_dish.npz", True),
    "dish-inferred": (f"{B}/joint_params_train_dish.npz", False),
    "lexhard": (f"{B}/joint_params_train_lexhard.npz", False),
}
LABEL = {"dish-given": "HSMM dish model, true dish given",
         "dish-inferred": "HSMM dish model, dish inferred",
         "lexhard": "HSMM 16-cluster model (shipped)"}
REFERENCE_OP = {"alpha": 0.005, "s_temporal": 1e-5}   # joint_params.npz's arm in the LLM report


# ---- HSMM sweeps ----------------------------------------------------------------------------


def hsmm_grids(hsmm, bad, streams, recipes, lags):
    """{arm: metric G grid} over one per-type pool. Hindsight is batched per group, as always."""
    trials = harness.trials_from_streams(streams, recipes)
    hind = HindsightHSMMDetector(hsmm, bad, OperatingPoint(0.005), chunk_size=8)
    masks = {"hindsight": []}
    for group in streams:
        group_trials = [t for g, t in trials if g == group]
        for tr in hind.traces(group_trials):
            masks["hindsight"].append(harness.hindsight_grid(hind, tr, harness.ALPHAS))
    op = OperatingPoint(0.5, 0.5)
    for lag in lags:
        det = (LiveHSMMDetector(hsmm, bad, op) if lag == 0 else
               LagHSMMDetector(hsmm, bad, op, math.inf if lag == "inf" else lag))
        masks[f"live_L{lag}"] = [harness.live_grid(det, det.trace(t), harness.ALPHAS,
                                                   harness.KAPPAS) for _, t in trials]
    return {arm: harness.sweep(m, trials) for arm, m in masks.items()}


def picked(grid, arm):
    idx = harness.pick(grid)
    return {"alpha": harness.ALPHAS[idx[0]],
            "kappa": harness.KAPPAS[idx[1]] if len(idx) > 1 else None, "index": idx}


def dish_of(seqs, labels):
    return recipe_label_assignment(seqs, labels)[0]


# ---- the headline table ----------------------------------------------------------------------


def hsmm_rows(model, hsmm, ops, streams, recipes):
    """Hindsight and live L=0 rows at the train-picked operating points, on the LLM pool."""
    rows = {}
    trials = harness.trials_from_streams(streams, recipes)
    a_h = ops["hindsight"]["alpha"]
    bad = BadSets(hsmm, harness.ALPHAS)
    hind = HindsightHSMMDetector(hsmm, bad, OperatingPoint(a_h), chunk_size=8)
    masks = [harness.alarm_mask(d, len(t.verbs)) for d, (_, t) in
             zip(hind.detect_many([t for _, t in trials]), trials)]
    rows[f"{LABEL[model]}: hindsight (α={a_h:g})"] = harness.score_masks(masks, trials)
    a_l, k_l = ops["live_L0"]["alpha"], ops["live_L0"]["kappa"]
    live = LiveHSMMDetector(hsmm, bad, OperatingPoint(a_l, k_l))
    masks = [harness.alarm_mask(live.detect_trial(t), len(t.verbs)) for _, t in trials]
    rows[f"{LABEL[model]}: live L=0 (α={a_l:g}, κ={k_l})"] = harness.score_masks(masks, trials)
    return rows


def reference_row(placement, d_max, streams):
    """joint_params.npz hindsight at alpha 0.005 with s_temporal 1e-5, batched per group (the
    HSMM arm the LLM report has always carried)."""
    hsmm = HSMM(placement, d_max)
    alphas = {c: REFERENCE_OP["alpha"] for c in harness.CHANNELS}
    alphas["s_temporal"] = REFERENCE_OP["s_temporal"]
    det = HindsightHSMMDetector(hsmm, BadSets(hsmm, sorted(set(alphas.values()))),
                                OperatingPoint(alphas), chunk_size=16)
    trials = harness.trials_from_streams(streams)
    masks = []
    for group in streams:
        for d, (_, t) in zip(det.detect_many([t for g, t in trials if g == group]),
                             [(g, t) for g, t in trials if g == group]):
            masks.append(harness.alarm_mask(d, len(t.verbs)))
    return harness.score_masks(masks, trials)


def llm_rows(args, vocab, labels, lexicon, train_seqs, trials):
    rows = {}
    train = [Trial(s["trial_id"], np.asarray(s["verb_ids"]), np.asarray(s["noun_ids"]))
             for s in train_seqs]
    cache = Path(args.cache_dir) / args.model.replace("/", "_")

    def client(temperature, max_tokens):
        return llm_client.ChatClient(model=args.model, base_url=args.base_url, cache_dir=cache,
                                     max_requests=args.max_requests, temperature=temperature,
                                     max_tokens=max_tokens, concurrency=args.concurrency)

    for shots, cot in (("random", False), ("autocot", True)):
        for variant in ("no-recipes", "with-recipes"):
            c = client(llm_pred.TEMPERATURE, llm_pred.COT_MAX_TOKENS if cot else llm_pred.MAX_TOKENS)
            det = llm_pred.LLMPredictiveDetector(
                c, vocab, lexicon, variant=variant, labels=labels, n_shot=3, shot_select=shots,
                shot_cot=cot, rationale_client=client(0.0, llm_pred.RATIONALE_MAX_TOKENS),
                prompt_unit="tick")
            det.fit(train)
            masks = [harness.alarm_mask(det.detect_trial(t), len(t.verbs)) for _, t in trials]
            name = f"LLM llm-predictive-{variant}" + ("+autocot+cot" if cot else "")
            rows[name] = harness.score_masks(masks, trials)
            print(f"  {name}: {c.n_cache_hits} cached, {c.n_would_request} uncached", flush=True)
    if "llm-verdict" in args.detectors:
        for variant in ("no-recipes", "with-recipes"):
            c = client(0.0, None)
            det = LLMVerdictDetector(c, vocab, lexicon, protocol="incremental", unit="tick",
                                     variant=variant, labels=labels)
            det.fit([])
            masks = [harness.alarm_mask(det.detect_trial(t), len(t.verbs)) for _, t in trials]
            name = f"LLM llm-verdict-{variant} (incremental, tick)"
            rows[name] = harness.score_masks(masks, trials)
            print(f"  {name}: {c.n_cache_hits} cached, {c.n_would_request} uncached", flush=True)
    return rows


# ---- main -----------------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="dish-given,dish-inferred,lexhard")
    ap.add_argument("--detectors", default="hsmm-hindsight,hsmm-live,hsmm-lag,llm-predictive,"
                                           "llm-verdict,nulls")
    ap.add_argument("--lags", default="5,inf", help="lags for hsmm-lag (swept and reported)")
    ap.add_argument("--skip-test-grid", action="store_true",
                    help="skip step 2 (the full test-pool sweep)")
    ap.add_argument("--out", default="runs/eval_breakfast.json")
    ap.add_argument("--np-hazard", type=float, nargs=2, metavar=("KAPPA", "BW"), default=None,
                    help="replace each model's NB durations with the nonparametric hazard fitted on its "
                         "decode of the train split (hsmm/train/np_hazard.py; testing)")
    ap.add_argument("--operating-points", default=None,
                    help="write the train-picked operating points into this YAML")
    ap.add_argument("--model", default="gemma3:27b")
    ap.add_argument("--base-url", default="http://localhost:11435/v1")
    ap.add_argument("--cache-dir", default=f"{B}/llm_cache")
    ap.add_argument("--max-requests", type=int, default=None,
                    help="cap on uncached requests; 0 = replay from the cache only")
    ap.add_argument("--concurrency", type=int, default=None)
    args = ap.parse_args()
    args.detectors = args.detectors.split(",")
    lags = [0] + ([x if x == "inf" else int(x) for x in args.lags.split(",")]
                  if "hsmm-lag" in args.detectors else [])

    d_max = load_config("configs/breakfast.yaml")["duration"]["d_max_ticks"]
    vocab = json.load(open(f"{B}/vocab.json"))
    labels = json.load(open(f"{B}/labels.json"))
    pool = {part: pools.load_pool(pools.pool_path("breakfast", part)) for part in ("train", "test")}
    streams = {part: pools.to_streams(entries) for part, (entries, _) in pool.items()}
    dishes = {part: dish_of([{"trial_id": e["trial_id"]} for e in entries], labels)
              for part, (entries, _) in pool.items()}
    train_seqs = pools.split_sequences(f"{B}/sequences.json", f"{B}/split.json", "train")
    placement = hparams.load(pool["test"][1]["placement"])

    out = {"metric": "G", "metric_g_definition": dict(metric_g.METRIC_G), "alphas": harness.ALPHAS,
           "kappas": harness.KAPPAS, "lags": lags, "grids": {}, "operating_points": {},
           "pools": {part: meta for part, (_, meta) in pool.items()}}
    def load_model(model):
        path, given = MODELS[model]
        if not args.np_hazard:
            return HSMM.load(path, d_max)
        from cook_ad.hsmm.train.np_hazard import np_hsmm   # nonparametric hazard (next_steps.md §3)
        kappa, bw = args.np_hazard
        return np_hsmm(hparams.load(path), d_max, train_seqs,
                       dish_of(train_seqs, labels) if given else None, kappa=kappa, bandwidth=bw)

    for model in args.models.split(","):
        path, given = MODELS[model]
        hsmm = load_model(model)
        bad = BadSets(hsmm, harness.ALPHAS)
        out["grids"][model] = {}
        for part in ("train", "test"):
            if part == "test" and args.skip_test_grid:
                continue
            t0 = time.time()
            out["grids"][model][part] = hsmm_grids(hsmm, bad, streams[part],
                                                   dishes[part] if given else None, lags)
            print(f"[{model}/{part}] {len(pool[part][0])} trials per group "
                  f"({time.time() - t0:.0f}s)", flush=True)
        out["operating_points"][model] = {arm: picked(g, arm)
                                          for arm, g in out["grids"][model]["train"].items()}

    streams, dishes = streams["test"], dishes["test"]
    trials = harness.trials_from_streams(streams)
    rows = {}
    if "nulls" in args.detectors:
        rows.update({k: harness.score_masks(m, trials)
                     for k, m in harness.null_masks(trials).items()})
    rows["HSMM joint_params.npz (old LLM-report arm)"] = reference_row(placement, d_max, streams)
    if "llm-predictive" in args.detectors:
        lexicon = Lexicon(vocab, placement)
        rows.update(llm_rows(args, vocab, labels, lexicon, train_seqs, trials))
    for model in args.models.split(","):
        hsmm = load_model(model)
        rows.update(hsmm_rows(model, hsmm, out["operating_points"][model], streams,
                              dishes if MODELS[model][1] else None))
    out["table"] = rows
    harness.table(rows, f"Breakfast test pool ({len(streams['healthy'])} trials per group; "
                        f"= the LLM pool: {pool['test'][1].get('equals_llm_pool')})")
    harness.dump(args.out, out)
    print(f"\nwritten to {args.out}")
    if args.operating_points:
        harness.write_operating_points(args.operating_points, {"breakfast": {
            m: {arm: {"alpha": op["alpha"], "kappa": op["kappa"]} for arm, op in ops.items()}
            for m, ops in out["operating_points"].items()}})
        print(f"operating points written to {args.operating_points}")


if __name__ == "__main__":
    main()
