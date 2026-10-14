"""One metric-G table for Breakfast: nulls, HSMM (hindsight and live), and the LLM predictive arm,
all on EXACTLY the same streams.

The LLM arm can only be scored on the pool its cached answers were produced for, so that pool is
the reference: `run_llm_eval.py --dump-streams` writes it (every trial's ids and spans) together
with every arm's per-trial flags. This tool adds

  * the nulls: flag every step (one flag on each run-length step's first tick) and flag every tick;
  * HSMM arms for each fitted model, hindsight and live, at the operating point each picked on its
    TRAIN split (G-F1, from run_live_eval.py's `live_<tag>G_train.json`) -- never tuned on this pool;

and scores every arm under metric G (span_metrics.METRIC_G): alarm onsets find spans (merge 5,
deadline 10, one alarm per run of constant (verb, noun)), precision over injected trials,
alarms per 10 min on healthy trials, run FPR.

    ./py tools_breakfast_g_table.py STREAMS.pkl [MORE_STREAMS.pkl ...]
"""
import json
import pickle
import sys

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from cook_ad.anomaly import live, quantile, surprise
from cook_ad.data import split as split_mod
from cook_ad.eval import batch, span_metrics
from cook_ad.hsmm import joint_params
from cook_ad.recipe import lexical_init

B = "dataset/processed/breakfast"
D_MAX = 200
MODELS = {   # name -> (params, train-pick json, give the true dish?)
    "HSMM dish model, true dish given": (f"{B}/joint_params_train_dish.npz", "runs/live_dishknownG_train.json", True),
    "HSMM dish model, dish inferred": (f"{B}/joint_params_train_dish.npz", "runs/live_dishG_train.json", False),
    "HSMM 16-cluster model (shipped)": (f"{B}/joint_params_train_lexhard.npz", "runs/live_lexhardG_train.json", False),
}


def rle_starts(verb, noun):
    """First tick of every maximal run of constant (verb, noun)."""
    v, n = np.asarray(verb), np.asarray(noun)
    change = np.r_[True, (v[1:] != v[:-1]) | (n[1:] != n[:-1])]
    return np.flatnonzero(change)


def correct_steps(verb, noun, spans):
    starts = rle_starts(verb, noun)
    ends = np.r_[starts[1:] - 1, len(verb) - 1]
    return [(int(a), int(b)) for a, b in zip(starts, ends)
            if not any(a <= s1 and b >= s0 for s0, s1 in spans)]


def score_g(per_trial_flags, trials):
    """per_trial_flags: list aligned with `trials` of (T,) bool. trials: [(group, verb, noun, spans)]."""
    tp = n_spans = fp = 0
    onsets = ticks = any_h = n_h = steps_fl = n_correct = 0
    per_type = {}
    for f, (g, v, n, spans) in zip(per_trial_flags, trials):
        cs = correct_steps(v, n, spans)
        r = span_metrics.score_onsets_dense(np.asarray(f)[:, None], [(s, g) for s in spans], cs,
                                            run_starts=span_metrics.run_starts(v, n))
        steps_fl += int(r["steps_flagged"][0])
        n_correct += len(cs)
        if g == "healthy":
            onsets += int(r["onsets"][0]); ticks += len(v); any_h += int(r["any_flag"][0]); n_h += 1
        else:
            h = int(r["hits"].sum())
            tp += h; n_spans += len(spans); fp += int(r["fp"][0])
            t = per_type.setdefault(g, [0, 0]); t[0] += h; t[1] += len(spans)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / n_spans if n_spans else 0.0
    return {"precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "alarms_per_10min": 600.0 * onsets / ticks, "run_fpr": steps_fl / max(n_correct, 1),
            "healthy_trial_fpr": any_h / max(n_h, 1),
            "per_type": {k: a / b for k, (a, b) in per_type.items()}}


def train_pick(path, arm):
    g = json.load(open(path))["metric_g"][arm]
    d = json.load(open(path))
    f1 = np.asarray(g["f1"])
    idx = np.unravel_index(int(np.argmax(f1)), f1.shape)
    alpha = d["alphas"][idx[0]]
    kappa = d["kappas"][idx[1]] if len(idx) > 1 else None
    return alpha, kappa


def true_dishes(healthy_ids):
    """Recover each pool trial's dish by matching its healthy ids to sequences.json (the pool
    carries no trial ids)."""
    seqs = json.load(open(f"{B}/sequences.json"))
    seqs = split_mod.filter_sequences(seqs, split_mod.load_split(f"{B}/split.json"), "test")
    key = {(tuple(s["verb_ids"]), tuple(s["noun_ids"])): s for s in seqs}
    matched = [key[(tuple(map(int, h["verb_ids"])), tuple(map(int, h["noun_ids"])))] for h in healthy_ids]
    r, _ = lexical_init.recipe_label_assignment(matched, json.load(open(f"{B}/labels.json")))
    return r


def main(*paths):
    payloads = [pickle.load(open(p, "rb")) for p in paths]
    pool = payloads[0]["pools"]["real"]
    groups = ["healthy"] + list(pool[0][1].keys())
    trials, seq_by_trial, trial_index = [], [], []
    for gi, g in enumerate(groups):
        for ti, (healthy, deg) in enumerate(pool):
            d = healthy if g == "healthy" else deg[g]
            spans = [] if g == "healthy" else [tuple(map(int, s)) for s in d["spans"]]
            trials.append((g, d["verb_ids"], d["noun_ids"], spans))
            trial_index.append(ti)
    print(f"pool: {len(pool)} trials x {len(groups)} groups (healthy + {len(groups) - 1} injected)")

    arms = {}
    arms["null: flag every run"] = [np.isin(np.arange(len(v)), rle_starts(v, n)) for _, v, n, _ in trials]
    arms["null: flag every tick"] = [np.ones(len(v), bool) for _, v, n, _ in trials]

    def stream_to_masks(stream):
        flat = list(stream["healthy"])
        for g in groups[1:]:
            flat += list(stream["degraded"][g])
        out = []
        for flags, (_, v, _, _) in zip(flat, trials):
            m = np.zeros(len(v), bool)
            ticks = [f if isinstance(f, (int, np.integer)) else f.tick for f in flags]
            m[[t for t in ticks if 0 <= t < len(v)]] = True
            out.append(m)
        return out

    for p in payloads:
        for name, stream in p["streams"].items():
            label = name.split("/", 1)[1]
            label = ("HSMM joint_params.npz (old LLM-report arm)" if label.startswith("hsmm")
                     else f"LLM {label}")
            arms[label] = stream_to_masks(stream)

    dishes = true_dishes([h for h, _ in pool])
    for name, (params, pick_path, given) in MODELS.items():
        jp = joint_params.load_params(params)
        a_h, _ = train_pick(pick_path, "hindsight")
        a_l, k_l = train_pick(pick_path, "live_L0")
        seqs = [{"verb_ids": v, "noun_ids": n} for _, v, n, _ in trials]
        known = np.asarray([dishes[i] for i in trial_index]) if given else None
        traces, lp, r_hat, ltm = batch.compute_traces_joint(jp, seqs, D_MAX, chunk_size=8, r_hat=known)
        cache = quantile.JointThresholdCache(lp, ltm)
        hs = []
        for tr, r in zip(traces, np.asarray(r_hat)):
            f = surprise.flag_joint_cached(cache, tr, lp, int(r), ltm, alpha=a_h)
            hs.append(np.any(np.stack([np.asarray(f[c]) for c in surprise.CHANNELS]), axis=0))
        arms[f"{name}: hindsight (α={a_h:g})"] = hs
        tb = live.build_tables(jp, D_MAX, [a_l])
        ls = []
        for i, s in enumerate(seqs):
            lt = live.compute_live_trace(tb, s["verb_ids"], s["noun_ids"], lags=(0,),
                                         recipe=None if known is None else int(known[i]))
            f = live.flags(lt, tb.alphas, a_l, k_l, lag=0)
            ls.append(np.any(np.stack([f[c] for c in surprise.CHANNELS]), axis=0))
        arms[f"{name}: live L=0 (α={a_l:g}, κ={k_l})"] = ls
        print(f"  scored {name}", flush=True)

    rows = {name: score_g(masks, trials) for name, masks in arms.items()}
    types = groups[1:]
    print("\nMetric G (alarm = first flag per run of constant (verb, noun); deadline 10). Breakfast test, "
          f"{len(pool)} trials per group, LLM pool (run_llm_eval.py).\n")
    print("| arm | precision | recall | F1 | alarms / 10 min | run FPR | healthy-trial FPR |")
    print("|---|---|---|---|---|---|---|")
    for name, r in rows.items():
        print(f"| {name} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | "
              f"{r['alarms_per_10min']:.2f} | {r['run_fpr']:.3f} | {r['healthy_trial_fpr']:.3f} |")
    print("\nRecall per injected type:\n")
    print("| arm | " + " | ".join(types) + " |\n|---|" + "---|" * len(types))
    for name, r in rows.items():
        print(f"| {name} | " + " | ".join(f"{r['per_type'].get(t, 0):.2f}" for t in types) + " |")
    json.dump(rows, open("runs/breakfast_g_table.json", "w"), indent=2)
    print("\nwritten to runs/breakfast_g_table.json")


if __name__ == "__main__":
    main(*sys.argv[1:])
