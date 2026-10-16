"""Does the HSMM anomaly detector work on the synthetic drift data (docs/drift_synth.md §10)?

One normal person (seed 0, person 6: baseline slow = fine_motor = 0.41) cooks for 320 days.
The HSMM is fitted on days 0-199 only (dish preset: recipe = the CC4D recipe). Then:

- injected errors: the five Breakfast injectors on held-out normal days, placed in the fitted
  model's segmentation (docs/data.md §6). Operating points picked on days 200-259 by best G-F1,
  reported on days 260-319, metric G.
- slow / fine_motor: the SAME held-out days re-sampled with that capability knob raised by delta
  (common random numbers: each anomalous trial has a normal twin that differs only in theta).
  slow has no span (the whole trial is slower), so it is scored by alarm rate against the twins;
  fine_motor's repetitions are logged by the generator and scored as metric G spans.

    ./py scripts/eval/eval_synth_ad.py --stage data     # generate + export the person
    ./py scripts/eval/eval_synth_ad.py --stage train    # 4-stage dish fit on days 0-199
    ./py scripts/eval/eval_synth_ad.py --stage eval     # pools, operating points, tables

--noise off generates the same person without recognizer noise (into drift_synth_ad_clean/), to
separate what the detector does with the behaviour from what it does with the recognizer.
--train-days, --habit and --noise-scale are the healthy-floor levers (docs/drift_synth.md §9);
each setting writes its own drift_synth_ad_d<days>_h<habit>_n<scale>/ and results file.
"""
import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())

ROOT = Path("dataset/processed/drift_synth_ad")
NOISE_SCALE = 1.0   # recognizer noise rates x this (0 = noise-free)
HABIT = 0.0         # person routine strength (configs/drift_synth/generator.yaml person.habit)
CONFIG = "configs/drift_synth/hsmm.yaml"
SEED, PERSON = 0, 6
N_DAYS, TRAIN_END, VAL_END = 320, 200, 260
LEVELS = (0.2, 0.4, 0.55)
KNOBS = ("slow", "fine_motor")
TICK_S = 0.5


def cond_name(knob, delta):
    return f"{knob}_{delta:g}"


# ---- data --------------------------------------------------------------------------------------

def stage_data():
    from cook_ad.drift_synth.assets import load_assets
    from cook_ad.drift_synth.export import export
    from cook_ad.drift_synth.sampler import ConditionCfg, generate_person
    from cook_ad.drift_synth.schedule import DriftRegime

    assets = load_assets()
    assets.gen["recognizer"] = {k: (v if k == "min_event_s" else v * NOISE_SCALE)
                                for k, v in assets.gen["recognizer"].items()}
    assets.gen["person"]["habit"] = HABIT
    conds = [ConditionCfg("null", n_persons=1, n_days=N_DAYS)]
    conds += [ConditionCfg(cond_name(k, d), DriftRegime("sudden", TRAIN_END), {k: d}, n_persons=1,
                           n_days=N_DAYS) for k in KNOBS for d in LEVELS]
    for cond in conds:
        person, sessions, gt = generate_person(PERSON, cond, SEED, assets)
        for k, d in gt["delta"].items():
            if d <= 0:
                raise SystemExit(f"{cond.name}: {k} moved down ({d}); pick a person with room")
        export(sessions, ROOT / cond.name, assets.vocab, gt, assets.gen["export"])
        print(f"{cond.name}: {len(sessions)} sessions, theta1 "
              f"{ {k: gt['theta1'][k] for k in KNOBS} }")
    seqs = json.loads((ROOT / "null" / "sequences.json").read_text())
    labels = {e["trial_id"]: e for e in json.loads((ROOT / "null" / "labels.json").read_text())}
    train = [s["trial_id"] for s in seqs if labels[s["trial_id"]]["day"] < TRAIN_END]
    test = [s["trial_id"] for s in seqs if labels[s["trial_id"]]["day"] >= TRAIN_END]
    (ROOT / "null" / "split.json").write_text(json.dumps(
        {"seed": SEED, "test_frac": None, "train_trial_ids": train, "test_trial_ids": test,
         "rule": f"train = days < {TRAIN_END}; test = the rest (val {TRAIN_END}-{VAL_END - 1}, "
                 f"test {VAL_END}-{N_DAYS - 1})"}, indent=1))
    print(f"split: {len(train)} train / {len(test)} held-out sessions")


# ---- train -------------------------------------------------------------------------------------

def stage_train(max_iters, hard_iters):
    d = ROOT / "null"
    common = ["--config", CONFIG, "--sequences", str(d / "sequences.json"),
              "--labels", str(d / "labels.json"), "--split-file", str(d / "split.json"),
              "--recipe-labels"]
    out = ROOT / "model"
    out.mkdir(parents=True, exist_ok=True)
    steps = [
        ["scripts/train/fit_lexical.py", *common, "--init-prior-scale", "0.0", "--anchor", "50",
         "--max-iters", str(max_iters), "--out", str(out / "soft.npz")],
        ["scripts/train/fit_hard_em.py", *common, "--init-from", str(out / "soft.npz"),
         "--keep-init-emissions", "--iters", str(hard_iters), "--out", str(out / "hard.npz")],
        ["scripts/train/smooth_params.py", "--in", str(out / "hard.npz"),
         "--out", str(out / "smooth.npz"), "--strength", "0.7", "--backoff-tau", "30"],
        ["scripts/train/refit_durations.py", *common, "--in", str(out / "smooth.npz"),
         "--out", str(out / "model.npz"), "--kappa", "0.001"],
    ]
    for cmd in steps:
        print("+", " ".join(cmd), flush=True)
        subprocess.run(["./py", *cmd], check=True)
    print(f"model -> {out / 'model.npz'}")


# ---- eval --------------------------------------------------------------------------------------

def _load(cond):
    d = ROOT / cond
    seqs = json.loads((d / "sequences.json").read_text())
    labels = {e["trial_id"]: e for e in json.loads((d / "labels.json").read_text())}
    return seqs, labels


def _days(seqs, labels, lo, hi):
    return [s for s in seqs if lo <= labels[s["trial_id"]]["day"] < hi]


def repeat_spans(cond, trial_id, n_ticks):
    """fine_motor repetitions (logged by the generator) as (start, end) tick spans."""
    sess = json.loads((ROOT / cond / "sessions" / f"{trial_id}.json").read_text())
    origin = sess["events"][0][0] if sess["events"] else 0.0
    clean = sess["meta"]["clean_events"]
    spans = []
    for j in sess["meta"].get("repeats", []):
        t0, t1 = clean[j][0] - origin, clean[j][1] - origin
        a = max(0, int(math.floor(t0 / TICK_S)))
        b = min(n_ticks - 1, max(a, int(math.ceil(t1 / TICK_S)) - 1))
        if a < n_ticks:
            spans.append((a, b))
    return spans


def _flagged_runs(mask, trial):
    from cook_ad.eval import metric_g
    import numpy as np
    starts = np.asarray(metric_g.run_starts(trial.verbs, trial.nouns))
    ticks = np.flatnonzero(mask)
    return len(set(np.searchsorted(starts, ticks, side="right") - 1)) if len(ticks) else 0


def stage_eval(out_path, chunk_size):
    from cook_ad.xla_env import disable_gpu_autotuning
    disable_gpu_autotuning()
    import jax
    import numpy as np
    jax.config.update("jax_enable_x64", True)
    from cook_ad.core import OperatingPoint, Trial
    from cook_ad.data.config import load_config
    from cook_ad.detectors.hsmm_hindsight import HindsightHSMMDetector
    from cook_ad.detectors.hsmm_live import LiveHSMMDetector
    from cook_ad.detectors.thresholds import BadSets
    from cook_ad.eval import harness, metric_g, pools
    from cook_ad.hsmm import params as hparams
    from cook_ad.hsmm.model import HSMM
    from cook_ad.hsmm.train.lexical_init import recipe_label_assignment

    d_max = load_config(CONFIG)["duration"]["d_max_ticks"]
    model_path = ROOT / "model" / "model.npz"
    params = hparams.load(str(model_path))
    hsmm = HSMM.load(str(model_path), d_max)
    bad = BadSets(hsmm, harness.ALPHAS)
    null_seqs, null_labels = _load("null")
    all_labels = list(null_labels.values())

    def dish(seqs):
        return recipe_label_assignment(seqs, all_labels)[0]

    def injected(lo, hi):
        seqs = _days(null_seqs, null_labels, lo, hi)
        pool, keep = pools.injected_per_trial(params, seqs, d_max, seed=SEED, chunk_size=chunk_size)
        return pools.per_trial_to_streams(pool), dish([seqs[i] for i in keep])

    def hind_masks(det, trials, alpha=None):
        """Per trial: the (T, A) alpha grid, or the (T,) mask at `alpha`. Batched per group."""
        out = []
        for group in dict.fromkeys(g for g, _ in trials):
            ts = [t for g, t in trials if g == group]
            if alpha is None:
                out += [harness.hindsight_grid(det, tr, harness.ALPHAS) for tr in det.traces(ts)]
            else:
                out += [harness.alarm_mask(d, len(t.verbs)) for d, t in zip(det.detect_many(ts), ts)]
        return out

    # 1. operating points on the validation days (injected errors), best G-F1
    streams, recipes = injected(TRAIN_END, VAL_END)
    trials = harness.trials_from_streams(streams, recipes)
    hind = HindsightHSMMDetector(hsmm, bad, OperatingPoint(0.005), chunk_size=chunk_size)
    live = LiveHSMMDetector(hsmm, bad, OperatingPoint(0.5, 0.5))
    grids = {"hindsight": harness.sweep(hind_masks(hind, trials), trials, TICK_S),
             "live_L0": harness.sweep([harness.live_grid(live, live.trace(t), harness.ALPHAS,
                                                         harness.KAPPAS) for _, t in trials], trials, TICK_S)}
    ops = {}
    for arm, g in grids.items():
        idx = harness.pick(g)
        ops[arm] = {"alpha": harness.ALPHAS[idx[0]],
                    "kappa": harness.KAPPAS[idx[1]] if len(idx) > 1 else None,
                    "val": harness.at(g, idx)}
    print(f"val ({len(streams['healthy'])} trials per group): " + ", ".join(
        f"{a} alpha={o['alpha']:g} kappa={o['kappa']} F1={o['val']['f1']:.3f}" for a, o in ops.items()))
    detectors = {
        "hindsight": HindsightHSMMDetector(hsmm, bad, OperatingPoint(ops["hindsight"]["alpha"]),
                                           chunk_size=chunk_size),
        "live_L0": LiveHSMMDetector(hsmm, bad, OperatingPoint(ops["live_L0"]["alpha"],
                                                               ops["live_L0"]["kappa"])),
    }

    def detect(name, trials):
        det = detectors[name]
        if name == "hindsight":
            dets = []
            for group in dict.fromkeys(g for g, _ in trials):
                dets += det.detect_many([t for g, t in trials if g == group])
            return dets
        return [det.detect_trial(t) for _, t in trials]

    # 2. injected errors on the test days, metric G
    streams, recipes = injected(VAL_END, N_DAYS)
    trials = harness.trials_from_streams(streams, recipes)
    rows = {k: harness.score_masks(m, trials, TICK_S) for k, m in harness.null_masks(trials).items()}
    dets_by_arm = {}
    for arm in detectors:
        dets_by_arm[arm] = detect(arm, trials)
        rows[f"HSMM {arm} (alpha={ops[arm]['alpha']:g}"
             + (f", kappa={ops[arm]['kappa']})" if ops[arm]["kappa"] else ")")] = harness.score_masks(
            [harness.alarm_mask(d, len(t.verbs)) for d, (_, t) in zip(dets_by_arm[arm], trials)], trials,
            TICK_S)
    harness.table(rows, f"Synthetic person p06 (train days {TRAIN_END}, habit {HABIT:g}, noise x{NOISE_SCALE:g}), injected errors, test days {VAL_END}-{N_DAYS - 1} "
                        f"({len(streams['healthy'])} trials per group)")

    # 3. capability pools vs their normal twins (the same test days)
    test_null = _days(null_seqs, null_labels, VAL_END, N_DAYS)
    ids = [s["trial_id"] for s in test_null]
    cap = {"healthy (twins)": ("null", test_null)}
    for k in KNOBS:
        for dlt in LEVELS:
            seqs, labels = _load(cond_name(k, dlt))
            by_id = {s["trial_id"]: s for s in seqs}
            cap[f"{k} +{dlt:g}"] = (cond_name(k, dlt), [by_id[i] for i in ids])
    rec = dish(test_null)
    cap_rows = {}
    for arm in detectors:
        twin_alarms, twin_trials, twin_masks = None, None, None
        for name, (cond, seqs) in cap.items():
            group = "healthy" if name.startswith("healthy") else name
            trials = [(group, Trial(f"{cond}/{s['trial_id']}", np.asarray(s["verb_ids"]),
                                    np.asarray(s["noun_ids"]), int(r),
                                    tuple(repeat_spans(cond, s["trial_id"], len(s["verb_ids"])))
                                    if cond.startswith("fine_motor") else (), group))
                      for s, r in zip(seqs, rec)]
            dets = detect(arm, trials)
            masks = [harness.alarm_mask(d, len(t.verbs)) for d, (_, t) in zip(dets, trials)]
            # metric G's alarm: the first flag inside a run, so count flagged runs per trial
            n_alarms = np.array([_flagged_runs(m, t) for m, (_, t) in zip(masks, trials)])
            minutes = sum(len(t.verbs) for _, t in trials) * TICK_S / 60
            chans = {}
            for d in dets:
                for a in d.alarms:
                    chans[a.channel] = chans.get(a.channel, 0) + 1
            row = {"trials": len(trials), "alarms_per_10min": float(n_alarms.sum() / minutes * 10),
                   "trials_with_alarm": float(np.mean(n_alarms > 0)),
                   "mean_minutes": float(minutes / len(trials)), "channels": chans}
            if twin_alarms is None:
                twin_alarms, twin_trials, twin_masks = n_alarms, trials, masks
            else:
                row["more_alarms_than_twin"] = float(np.mean(n_alarms > twin_alarms))
                row["fewer_alarms_than_twin"] = float(np.mean(n_alarms < twin_alarms))
                row["delta_alarms_per_trial"] = float(np.mean(n_alarms - twin_alarms))
            if cond.startswith("fine_motor"):
                # metric G needs the healthy group: score the twins and this pool together
                g = harness.score_masks(twin_masks + masks, twin_trials + trials, TICK_S)
                row["repeat_spans"] = int(sum(len(t.spans) for _, t in trials))
                row["repeat_recall_G"] = g["recall"]
            cap_rows[f"{arm} | {name}"] = row

    print(f"\n### Capability pools vs their normal twins, test days {VAL_END}-{N_DAYS - 1} "
          f"(metric G alarm-rate columns; repetition recall is metric G)\n")
    print("| arm, pool | trials | min / trial | alarms / 10 min | trials with alarm | more alarms than twin "
          "| fewer | Δ alarms / trial | repetition spans | repetition recall (G) | top channels (raw flags) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for k, r in cap_rows.items():
        top = ", ".join(f"{c} {n}" for c, n in sorted(r["channels"].items(), key=lambda x: -x[1])[:3])
        print(f"| {k} | {r['trials']} | {r['mean_minutes']:.1f} | {r['alarms_per_10min']:.2f} | "
              f"{r['trials_with_alarm']:.2f} | {r.get('more_alarms_than_twin', float('nan')):.2f} | "
              f"{r.get('fewer_alarms_than_twin', float('nan')):.2f} | "
              f"{r.get('delta_alarms_per_trial', float('nan')):+.2f} | {r.get('repeat_spans', '')} | "
              f"{r.get('repeat_recall_G', float('nan')):.2f} | {top} |")

    harness.dump(out_path, {"metric": "G", "metric_g_definition": dict(metric_g.METRIC_G),
                            "person": {"seed": SEED, "person": PERSON, "habit": HABIT,
                                       "noise_scale": NOISE_SCALE}, "days": {
                                "train": [0, TRAIN_END], "val": [TRAIN_END, VAL_END],
                                "test": [VAL_END, N_DAYS]},
                            "operating_points": ops, "injected_table": rows,
                            "capability": cap_rows})
    print(f"\nwritten to {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["data", "train", "eval"], required=True)
    ap.add_argument("--max-iters", type=int, default=60)
    ap.add_argument("--hard-iters", type=int, default=5)
    ap.add_argument("--chunk-size", type=int, default=8)
    ap.add_argument("--noise", choices=["on", "off"], default="on", help="off = --noise-scale 0")
    ap.add_argument("--noise-scale", type=float, default=None)
    ap.add_argument("--train-days", type=int, default=200, help="val and test are 60 days each after")
    ap.add_argument("--habit", type=float, default=0.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    global ROOT, NOISE_SCALE, HABIT, TRAIN_END, VAL_END, N_DAYS
    NOISE_SCALE = a.noise_scale if a.noise_scale is not None else (1.0 if a.noise == "on" else 0.0)
    HABIT = a.habit
    TRAIN_END, VAL_END, N_DAYS = a.train_days, a.train_days + 60, a.train_days + 120
    # The two first runs kept their names; every other setting gets its own directory.
    if (TRAIN_END, HABIT) == (200, 0.0) and NOISE_SCALE in (0.0, 1.0):
        tag = "" if NOISE_SCALE else "_clean"
    else:
        tag = f"_d{TRAIN_END}_h{HABIT:g}_n{NOISE_SCALE:g}"
    ROOT = ROOT.with_name(ROOT.name + tag)
    a.out = a.out or f"runs/drift_synth/eval_synth_ad{tag}.json"
    if a.stage == "data":
        stage_data()
    elif a.stage == "train":
        stage_train(a.max_iters, a.hard_iters)
    else:
        stage_eval(a.out, a.chunk_size)


if __name__ == "__main__":
    main()
