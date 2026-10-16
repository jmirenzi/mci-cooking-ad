"""A global prior from a simulated population, fine-tuned per person (docs/drift_synth.md §9.2).

Population: 60 people (seed 1000, never test subjects), 30 days each, preference knobs
theta0 ~ U(0.05, 0.95), capability knobs ~ U(0.4, 0.6), habit ~ U(0, 1). Their tick streams are
run-length counted per recipe (hsmm/train/global_prior.count_global): ONE global prior for every
person model.

Test persons: seed 0, persons 0-3, habit 2, capability ~ U(0.4, 0.6), 320 days: train on the
first N in {10, 30, 100, 200}, validation days 200-259, test days 260-319. Each person model is
the four-stage dish fit (configs/drift_synth/hsmm.yaml), then the global prior is added post-fit
at strength lambda (apply_global_prior; lambda = 0 is the person-only model, "global only" drops
the person's own init/transition counts). Injected errors are placed once per person (by its
N = 200 person-only model), so every arm and N is scored on the same trials. Hindsight's alpha
is picked per arm on validation by best G-F1 and reported on test. Recognizer noise is off.

    ./py scripts/eval/eval_synth_prior.py --stage population
    ./py scripts/eval/eval_synth_prior.py --stage persons
    ./py scripts/eval/eval_synth_prior.py --stage train [--person 0 --days 200]
    ./py scripts/eval/eval_synth_prior.py --stage eval [--person 0]

--variant cal is the CC4D-calibrated generator (configs/drift_synth/step_time_scale.json, events
capped at 600 s): 1 s ticks, d_max 300 (configs/drift_synth/hsmm_cal.yaml), training sizes 10 /
30 / 200 days and 20 soft-EM iterations (the objective moves < 0.2% after iteration 20 in 13 of
16 uncalibrated fits), to fit in a night. Everything goes to drift_synth_prior_cal/ and
runs/drift_synth/*_cal_*.json.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())

ROOT = Path("dataset/processed/drift_synth_prior")
CONFIG = "configs/drift_synth/hsmm.yaml"
POP_SEED, POP_N, POP_DAYS = 1000, 60, 30
SEED, PERSONS = 0, (0, 1, 2, 3)
N_DAYS, VAL_START, TEST_START = 320, 200, 260
TRAIN_DAYS = (10, 30, 100, 200)
LAMBDAS = (0.0, 1.0, 3.0, 10.0, 30.0, 100.0)
GLOBAL_ONLY_LAMBDA = 30.0
CAPABILITY = {"slow": [0.4, 0.6], "fine_motor": [0.4, 0.6]}
TICK = 0.5
SOFT_ITERS = 60
TAG = ""          # results-file suffix
VARIANT = "base"


def set_variant(name):
    global ROOT, CONFIG, TICK, TRAIN_DAYS, SOFT_ITERS, TAG, VARIANT
    VARIANT = name
    if name == "cal":
        ROOT = Path("dataset/processed/drift_synth_prior_cal")
        CONFIG = "configs/drift_synth/hsmm_cal.yaml"
        TICK, TRAIN_DAYS, SOFT_ITERS, TAG = 1.0, (10, 30, 200), 20, "_cal"


def _assets(person_cfg):
    from cook_ad.drift_synth.assets import load_assets
    a = load_assets()
    a.gen["recognizer"] = {k: (v if k == "min_event_s" else 0.0) for k, v in a.gen["recognizer"].items()}
    a.gen["person"] = dict(a.gen["person"], **person_cfg)
    if VARIANT == "cal":
        from cook_ad.drift_synth.assets import load_step_scale
        a.gen["durations"] = dict(a.gen["durations"], step_scale="configs/drift_synth/step_time_scale.json",
                                  max_event_s=600.0)
        a.step_scale = load_step_scale(a.gen)
        a.gen["export"] = dict(a.gen["export"], tick_seconds=TICK)
    return a


def stage_population():
    from cook_ad.drift_synth.export import export
    from cook_ad.drift_synth.sampler import ConditionCfg, generate_person
    from cook_ad.hsmm.train.global_prior import count_global

    assets = _assets({"theta0": [0.05, 0.95], "theta0_by_knob": CAPABILITY, "habit": [0.0, 1.0]})
    cond = ConditionCfg("null", n_persons=POP_N, n_days=POP_DAYS)
    seqs, recipes, n = [], [], 0
    for i in range(POP_N):
        person, sessions, gt = generate_person(i, cond, POP_SEED, assets)
        out = ROOT / "population" / person.person_id
        export(sessions, out, assets.vocab, gt, assets.gen["export"])
        labels = {e["trial_id"]: e["recipe_label"] for e in json.loads((out / "labels.json").read_text())}
        for s in json.loads((out / "sequences.json").read_text()):
            seqs.append(s)
            recipes.append(labels[s["trial_id"]])
        n += len(sessions)
    counts = count_global(seqs, recipes)
    (ROOT / "global_counts.json").write_text(json.dumps(counts, sort_keys=True))
    print(f"population: {POP_N} people, {n} sessions; sessions per recipe "
          f"{dict(sorted(counts['n_trials'].items()))}")


def stage_persons():
    from cook_ad.drift_synth.export import export
    from cook_ad.drift_synth.sampler import ConditionCfg, generate_person

    assets = _assets({"theta0_by_knob": CAPABILITY, "habit": 2.0})
    cond = ConditionCfg("null", n_persons=1, n_days=N_DAYS)
    for i in PERSONS:
        person, sessions, gt = generate_person(i, cond, SEED, assets)
        d = ROOT / "persons" / person.person_id
        export(sessions, d / "null", assets.vocab, gt, assets.gen["export"])
        seqs = json.loads((d / "null" / "sequences.json").read_text())
        day = {e["trial_id"]: e["day"] for e in json.loads((d / "null" / "labels.json").read_text())}
        for n in TRAIN_DAYS:
            train = [s["trial_id"] for s in seqs if day[s["trial_id"]] < n]
            test = [s["trial_id"] for s in seqs if day[s["trial_id"]] >= VAL_START]
            (d / "null" / f"split_{n}.json").write_text(json.dumps(
                {"seed": SEED, "test_frac": None, "train_trial_ids": train, "test_trial_ids": test,
                 "rule": f"train = days < {n}; held out = days >= {VAL_START}"}, indent=1))
        print(f"{person.person_id}: {len(sessions)} sessions, repertoire {list(person.repertoire)}, "
              f"habit {person.habit}")


def stage_train(persons, days):
    for i in persons:
        d = ROOT / "persons" / f"p{i:02d}"
        for n in days:
            out = d / f"model_{n}"
            if (out / "model.npz").exists():
                print(f"p{i:02d} N={n}: exists")
                continue
            out.mkdir(parents=True, exist_ok=True)
            common = ["--config", CONFIG, "--sequences", str(d / "null" / "sequences.json"),
                      "--labels", str(d / "null" / "labels.json"),
                      "--split-file", str(d / "null" / f"split_{n}.json"), "--recipe-labels"]
            steps = [
                ["scripts/train/fit_lexical.py", *common, "--init-prior-scale", "0.0", "--anchor",
                 "50", "--max-iters", str(SOFT_ITERS), "--out", str(out / "soft.npz")],
                ["scripts/train/fit_hard_em.py", *common, "--init-from", str(out / "soft.npz"),
                 "--keep-init-emissions", "--iters", "5", "--out", str(out / "hard.npz")],
                ["scripts/train/smooth_params.py", "--in", str(out / "hard.npz"),
                 "--out", str(out / "smooth.npz"), "--strength", "0.7", "--backoff-tau", "30"],
                ["scripts/train/refit_durations.py", *common, "--in", str(out / "smooth.npz"),
                 "--out", str(out / "model.npz"), "--kappa", "0.001"],
            ]
            with open(out / "train.log", "w") as log:
                for cmd in steps:
                    subprocess.run(["./py", *cmd], check=True, stdout=log, stderr=subprocess.STDOUT)
            print(f"p{i:02d} N={n}: trained", flush=True)


def stage_eval(persons, chunk_size=8):
    from cook_ad.xla_env import disable_gpu_autotuning
    disable_gpu_autotuning()
    import jax
    import numpy as np
    jax.config.update("jax_enable_x64", True)
    from cook_ad.core import OperatingPoint
    from cook_ad.data.config import load_config
    from cook_ad.detectors.hsmm_hindsight import HindsightHSMMDetector
    from cook_ad.detectors.thresholds import BadSets
    from cook_ad.eval import harness, pools
    from cook_ad.hsmm import params as hparams
    from cook_ad.hsmm.model import HSMM
    from cook_ad.hsmm.train.global_prior import apply_global_prior
    from cook_ad.hsmm.train.lexical_init import recipe_label_assignment

    d_max = load_config(CONFIG)["duration"]["d_max_ticks"]
    gcounts = json.loads((ROOT / "global_counts.json").read_text())
    results = {}
    for i in persons:
        pid = f"p{i:02d}"
        d = ROOT / "persons" / pid
        seqs = json.loads((d / "null" / "sequences.json").read_text())
        labels = json.loads((d / "null" / "labels.json").read_text())
        day = {e["trial_id"]: e["day"] for e in labels}
        _, names = recipe_label_assignment(seqs, labels)
        placement = hparams.load(str(d / "model_200" / "model.npz"))

        def pool(lo, hi):
            part = [s for s in seqs if lo <= day[s["trial_id"]] < hi]
            entries, keep = pools.injected_per_trial(placement, part, d_max, seed=SEED,
                                                     chunk_size=chunk_size)
            streams = pools.per_trial_to_streams(entries)
            dish = recipe_label_assignment([part[k] for k in keep], labels)[0]
            return harness.trials_from_streams(streams, dish)

        val, test = pool(VAL_START, TEST_START), pool(TEST_START, N_DAYS)

        def masks(det, trials, grid):
            out = []
            for group in dict.fromkeys(g for g, _ in trials):
                ts = [t for g, t in trials if g == group]
                if grid:
                    out += [harness.hindsight_grid(det, tr, harness.ALPHAS) for tr in det.traces(ts)]
                else:
                    out += [harness.alarm_mask(x, len(t.verbs)) for x, t in zip(det.detect_many(ts), ts)]
            return out

        results[pid] = {}
        for n in TRAIN_DAYS:
            base = hparams.load(str(d / f"model_{n}" / "model.npz"))
            arms = {f"lambda={lam:g}": apply_global_prior(base, gcounts, names, lam) for lam in LAMBDAS}
            arms["global only"] = apply_global_prior(base, gcounts, names, GLOBAL_ONLY_LAMBDA, replace=True)
            results[pid][n] = {}
            for arm, p in arms.items():
                hsmm = HSMM(p, d_max)
                bad = BadSets(hsmm, harness.ALPHAS)
                det = HindsightHSMMDetector(hsmm, bad, OperatingPoint(0.005), chunk_size=chunk_size)
                grid = harness.sweep(masks(det, val, True), val, TICK)
                idx = harness.pick(grid)
                alpha = harness.ALPHAS[idx[0]]
                det = HindsightHSMMDetector(hsmm, bad, OperatingPoint(alpha), chunk_size=chunk_size)
                row = harness.score_masks(masks(det, test, False), test, TICK)
                row["alpha"] = alpha
                results[pid][n][arm] = row
                print(f"{pid} N={n:3d} {arm:12s} alpha={alpha:<7g} F1={row['f1']:.3f} "
                      f"P={row['precision']:.3f} R={row['recall']:.3f} "
                      f"healthy-trial FPR={row['healthy_trial_fpr']:.2f} "
                      f"alarms/10min={row['alarms_per_10min']:.2f}", flush=True)
        harness.dump(f"runs/drift_synth/eval_synth_prior{TAG}_{pid}.json",
                     {"metric": "G", "person": pid, "results": results[pid]})
    return results


FADE_ARMS = {  # arm -> (lam, n0); every arm picks the transition alpha separately
    "person only": (0.0, None),
    "prior lambda=1": (1.0, None),
    "fade lambda0=10 n0=3": (10.0, 3.0),
    "fade lambda0=10 n0=10": (10.0, 10.0),
    "fade lambda0=30 n0=3": (30.0, 3.0),
}


def stage_eval_split(persons, chunk_size=8):
    """Every arm with a split operating point: the transition channels' alpha picked separately
    from the other channels' (harness.hindsight_split_grid), both on validation by G-F1."""
    from cook_ad.xla_env import disable_gpu_autotuning
    disable_gpu_autotuning()
    import jax
    jax.config.update("jax_enable_x64", True)
    from cook_ad.core import OperatingPoint
    from cook_ad.data.config import load_config
    from cook_ad.detectors.hsmm_hindsight import HindsightHSMMDetector
    from cook_ad.detectors.thresholds import BadSets
    from cook_ad.eval import harness, pools
    from cook_ad.hsmm import params as hparams
    from cook_ad.hsmm.model import HSMM
    from cook_ad.hsmm.train.global_prior import apply_global_prior, count_global
    from cook_ad.hsmm.train.lexical_init import recipe_label_assignment
    import numpy as np

    d_max = load_config(CONFIG)["duration"]["d_max_ticks"]
    gcounts = json.loads((ROOT / "global_counts.json").read_text())
    for i in persons:
        pid = f"p{i:02d}"
        d = ROOT / "persons" / pid
        seqs = json.loads((d / "null" / "sequences.json").read_text())
        labels = json.loads((d / "null" / "labels.json").read_text())
        day = {e["trial_id"]: e["day"] for e in labels}
        rec = {e["trial_id"]: e["recipe_label"] for e in labels}
        _, names = recipe_label_assignment(seqs, labels)
        placement = hparams.load(str(d / "model_200" / "model.npz"))

        def pool(lo, hi):
            part = [s for s in seqs if lo <= day[s["trial_id"]] < hi]
            entries, keep = pools.injected_per_trial(placement, part, d_max, seed=SEED,
                                                     chunk_size=chunk_size)
            dish = recipe_label_assignment([part[k] for k in keep], labels)[0]
            return harness.trials_from_streams(pools.per_trial_to_streams(entries), dish)

        val, test = pool(VAL_START, TEST_START), pool(TEST_START, N_DAYS)
        out = {}
        for n in TRAIN_DAYS:
            base = hparams.load(str(d / f"model_{n}" / "model.npz"))
            train = [s for s in seqs if day[s["trial_id"]] < n]
            person = count_global(train, [rec[s["trial_id"]] for s in train])
            out[n] = {}
            for arm, (lam, n0) in FADE_ARMS.items():
                p = apply_global_prior(base, gcounts, names, lam, person_counts=person if n0 else None,
                                       n0=n0)
                hsmm = HSMM(p, d_max)
                bad = BadSets(hsmm, harness.ALPHAS)
                det = HindsightHSMMDetector(hsmm, bad, OperatingPoint(0.005), chunk_size=chunk_size)

                def grid(trials):
                    m = []
                    for group in dict.fromkeys(g for g, _ in trials):
                        ts = [t for g, t in trials if g == group]
                        m += [harness.hindsight_split_grid(det, tr, harness.ALPHAS) for tr in det.traces(ts)]
                    return harness.sweep(m, trials, TICK)
                gv, gt = grid(val), grid(test)
                idx = harness.pick(gv)
                row = harness.at(gt, idx)
                row["alpha_transition"] = harness.ALPHAS[idx[0]]
                row["alpha_rest"] = harness.ALPHAS[idx[1]]
                out[n][arm] = row
                # the single-alpha pick is the grid's diagonal (same alpha for every channel)
                diag = [float(np.asarray(gv["f1"])[j, j]) for j in range(len(harness.ALPHAS))]
                j = int(np.argmax(diag))
                single = harness.at(gt, (j, j))
                single["alpha_transition"] = single["alpha_rest"] = harness.ALPHAS[j]
                out[n][arm + " (single alpha)"] = single
                print(f"{pid} N={n:3d} {arm:22s} a_trans={row['alpha_transition']:<7g} "
                      f"a_rest={row['alpha_rest']:<7g} F1={row['f1']:.3f} P={row['precision']:.3f} "
                      f"R={row['recall']:.3f} FPR={row['healthy_trial_fpr']:.2f} "
                      f"alarms/10min={row['alarms_per_10min']:.2f}", flush=True)
        harness.dump(f"runs/drift_synth/eval_synth_prior_split{TAG}_{pid}.json",
                     {"metric": "G", "person": pid, "arms": {k: list(v) for k, v in FADE_ARMS.items()},
                      "results": out})


def summarize_split():
    import numpy as np
    rows = {}
    for f in sorted(Path("runs/drift_synth").glob(f"eval_synth_prior_split{TAG}_p*.json")):
        for n, arms in json.loads(f.read_text())["results"].items():
            for arm, r in arms.items():
                rows.setdefault((int(n), arm), []).append(r)
    print("\n### Split alpha (transition channels picked separately), mean over persons, test "
          "days, metric G (Hindsight)\n")
    print("| train days | arm | persons | F1 | precision | recall | alarms / 10 min | healthy-trial FPR "
          "| subst. | aband. | omission | transp. | repet. |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    order = [x for a in FADE_ARMS for x in (a + " (single alpha)", a)]
    for (n, arm), rs in sorted(rows.items(), key=lambda kv: (kv[0][0], order.index(kv[0][1])
                                                              if kv[0][1] in order else 99)):
        m = lambda k: np.mean([r[k] for r in rs])  # noqa: E731
        t = lambda k: np.mean([r["per_type"][k] for r in rs])  # noqa: E731
        print(f"| {n} | {arm} | {len(rs)} | {m('f1'):.3f} | {m('precision'):.3f} | {m('recall'):.3f} | "
              f"{m('alarms_per_10min'):.2f} | {m('healthy_trial_fpr'):.2f} | {t('substitution'):.2f} | "
              f"{t('abandonment'):.2f} | {t('omission'):.2f} | {t('transposition'):.2f} | "
              f"{t('repetition'):.2f} |")


def summarize():
    import numpy as np
    rows = {}
    for f in sorted(Path("runs/drift_synth").glob(f"eval_synth_prior{TAG}_p*.json")):
        res = json.loads(f.read_text())["results"]
        for n, arms in res.items():
            for arm, r in arms.items():
                rows.setdefault((int(n), arm), []).append(r)
    print("\n### Global prior + fine-tune, mean over persons, test days, metric G (Hindsight)\n")
    print("| train days | arm | persons | F1 | precision | recall | alarms / 10 min | healthy-trial FPR "
          "| subst. | aband. | omission | transp. | repet. |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for (n, arm), rs in sorted(rows.items(), key=lambda kv: (kv[0][0], list(rows).index(kv[0]))):
        m = lambda k: np.mean([r[k] for r in rs])  # noqa: E731
        t = lambda k: np.mean([r["per_type"][k] for r in rs])  # noqa: E731
        print(f"| {n} | {arm} | {len(rs)} | {m('f1'):.3f} | {m('precision'):.3f} | {m('recall'):.3f} | "
              f"{m('alarms_per_10min'):.2f} | {m('healthy_trial_fpr'):.2f} | {t('substitution'):.2f} | "
              f"{t('abandonment'):.2f} | {t('omission'):.2f} | {t('transposition'):.2f} | "
              f"{t('repetition'):.2f} |")


def plot(path=None):
    """Cold-start curves: F1, healthy-trial FPR and recall vs training days, per person (thin)
    and mean (thick), for person-only, + prior (lambda = 1) and global-only."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    arms = [("lambda=0", "person only", "#2a78d6"), ("lambda=1", "+ global prior (λ = 1)", "#eb6834"),
            ("global only", "global only", "#1baf7a")]
    path = path or f"runs/drift_synth/reports/prior_curves{TAG}.png"
    res = {f.stem.split("_")[-1]: json.loads(f.read_text())["results"]
           for f in sorted(Path("runs/drift_synth").glob(f"eval_synth_prior{TAG}_p*.json"))}
    days = list(TRAIN_DAYS)
    ink, ink2, grid = "#1a1a19", "#5f5e58", "#e4e3dd"
    panels = [("f1", "F1 (injected errors)"), ("healthy_trial_fpr", "healthy-trial FPR"),
              ("recall", "recall")]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.4), constrained_layout=True)
    for ax, (key, title) in zip(axes[:3], panels):
        ax.grid(True, color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for arm, label, color in arms:
            per = np.array([[res[p][str(n)][arm][key] for n in days] for p in sorted(res)])
            for row in per:
                ax.plot(days, row, color=color, linewidth=0.8, alpha=0.35)
            ax.plot(days, per.mean(axis=0), color=color, linewidth=2.2, marker="o", markersize=5,
                    label=label)
        ax.set_xscale("log")
        ax.set_xticks(days, [f"{n}\n(~{round(n * 1.3)} sess.)" for n in days])
        ax.tick_params(colors=ink2, labelsize=8)
        ax.set_title(title, color=ink, fontsize=10, loc="left")
        ax.set_xlabel("person's own training days", color=ink2, fontsize=9)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=ink)
    # per person: F1(+prior) - F1(person only); hollow = the prior arm's validation pick left
    # the strict end of the alpha grid (alpha > 1e-7)
    ax = axes[3]
    ax.grid(True, color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.axhline(0, color=ink2, linewidth=1)
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    for color, pid in zip(colors, sorted(res)):
        delta = [res[pid][str(n)]["lambda=1"]["f1"] - res[pid][str(n)]["lambda=0"]["f1"] for n in days]
        loose = [res[pid][str(n)]["lambda=1"]["alpha"] > 1e-7 for n in days]
        ax.plot(days, delta, color=color, linewidth=1.6)
        for x, y, l in zip(days, delta, loose):
            ax.plot(x, y, marker="o", markersize=7, color=color, markerfacecolor="white" if l else color,
                    markeredgewidth=2)
        ax.text(days[-1] * 1.08, delta[-1], pid, color=ink2, fontsize=8, va="center")
    ax.plot([], [], marker="o", color=ink2, linestyle="none", markerfacecolor="white",
            label="prior arm picked a loose α (> 1e-7)")
    ax.plot([], [], marker="o", color=ink2, linestyle="none", label="picked α ≤ 1e-7")
    ax.legend(frameon=False, fontsize=8, labelcolor=ink, loc="lower left")
    ax.set_xscale("log")
    ax.set_xticks(days, [str(n) for n in days])
    ax.set_xlim(days[0] * 0.8, days[-1] * 1.5)
    ax.tick_params(colors=ink2, labelsize=8)
    ax.set_title("F1 change from the prior, per person", color=ink, fontsize=10, loc="left")
    ax.set_xlabel("person's own training days", color=ink2, fontsize=9)
    fig.suptitle(f"Global prior + fine-tune, {len(res)} synthetic persons (thin) and mean (thick); "
                 "Hindsight, test days, metric G", color=ink, fontsize=11, x=0.01, ha="left")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


def plot_split(path=None):
    """Mean over persons vs training days for the split-alpha arms (and person-only, single alpha)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    path = path or f"runs/drift_synth/reports/prior_split_curves{TAG}.png"
    res = {f.stem.split("_")[-1]: json.loads(f.read_text())["results"]
           for f in sorted(Path("runs/drift_synth").glob(f"eval_synth_prior_split{TAG}_p*.json"))}
    if not res:
        raise SystemExit("no split results")
    days = sorted(int(n) for n in next(iter(res.values())))
    palette = ["#5f5e58", "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    arms = [a for a in ["person only (single alpha)", *FADE_ARMS]
            if all(a in res[p][str(n)] for p in res for n in days)]
    ink, ink2, grid = "#1a1a19", "#5f5e58", "#e4e3dd"
    panels = [("f1", "F1 (injected errors)"), ("healthy_trial_fpr", "healthy-trial FPR"), ("recall", "recall")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)
    for ax, (key, title) in zip(axes, panels):
        ax.grid(True, color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for color, arm in zip(palette, arms):
            m = [np.mean([res[p][str(n)][arm][key] for p in res]) for n in days]
            ax.plot(days, m, color=color, linewidth=2.2 if "single" not in arm else 1.4,
                    linestyle="--" if "single" in arm else "-", marker="o", markersize=5,
                    label=arm + ("" if "single" in arm else ", split alpha"))
        ax.set_xscale("log")
        ax.set_xticks(days, [str(n) for n in days])
        ax.tick_params(colors=ink2, labelsize=8)
        ax.set_title(title, color=ink, fontsize=10, loc="left")
        ax.set_xlabel("person's own training days", color=ink2, fontsize=9)
    axes[0].legend(frameon=False, fontsize=7.5, labelcolor=ink)
    fig.suptitle(f"Fading prior + split alpha, mean of {len(res)} persons{' (CC4D-calibrated)' if TAG else ''}; "
                 "Hindsight, test days, metric G", color=ink, fontsize=11, x=0.01, ha="left")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["population", "persons", "train", "eval", "summary", "plot",
                                        "eval-split", "summary-split", "plot-split"],
                    required=True)
    ap.add_argument("--person", type=int, action="append")
    ap.add_argument("--days", type=int, action="append")
    ap.add_argument("--variant", choices=["base", "cal"], default="base")
    a = ap.parse_args()
    set_variant(a.variant)
    persons = a.person or PERSONS
    if a.stage == "population":
        stage_population()
    elif a.stage == "persons":
        stage_persons()
    elif a.stage == "train":
        stage_train(persons, a.days or TRAIN_DAYS)
    elif a.stage == "eval":
        stage_eval(persons)
        summarize()
    elif a.stage == "plot":
        plot()
    elif a.stage == "eval-split":
        stage_eval_split(persons)
        summarize_split()
    elif a.stage == "summary-split":
        summarize_split()
    elif a.stage == "plot-split":
        plot_split()
    else:
        summarize()


if __name__ == "__main__":
    main()
