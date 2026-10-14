"""Summarise run_live_eval.py: pick each arm's operating point on TRAIN (max trial_loc accuracy,
the repo's convention -- always-flag scores F1 0.625, so accuracy is what we quote), then report
that same point on TEST. Prints a markdown table, per-type recall, and per-channel healthy FPR.

    ./py tools_live_summary.py --metric-g runs/live_dishknownG_train.json runs/live_dishknownG_test.json
    ./py tools_live_summary.py runs/live_lexhard_train.json runs/live_lexhard_test.json \
        runs/live_oracle_train.json:oracle runs/live_oracle_test.json:oracle
"""
import json
import sys

import numpy as np

CHANNELS = ["s_emit", "s_verb", "s_noun", "s_temporal", "s_dur_two", "s_transition",
            "s_recipe_transition"]


def _pick(arm):
    acc = np.asarray(arm["raw"]["accuracy"])
    return np.unravel_index(int(np.argmax(acc)), acc.shape)


def _at(block, idx):
    return float(np.asarray(block)[idx])


def _load(spec):
    """'path' or 'path:prefix' -- several runs can be merged into one table; a prefix keeps
    their arm names apart (e.g. oracle:...)."""
    path, _, prefix = spec.partition(":")
    d = json.load(open(path))
    if prefix:
        d["arms"] = {f"{prefix}/{k}": v for k, v in d["arms"].items()}
    return d


def main(*specs):
    pairs = [(_load(specs[i]), _load(specs[i + 1])) for i in range(0, len(specs), 2)]
    tr, te = pairs[0]
    for a, b in pairs[1:]:
        tr["arms"].update(a["arms"])
        te["arms"].update(b["arms"])
        for lag, per in b["mean_switch_mass_per_trial"].items():
            te["mean_switch_mass_per_trial"].setdefault(lag, per)
    alphas, kappas = te["alphas"], te["kappas"]
    print(f"train n={tr['n_usable']}  test n={te['n_usable']}  (trials per group)\n")
    print("| arm | alpha | kappa | train acc | test acc | test prec | test rec | test healthy FPR |")
    print("|---|---|---|---|---|---|---|---|")
    picks = {}
    for name in te["arms"]:
        idx = _pick(tr["arms"][name])
        picks[name] = idx
        r = te["arms"][name]["raw"]
        kap = f"{kappas[idx[1]]}" if len(idx) > 1 else "—"
        print(f"| {name} | {alphas[idx[0]]:g} | {kap} | {_at(tr['arms'][name]['raw']['accuracy'], idx):.3f} "
              f"| {_at(r['accuracy'], idx):.3f} | {_at(r['precision'], idx):.3f} | {_at(r['recall'], idx):.3f} "
              f"| {_at(r['healthy_fpr'], idx):.3f} |")

    types = [g for g in next(iter(te["arms"].values()))["raw"]["per_type"] if g != "healthy"]
    print("\nTest recall per injected type, at the train-picked point:\n")
    print("| arm | " + " | ".join(types) + " |")
    print("|---|" + "---|" * len(types))
    for name, idx in picks.items():
        pt = te["arms"][name]["raw"]["per_type"]
        print(f"| {name} | " + " | ".join(f"{_at(pt[g]['recall'], idx):.2f}" for g in types) + " |")

    print("\nTest healthy FPR per channel, at the train-picked point:\n")
    print("| arm | " + " | ".join(c[2:] for c in CHANNELS) + " |")
    print("|---|" + "---|" * len(CHANNELS))
    for name, idx in picks.items():
        a = te["arms"][name]
        print(f"| {name} | " + " | ".join(f"{_at(a[c]['healthy_fpr'], idx):.2f}" for c in CHANNELS) + " |")

    print("\nMean total switch probability per trial (live filter, sum over ticks), by group:")
    for lag, per in te["mean_switch_mass_per_trial"].items():
        print(f"  L={lag}: " + "  ".join(f"{g}={v:.2f}" for g, v in per.items()))


BUDGETS = (0.5, 1.0, 2.0)   # alarm onsets per 10 min on healthy trials


def main_g(*specs):
    """Metric G (span_metrics.METRIC_G). Operating point picked on TRAIN by G-F1, and separately
    the best TRAIN recall within each healthy false-alarm budget; both reported on TEST."""
    pairs = [(_load_g(specs[i]), _load_g(specs[i + 1])) for i in range(0, len(specs), 2)]
    tr = {k: v for a, _ in pairs for k, v in a["metric_g"].items()}
    te = {k: v for _, b in pairs for k, v in b["metric_g"].items()}
    alphas, kappas = pairs[0][1]["alphas"], pairs[0][1]["kappas"]

    def op(idx):
        return f"{alphas[idx[0]]:g}" + (f", κ={kappas[idx[1]]}" if len(idx) > 1 else "")

    def row(name, idx, a):
        g = lambda k: float(np.asarray(a[k])[idx])
        return (f"| {name} | {op(idx)} | {g('precision'):.3f} | {g('recall'):.3f} | {g('f1'):.3f} "
                f"| {g('alarms_per_10min'):.2f} | {g('run_fpr' if 'run_fpr' in a else 'step_fpr'):.3f} | {g('healthy_trial_fpr'):.3f} |")

    head = ("| arm | α (κ) picked on train | precision | recall | F1 | alarms / 10 min | run FPR "
            "| healthy-trial FPR |\n|---|---|---|---|---|---|---|---|")
    print("Metric G (alarm = first flag per run of constant (verb, noun); deadline 10). "
          "Picked on train by F1, reported on test.\n")
    print(head)
    picks = {}
    for name in te:
        f1 = np.asarray(tr[name]["f1"])
        idx = np.unravel_index(int(np.argmax(f1)), f1.shape)
        picks[name] = idx
        print(row(name, idx, te[name]))
    types = list(next(iter(te.values()))["per_type_recall"])
    print("\nTest recall per injected type at the F1 pick:\n")
    print("| arm | " + " | ".join(types) + " |\n|---|" + "---|" * len(types))
    for name, idx in picks.items():
        pt = te[name]["per_type_recall"]
        print(f"| {name} | " + " | ".join(f"{float(np.asarray(pt[t])[idx]):.2f}" for t in types) + " |")
    for b in BUDGETS:
        print(f"\nBest train recall with train alarms/10min <= {b}, reported on test:\n")
        print(head)
        for name in te:
            ok = np.asarray(tr[name]["alarms_per_10min"]) <= b
            if not ok.any():
                print(f"| {name} | (no point within budget) | | | | | | |")
                continue
            rec = np.where(ok, np.asarray(tr[name]["recall"]), -1)
            idx = np.unravel_index(int(np.argmax(rec)), rec.shape)
            print(row(name, idx, te[name]))


def _load_g(spec):
    path, _, prefix = spec.partition(":")
    d = json.load(open(path))
    if prefix:
        d["metric_g"] = {f"{prefix}/{k}": v for k, v in d["metric_g"].items()}
    return d


if __name__ == "__main__":
    if sys.argv[1:2] == ["--metric-g"]:
        main_g(*sys.argv[2:])
    else:
        main(*sys.argv[1:])
