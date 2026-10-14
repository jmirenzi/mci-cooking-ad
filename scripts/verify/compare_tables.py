"""Compare the new eval scripts' tables with the golden ones (docs/restructure.md §9.2).
TEMPORARY: deleted in Phase G.

    ./py scripts/verify/compare_tables.py runs/golden runs/golden_E

  * EPIC-Tent reports (et_*.json): every golden report row the new script still produces, to 3
    decimals. Rows of deleted metrics are skipped and listed: `@merge_tol=0` (G-gap) and
    `control/hsmm-injected` (it was scored under G-gap; it is now scored under G).
  * Breakfast: eval_breakfast.json's train/test grids against run_live_eval's metric_g grids
    (live_golden_<tag>_<part>.json), its train-picked operating points against the same argmax
    on the golden train grids, and its LLM-pool table against breakfast_g_table.json.
"""
import json
import os
import sys

import numpy as np

SKIP_ROWS = ("@merge_tol=0", "control/hsmm-injected")
IGNORE = {"merge_tol", "client"}
TAGS = {"lexhard": "golden_lexhard", "dish-given": "golden_dishknown", "dish-inferred": "golden_dish"}
GRID_KEYS = ("precision", "recall", "f1", "alarms_per_10min", "healthy_trial_fpr", "run_fpr")


def cmp(a, b, path, bad):
    if isinstance(a, dict):
        for k in a:
            if k in IGNORE:
                continue
            if not isinstance(b, dict) or k not in b:
                bad.append(f"{path}/{k}: missing in new")
            else:
                cmp(a[k], b[k], f"{path}/{k}", bad)
    elif isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b):
            bad.append(f"{path}: list length")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                cmp(x, y, f"{path}[{i}]", bad)
    elif isinstance(a, float) or isinstance(b, float):
        if a is None or b is None:
            if a is not b:
                bad.append(f"{path}: {a} vs {b}")
        elif not (np.isnan(a) and np.isnan(b)) and round(float(a), 3) != round(float(b), 3):
            bad.append(f"{path}: {a} vs {b}")
    elif a != b:
        bad.append(f"{path}: {a!r} vs {b!r}")


def epic_tent(gold, new):
    fails = 0
    for f in sorted(os.listdir(gold)):
        if not (f.startswith("et_") and f.endswith(".json")) or not os.path.exists(f"{new}/{f}"):
            continue
        a, b = json.load(open(f"{gold}/{f}")), json.load(open(f"{new}/{f}"))
        bad, skipped = [], []
        for key, rep in a["reports"].items():
            if any(s in key for s in SKIP_ROWS):
                skipped.append(key)
                continue
            if key not in b["reports"]:
                bad.append(f"{key}: row missing in new")
                continue
            cmp(rep, b["reports"][key], key, bad)
        fails += bool(bad)
        print(f"{'PASS' if not bad else 'FAIL'}  {f}  ({len(a['reports']) - len(skipped)} rows; "
              f"skipped {len(skipped)}: {sorted(set(s.split('/')[-1] for s in skipped))})")
        for line in bad[:15]:
            print(f"    {line}")
    return fails


def breakfast(gold, new):
    path = f"{new}/eval_breakfast.json"
    if not os.path.exists(path):
        return 0
    e = json.load(open(path))
    fails = 0
    for model, tag in TAGS.items():
        for part in ("train", "test"):
            gpath = f"{gold}/live_{tag}_{part}.json"
            if model not in e["grids"] or part not in e["grids"][model] or not os.path.exists(gpath):
                continue
            g = json.load(open(gpath))["metric_g"]
            bad = []
            for arm, grid in e["grids"][model][part].items():
                if arm not in g:
                    continue
                for k in GRID_KEYS:
                    x, y = np.round(np.asarray(g[arm][k], float), 3), np.round(np.asarray(grid[k], float), 3)
                    if x.shape != y.shape or not np.array_equal(x, y):
                        bad.append(f"{arm}/{k}: max diff {np.max(np.abs(x - y)) if x.shape == y.shape else 'shape'}")
                for t, v in g[arm]["per_type_recall"].items():
                    if not np.array_equal(np.round(np.asarray(v), 3), np.round(np.asarray(grid["per_type_recall"][t]), 3)):
                        bad.append(f"{arm}/per_type_recall/{t}")
            if part == "train":
                for arm, op in e["operating_points"][model].items():
                    if arm in g:
                        idx = np.unravel_index(int(np.argmax(np.asarray(g[arm]["f1"]))),
                                               np.shape(g[arm]["f1"]))
                        if list(idx) != list(op["index"]):
                            bad.append(f"pick {arm}: golden {idx} vs new {op['index']}")
            fails += bool(bad)
            print(f"{'PASS' if not bad else 'FAIL'}  breakfast grids {model}/{part}")
            for line in bad[:10]:
                print(f"    {line}")
    gt = f"{gold}/breakfast_g_table.json"
    if os.path.exists(gt):
        a, bad = json.load(open(gt)), []
        for row, r in a.items():
            if row not in e["table"]:
                bad.append(f"{row}: row missing in new")
                continue
            cmp(r, e["table"][row], row, bad)
        fails += bool(bad)
        print(f"{'PASS' if not bad else 'FAIL'}  breakfast headline table ({len(a)} golden rows; "
              f"new-only rows: {sorted(set(e['table']) - set(a))})")
        for line in bad[:15]:
            print(f"    {line}")
    return fails


def main():
    gold, new = sys.argv[1:3]
    fails = epic_tent(gold, new) + breakfast(gold, new)
    print("\nALL PASS" if not fails else f"\n{fails} FAIL")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
