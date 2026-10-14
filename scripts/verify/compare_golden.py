"""Compare two golden dumps (docs/restructure.md §9.2). TEMPORARY: deleted in Phase G.

    ./py scripts/verify/compare_golden.py runs/golden runs/golden_B

Rules, per file present in BOTH dirs:
  *.pkl  bool and int arrays, ints, strings, flag lists: exact. Floats named in LL_KEYS: relative
         1e-9. Every other float (diagnostics): reported if it differs by more than 1e-9 relative,
         but does not fail the check on its own -- the flags it feeds are what must agree.
  *.json numbers rounded to 3 decimals must agree (the printed tables' precision); other leaves
         exact. Keys in IGNORE_JSON (paths, timings) are skipped.
Exit status 1 if any hard check fails. Files only in one dir are listed.
"""
import json
import os
import pickle
import sys

import numpy as np

LL_KEYS = {"trial_ll", "live_ll", "log_norm"}
LL_RTOL = 1e-9
DIAG_RTOL = 1e-9
IGNORE_JSON = {"out", "out_dir", "figures_dir", "dump_streams", "cache_dir", "tag", "seconds",
               "elapsed", "client"}


class _Unpickler(pickle.Unpickler):
    """Goldens made before the restructure name cook_ad.eval.span_metrics.Flag."""

    def find_class(self, module, name):
        if module == "cook_ad.eval.span_metrics":
            module = "cook_ad.eval.metric_g"
        return super().find_class(module, name)


class Report:
    def __init__(self):
        self.hard, self.soft, self.checked = [], [], 0

    def fail(self, path, msg):
        self.hard.append(f"{path}: {msg}")

    def note(self, path, msg):
        self.soft.append(f"{path}: {msg}")


def _rel(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    both_inf = np.isinf(a) & np.isinf(b) & (np.sign(a) == np.sign(b))
    both_nan = np.isnan(a) & np.isnan(b)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.abs(a - b) / np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-300)
    r = np.where(both_inf | both_nan, 0.0, r)
    return float(np.nanmax(np.where(np.isnan(r), np.inf, r))) if r.size else 0.0


def cmp_pkl(a, b, path, rep, key=""):
    rep.checked += 1
    if isinstance(a, dict):
        if not isinstance(b, dict) or set(a) != set(b):
            return rep.fail(path, f"keys differ: {sorted(map(str, set(a) ^ set(b)))[:10]}")
        for k in a:
            cmp_pkl(a[k], b[k], f"{path}/{k}", rep, str(k))
        return
    if isinstance(a, (list, tuple)) and not (a and isinstance(a[0], (float, np.floating))):
        if not isinstance(b, (list, tuple)) or len(a) != len(b):
            return rep.fail(path, f"length {len(a)} vs {len(b) if hasattr(b, '__len__') else b}")
        for i, (x, y) in enumerate(zip(a, b)):
            cmp_pkl(x, y, f"{path}[{i}]", rep, key)
        return
    aa, bb = np.asarray(a), np.asarray(b) if b is not None else None
    if a is None or b is None:
        if a is not b:
            rep.fail(path, f"{a!r} vs {b!r}")
        return
    if aa.dtype.kind in "fc" or bb.dtype.kind in "fc":
        if aa.shape != bb.shape:
            return rep.fail(path, f"shape {aa.shape} vs {bb.shape}")
        r = _rel(aa, bb)
        if key in LL_KEYS and r > LL_RTOL:
            rep.fail(path, f"log-likelihood rel diff {r:.2e}")
        elif r > DIAG_RTOL:
            rep.note(path, f"float rel diff {r:.2e}")
        return
    if aa.shape != bb.shape or not np.array_equal(aa, bb):
        if aa.dtype == bool and aa.shape == bb.shape:
            n = int(np.sum(aa != bb))
            return rep.fail(path, f"{n} of {aa.size} flags differ")
        rep.fail(path, f"differs ({aa.dtype}{aa.shape} vs {bb.dtype}{bb.shape})")


def cmp_json(a, b, path, rep):
    rep.checked += 1
    if isinstance(a, dict):
        keys = (set(a) | set(b if isinstance(b, dict) else {})) - IGNORE_JSON
        if not isinstance(b, dict):
            return rep.fail(path, "dict vs non-dict")
        for k in sorted(keys):
            if k not in a or k not in b:
                rep.fail(f"{path}/{k}", "missing on one side")
            else:
                cmp_json(a[k], b[k], f"{path}/{k}", rep)
    elif isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b):
            return rep.fail(path, "list length differs")
        for i, (x, y) in enumerate(zip(a, b)):
            cmp_json(x, y, f"{path}[{i}]", rep)
    elif isinstance(a, float) or isinstance(b, float):
        if a is not None and b is not None and np.isnan(a) and np.isnan(b):
            return
        if a is None or b is None or round(float(a), 3) != round(float(b), 3):
            rep.fail(path, f"{a} vs {b}")
    elif a != b:
        rep.fail(path, f"{a!r} vs {b!r}")


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    da, db = sys.argv[1:]
    fa = {f for f in os.listdir(da) if f.endswith((".pkl", ".json"))}
    fb = {f for f in os.listdir(db) if f.endswith((".pkl", ".json"))}
    total_fail = 0
    for f in sorted(fa & fb):
        rep = Report()
        if f.endswith(".pkl"):
            with open(os.path.join(da, f), "rb") as ha, open(os.path.join(db, f), "rb") as hb:
                a, b = _Unpickler(ha).load(), _Unpickler(hb).load()
            # run_llm_eval's dump records its own argv (paths differ run to run)
            for x in (a, b):
                if isinstance(x, dict):
                    x.pop("args", None)
            cmp_pkl(a, b, f, rep)
        else:
            cmp_json(json.load(open(os.path.join(da, f))), json.load(open(os.path.join(db, f))),
                     f, rep)
        status = "PASS" if not rep.hard else "FAIL"
        print(f"{status}  {f}  ({rep.checked} nodes, {len(rep.hard)} hard, {len(rep.soft)} soft)")
        for line in rep.hard[:20]:
            print(f"    HARD {line}")
        for line in rep.soft[:5]:
            print(f"    soft {line}")
        total_fail += bool(rep.hard)
    for f in sorted(fa - fb):
        print(f"ONLY IN {da}: {f}")
    for f in sorted(fb - fa):
        print(f"ONLY IN {db}: {f}")
    print(f"\n{'ALL PASS' if not total_fail else f'{total_fail} FILE(S) FAIL'}")
    sys.exit(1 if total_fail else 0)


if __name__ == "__main__":
    main()
