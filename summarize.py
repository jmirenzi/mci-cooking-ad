"""Compact side-by-side of every runs/detect_<tag>_<part>.json scorecard.

Ranked by trial_loc ACCURACY, and against the two degenerate references the metric admits:
flagging every trial and flagging none. With a 1-healthy : 5-degraded pool and strays charged
independently, always-flag scores accuracy 5/11 = 0.455 and F1 0.625 -- so an F1 near 0.62 is
not evidence of a working detector, and accuracy is the number that separates one.
"""
import glob, json, re, sys

part = sys.argv[1] if len(sys.argv) > 1 else "train"
key = sys.argv[2] if len(sys.argv) > 2 else "accuracy"

rows = []
for f in sorted(glob.glob(f"runs/detect_*_{part}.json")):
    tag = re.match(rf"runs/detect_(.+)_{part}\.json", f).group(1)
    res = json.load(open(f))["results"]
    rows.append((tag, max(res, key=lambda r: r["raw"][key]), {r["alpha"]: r for r in res}))


def line(tag, r):
    x = r["raw"]; pt = x["per_type"]
    stray = sum(pt[k]["stray"] for k in pt if k != "healthy") / 5.0
    return (f"{tag:14s} a={r['alpha']:<7.0e} acc={x['accuracy']:.3f} P={x['precision']:.3f} "
            f"R={x['recall']:.3f} F1={x['f1']:.3f} hFP={pt['healthy']['stray']:.3f} "
            f"stray={stray:.3f} | "
            + " ".join(f"{k[:4]}={pt[k]['recall']:.2f}" for k in
                       ("substitution", "abandonment", "omission", "transposition", "repetition")))


print(f"===== best trial_loc {key} ({part}) =====")
print(f"{'always-flag':14s} a=-       acc=0.455 P=0.455 R=1.000 F1=0.625 hFP=1.000 stray=1.000")
for tag, best, _ in sorted(rows, key=lambda z: -z[1]["raw"][key]):
    print(line(tag, best))
for alpha in (0.05, 0.02, 0.005):
    sel = [(tag, at[alpha]) for tag, _, at in rows if alpha in at]
    if not sel:
        continue
    print(f"\n===== fixed alpha={alpha:.0e} ({part}) =====")
    for tag, r in sorted(sel, key=lambda z: -z[1]["raw"][key]):
        print(line(tag, r))
