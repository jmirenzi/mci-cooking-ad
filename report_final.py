"""Honest train->test protocol: pick each model's alpha on the TRAIN split by trial_loc
accuracy, then report that model at that same alpha on TEST. Picking alpha per split would
report a number no deployment could reproduce, since alpha has to be fixed before the test
trials are seen."""
import glob, json, os, re, sys

tags = sys.argv[1:] or sorted(
    re.match(r"runs/detect_(.+)_test\.json", f).group(1) for f in glob.glob("runs/detect_*_test.json")
)


def load(tag, part):
    p = f"runs/detect_{tag}_{part}.json"
    return json.load(open(p))["results"] if os.path.exists(p) else None


def fmt(r):
    x = r["raw"]; pt = x["per_type"]
    stray = sum(pt[k]["stray"] for k in pt if k != "healthy") / 5.0
    return (f"acc={x['accuracy']:.3f} P={x['precision']:.3f} R={x['recall']:.3f} "
            f"F1={x['f1']:.3f} hFP={pt['healthy']['stray']:.3f} stray={stray:.3f} | "
            + " ".join(f"{k[:4]}={pt[k]['recall']:.2f}" for k in
                       ("substitution", "abandonment", "omission", "transposition", "repetition")))


print("alpha chosen on TRAIN by trial_loc accuracy, then applied unchanged to TEST")
print(f"{'always-flag':16s} a=-       TEST  acc=0.455 P=0.455 R=1.000 F1=0.625")
rows = []
for tag in tags:
    tr, te = load(tag, "train"), load(tag, "test")
    if not tr or not te:
        continue
    best = max(tr, key=lambda r: r["raw"]["accuracy"])
    match = min(te, key=lambda r: abs(r["alpha"] - best["alpha"]))
    rows.append((match["raw"]["accuracy"], tag, best, match))
for _acc, tag, best, match in sorted(rows, reverse=True):
    print(f"\n{tag:16s} alpha={best['alpha']:.0e}")
    print(f"  {'train':6s} {fmt(best)}")
    print(f"  {'TEST':6s} {fmt(match)}")
