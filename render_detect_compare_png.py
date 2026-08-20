"""Figures for run_detect_eval.py scorecards: the trial_loc PR curve and accuracy-vs-alpha for
a set of tagged models, plus the two degenerate references the metric admits.

    ./py render_detect_compare_png.py --tags baseline la_s05t1 --part test --out runs/compare.png

Layout only; it performs no inference and reads nothing but the JSON scorecards, mirroring the
export/render split the rest of the repo uses.
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# always-flag: every degraded trial is a hit and also a stray, every healthy trial an alarm, so
# TP=5N, FP=6N, FN=0, TN=0 -- accuracy 5/11, precision 5/11, recall 1. Worth drawing, because an
# F1 of 0.625 sits exactly there and a detector has to clear it to mean anything.
ALWAYS_FLAG_ACC = 5.0 / 11.0
ALWAYS_FLAG_PREC = 5.0 / 11.0
NEVER_FLAG_ACC = 1.0 / 6.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--part", default="test")
    ap.add_argument("--out", default="runs/detect_compare.png")
    args = ap.parse_args()
    labels = args.labels or args.tags

    fig, (ax_pr, ax_acc) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    for tag, label in zip(args.tags, labels):
        res = json.load(open(f"runs/detect_{tag}_{args.part}.json"))["results"]
        rec = [r["raw"]["recall"] for r in res]
        prec = [r["raw"]["precision"] for r in res]
        acc = [r["raw"]["accuracy"] for r in res]
        alpha = [r["alpha"] for r in res]
        ax_pr.plot(rec, prec, "o-", ms=3, label=label)
        ax_acc.plot(alpha, acc, "o-", ms=3, label=label)
        best = max(range(len(res)), key=lambda i: acc[i])
        ax_acc.plot(alpha[best], acc[best], "*", ms=14, color=ax_acc.lines[-1].get_color())
        ax_pr.plot(rec[best], prec[best], "*", ms=14, color=ax_pr.lines[-1].get_color())

    ax_pr.axhline(ALWAYS_FLAG_PREC, ls="--", c="grey", lw=1)
    ax_pr.text(0.02, ALWAYS_FLAG_PREC + 0.01, "always-flag precision", fontsize=8, color="grey")
    ax_pr.set_xlabel("trial_loc recall")
    ax_pr.set_ylabel("trial_loc precision")
    ax_pr.set_title(f"precision-recall ({args.part} split)\nstar = each model's best-accuracy alpha")
    ax_pr.grid(alpha=0.3)
    ax_pr.legend(fontsize=8)

    ax_acc.axhline(ALWAYS_FLAG_ACC, ls="--", c="grey", lw=1)
    ax_acc.text(1e-9, ALWAYS_FLAG_ACC + 0.005, "always-flag", fontsize=8, color="grey")
    ax_acc.axhline(NEVER_FLAG_ACC, ls=":", c="grey", lw=1)
    ax_acc.text(1e-9, NEVER_FLAG_ACC + 0.005, "never-flag", fontsize=8, color="grey")
    ax_acc.set_xscale("log")
    ax_acc.invert_xaxis()
    ax_acc.set_xlabel("alpha (tighter to the right)")
    ax_acc.set_ylabel("trial_loc accuracy")
    ax_acc.set_title(f"accuracy vs alpha ({args.part} split)")
    ax_acc.grid(alpha=0.3)
    ax_acc.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
