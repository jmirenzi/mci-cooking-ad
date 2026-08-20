"""Per-channel breakdown of run_threshold_sweep.py's alpha sweep.

The shipped detector unions all seven surprise channels under one shared alpha
(surprise.flag_joint). This renders each channel's OWN trial_loc precision/recall/F1 and
healthy false-positive rate across the same alpha grid, scored ALONE rather than unioned --
the diagnostic for whether channels have different enough natural operating points to justify
per-channel thresholds, as opposed to the one shared alpha every channel currently gets.

    python run_threshold_sweep.py --out dataset/processed/breakfast/threshold_sweep_by_channel.json
    python render_threshold_sweep_by_channel.py \
        --sweep dataset/processed/breakfast/threshold_sweep_by_channel.json
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

CHANNELS = (
    "s_emit", "s_verb", "s_noun", "s_temporal", "s_dur_two", "s_transition", "s_recipe_transition",
)
COLORS = dict(zip(CHANNELS, plt.get_cmap("tab10").colors))
DEFAULT_ALPHA = 5e-3


def _f1(precision, recall):
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def _pr_auc(precision, recall):
    """Area under the precision-recall curve, trapezoidal, over the swept alpha points sorted
    by recall ascending. Not sklearn's step-interpolated average precision -- these points come
    from a fixed alpha grid rather than every distinct score value -- but the same idea: a
    single threshold-independent number summarizing the whole curve, for ranking channels
    against each other rather than reading 21 points by eye."""
    order = np.argsort(recall)
    r = np.asarray(recall)[order]
    p = np.asarray(precision)[order]
    return float(np.trapezoid(p, r))


def _best_alpha(alphas, precision, recall, f1):
    """argmax F1 over the swept grid -- the best OBSERVED operating point, not an interpolated
    optimum. Reports the point as-is rather than searching between grid alphas."""
    i = int(np.argmax(f1))
    return {"alpha": alphas[i], "precision": precision[i], "recall": recall[i], "f1": f1[i]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", default="dataset/processed/breakfast/threshold_sweep_by_channel.json")
    ap.add_argument("--out", default="dataset/processed/breakfast/figures/threshold_sweep_by_channel.png")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    data = json.load(open(args.sweep))
    results = sorted(data["results"], key=lambda r: r["alpha"])
    alphas = [r["alpha"] for r in results]

    n_real = data["config"].get("max_real")
    title = args.title or (
        f"HSMM surprise channels, scored INDIVIDUALLY (not unioned) -- {n_real} real trials x "
        f"(healthy + 5 injections). Dashed vertical = current shared default alpha={DEFAULT_ALPHA:g}."
    )

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    (ax_prec, ax_rec), (ax_fpr, ax_f1) = axes

    summary = []
    for bucket in ("raw", *CHANNELS):
        precision = [r[f"{bucket}_trial_loc"]["precision"] for r in results]
        recall = [r[f"{bucket}_trial_loc"]["recall"] for r in results]
        # "healthy FPR" specifically -- the plain `trial` bucket's fp/tn population is healthy
        # trials only (degraded trials there only ever contribute tp/fn on "any flag anywhere"),
        # unlike trial_loc's fp, which pools healthy false alarms with degraded strays.
        healthy_fpr = [r[f"{bucket}_trial"]["fpr"] for r in results]
        f1 = [_f1(p, rc) for p, rc in zip(precision, recall)]

        summary.append({
            "channel": bucket,
            "pr_auc": _pr_auc(precision, recall),
            "best": _best_alpha(alphas, precision, recall, f1),
        })

        is_raw = bucket == "raw"
        style = dict(
            color="black" if is_raw else COLORS[bucket],
            linewidth=2.4 if is_raw else 1.6,
            linestyle="--" if is_raw else "-",
            marker="o" if is_raw else ".",
            markersize=5 if is_raw else 4,
            label="raw (union, current detector)" if is_raw else bucket,
            zorder=3 if is_raw else 2,
            alpha=1.0 if is_raw else 0.9,
        )
        ax_prec.plot(alphas, precision, **style)
        ax_rec.plot(alphas, recall, **style)
        ax_fpr.plot(alphas, healthy_fpr, **style)
        ax_f1.plot(alphas, f1, **style)

    for ax, ylabel, title_ in (
        (ax_prec, "precision", "trial_loc precision"),
        (ax_rec, "recall", "trial_loc recall"),
        (ax_fpr, "healthy FPR (nag rate)", "healthy false-positive rate"),
        (ax_f1, "F1", "trial_loc F1"),
    ):
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.axvline(DEFAULT_ALPHA, color="#888", linestyle=":", linewidth=1)
        ax.set_xlabel(r"$\alpha$ (smaller = stricter $\rightarrow$)")
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(title_, fontsize=11)
        ax.grid(alpha=0.3)

    handles, labels = ax_prec.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02), fontsize=9)
    fig.suptitle(title, fontsize=10, y=0.995)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"written to {args.out}")

    summary.sort(key=lambda s: s["pr_auc"], reverse=True)
    print(f"\n{'channel':>20}  {'PR-AUC':>7}  |  {'best alpha':>10}  {'precision':>9}  "
          f"{'recall':>7}  {'F1':>6}")
    for s in summary:
        b = s["best"]
        print(f"{s['channel']:>20}  {s['pr_auc']:>7.3f}  |  {b['alpha']:>10.2e}  "
              f"{b['precision']:>9.3f}  {b['recall']:>7.3f}  {b['f1']:>6.3f}")

    summary_path = args.out.rsplit(".", 1)[0] + "_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsummary written to {summary_path}")


if __name__ == "__main__":
    main()
