"""Render run_threshold_sweep_coordinate.py's output: for each channel, the UNION's trial_loc
metrics as that channel's alpha varies with the other six held at surprise.DEFAULT_ALPHA.

    python render_threshold_sweep_coordinate.py
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", default="dataset/processed/breakfast/threshold_sweep_coordinate.json")
    ap.add_argument("--out", default="dataset/processed/breakfast/figures/threshold_sweep_coordinate.png")
    args = ap.parse_args()

    data = json.load(open(args.sweep))
    baseline = data["baseline"]
    base_prec = baseline["trial_loc"]["precision"]
    base_rec = baseline["trial_loc"]["recall"]
    base_fpr = baseline["trial"]["fpr"]
    base_f1 = _f1(base_prec, base_rec)
    n_real = data["config"].get("max_real")
    split_part = data["config"].get("split_part")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    (ax_prec, ax_rec), (ax_fpr, ax_f1) = axes

    summary = []
    for ch in CHANNELS:
        rows = sorted(data["per_channel"][ch], key=lambda r: r["alpha"])
        alphas = [r["alpha"] for r in rows]
        precision = [r["trial_loc"]["precision"] for r in rows]
        recall = [r["trial_loc"]["recall"] for r in rows]
        healthy_fpr = [r["trial"]["fpr"] for r in rows]
        f1 = [_f1(p, rc) for p, rc in zip(precision, recall)]

        style = dict(color=COLORS[ch], linewidth=1.6, marker=".", markersize=4, label=ch)
        ax_prec.plot(alphas, precision, **style)
        ax_rec.plot(alphas, recall, **style)
        ax_fpr.plot(alphas, healthy_fpr, **style)
        ax_f1.plot(alphas, f1, **style)

        i = int(np.argmax(f1))
        summary.append({
            "channel": ch,
            "best_alpha": alphas[i],
            "precision": precision[i], "recall": recall[i], "healthy_fpr": healthy_fpr[i], "f1": f1[i],
            "delta_f1_vs_baseline": f1[i] - base_f1,
        })

    for ax, base_y, ylabel, title_ in (
        (ax_prec, base_prec, "precision", "union trial_loc precision"),
        (ax_rec, base_rec, "recall", "union trial_loc recall"),
        (ax_fpr, base_fpr, "healthy FPR (nag rate)", "union healthy false-positive rate"),
        (ax_f1, base_f1, "F1", "union trial_loc F1"),
    ):
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.axvline(DEFAULT_ALPHA, color="#888", linestyle=":", linewidth=1)
        ax.axhline(base_y, color="black", linestyle="--", linewidth=1.4, zorder=1,
                   label="all-default baseline" if ax is ax_prec else None)
        ax.set_xlabel(r"$\alpha$ of the ONE varied channel (others fixed at default)")
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(title_, fontsize=11)
        ax.grid(alpha=0.3)

    handles, labels = ax_prec.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02), fontsize=9)
    fig.suptitle(
        f"Coordinate-descent: ONE channel's alpha varied, other six held at {DEFAULT_ALPHA:g} -- "
        f"UNION scored each time. {n_real} real trials ({split_part} split) x (healthy + 5 injections). "
        f"Black dashed = all-channels-at-default baseline (F1={base_f1:.3f}); "
        f"every curve crosses it at the dotted vertical (alpha={DEFAULT_ALPHA:g}).",
        fontsize=9.5, y=0.995,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"written to {args.out}")

    summary.sort(key=lambda s: s["delta_f1_vs_baseline"], reverse=True)
    print(f"\nbaseline (all channels at alpha={DEFAULT_ALPHA:g}): "
          f"precision={base_prec:.3f}  recall={base_rec:.3f}  healthy_fpr={base_fpr:.3f}  F1={base_f1:.3f}\n")
    print(f"{'channel':>20}  {'best alpha':>10}  {'precision':>9}  {'recall':>7}  "
          f"{'healthy_fpr':>11}  {'F1':>6}  {'d(F1)':>7}")
    for s in summary:
        print(f"{s['channel']:>20}  {s['best_alpha']:>10.2e}  {s['precision']:>9.3f}  "
              f"{s['recall']:>7.3f}  {s['healthy_fpr']:>11.3f}  {s['f1']:>6.3f}  "
              f"{s['delta_f1_vs_baseline']:>+7.3f}")

    summary_path = args.out.rsplit(".", 1)[0] + "_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"baseline": {"precision": base_prec, "recall": base_rec,
                                "healthy_fpr": base_fpr, "f1": base_f1},
                   "per_channel": summary}, f, indent=2)
    print(f"\nsummary written to {summary_path}")


if __name__ == "__main__":
    main()
