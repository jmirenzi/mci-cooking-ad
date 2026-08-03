"""Cascade-vs-joint comparison figure for dataset/processed/breakfast_mini/flow/flow_compare.json
(see export_flow_joint.py). One observations row (shared, since the input is identical), then
subtask and recipe rows for each model stacked directly beneath each other so differences are a
straight vertical read, not a side-by-side guess."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402

from render_flow_png import _place_fitted_label  # noqa: E402

FLAME = "#B85A17"
POT = "#0E6E63"
STEEL = "#2F6690"
PLUM = "#6B4472"
GOLD = "#A9791F"
INK = "#20241F"
INK_SOFT = "#5B615C"
LINE = "#D8D6CC"

ROW_LABELS = [
    "observations",
    "subtask — cascade",
    "subtask — joint (conditioned)",
    "recipe — cascade (per segment)",
    "recipe — joint (per trial)",
]


def plot_comparison(rec, out_path):
    c, j = rec["cascade"], rec["joint"]
    t_max = c["T"]
    runs = c["runs"]
    c_segs = c["segments"]
    j_segs = j["segments"]
    c_recipe = {rp["seg"]: rp["r"] for rp in c["recipe_path"]}

    fig, ax = plt.subplots(figsize=(max(7, 0.05 * t_max), 4.4))

    bar_h = 8
    gap = 2
    row_y = {label: (len(ROW_LABELS) - 1 - i) * (bar_h + gap) for i, label in enumerate(ROW_LABELS)}

    def draw_row(label, spans, color):
        y0 = row_y[label]
        ax.broken_barh(spans, (y0, bar_h), facecolors=color, alpha=0.85, edgecolor="white", linewidth=0.6)
        return y0, y0 + bar_h

    draw_row("observations", [(r["start"], r["end"] - r["start"]) for r in runs], FLAME)
    draw_row("subtask — cascade", [(s["start"], s["end"] - s["start"]) for s in c_segs], POT)
    draw_row("subtask — joint (conditioned)", [(s["start"], s["end"] - s["start"]) for s in j_segs], STEEL)
    draw_row("recipe — cascade (per segment)", [(s["start"], s["end"] - s["start"]) for s in c_segs], PLUM)
    draw_row("recipe — joint (per trial)", [(0, t_max)], GOLD)

    ax.set_xlim(0, t_max)
    ax.set_ylim(-2, row_y["observations"] + bar_h + 2)
    ax.set_yticks([row_y[label] + bar_h / 2 for label in ROW_LABELS])
    ax.set_yticklabels(ROW_LABELS, fontsize=8.5, color=INK_SOFT)
    ax.set_xlabel("tick", fontsize=9, color=INK_SOFT)
    ax.tick_params(axis="x", labelsize=8, colors=INK_SOFT)
    ax.tick_params(axis="y", length=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(LINE)
    ax.set_title(
        f"{c['true_recipe']} — {rec['trial_id']} ({t_max} ticks)",
        fontsize=11, color=INK, loc="left", fontweight="bold",
    )

    fig.tight_layout()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    y0, y1 = row_y["observations"], row_y["observations"] + bar_h
    for r in runs:
        _place_fitted_label(ax, renderer, r["phrase"], r["start"], r["end"], y0, y1, "white")

    y0, y1 = row_y["subtask — cascade"], row_y["subtask — cascade"] + bar_h
    for s in c_segs:
        _place_fitted_label(ax, renderer, f"z{s['z']}", s["start"], s["end"], y0, y1, "white")

    y0, y1 = row_y["subtask — joint (conditioned)"], row_y["subtask — joint (conditioned)"] + bar_h
    for s in j_segs:
        _place_fitted_label(ax, renderer, f"z{s['z']}", s["start"], s["end"], y0, y1, "white")

    y0, y1 = row_y["recipe — cascade (per segment)"], row_y["recipe — cascade (per segment)"] + bar_h
    for i, s in enumerate(c_segs):
        r = c_recipe.get(i)
        _place_fitted_label(ax, renderer, f"r{r}" if r is not None else "-", s["start"], s["end"], y0, y1, "white")

    y0, y1 = row_y["recipe — joint (per trial)"], row_y["recipe — joint (per trial)"] + bar_h
    _place_fitted_label(
        ax, renderer, f"r{j['r_hat']}  (posterior {j['confidence']:.2f})", 0, t_max, y0, y1, "white",
    )

    legend_handles = [
        mpatches.Patch(color=FLAME, label="observations"),
        mpatches.Patch(color=POT, label="subtask, cascade"),
        mpatches.Patch(color=STEEL, label="subtask, joint"),
        mpatches.Patch(color=PLUM, label="recipe, cascade"),
        mpatches.Patch(color=GOLD, label="recipe, joint"),
    ]
    ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=5, fontsize=7, frameon=False)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow-json", default="dataset/processed/breakfast_mini/flow/flow_compare.json")
    parser.add_argument("--out-dir", default="dataset/processed/breakfast_mini/figures")
    args = parser.parse_args()

    with open(args.flow_json) as f:
        data = json.load(f)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for rec in data["trials"]:
        out_path = out_dir / f"compare_{rec['trial_id']}.png"
        plot_comparison(rec, out_path)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
