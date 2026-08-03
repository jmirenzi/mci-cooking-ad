"""Render two representative trials from dataset/processed/breakfast/flow/flow.json as
matplotlib broken-barh figures, following eval/plotting.py's conventions (Agg backend,
dpi=150, figures/ output dir)."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402

FLAME = "#B85A17"
POT = "#0E6E63"
INK = "#20241F"
INK_SOFT = "#5B615C"
LINE = "#D8D6CC"

TIER_LABELS = ["observations", "subtask (HSMM)"]

BASE_FONTSIZE = 6.5
MIN_FONTSIZE = 5.0
MARGIN = 0.88  # fraction of the bar a label may occupy before we shrink/rotate/truncate


def _place_fitted_label(ax, renderer, text_str, x0, x1, y0, y1, color):
    """Draw text_str centered in the (x0,x1)x(y0,y1) data-space bar, trying in order: full
    text horizontal, shrunk horizontal, shrunk vertical (rotated 90), then a truncated
    horizontal fallback -- always at least one strategy fits, since truncation can always
    be shortened to a single character. A clip path matching the bar is set regardless, as
    a hard guarantee against bleed into a neighboring bar if this estimate is ever off."""
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    p0 = ax.transData.transform((x0, y0))
    p1 = ax.transData.transform((x1, y1))
    width_px = abs(p1[0] - p0[0]) * MARGIN
    height_px = abs(p1[1] - p0[1]) * MARGIN

    clip_rect = mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0, transform=ax.transData)

    def _fits(txt, fontsize, rotation):
        t = ax.text(cx, cy, txt, ha="center", va="center", fontsize=fontsize,
                     color=color, rotation=rotation)
        bbox = t.get_window_extent(renderer=renderer)
        ok = bbox.width <= width_px and bbox.height <= height_px
        return t, ok

    # 1) full text, shrinking horizontally
    for fontsize in (BASE_FONTSIZE, 6.0, 5.5, MIN_FONTSIZE):
        t, ok = _fits(text_str, fontsize, 0)
        if ok:
            t.set_clip_path(clip_rect)
            return
        t.remove()

    # 2) full text, rotated 90 (narrow-but-tall bars)
    for fontsize in (BASE_FONTSIZE, 6.0, 5.5, MIN_FONTSIZE):
        t, ok = _fits(text_str, fontsize, 90)
        if ok:
            t.set_clip_path(clip_rect)
            return
        t.remove()

    # 3) truncate at the smallest fontsize until it fits horizontally; a 1-char label
    # always fits some nonzero bar, so this terminates.
    for n in range(len(text_str) - 1, 0, -1):
        truncated = text_str[:n].rstrip() + "…"
        t, ok = _fits(truncated, MIN_FONTSIZE, 0)
        if ok:
            t.set_clip_path(clip_rect)
            return
        t.remove()

    t, _ = _fits(text_str[0], MIN_FONTSIZE, 0)
    t.set_clip_path(clip_rect)


def plot_trial(trial, out_path, title=None):
    segments = trial["segments"]
    runs = trial["runs"]
    t_max = trial["T"]

    fig, ax = plt.subplots(figsize=(max(7, 0.028 * t_max), 2.3))

    row_y = {"observations": 10, "subtask (HSMM)": 0}
    bar_h = 8

    obs_spans = [(r["start"], r["end"] - r["start"]) for r in runs]
    ax.broken_barh(obs_spans, (row_y["observations"], bar_h), facecolors=FLAME, alpha=0.85,
                    edgecolor="white", linewidth=0.6)

    sub_spans = [(s["start"], s["end"] - s["start"]) for s in segments]
    sub_colors = [POT if not s["clipped"] else "#3FA79A" for s in segments]
    ax.broken_barh(sub_spans, (row_y["subtask (HSMM)"], bar_h), facecolors=sub_colors, alpha=0.85,
                    edgecolor="white", linewidth=0.6)

    ax.set_xlim(0, t_max)
    ax.set_ylim(-2, 20)
    ax.set_yticks([row_y[k] + bar_h / 2 for k in TIER_LABELS])
    ax.set_yticklabels(TIER_LABELS, fontsize=9, color=INK_SOFT)
    ax.set_xlabel("tick", fontsize=9, color=INK_SOFT)
    ax.tick_params(axis="x", labelsize=8, colors=INK_SOFT)
    ax.tick_params(axis="y", length=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(LINE)
    ax.set_title(
        title or f"{trial['true_recipe']} — {trial['trial_id']} ({t_max} ticks)",
        fontsize=11, color=INK, loc="left", fontweight="bold",
    )

    legend_handles = [
        mpatches.Patch(color=FLAME, label="observations"),
        mpatches.Patch(color=POT, label="subtask (HSMM state)"),
    ]
    ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.35),
              ncol=2, fontsize=7.5, frameon=False)

    fig.tight_layout()
    # Labels are placed after tight_layout/first draw, using the renderer, so text-extent
    # measurement (in _place_fitted_label) reflects the final axes size in pixels.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    y0, y1 = row_y["observations"], row_y["observations"] + bar_h
    for r in runs:
        _place_fitted_label(ax, renderer, r["phrase"], r["start"], r["end"], y0, y1, "white")

    y0, y1 = row_y["subtask (HSMM)"], row_y["subtask (HSMM)"] + bar_h
    for s in segments:
        label = f"z{s['z']}" + (" ⋯" if s["clipped"] else "")
        _place_fitted_label(ax, renderer, label, s["start"], s["end"], y0, y1, "white")

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow-json", default="dataset/processed/breakfast/flow/flow.json")
    parser.add_argument("--out-dir", default="dataset/processed/breakfast/figures")
    parser.add_argument("--trials", nargs="+", default=None,
                         help="trial_ids to render; defaults to the shortest and longest selected trials")
    args = parser.parse_args()

    with open(args.flow_json) as f:
        data = json.load(f)
    by_id = {t["trial_id"]: t for t in data["trials"]}

    if args.trials:
        chosen = args.trials
    else:
        by_len = sorted(data["trials"], key=lambda t: t["T"])
        chosen = [by_len[0]["trial_id"], by_len[-1]["trial_id"]]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for trial_id in chosen:
        trial = by_id[trial_id]
        out_path = out_dir / f"flow_{trial_id}.png"
        plot_trial(trial, out_path)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
