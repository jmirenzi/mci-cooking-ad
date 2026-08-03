"""Render dataset/processed/breakfast/flow/anomaly.json (see export_anomaly.py) as matplotlib
figures: the same observations/subtask flow blocks as render_flow_png.py, plus the injected
ground-truth window, per-channel flagged ticks, and the narrated Query cards -- a visual
version of run_rollout_demo.py's own scenario sweep and query printout."""
import argparse
import json
import textwrap
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from render_flow_png import FLAME, INK, INK_SOFT, LINE, POT, _place_fitted_label  # noqa: E402

ALARM = "#B3261E"
WINDOW_FILL = "#B3261E"

SEVERITY_COLOR = {"low": "#4C7A5E", "medium": FLAME, "high": ALARM}
SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2}

# Which flow row a channel's flagged ticks get marked on. s_recipe_transition has no row to
# mark against here (Part 1 dropped the recipe row entirely), so it's omitted -- narrate.py
# never renders it either (K_recipe has no learned cluster->name map, see narrate.py's caveat).
CHANNEL_ROW = {
    "s_emit": "observations", "s_verb": "observations", "s_noun": "observations",
    "s_temporal": "subtask", "s_dur_two": "subtask", "s_transition": "subtask",
}

CARD_FONTSIZE = 8.3
CARD_LINE_HEIGHT = 0.185  # inches
CARD_WRAP = 96


def _fmt_ratio(ratio):
    """s_transition has no EMIT_THRESHOLD_FLOOR-style floor (surprise.py only floors
    s_emit/s_verb/s_noun), so a near-deterministic transition row's quantile threshold can be
    a tiny-but-positive float that _severity's `threshold <= 0` guard doesn't catch -- ratio
    then blows up to something like 2.6e13. That's a real pre-existing calibration gap, not a
    display bug; this just keeps the card legible instead of printing the raw absurd float."""
    if not np.isfinite(ratio):
        return "×inf"
    if ratio > 999:
        return "×999+"
    return f"×{ratio:.1f}"


def _wrap_query(i, q):
    tag = "TRUE POSITIVE" if q["true_positive"] else "FALSE ALARM"
    head = f"{i}. [{tag}] t={q['tick']}  {q['channel']}  {q['severity']} ({_fmt_ratio(q['ratio'])})"
    body = textwrap.wrap(q["text"], width=CARD_WRAP, subsequent_indent="    ")
    return [head] + [f"    {line}" for line in body]


def plot_scenario(rec, error_type, out_path):
    segments = rec["segments"]
    runs = rec["runs"]
    t_max = rec["T"]
    healthy_runs = rec["healthy_runs"]
    healthy_t_max = rec["healthy_T"]
    window = rec["window"]
    flagged = rec["flagged_channels"]
    queries = sorted(rec["queries"], key=lambda q: q["tick"])

    card_lines = []
    for i, q in enumerate(queries, start=1):
        card_lines += _wrap_query(i, q)
    if not card_lines:
        card_lines = ["(no narrated queries)"]

    chart_h = 3.2
    card_h = 0.5 + CARD_LINE_HEIGHT * len(card_lines)
    fig_h = chart_h + card_h
    fig = plt.figure(figsize=(max(7.5, 0.03 * max(t_max, healthy_t_max)), fig_h))
    ax = fig.add_axes([0.09, (card_h + 0.55) / fig_h, 0.88, (chart_h - 0.55) / fig_h])
    card_ax = fig.add_axes([0.02, 0.02, 0.96, (card_h - 0.1) / fig_h])
    card_ax.axis("off")

    # A wide gap between the observations/subtask bars leaves room for each row's own
    # flagged-tick strip and query-star row directly above it, so a marker's row membership
    # reads from proximity instead of needing a legend lookup. "unaltered" sits well clear of
    # observations' own marker rows (which top out around 26), and shares observations' tick
    # scale up to the injection point so the two are directly comparable by eye.
    row_y = {"unaltered": 30, "observations": 15, "subtask": 0}
    bar_h = 8

    if window is not None:
        t0, t1 = window
        hi = t1 + 5  # metrics.DEFAULT_LATENCY_TOL, mirrored here to avoid importing eval.metrics for one constant
        ax.axvspan(t0, hi, color=WINDOW_FILL, alpha=0.10, zorder=0)
        ax.axvline(t0, color=ALARM, alpha=0.55, linewidth=1.1, linestyle="--", zorder=1)
        ax.text(t0, row_y["unaltered"] + bar_h + 1.2, " injected", color=ALARM, fontsize=7,
                va="bottom", ha="left", alpha=0.85)

    unalt_spans = [(r["start"], r["end"] - r["start"]) for r in healthy_runs]
    ax.broken_barh(unalt_spans, (row_y["unaltered"], bar_h), facecolors=FLAME, alpha=0.35,
                    edgecolor="white", linewidth=0.6, zorder=2)
    obs_spans = [(r["start"], r["end"] - r["start"]) for r in runs]
    ax.broken_barh(obs_spans, (row_y["observations"], bar_h), facecolors=FLAME, alpha=0.85,
                    edgecolor="white", linewidth=0.6, zorder=2)
    sub_spans = [(s["start"], s["end"] - s["start"]) for s in segments]
    ax.broken_barh(sub_spans, (row_y["subtask"], bar_h), facecolors=POT, alpha=0.85,
                    edgecolor="white", linewidth=0.6, zorder=2)

    for ch, ticks in flagged.items():
        row = CHANNEL_ROW.get(ch)
        if row is None or not ticks:
            continue
        y = row_y[row] + bar_h + 0.9
        ax.scatter(ticks, [y] * len(ticks), marker="v", s=16, color=ALARM, alpha=0.75, zorder=3, linewidths=0)

    for i, q in enumerate(queries, start=1):
        row = CHANNEL_ROW.get(q["channel"], "observations")
        y = row_y[row] + bar_h + 2.6
        ax.scatter([q["tick"]], [y], marker="*", s=130, color=SEVERITY_COLOR[q["severity"]],
                   edgecolor="white", linewidth=0.6, zorder=4)
        ax.annotate(str(i), (q["tick"], y), textcoords="offset points", xytext=(0, 5),
                    ha="center", fontsize=7, fontweight="bold", color=INK, zorder=5)

    ax.set_xlim(0, max(t_max, healthy_t_max))
    ax.set_ylim(-2, row_y["unaltered"] + bar_h + 3)
    ax.set_yticks([row_y[k] + bar_h / 2 for k in ("unaltered", "observations", "subtask")])
    ax.set_yticklabels(
        ["unaltered\n(what happened)", "observations\n(fed to detector)", "subtask (HSMM)"],
        fontsize=8.5, color=INK_SOFT,
    )
    ax.set_xlabel("tick", fontsize=9, color=INK_SOFT)
    ax.tick_params(axis="x", labelsize=8, colors=INK_SOFT)
    ax.tick_params(axis="y", length=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(LINE)

    subtitle = f"{rec['trial_id']} ({t_max} ticks)"
    if rec.get("select"):
        subtitle += f" — select={rec['select']} seed={rec['seed']}"
    ax.set_title(f"{error_type}  ·  {subtitle}", fontsize=11, color=INK, loc="left", fontweight="bold")

    legend_handles = [
        mpatches.Patch(color=FLAME, alpha=0.35, label="observations, unaltered"),
        mpatches.Patch(color=FLAME, label="observations, fed to detector"),
        mpatches.Patch(color=POT, label="subtask (HSMM state)"),
        mpatches.Patch(color=ALARM, label="flagged tick (any channel)"),
    ]
    for sev, color in SEVERITY_COLOR.items():
        legend_handles.append(plt.Line2D([0], [0], marker="*", color="w", markerfacecolor=color,
                                          markersize=9, label=f"query: {sev}"))
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.005, 1.05),
              fontsize=7, frameon=False, borderaxespad=0)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    y0, y1 = row_y["unaltered"], row_y["unaltered"] + bar_h
    for r in healthy_runs:
        _place_fitted_label(ax, renderer, r["phrase"], r["start"], r["end"], y0, y1, INK)
    y0, y1 = row_y["observations"], row_y["observations"] + bar_h
    for r in runs:
        _place_fitted_label(ax, renderer, r["phrase"], r["start"], r["end"], y0, y1, "white")
    y0, y1 = row_y["subtask"], row_y["subtask"] + bar_h
    for s in segments:
        _place_fitted_label(ax, renderer, f"z{s['z']}", s["start"], s["end"], y0, y1, "white")

    card_ax.text(0, 1, "\n".join(card_lines), transform=card_ax.transAxes, fontsize=CARD_FONTSIZE,
                 family="monospace", va="top", ha="left", color=INK,
                 linespacing=1.35)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anomaly-json", default="dataset/processed/breakfast/flow/anomaly.json")
    parser.add_argument("--out-dir", default="dataset/processed/breakfast/figures")
    args = parser.parse_args()

    with open(args.anomaly_json) as f:
        data = json.load(f)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for error_type, rec in data["scenarios"].items():
        out_path = out_dir / f"anomaly_{error_type}.png"
        plot_scenario(rec, error_type, out_path)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
