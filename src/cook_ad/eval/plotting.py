from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _error_types(report):
    return list(report["per_type"].keys())


def plot_precision_recall_latency(report, out_path, title=None):
    """Grouped recall/precision bars per error type, with mean detection latency annotated."""
    types = _error_types(report)
    recall = [report["per_type"][e]["recall"] for e in types]
    precision = [report["per_type"][e]["precision"] for e in types]
    latency = [report["per_type"][e]["mean_latency"] for e in types]

    x = np.arange(len(types))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(types)), 4.5))
    ax.bar(x - width / 2, recall, width, label="recall", color="#4c72b0")
    ax.bar(x + width / 2, precision, width, label="precision", color="#dd8452")

    for xi, lat in zip(x, latency):
        label = "n/a" if np.isnan(lat) else f"lat {lat:.1f}"
        ax.text(xi, 1.02, label, ha="center", va="bottom", fontsize=8, color="#555")

    ax.set_xticks(x)
    ax.set_xticklabels(types, rotation=20, ha="right")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("rate")
    ax.set_title(title or "Per-error-type detection")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_attribution_heatmap(report, out_path, title=None):
    """channel x error-type: fraction of detected trials each channel fired in-window. Shows the
    isolation claim -- substitution -> s_noun, abandonment -> s_dur_short, etc."""
    types = _error_types(report)
    channels = report["channels"]
    matrix = np.array([[report["attribution"][e][ch] for e in types] for ch in channels])

    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(types)), max(4, 0.5 * len(channels))))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(types)))
    ax.set_xticklabels(types, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(channels)))
    ax.set_yticklabels(channels)
    for i in range(len(channels)):
        for j in range(len(types)):
            v = matrix[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v < 0.6 else "black", fontsize=8)
    ax.set_title(title or "Channel x error-type attribution")
    fig.colorbar(im, ax=ax, label="fraction of detections")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_figures(report, out_dir, tag):
    """Writes both figures for one healthy source (tag='synthetic' or 'real'). Returns paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pr_path = out_dir / f"detection_{tag}.png"
    attr_path = out_dir / f"attribution_{tag}.png"
    plot_precision_recall_latency(report, pr_path, title=f"Per-error-type detection ({tag})")
    plot_attribution_heatmap(report, attr_path, title=f"Channel x error attribution ({tag})")
    return [pr_path, attr_path]
