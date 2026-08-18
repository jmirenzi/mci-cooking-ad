"""Comparison figures for the LLM baseline vs the HSMM, from run_llm_eval.py's report JSON.

Layout only -- performs no inference and reads nothing but the report, matching the export/render
split the rest of the repo uses (see docs/README.md): figures can be restyled without re-running a
three-hour sweep.

    python render_llm_compare_png.py --report dataset/processed/breakfast/llm_full_report.json
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HSMM_C = "#4c72b0"
LLM_C = "#dd8452"
DPI = 150


def _arms(reports):
    """{source: [arm_key, ...]} from keys like 'real/hsmm-joint'.

    Any number of arms per source: one HSMM plus however many LLM configurations were scored on
    the same pool (no-recipes, with-recipes, another model...). The HSMM is ordered first so it
    reads as the reference everything else is compared against.
    """
    out = {}
    for key in reports:
        if reports[key].get("incomplete") or "step_level" not in reports[key]:
            continue
        source = key.partition("/")[0]
        out.setdefault(source, []).append(key)
    return {src: sorted(keys, key=lambda k: (0 if "/hsmm" in k else 1, k))
            for src, keys in out.items() if keys}


# One colour per arm, stable across every figure so a reader learns the legend once.
ARM_COLORS = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3", "#937860"]


def _color(i):
    return ARM_COLORS[i % len(ARM_COLORS)]


def _label(reports, key):
    r = reports[key]
    if "/hsmm" in key:
        return f"HSMM ({key.split('hsmm-')[-1]})"
    model = r.get("client", {}).get("model", "llm")
    variant = r.get("prompt_variant", "")
    tag = {"no-recipes": "no recipes", "with-recipes": "+ recipes"}.get(variant, variant)
    return f"{model} ({tag})" if tag else model


def _types(report):
    return list(report["per_type"].keys())


def detection_panel(reports, arms, source, out_path):
    """Recall and precision per error type, every arm, side by side."""
    types = _types(reports[arms[0]])
    x = np.arange(len(types))
    w = 0.8 / len(arms)

    fig, axes = plt.subplots(1, 2, figsize=(max(12, 2.1 * len(types)), 4.8), sharey=True)
    for ax, metric in zip(axes, ("recall", "precision")):
        for i, key in enumerate(arms):
            vals = [reports[key]["per_type"][t][metric] for t in types]
            ax.bar(x + (i - (len(arms) - 1) / 2) * w, vals, w * 0.92,
                   label=_label(reports, key), color=_color(i))
        ax.set_xticks(x)
        ax.set_xticklabels(types, rotation=20, ha="right")
        ax.set_ylim(0, 1.08)
        ax.set_title(metric)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("rate")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle(f"Per-error-type detection, step-level -- {source} "
                 f"(n={reports[arms[0]]['per_type'][types[0]]['n']} trials/type)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def steplevel_panel(reports, arms_by_source, out_path):
    """Pooled step-level precision / recall / F1, every arm, plus the chance-precision line.

    The chance line is the point of this figure: precision has to be read against the base rate of
    anomalous steps, or a detector that simply flags everything looks respectable.
    """
    labels, prec, rec, f1, chance, colors = [], [], [], [], [], []
    for source, arms in arms_by_source.items():
        for i, key in enumerate(arms):
            r = reports[key]
            sl = r["step_level"]
            p, q = sl["precision"], sl["recall"]
            labels.append(f"{source}\n{_label(reports, key).replace(' (', chr(10) + '(')}")
            colors.append(_color(i))
            prec.append(p)
            rec.append(q)
            f1.append(2 * p * q / (p + q) if (p + q) else 0.0)
            # Reported by evaluate_steps rather than derived here: tp+fp+fn is NOT the step count
            # and would give the two arms different baselines for the same data.
            chance.append(sl.get("chance_precision", np.nan))

    x = np.arange(len(labels))
    w = 0.26
    fig, ax = plt.subplots(figsize=(max(9, 2.3 * len(labels)), 5.0))
    ax.bar(x - w, prec, w, label="precision", color=HSMM_C)
    ax.bar(x, rec, w, label="recall", color=LLM_C)
    ax.bar(x + w, f1, w, label="F1", color="#55a868")
    for xi, c in zip(x, chance):
        ax.plot([xi - 1.5 * w, xi + 1.5 * w], [c, c], color="#c44e52", ls="--", lw=1.4,
                label="precision at chance" if xi == 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("rate")
    ax.set_title("Pooled step-level performance (every step is one test)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def confusion_panel(reports, arms, source, out_path):
    """Predicted x true anomaly type, both detectors. Fixed [0,1] colourbar so the two panels --
    and figures from different runs -- are directly comparable."""
    types = _types(reports[arms[0]])
    rows = list(reports[arms[0]]["channels"])  # predicted types + "none"

    fig, axes = plt.subplots(1, len(arms),
                             figsize=(max(5.5 * len(arms), 11), max(4.2, 0.62 * len(rows))),
                             squeeze=False)
    axes = axes[0]
    for ax, key in zip(axes, arms):
        rep = reports[key]
        m = np.array([[rep["type_confusion"][t].get(p, 0.0) for t in types] for p in rows])
        im = ax.imshow(m, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(np.arange(len(types)))
        ax.set_xticklabels(types, rotation=20, ha="right")
        ax.set_yticks(np.arange(len(rows)))
        ax.set_yticklabels(rows)
        for i in range(len(rows)):
            for j in range(len(types)):
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if m[i, j] < 0.6 else "black")
        ax.set_title(_label(reports, key), fontsize=10)
        ax.set_xlabel("true type")
    axes[0].set_ylabel("predicted type")
    fig.colorbar(im, ax=axes, label="fraction of detections", fraction=0.025)
    fig.suptitle(f"Predicted x true anomaly type -- {source}")
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _diag(rep, types):
    """Fraction of detections given the CORRECT type, per true type -- the confusion diagonal."""
    return [rep["type_confusion"][t].get(t, 0.0) for t in types]


def type_accuracy_panel(reports, arms_by_source, out_path):
    """The confusion diagonal on its own: given that a detector fired, did it name the right
    error? This is what the LLM baseline exists to test, and the tick-level channel-attribution
    matrix can only proxy."""
    sources = list(arms_by_source)
    fig, axes = plt.subplots(1, len(sources), figsize=(6.2 * len(sources), 4.4), squeeze=False)
    for ax, source in zip(axes[0], sources):
        arms = arms_by_source[source]
        types = _types(reports[arms[0]])
        x = np.arange(len(types))
        w = 0.8 / len(arms)
        for i, key in enumerate(arms):
            ax.bar(x + (i - (len(arms) - 1) / 2) * w, _diag(reports[key], types), w * 0.92,
                   label=_label(reports, key), color=_color(i))
        ax.set_xticks(x)
        ax.set_xticklabels(types, rotation=20, ha="right")
        ax.set_ylim(0, 1.08)
        ax.set_title(source)
        ax.grid(axis="y", alpha=0.3)
    axes[0][0].set_ylabel("fraction of detections given the correct type")
    axes[0][0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Type-identification accuracy (confusion diagonal)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def correction_panel(reports, arms_by_source, out_path):
    """Does the proposed 'correct move' match the pre-injection step? Token match and duration
    match are separate bars because abandonment leaves verb/noun untouched -- a combined score
    would credit naming a step the detector never had to identify."""
    sources = list(arms_by_source)
    fig, axes = plt.subplots(1, len(sources), figsize=(6.6 * len(sources), 4.4), squeeze=False)
    for ax, source in zip(axes[0], sources):
        arms = arms_by_source[source]
        types = _types(reports[arms[0]])
        x = np.arange(len(types))
        n_bars = 2 * len(arms)
        w = 0.86 / n_bars
        off = 0
        for i, key in enumerate(arms):
            for metric, hatch in (("verb_noun_accuracy", None), ("duration_accuracy", "//")):
                vals = [reports[key]["correction_accuracy"][t][metric] for t in types]
                vals = [0.0 if (v is None or np.isnan(v)) else v for v in vals]
                ax.bar(x + (off - (n_bars - 1) / 2) * w, vals, w * 0.92, color=_color(i),
                       hatch=hatch, edgecolor="white",
                       label=f"{_label(reports, key)} "
                             f"{'verb+noun' if 'verb' in metric else 'duration'}")
                off += 1
        ax.set_xticks(x)
        ax.set_xticklabels(types, rotation=20, ha="right")
        ax.set_ylim(0, 1.08)
        ax.set_title(source)
        ax.grid(axis="y", alpha=0.3)
    axes[0][0].set_ylabel("accuracy of proposed correction")
    axes[0][0].legend(fontsize=6, ncol=2)
    fig.suptitle("'Correct move would have been ...' -- accuracy against the pre-injection step")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def latency_panel(reports, arms_by_source, out_path):
    sources = list(arms_by_source)
    fig, axes = plt.subplots(1, len(sources), figsize=(6.2 * len(sources), 4.2), squeeze=False)
    for ax, source in zip(axes[0], sources):
        arms = arms_by_source[source]
        types = _types(reports[arms[0]])
        x = np.arange(len(types))
        w = 0.8 / len(arms)
        for i, key in enumerate(arms):
            vals = [reports[key]["per_type"][t]["mean_latency"] for t in types]
            vals = [0.0 if (v is None or np.isnan(v)) else v for v in vals]
            ax.bar(x + (i - (len(arms) - 1) / 2) * w, vals, w * 0.92, color=_color(i),
                   label=_label(reports, key))
        ax.set_xticks(x)
        ax.set_xticklabels(types, rotation=20, ha="right")
        ax.set_title(source)
        ax.grid(axis="y", alpha=0.3)
    axes[0][0].set_ylabel("mean detection latency (steps)")
    axes[0][0].legend(fontsize=8)
    fig.suptitle("Detection latency, in steps from the first ground-truth step")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def healthy_panel(reports, arms_by_source, out_path):
    """Trial-level false-positive rate on healthy controls -- the specificity number that per-type
    precision cannot show, since every type shares the same healthy pool."""
    labels, vals, colors = [], [], []
    for source, arms in arms_by_source.items():
        for i, key in enumerate(arms):
            labels.append(f"{source}\n{_label(reports, key).replace(' (', chr(10) + '(')}")
            vals.append(reports[key]["healthy"]["false_positive_rate"])
            colors.append(_color(i))
    fig, ax = plt.subplots(figsize=(max(7, 2.1 * len(labels)), 4.6))
    bars = ax.bar(np.arange(len(labels)), vals, 0.55, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, max(1.0, max(vals) * 1.2))
    ax.set_ylabel("healthy trials flagged")
    ax.set_title("False-positive rate on healthy control trials (lower is better)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="dataset/processed/breakfast/llm_full_report.json")
    ap.add_argument("--out-dir", default="dataset/processed/breakfast/figures")
    args = ap.parse_args()

    payload = json.loads(Path(args.report).read_text())
    reports = payload["reports"]
    arms_by_source = _arms(reports)
    if not arms_by_source:
        raise SystemExit(f"no complete arms found in {args.report}; saw: {list(reports)}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []

    for source, arms in arms_by_source.items():
        p = out / f"compare_detection_{source}.png"
        detection_panel(reports, arms, source, p)
        written.append(p)
        p = out / f"compare_type_confusion_{source}.png"
        confusion_panel(reports, arms, source, p)
        written.append(p)

    for fn, name in ((steplevel_panel, "compare_steplevel.png"),
                     (type_accuracy_panel, "compare_type_accuracy.png"),
                     (correction_panel, "compare_correction.png"),
                     (latency_panel, "compare_latency.png"),
                     (healthy_panel, "compare_healthy_fpr.png")):
        p = out / name
        fn(reports, arms_by_source, p)
        written.append(p)

    for src, keys in arms_by_source.items():
        print(f"{src}: {len(keys)} arms -> {', '.join(_label(reports, k) for k in keys)}")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
