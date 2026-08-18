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
    """{source: {"hsmm": key, "llm": key}} from arm keys like 'real/hsmm-joint'."""
    out = {}
    for key in reports:
        if reports[key].get("incomplete"):
            continue
        source, _, arm = key.partition("/")
        out.setdefault(source, {})[("hsmm" if arm.startswith("hsmm") else "llm")] = key
    return {s: a for s, a in out.items() if "hsmm" in a and "llm" in a}


def _label(reports, key):
    r = reports[key]
    if "/hsmm" in key:
        return f"HSMM ({key.split('hsmm-')[-1]})"
    return f"LLM ({r.get('client', {}).get('model', r.get('prompt_variant', 'llm'))})"


def _types(report):
    return list(report["per_type"].keys())


def detection_panel(reports, arms, source, out_path):
    """Recall and precision per error type, both detectors, side by side."""
    hs, ll = reports[arms["hsmm"]], reports[arms["llm"]]
    types = _types(hs)
    x = np.arange(len(types))
    w = 0.2

    fig, axes = plt.subplots(1, 2, figsize=(max(11, 1.9 * len(types)), 4.6), sharey=True)
    for ax, metric in zip(axes, ("recall", "precision")):
        ax.bar(x - 1.5 * w, [hs["per_type"][t][metric] for t in types], w * 1.4,
               label=_label(reports, arms["hsmm"]), color=HSMM_C)
        ax.bar(x + 0.5 * w, [ll["per_type"][t][metric] for t in types], w * 1.4,
               label=_label(reports, arms["llm"]), color=LLM_C)
        ax.set_xticks(x)
        ax.set_xticklabels(types, rotation=20, ha="right")
        ax.set_ylim(0, 1.08)
        ax.set_title(metric)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("rate")
    axes[0].legend(loc="lower left", fontsize=9)
    fig.suptitle(f"Per-error-type detection, step-level -- {source} "
                 f"(n={hs['per_type'][types[0]]['n']} trials/type)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def steplevel_panel(reports, arms_by_source, out_path):
    """Pooled step-level precision / recall / F1, every arm, plus the chance-precision line.

    The chance line is the point of this figure: precision has to be read against the base rate of
    anomalous steps, or a detector that simply flags everything looks respectable.
    """
    labels, prec, rec, f1, chance = [], [], [], [], []
    for source, arms in arms_by_source.items():
        for which in ("hsmm", "llm"):
            r = reports[arms[which]]
            sl = r["step_level"]
            p, q = sl["precision"], sl["recall"]
            labels.append(f"{source}\n{'HSMM' if which == 'hsmm' else 'LLM'}")
            prec.append(p)
            rec.append(q)
            f1.append(2 * p * q / (p + q) if (p + q) else 0.0)
            # Reported by evaluate_steps rather than derived here: tp+fp+fn is NOT the step count
            # and would give the two arms different baselines for the same data.
            chance.append(sl.get("chance_precision", np.nan))

    x = np.arange(len(labels))
    w = 0.26
    fig, ax = plt.subplots(figsize=(max(8, 2.0 * len(labels)), 4.6))
    ax.bar(x - w, prec, w, label="precision", color=HSMM_C)
    ax.bar(x, rec, w, label="recall", color=LLM_C)
    ax.bar(x + w, f1, w, label="F1", color="#55a868")
    for xi, c in zip(x, chance):
        ax.plot([xi - 1.5 * w, xi + 1.5 * w], [c, c], color="#c44e52", ls="--", lw=1.4,
                label="precision at chance" if xi == 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
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
    hs = reports[arms["hsmm"]]
    types = _types(hs)
    rows = list(hs["channels"])  # predicted types + "none"

    fig, axes = plt.subplots(1, 2, figsize=(max(11, 2.0 * len(types)), max(4.2, 0.62 * len(rows))))
    for ax, key in zip(axes, ("hsmm", "llm")):
        rep = reports[arms[key]]
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
        ax.set_title(_label(reports, arms[key]))
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
        types = _types(reports[arms["hsmm"]])
        x = np.arange(len(types))
        w = 0.36
        ax.bar(x - w / 2, _diag(reports[arms["hsmm"]], types), w,
               label=_label(reports, arms["hsmm"]), color=HSMM_C)
        ax.bar(x + w / 2, _diag(reports[arms["llm"]], types), w,
               label=_label(reports, arms["llm"]), color=LLM_C)
        ax.set_xticks(x)
        ax.set_xticklabels(types, rotation=20, ha="right")
        ax.set_ylim(0, 1.08)
        ax.set_title(source)
        ax.grid(axis="y", alpha=0.3)
    axes[0][0].set_ylabel("fraction of detections given the correct type")
    axes[0][0].legend(fontsize=9, loc="upper right")
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
        types = _types(reports[arms["hsmm"]])
        x = np.arange(len(types))
        w = 0.2
        for off, (which, metric, c, hatch) in enumerate([
            ("hsmm", "verb_noun_accuracy", HSMM_C, None),
            ("hsmm", "duration_accuracy", HSMM_C, "//"),
            ("llm", "verb_noun_accuracy", LLM_C, None),
            ("llm", "duration_accuracy", LLM_C, "//"),
        ]):
            vals = [reports[arms[which]]["correction_accuracy"][t][metric] for t in types]
            vals = [0.0 if (v is None or np.isnan(v)) else v for v in vals]
            ax.bar(x + (off - 1.5) * w, vals, w, color=c, hatch=hatch, edgecolor="white",
                   label=f"{'HSMM' if which == 'hsmm' else 'LLM'} "
                         f"{'verb+noun' if 'verb' in metric else 'duration'}")
        ax.set_xticks(x)
        ax.set_xticklabels(types, rotation=20, ha="right")
        ax.set_ylim(0, 1.08)
        ax.set_title(source)
        ax.grid(axis="y", alpha=0.3)
    axes[0][0].set_ylabel("accuracy of proposed correction")
    axes[0][0].legend(fontsize=8, ncol=2)
    fig.suptitle("'Correct move would have been ...' -- accuracy against the pre-injection step")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def latency_panel(reports, arms_by_source, out_path):
    sources = list(arms_by_source)
    fig, axes = plt.subplots(1, len(sources), figsize=(6.2 * len(sources), 4.2), squeeze=False)
    for ax, source in zip(axes[0], sources):
        arms = arms_by_source[source]
        types = _types(reports[arms["hsmm"]])
        x = np.arange(len(types))
        w = 0.36
        for off, which in ((-0.5, "hsmm"), (0.5, "llm")):
            vals = [reports[arms[which]]["per_type"][t]["mean_latency"] for t in types]
            vals = [0.0 if (v is None or np.isnan(v)) else v for v in vals]
            ax.bar(x + off * w, vals, w, color=HSMM_C if which == "hsmm" else LLM_C,
                   label=_label(reports, arms[which]))
        ax.set_xticks(x)
        ax.set_xticklabels(types, rotation=20, ha="right")
        ax.set_title(source)
        ax.grid(axis="y", alpha=0.3)
    axes[0][0].set_ylabel("mean detection latency (steps)")
    axes[0][0].legend(fontsize=9)
    fig.suptitle("Detection latency, in steps from the first ground-truth step")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def healthy_panel(reports, arms_by_source, out_path):
    """Trial-level false-positive rate on healthy controls -- the specificity number that per-type
    precision cannot show, since every type shares the same healthy pool."""
    labels, vals, colors = [], [], []
    for source, arms in arms_by_source.items():
        for which in ("hsmm", "llm"):
            labels.append(f"{source}\n{'HSMM' if which == 'hsmm' else 'LLM'}")
            vals.append(reports[arms[which]]["healthy"]["false_positive_rate"])
            colors.append(HSMM_C if which == "hsmm" else LLM_C)
    fig, ax = plt.subplots(figsize=(max(6, 1.7 * len(labels)), 4.2))
    bars = ax.bar(np.arange(len(labels)), vals, 0.55, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
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
        raise SystemExit(f"no source in {args.report} has BOTH a complete hsmm and llm arm; "
                         f"found: {list(reports)}")

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

    print(f"sources compared: {', '.join(arms_by_source)}")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
