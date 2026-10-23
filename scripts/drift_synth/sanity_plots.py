"""Sanity plots for the drift generator, before any detector touches the data
(docs/drift_synth.md). Writes to runs/drift_synth/reports/sanity/:

- dose_response.png   drift_magnitude vs |delta theta| per knob, averaged over the recipes,
                      with the Monte Carlo noise floor (same theta, another seed);
- isolation.png       knob x feature group, change at |delta| = 0.45 as a multiple of that floor;
- lengths.png         session length and action duration, synthetic vs CC4D / EPIC;
- examples.md         10 synthetic sessions next to real EPIC sequences and CC4D step lists;
- sanity.json         the numbers behind the figures.

    ./py scripts/drift_synth/sanity_plots.py [--n-mc 150]
"""
import argparse
import csv
import glob
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from cook_ad.drift_synth import ground_truth as GT  # noqa: E402
from cook_ad.drift_synth.assets import load_assets  # noqa: E402
from cook_ad.drift_synth.knobs import KNOBS  # noqa: E402

LEVELS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.45)
BASE = 0.5
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]  # dataviz slots 1-6
INK, INK2, GRID = "#1a1a19", "#5f5e58", "#e4e3dd"
LOGISTIC_VERBS = ("take", "put", "wash")


def _style(ax):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)


def _subset_js(s0, s1, keep):
    p = {k: v for k, v in s0.marginal.items() if keep(k)}
    q = {k: v for k, v in s1.marginal.items() if keep(k)}
    zp, zq = sum(p.values()), sum(q.values())
    if not zp or not zq:
        return 0.0
    return GT.js_divergence({k: v / zp for k, v in p.items()}, {k: v / zq for k, v in q.items()})


def _features(s0, s1):
    m = GT.drift_magnitude(s0, s1)
    is_log = lambda t: t.split(" ")[0] in LOGISTIC_VERBS  # noqa: E731
    return {"marginal_js": m["marginal_js"], "bigram_js": m["bigram_js"],
            "duration_w1": m["duration_w1_mean"],
            "core_marginal_js": _subset_js(s0, s1, lambda t: not is_log(t)),
            "logistics_marginal_js": _subset_js(s0, s1, is_log)}


def measure(assets, n_mc):
    """{knob: {level: {feature: mean over recipes}}} plus the null floor."""
    recipes = sorted(assets.graphs)
    base = {k: BASE for k in KNOBS}
    s_base = {r: GT.observable_stats(r, base, n_mc, 1, assets) for r in recipes}
    s_null = {r: GT.observable_stats(r, base, n_mc, 2, assets) for r in recipes}

    def mean_features(other):
        rows = [_features(s_base[r], other[r]) for r in recipes]
        return {k: float(np.mean([row[k] for row in rows])) for k in rows[0]}

    out = {"null": mean_features(s_null), "knobs": {}}
    for knob in KNOBS:
        out["knobs"][knob] = {}
        for lvl in LEVELS:
            th = dict(base, **{knob: BASE + lvl})
            other = s_null if lvl == 0 else {r: GT.observable_stats(r, th, n_mc, 2, assets) for r in recipes}
            out["knobs"][knob][lvl] = mean_features(other)
        print(f"  {knob} done")
    return out


def plot_dose(res, path):
    feats = [("marginal_js", "token marginals, JS (bits)"), ("bigram_js", "bigrams, JS (bits)"),
             ("duration_w1", "log duration, mean W1")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    for ax, (f, title) in zip(axes, feats):
        _style(ax)
        for color, knob in zip(SERIES, KNOBS):
            ys = [res["knobs"][knob][lv][f] for lv in LEVELS]
            ax.plot(LEVELS, ys, color=color, linewidth=2, marker="o", markersize=4, label=knob)
        ax.axhline(res["null"][f], color=INK2, linestyle="--", linewidth=1,
                   label="MC noise (rerun at θ = 0.5)")
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        ax.set_xlabel("|Δθ| from θ = 0.5", color=INK2, fontsize=9)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle("Dose-response: observable drift per knob (mean over 18 recipes)", color=INK,
                 fontsize=11, x=0.01, ha="left")
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_isolation(res, path):
    feats = ["core_marginal_js", "logistics_marginal_js", "bigram_js", "duration_w1"]
    labels = ["core\ntokens", "take / put\n/ wash", "bigrams", "log\ndurations"]
    mat = np.array([[res["knobs"][k][LEVELS[-1]][f] / max(res["null"][f], 1e-12) for f in feats]
                    for k in KNOBS])
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    im = ax.imshow(np.log10(np.maximum(mat, 1e-3)), cmap="Blues", vmin=0, vmax=max(2, np.log10(mat.max())))
    ax.set_xticks(range(len(feats)), labels, fontsize=9, color=INK2)
    ax.set_yticks(range(len(KNOBS)), KNOBS, fontsize=9, color=INK2)
    for i in range(len(KNOBS)):
        for j in range(len(feats)):
            dark = np.log10(max(mat[i, j], 1e-3)) > 1.2
            ax.text(j, i, f"{mat[i, j]:.1f}×", ha="center", va="center", fontsize=9,
                    color="#ffffff" if dark else INK)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("log10(change / MC noise)", color=INK2, fontsize=9)
    cb.ax.tick_params(colors=INK2, labelsize=8)
    fig.suptitle(f"Isolation: change at |Δθ| = {LEVELS[-1]} as a multiple of Monte Carlo noise\n"
                 "(1× = indistinguishable from a rerun at the same θ)", color=INK, fontsize=10,
                 x=0.01, ha="left")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return mat


def real_lengths(cc4d, epic_processed):
    errors = json.loads((Path(cc4d) / "annotation_json" / "error_annotations.json").read_text())
    normal = {e["recording_id"] for e in errors if not e["is_error"]}
    with open(Path(cc4d) / "metadata" / "video_information.csv", newline="") as f:
        cc = [float(r["duration(sec)"]) / 60 for r in csv.DictReader(f) if r["recording_id"] in normal]
    seqs = json.loads((Path(epic_processed) / "sequences.json").read_text())
    ep = [len(s["verb_ids"]) * 0.5 / 60 for s in seqs]
    return np.array(cc), np.array(ep)


def synth_sessions(root):
    out = []
    for f in sorted(glob.glob(f"{root}/null/seed0/*/sessions/*.json")):
        out.append(json.loads(Path(f).read_text()))
    return out


def plot_lengths(syn, cc, ep, epic_durs, path):
    syn_len = np.array([max((e[1] for e in s["events"]), default=0) / 60 for s in syn])
    syn_dur = np.array([e[1] - e[0] for s in syn for e in s["events"]])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    ax = axes[0]
    _style(ax)
    bins = np.linspace(0, 40, 41)
    for data, color, label in ((syn_len, SERIES[0], f"synthetic (null, n={len(syn_len)})"),
                               (cc, SERIES[1], f"CC4D normal recordings (n={len(cc)})"),
                               (ep, SERIES[2], f"EPIC-100 sessions, 2-15 min (n={len(ep)})")):
        ax.hist(np.clip(data, 0, 40), bins=bins, density=True, histtype="step", linewidth=2,
                color=color, label=f"{label}, median {np.median(data):.1f}")
    ax.set_xlabel("session length (min, clipped at 40)", color=INK2, fontsize=9)
    ax.set_title("Session length", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    ax = axes[1]
    _style(ax)
    bins = np.logspace(-1, 2.5, 50)
    for data, color, label in ((syn_dur, SERIES[0], "synthetic events"),
                               (epic_durs, SERIES[2], "EPIC-100 narrations")):
        ax.hist(data, bins=bins, density=True, histtype="step", linewidth=2, color=color,
                label=f"{label}, median {np.median(data):.1f} s")
    ax.set_xscale("log")
    ax.set_xlabel("action duration (s, log)", color=INK2, fontsize=9)
    ax.set_title("Action duration", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return {"synthetic_min_median": float(np.median(syn_len)), "cc4d_min_median": float(np.median(cc)),
            "epic_min_median": float(np.median(ep)), "synthetic_event_s_median": float(np.median(syn_dur)),
            "epic_event_s_median": float(np.median(epic_durs))}


def examples(syn, assets, epic_root, path, n=10, seed=0):
    from cook_ad.data.parse_epic import load_classes, load_narrations
    rng = random.Random(seed)
    picks = rng.sample(syn, n)
    vk, nk, _ = load_classes(epic_root)
    videos = load_narrations(epic_root)
    lines = ["# Example sessions: synthetic vs real", "",
             "Synthetic sessions from `null`, seed 0 (events after recognizer noise, consecutive "
             "repeats collapsed). Each is followed by the recipe's CC4D step list in the sampled "
             "order. Real EPIC-100 sequences (first 25 narrations of random videos) are at the end, "
             "for granularity.", ""]
    for s in picks:
        toks = [f"{e[2]} {e[3]}" for e in s["events"]]
        runs = [t for i, t in enumerate(toks) if i == 0 or t != toks[i - 1]]
        g = assets.graphs[s["recipe_id"]]
        lines += [f"## {s['trial_id']}: {s['recipe_id']} ({len(runs)} runs, "
                  f"{max(e[1] for e in s['events']) / 60:.1f} min)", "",
                  "**synthetic:** " + ", ".join(runs), "",
                  "**CC4D steps (sampled order):** " + " → ".join(
                      g.steps[k].description.split("-", 1)[-1][:50] for k in s["meta"]["order"]), ""]
    lines += ["## Real EPIC-100 sequences", ""]
    for vid in rng.sample(sorted(videos), 5):
        narr = videos[vid]["narrations"][:25]
        lines.append(f"- **{vid}:** " + ", ".join(f"{vk[v]} {nk[x]}" for _, _, v, x in narr))
    Path(path).write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", default="configs/drift_synth/generator.yaml")
    ap.add_argument("--n-mc", type=int, default=150)
    ap.add_argument("--out", default="runs/drift_synth/reports/sanity")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    assets = load_assets(args.generator)
    p = assets.gen["paths"]

    print("dose-response ...")
    res = measure(assets, args.n_mc)
    plot_dose(res, out / "dose_response.png")
    mat = plot_isolation(res, out / "isolation.png")

    print("lengths ...")
    from cook_ad.data.parse_epic import load_narrations
    epic_durs = np.array([t1 - t0 for e in load_narrations(p["epic_annotations"]).values()
                          for t0, t1, _, _ in e["narrations"] if t1 > t0])
    cc, ep = real_lengths(p["cc4d"], Path(p["vocab"]).parent)
    syn = synth_sessions(p["out"])
    lengths = plot_lengths(syn, cc, ep, epic_durs, out / "lengths.png")
    examples(syn, assets, p["epic_annotations"], out / "examples.md")

    (out / "sanity.json").write_text(json.dumps(
        {"levels": LEVELS, "null": res["null"],
         "knobs": {k: {str(lv): v for lv, v in d.items()} for k, d in res["knobs"].items()},
         "isolation_ratio": {k: dict(zip(["core_marginal", "logistics_marginal", "bigram",
                                           "duration"], map(float, row)))
                             for k, row in zip(KNOBS, mat)},
         "lengths": lengths}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
