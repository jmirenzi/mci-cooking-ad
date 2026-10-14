"""One Breakfast trial, every detector, with the Hindsight narration (docs/eval.md §4).

Lanes, top to bottom: the observed runs (labelled), the injected error's span, then one lane per
detector with a marker at every alarm tick. Hindsight's narrated questions are numbered on its
lane and printed as cards below. The trial comes from the LLM pool, so every LLM lane is read from
the response cache (no requests with --max-requests 0, the default).

    ./py scripts/figures/trial_figure.py --trial 0 --condition omission --out runs/fig_trial0.png
"""
import argparse
import json
import os
import textwrap
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

jax.config.update("jax_enable_x64", True)

from cook_ad.core import OperatingPoint, Trial  # noqa: E402
from cook_ad.data.config import load_config  # noqa: E402
from cook_ad.detectors import llm_predictive as llm_pred  # noqa: E402
from cook_ad.detectors.hsmm_hindsight import HindsightHSMMDetector  # noqa: E402
from cook_ad.detectors.hsmm_lag import LagHSMMDetector  # noqa: E402
from cook_ad.detectors.hsmm_live import LiveHSMMDetector  # noqa: E402
from cook_ad.detectors.llm_verdict import LLMVerdictDetector  # noqa: E402
from cook_ad.detectors.narrate import Lexicon, narrate  # noqa: E402
from cook_ad.detectors.thresholds import BadSets  # noqa: E402
from cook_ad.eval import metric_g, pools  # noqa: E402
from cook_ad.hsmm import params as hparams  # noqa: E402
from cook_ad.hsmm.model import HSMM  # noqa: E402
from cook_ad.llm import client as llm_client  # noqa: E402

B = "dataset/processed/breakfast"
SEV_C = {"low": "#55a868", "medium": "#dd8452", "high": "#c44e52"}
LANE_C = {"hindsight": "#2e7d72", "live": "#3b6fb6", "lag": "#7a5cb8", "llm": "#b5651d"}


def detectors(args, hsmm, vocab, labels, lex):
    """{lane name: detector} at the operating points from configs/operating_points.yaml, or the
    command-line ones if the file has none."""
    import yaml
    ops = (yaml.safe_load(open(args.operating_points)) or {}).get("breakfast", {}).get(
        args.model_name, {}) if Path(args.operating_points).exists() else {}
    a_h = ops.get("hindsight", {}).get("alpha", args.alpha)
    l0 = ops.get("live_L0", {"alpha": args.alpha, "kappa": args.kappa})
    l5 = ops.get("live_L5", l0)
    bad = BadSets(hsmm, sorted({a_h, l0["alpha"], l5["alpha"]}))
    cache = Path(f"{B}/llm_cache") / "gemma3:27b"
    client = lambda t, m: llm_client.ChatClient(  # noqa: E731
        model="gemma3:27b", base_url=args.base_url, cache_dir=cache,
        max_requests=args.max_requests, temperature=t, max_tokens=m)
    pred = llm_pred.LLMPredictiveDetector(client(llm_pred.TEMPERATURE, llm_pred.MAX_TOKENS), vocab,
                                          lex, variant="no-recipes", labels=labels, n_shot=3,
                                          prompt_unit="tick")
    train = pools.split_sequences(f"{B}/sequences.json", f"{B}/split.json", "train")
    pred.fit([Trial(s["trial_id"], np.asarray(s["verb_ids"]), np.asarray(s["noun_ids"]))
              for s in train])
    verdict = LLMVerdictDetector(client(0.0, None), vocab, lex, unit="tick", variant="no-recipes",
                                 labels=labels)
    verdict.fit([])
    return {
        f"HSMM hindsight (α={a_h:g})": HindsightHSMMDetector(hsmm, bad, OperatingPoint(a_h)),
        f"HSMM live (α={l0['alpha']:g}, κ={l0['kappa']})":
            LiveHSMMDetector(hsmm, bad, OperatingPoint(l0["alpha"], l0["kappa"])),
        f"HSMM lag 5 (α={l5['alpha']:g}, κ={l5['kappa']})":
            LagHSMMDetector(hsmm, bad, OperatingPoint(l5["alpha"], l5["kappa"]), 5),
        "LLM predictive (no recipes, 3-shot)": pred,
        "LLM verdict (incremental, tick, no recipes)": verdict,
    }


def draw(trial, lex, spans, lanes, queries, title, out):
    n = len(trial.verbs)
    fig, ax = plt.subplots(figsize=(max(10, n / 12), 1.2 + 0.55 * (len(lanes) + 2)))
    starts = metric_g.run_starts(trial.verbs, trial.nouns)
    ends = list(starts[1:]) + [n]
    for i, (a, b) in enumerate(zip(starts, ends)):
        ax.add_patch(plt.Rectangle((a, 0.1), b - a, 0.8, color=plt.cm.tab20(i % 20), alpha=0.6))
        if b - a >= 4:
            ax.text((a + b) / 2, 0.5, lex.phrase(trial.verbs[a], trial.nouns[a]), ha="center",
                    va="center", fontsize=6, rotation=0 if b - a > 20 else 90)
    for s0, s1 in spans:
        ax.axvspan(s0, s1 + 1, color="#f2c6c6", alpha=0.6, zorder=0)
    for row, (name, alarms) in enumerate(lanes.items(), start=1):
        y = -row
        color = next(c for k, c in LANE_C.items() if k in name.lower())
        ax.scatter([a.tick + 0.5 for a in alarms], [y] * len(alarms), marker="v", s=14, color=color)
        if "hindsight" in name:
            for j, q in enumerate(queries, start=1):
                ax.scatter(q.tick + 0.5, y, marker="v", s=90, color=SEV_C[q.severity], zorder=3)
                ax.text(q.tick + 0.5, y + 0.3, str(j), ha="center", fontsize=7)
    ax.set_yticks([0.5] + [-r for r in range(1, len(lanes) + 1)])
    ax.set_yticklabels(["observed runs"] + list(lanes))
    ax.set_xlim(0, n)
    ax.set_ylim(-len(lanes) - 0.7, 1.2)
    ax.set_xlabel("tick (s)")
    ax.set_title(title, fontsize=9)
    cards = "\n".join(textwrap.fill(f"{j}. [t={q.tick}, {q.severity}] {q.text}", 140)
                      for j, q in enumerate(queries, start=1)) or "(no narrated questions)"
    fig.text(0.01, -0.02, cards, fontsize=7, va="top", family="monospace")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"written to {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trial", type=int, default=0, help="index into the LLM pool (97 usable)")
    ap.add_argument("--condition", default="healthy", choices=["healthy", *pools.ERROR_TYPES])
    ap.add_argument("--params", default=f"{B}/joint_params_train_lexhard.npz")
    ap.add_argument("--model-name", default="lexhard", help="key in operating_points.yaml")
    ap.add_argument("--operating-points", default="configs/operating_points.yaml")
    ap.add_argument("--alpha", type=float, default=0.02)
    ap.add_argument("--kappa", type=float, default=0.5)
    ap.add_argument("--base-url", default="http://localhost:11435/v1")
    ap.add_argument("--max-requests", type=int, default=0)
    ap.add_argument("--out", default="runs/trial_figure.png")
    args = ap.parse_args()

    d_max = load_config("configs/breakfast.yaml")["duration"]["d_max_ticks"]
    vocab, labels = json.load(open(f"{B}/vocab.json")), json.load(open(f"{B}/labels.json"))
    test = pools.split_sequences(f"{B}/sequences.json", f"{B}/split.json", "test", limit=150)
    pool, keep = pools.injected_per_trial(hparams.load(f"{B}/joint_params.npz"), test, d_max)
    healthy, injected = pool[args.trial]
    traj = healthy if args.condition == "healthy" else injected[args.condition]
    trial = Trial(test[keep[args.trial]]["trial_id"], np.asarray(traj["verb_ids"]),
                  np.asarray(traj["noun_ids"]))
    spans = [tuple(map(int, s)) for s in traj.get("spans", [])] if args.condition != "healthy" else []

    hsmm = HSMM.load(args.params, d_max)
    lex = Lexicon(vocab, hsmm.params)
    lanes, queries = {}, []
    for name, det in detectors(args, hsmm, vocab, labels, lex).items():
        detection = det.detect_trial(trial)
        lanes[name] = detection.alarms
        if isinstance(det, HindsightHSMMDetector):
            queries = narrate(detection, det, vocab)
    draw(trial, lex, spans, lanes, queries,
         f"{trial.trial_id}, {args.condition}" + (f" at {spans}" if spans else ""), args.out)


if __name__ == "__main__":
    main()
