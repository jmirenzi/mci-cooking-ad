"""Per-trial narration figures for the LLM baseline -- the analogue of render_anomaly_png.py.

Same idea as the HSMM version: one trial, the ground-truth window, what the detector flagged, and
the text it produced. Layout only, and it makes no requests: every reply is read back out of the
response cache, so this can be re-run and restyled for free.

    python render_llm_narrate_png.py --error-type substitution
"""
import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

OK_C, HIT_C, FP_C, MISS_C, GT_C = "#c9d3e0", "#55a868", "#c44e52", "#dd8452", "#f2e2b6"
LINE_H = 0.20


def _cards(rows):
    out = []
    for label, step_txt, reply, kind in rows:
        tag = {"hit": "CORRECT DETECTION", "fp": "FALSE ALARM", "miss": "MISSED"}[kind]
        head = f"[{label}] step {step_txt}  --  {tag}"
        out.append(head)
        for ln in textwrap.wrap(reply.strip().replace("\n", " ") or "(no anomaly reported)",
                                width=104, initial_indent="      ", subsequent_indent="        "):
            out.append(ln)
        out.append("")
    return out or ["(nothing flagged and nothing missed)"]


def plot_trial(steps, gt, arms, title, out_path):
    """steps: textify Steps. gt: ground-truth step indices. arms: [(label, verdicts)]."""
    t_max = steps[-1].tick_end
    card_rows = []
    for label, verdicts in arms:
        for v in verdicts:
            st = steps[v.step_index]
            txt = f"{v.step_index + 1}. {st.verb} {st.noun} for {st.duration}s"
            if v.is_anomaly:
                card_rows.append((label, txt, v.raw, "hit" if v.step_index in gt else "fp"))
            elif v.step_index in gt:
                card_rows.append((label, txt, "reported No Anomaly", "miss"))
    lines = _cards(card_rows)

    chart_h = 1.5 + 0.62 * len(arms)
    card_h = 0.42 + LINE_H * len(lines)
    fig_h = chart_h + card_h
    fig = plt.figure(figsize=(15, fig_h))
    ax = fig.add_axes([0.105, (card_h + 0.42) / fig_h, 0.875, (chart_h - 0.62) / fig_h])
    cax = fig.add_axes([0.015, 0.01, 0.97, (card_h - 0.08) / fig_h])
    cax.axis("off")

    # ground-truth window shading, drawn first so everything else sits on top
    for i in gt:
        ax.add_patch(Rectangle((steps[i].tick_start, -0.5), steps[i].duration,
                               len(arms) + 1.0, color=GT_C, zorder=0))
    # the observation stream, one box per step
    for st in steps:
        ax.add_patch(Rectangle((st.tick_start, len(arms) - 0.28), st.duration, 0.56,
                               facecolor=OK_C, edgecolor="white", lw=1.2, zorder=2))
        ax.text(st.tick_start + st.duration / 2, len(arms), f"{st.verb} {st.noun}\n{st.duration}s",
                ha="center", va="center", fontsize=7.2, zorder=3)
    # one verdict row per arm
    for r, (label, verdicts) in enumerate(arms):
        y = len(arms) - 1 - r
        for v in verdicts:
            st = steps[v.step_index]
            if v.is_anomaly:
                c = HIT_C if v.step_index in gt else FP_C
            elif v.step_index in gt:
                c = MISS_C
            else:
                continue
            ax.add_patch(Rectangle((st.tick_start, y - 0.2), st.duration, 0.4,
                                   facecolor=c, edgecolor="white", lw=1.0, zorder=2))
        ax.text(-t_max * 0.015, y, label, ha="right", va="center", fontsize=8.5)
    ax.text(-t_max * 0.015, len(arms), "observed", ha="right", va="center", fontsize=8.5,
            fontweight="bold")

    ax.set_xlim(0, t_max)
    ax.set_ylim(-0.6, len(arms) + 0.6)
    ax.set_yticks([])
    ax.set_xlabel("time (seconds)")
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    handles = [Rectangle((0, 0), 1, 1, color=GT_C), Rectangle((0, 0), 1, 1, color=HIT_C),
               Rectangle((0, 0), 1, 1, color=FP_C), Rectangle((0, 0), 1, 1, color=MISS_C)]
    ax.legend(handles, ["ground-truth anomaly", "flagged (correct)", "flagged (false alarm)",
                        "missed"], loc="upper center", bbox_to_anchor=(0.5, 1.42), ncol=4,
              fontsize=8, frameon=False)
    ax.set_title(title, fontsize=11, pad=26)
    cax.text(0, 1, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=7.6)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/breakfast.yaml")
    ap.add_argument("--joint-params", default="dataset/processed/breakfast/joint_params.npz")
    ap.add_argument("--sequences", default="dataset/processed/breakfast/sequences.json")
    ap.add_argument("--labels", default="dataset/processed/breakfast/labels.json")
    ap.add_argument("--vocab", default="dataset/processed/breakfast/vocab.json")
    ap.add_argument("--cache-dir", default="dataset/processed/breakfast/llm_cache")
    ap.add_argument("--model", default="gemma3:27b")
    ap.add_argument("--protocol", default="conversational")
    ap.add_argument("--max-real", type=int, default=100,
                    help="MUST match the run whose replies are cached, or the rng draws differ "
                         "and the injections will not be the same ones the model answered about")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="dataset/processed/breakfast/figures_conv100")
    args = ap.parse_args()

    import os
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    jax.config.update("jax_enable_x64", True)
    import numpy as _np
    from cook_ad.anomaly import narrate
    from cook_ad.hsmm import joint_params
    from cook_ad.llm import client as llm_client
    from cook_ad.llm import detect, prompts
    from cook_ad.synthetic import error_injection, generate
    import run_llm_eval as R

    from cook_ad.data.config import load_config
    d_max = load_config(args.config)["duration"]["d_max_ticks"]
    vocab = json.load(open(args.vocab))
    labels = json.load(open(args.labels))
    jp = joint_params.load_params(args.joint_params)
    marg = joint_params.collapse_to_marginal(jp)
    lex = narrate.Lexicon(vocab, marg)
    seqs = json.load(open(args.sequences))[: args.max_real]

    traj = [generate.trajectory_from_real_joint(jp, s["verb_ids"], s["noun_ids"], d_max)
            for s in seqs]
    pool = R.build_pool(traj, _np.random.default_rng(args.seed), marg)
    print(f"{len(pool)} usable trials", flush=True)

    # max_requests=0: every reply must already be cached. A miss raises instead of silently
    # spending, which also catches a --max-real that does not match the cached run.
    clients = {}
    for variant in ("no-recipes", "with-recipes"):
        clients[variant] = (
            llm_client.ChatClient(model=args.model, base_url="http://localhost:11434/v1",
                                  cache_dir=Path(args.cache_dir) / args.model.replace("/", "_"),
                                  rpm=0, concurrency=1, max_requests=0),
            prompts.build_variant(variant, vocab, labels, args.protocol),
        )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for error_type in error_injection.ERROR_TYPES:
        best = None
        for traj_i, degraded in pool:
            steps, gt, _ = R.steps_and_truth(traj_i, degraded[error_type], lex)
            if not gt or len(steps) > 9:      # keep figures readable
                continue
            arms, ok = [], True
            for variant, (client, sysp) in clients.items():
                try:
                    vs = detect.run_trial(client, sysp, steps, vocab, args.protocol)
                except Exception:
                    ok = False
                    break
                arms.append((variant, vs))
            if not ok:
                continue
            # prefer a trial where at least one arm gets it right, and few false alarms
            hits = sum(any(v.is_anomaly and v.step_index in gt for v in vs) for _, vs in arms)
            fps = sum(sum(v.is_anomaly and v.step_index not in gt for v in vs) for _, vs in arms)
            score = (hits, -fps, -len(steps))
            if best is None or score > best[0]:
                best = (score, steps, gt, arms, degraded[error_type])
        if best is None:
            print(f"  {error_type}: no fully-cached readable trial found, skipped")
            continue
        _, steps, gt, arms, deg = best
        title = (f"{error_type} -- gemma3:27b, conversational  "
                 f"(ground-truth step{'s' if len(gt) > 1 else ''} "
                 f"{', '.join(str(i + 1) for i in gt)} of {len(steps)})")
        p = out / f"llm_narrate_{error_type}.png"
        plot_trial(steps, gt, arms, title, p)
        written.append(p)
        print(f"  {error_type}: wrote {p}")
    for w in written:
        print(w)


if __name__ == "__main__":
    main()
