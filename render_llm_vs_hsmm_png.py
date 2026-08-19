"""One trial, all three detectors, in render_anomaly_png.py's layout.

Rows: what actually happened, what the detector was fed, then one row per detector -- the HSMM's
flagged ticks and narrated queries, and each LLM prompt variant's per-step verdicts. Cards below
carry each detector's own words.

The two are not natively comparable and the figure says so rather than hiding it: the HSMM emits a
value every TICK, the LLM answers once per STEP, so the HSMM row is drawn at tick resolution and
the LLM rows at step resolution over the same time axis. Everything an LLM row shows is read from
the response cache -- no requests.
"""
import argparse
import json
import textwrap
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

UNALT_C, FED_C, HSMM_C = "#d9bda8", "#c1712c", "#2e7d72"
HIT_C, FP_C, MISS_C, WIN_C = "#55a868", "#c44e52", "#dd8452", "#f6e3e3"
DEBRIS_C = "#9e9e9e"
SEV_C = {"low": "#55a868", "medium": "#dd8452", "high": "#c44e52"}
ROW_H, LINE_H = 1.0, 0.205


def _runs(verb_ids, noun_ids, lexicon):
    from cook_ad.llm import textify
    return textify.steps_from_ids(verb_ids, noun_ids, lexicon)


def _bar(ax, y, steps, color, fontsize=7.0):
    for st in steps:
        ax.add_patch(Rectangle((st.tick_start, y - 0.26), st.duration, 0.52,
                               facecolor=color, edgecolor="white", lw=1.1, zorder=2))
        if st.duration >= 4:
            ax.text(st.tick_start + st.duration / 2, y, f"{st.verb} {st.noun}",
                    ha="center", va="center", fontsize=fontsize, color="white", zorder=3)


def plot(unaltered, fed, gt_ticks, hsmm, llm_arms, title, out_path):
    """hsmm: (flag_ticks_by_sev, queries)
    llm_arms: [(label, steps, verdicts, gt_steps, debris_steps), ...]

    `debris_steps` (textify.injection_touched_steps) are steps the injection itself created or
    reshaped but which are not the ground-truth anomaly -- element_metrics excludes them from
    false-positive scoring entirely (docs/llm.md, eval/element_metrics.py). Drawn as a distinct
    grey "debris (excluded)" rather than red FALSE ALARM, so this figure and the corrected score
    agree: a flag here is neither a hit nor a false alarm, it's not scored at all.
    """
    t_max = fed[-1].tick_end
    rows = ["unaltered\n(what happened)", "observations\n(fed to detector)", "HSMM (joint)"]
    rows += [lab for lab, _, _, _, _ in llm_arms]
    n = len(rows)

    cards = []
    for i, q in enumerate(hsmm[1], 1):
        cards.append(f"{i}. [HSMM]  t={q['tick']}  {q['channel']}  {q['severity']}")
        cards += textwrap.wrap(q["text"], 104, initial_indent="     ", subsequent_indent="       ")
    k = len(hsmm[1])
    for lab, steps, verdicts, gt_steps, debris in llm_arms:
        for v in verdicts:
            if not v.is_anomaly and v.step_index not in gt_steps:
                continue
            k += 1
            st = steps[v.step_index]
            if v.is_anomaly and v.step_index in gt_steps:
                mark = "HIT"
            elif v.is_anomaly and v.step_index in debris:
                mark = "DEBRIS (excluded, not scored)"
            elif v.is_anomaly:
                mark = "FALSE ALARM"
            else:
                mark = "MISSED"
            cards.append(f"{k}. [{lab}]  step {v.step_index + 1} "
                         f"({st.verb} {st.noun}, {st.duration}s)  {mark}")
            cards += textwrap.wrap(v.raw.strip().replace("\n", " ") or "reported No Anomaly",
                                   104, initial_indent="     ", subsequent_indent="       ")
    cards = cards or ["(nothing to report)"]

    chart_h = 2.3 + ROW_H * n
    card_h = 0.4 + LINE_H * len(cards)
    fig_h = chart_h + card_h
    fig = plt.figure(figsize=(15.5, fig_h))
    ax = fig.add_axes([0.135, (card_h + 0.55) / fig_h, 0.845, (chart_h - 1.75) / fig_h])
    cax = fig.add_axes([0.02, 0.01, 0.96, (card_h - 0.06) / fig_h])
    cax.axis("off")

    y_of = {name: n - 1 - i for i, name in enumerate(rows)}
    if gt_ticks:
        ax.add_patch(Rectangle((gt_ticks[0], -0.55), gt_ticks[1] - gt_ticks[0] + 1, n + 0.1,
                               color=WIN_C, zorder=0))
        ax.axvline(gt_ticks[0], color="#c44e52", ls="--", lw=1.2, zorder=1)
        ax.text(gt_ticks[0] + 1, n - 0.45, "injected", color="#c44e52", fontsize=8, zorder=4)

    _bar(ax, y_of[rows[0]], unaltered, UNALT_C)
    _bar(ax, y_of[rows[1]], fed, FED_C)

    # HSMM row: tick-resolution flags as triangles, narrated queries as stars
    yh = y_of["HSMM (joint)"]
    ax.add_patch(Rectangle((0, yh - 0.16), t_max, 0.32, facecolor=HSMM_C, alpha=0.30,
                           edgecolor="none", zorder=1))
    for sev, ticks in hsmm[0].items():
        if ticks:
            ax.plot(ticks, [yh + 0.30] * len(ticks), marker="v", ls="none", ms=5,
                    color=SEV_C[sev], zorder=3)
    for i, q in enumerate(hsmm[1], 1):
        ax.plot([q["tick"]], [yh - 0.34], marker="*", ms=13, ls="none",
                color=SEV_C.get(q["severity"], "#555"), zorder=4)
        ax.text(q["tick"], yh - 0.60, str(i), ha="center", fontsize=7.5, fontweight="bold")

    # LLM rows: step-resolution verdicts
    for lab, steps, verdicts, gt_steps, debris in llm_arms:
        y = y_of[lab]
        for v in verdicts:
            st = steps[v.step_index]
            if v.is_anomaly and v.step_index in gt_steps:
                c = HIT_C
            elif v.is_anomaly and v.step_index in debris:
                c = DEBRIS_C
            elif v.is_anomaly:
                c = FP_C
            elif v.step_index in gt_steps:
                c = MISS_C
            else:
                continue
            ax.add_patch(Rectangle((st.tick_start, y - 0.24), st.duration, 0.48,
                                   facecolor=c, edgecolor="white", lw=1.0, zorder=2))

    for name, y in y_of.items():
        ax.text(-t_max * 0.012, y, name, ha="right", va="center", fontsize=8.5,
                fontweight="bold" if "HSMM" in name or "gemma" in name else "normal")
    ax.set_xlim(0, t_max)
    ax.set_ylim(-0.75, n - 0.35)
    ax.set_yticks([])
    ax.set_xlabel("tick (seconds)")
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    handles = [Rectangle((0, 0), 1, 1, color=WIN_C), Rectangle((0, 0), 1, 1, color=HIT_C),
               Rectangle((0, 0), 1, 1, color=FP_C), Rectangle((0, 0), 1, 1, color=DEBRIS_C),
               Rectangle((0, 0), 1, 1, color=MISS_C),
               plt.Line2D([], [], marker="v", ls="none", color="#888"),
               plt.Line2D([], [], marker="*", ls="none", color="#888", ms=11)]
    ax.legend(handles, ["injected window", "flagged (correct)", "flagged (false alarm)",
                        "flagged (debris, excluded)", "missed", "HSMM flagged tick",
                        "HSMM query"],
              loc="upper center", bbox_to_anchor=(0.5, 1.19), ncol=4, fontsize=7.6, frameon=False)
    ax.set_title(title, fontsize=11.5, pad=34)
    cax.text(0, 1, "\n".join(cards), va="top", ha="left", family="monospace", fontsize=7.5)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _hsmm_bits(jp, vocab, v_ids, n_ids, d_max):
    """Exactly export_anomaly.py's _score_and_narrate_joint call sequence, so the HSMM row here is
    the same computation the existing anomaly_*.png figures show, not a reimplementation.
    """
    from cook_ad.anomaly import narrate, quantile, surprise
    trace, jlp, r_hat, ltm, rho = surprise.compute_trace_joint(jp, v_ids, n_ids, d_max)
    flags = surprise.flag_joint(trace, jlp, r_hat, ltm)
    pi_all = surprise.compute_pi_all_joint(jlp, r_hat, v_ids, n_ids, d_max)
    queries = narrate.narrate_joint(trace, flags, vocab, jp, r_hat, v_ids, n_ids, jlp, ltm, pi_all)
    tables = quantile.threshold_tables_joint(jlp, r_hat, ltm, surprise.DEFAULT_ALPHA)
    sev = surprise.flagged_tick_severity(trace, flags, tables)
    by_sev = {"low": [], "medium": [], "high": []}
    for _ch, ticks in sev.items():
        for t, sv in ticks.items():
            by_sev.setdefault(sv, []).append(int(t))
    qs = [{"tick": int(q.tick), "channel": q.channel, "severity": q.severity, "text": q.text}
          for q in queries]
    return r_hat, rho, by_sev, qs


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
    ap.add_argument("--max-real", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--out-dir", default="dataset/processed/breakfast/figures_conv100")
    args = ap.parse_args()

    import os
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    jax.config.update("jax_enable_x64", True)
    from cook_ad.anomaly import narrate
    from cook_ad.data.config import load_config
    from cook_ad.hsmm import joint_params
    from cook_ad.llm import client as llm_client
    from cook_ad.llm import detect, prompts
    from cook_ad.synthetic import error_injection, generate
    import run_llm_eval as R

    d_max = load_config(args.config)["duration"]["d_max_ticks"]
    vocab = json.load(open(args.vocab))
    labels = json.load(open(args.labels))
    jp = joint_params.load_params(args.joint_params)
    marg = joint_params.collapse_to_marginal(jp)
    lex = narrate.Lexicon(vocab, marg)
    seqs = json.load(open(args.sequences))[: args.max_real]
    traj = [generate.trajectory_from_real_joint(jp, s["verb_ids"], s["noun_ids"], d_max)
            for s in seqs]
    pool = R.build_pool(traj, np.random.default_rng(args.seed), marg)
    print(f"{len(pool)} usable trials", flush=True)

    clients = {}
    for variant, lab in (("no-recipes", "gemma3 (no recipes)"),
                         ("with-recipes", "gemma3 (+ recipes)")):
        clients[lab] = (llm_client.ChatClient(
            model=args.model, base_url="http://localhost:11434/v1",
            cache_dir=Path(args.cache_dir) / args.model.replace("/", "_"),
            rpm=0, concurrency=1, max_requests=0),
            prompts.build_variant(variant, vocab, labels, args.protocol))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for error_type in error_injection.ERROR_TYPES:
        shortlist = []
        for idx, (traj_i, degraded) in enumerate(pool):
            deg = degraded[error_type]
            steps, gt, _, debris = R.steps_and_truth(traj_i, deg, lex)
            if not gt or len(steps) > args.max_steps:
                continue
            arms, ok = [], True
            for lab, (client, sysp) in clients.items():
                try:
                    arms.append((lab, steps, detect.run_trial(client, sysp, steps, vocab,
                                                              args.protocol), set(gt), debris))
                except Exception:
                    ok = False
                    break
            if not ok:
                continue
            hits = sum(any(v.is_anomaly and v.step_index in set(gt) for v in vs)
                       for _, _, vs, _, _ in arms)
            shortlist.append(((hits, -len(steps)), idx, traj_i, deg, steps, gt, arms))
        shortlist.sort(key=lambda r: r[0], reverse=True)

        best = None
        for cand in shortlist[:14]:
            _, _, _, dg, _, _, _ = cand
            nq = len(_hsmm_bits(jp, vocab, dg["verb_ids"], dg["noun_ids"], d_max)[3])
            # a comparison figure needs all three detectors to have said something
            if nq and (best is None or cand[0] > best[0]):
                best = cand
        if best is None:
            best = shortlist[0] if shortlist else None
        if best is None:
            print(f"  {error_type}: no fully-cached readable trial, skipped")
            continue
        _, idx, traj_i, deg, steps, gt, arms = best

        # HSMM on the SAME degraded stream
        v_ids, n_ids = deg["verb_ids"], deg["noun_ids"]
        r_hat, rho, by_sev, queries = _hsmm_bits(jp, vocab, v_ids, n_ids, d_max)
        unaltered = _runs(traj_i["verb_ids"], traj_i["noun_ids"], lex)
        fed = _runs(v_ids, n_ids, lex)
        trial_id = seqs[idx]["trial_id"]
        title = (f"{error_type} -- {trial_id} ({len(v_ids)} ticks, recipe r={int(r_hat)} "
                 f"conf={float(rho[r_hat]):.2f})  |  HSMM (tick-level) vs "
                 f"gemma3:27b conversational (step-level)")
        p = out / f"combined_narrate_{error_type}.png"
        plot(unaltered, fed, deg["window"], (by_sev, queries), arms, title, p)
        print(f"  {error_type}: {trial_id} -> {p}")


if __name__ == "__main__":
    main()
