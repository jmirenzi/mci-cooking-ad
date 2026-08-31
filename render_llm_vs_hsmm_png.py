"""One trial, all three detectors, in render_anomaly_png.py's layout.

Rows: what actually happened, what the detector was fed, then one row per detector -- the HSMM's
flagged ticks and narrated queries, and each LLM prompt variant's verdicts. Cards below carry each
detector's own words. Everything an LLM row shows is read from the response cache -- no requests.

TWO DISPLAY MODES for the LLM lanes, selected by `--unit` / `plot(..., unit=)`:

  unit="step" (default, unchanged): the LLM answered once per llm/textify STEP (a run-length-
    encoded action), so its lane is drawn as a line spanning the step's full duration with a
    triangle at the end -- there is no shorter unit to mark. This is what every figure under
    figures_conv100/ and similar step-unit directories was built with, and it still is: nothing
    about this mode changed.

  unit="tick": the LLM answered once per SECOND (llm/textify.ticks_from_ids), the same resolution
    the HSMM's channels emit at, so both lanes can finally share ONE time scale -- which is the
    whole reason this mode exists. Its lane borrows the HSMM lane's own visual grammar rather than
    stretching the step-mode line-and-triangle over 1-tick spans: a small unlabeled triangle for
    every flagged tick (HSMM's baseline marker), and one BIGGER numbered triangle per maximal RUN
    of consecutive same-verdict ticks (HSMM's narrated-query marker), carded once per run rather
    than once per tick. That collapse is necessary, not cosmetic: at the tick unit the LLM can
    flag a third of a trial (measured on the repetition condition), and numbering every one of
    those ticks -- which is what the step-mode code does, correctly, for the ~7 steps/trial it was
    built for -- would produce hundreds of overlapping numbers and a card list nobody would read.

Loading tick-unit data for a REAL run (rather than the first --max-real rows of sequences.json)
needs the same split file, joint-params fit, and --base-url the run itself used, because the
response cache is keyed on exactly that: cook_ad.llm.client.ChatClient._cache_key hashes
(base_url, model, temperature, messages). Pass --split-file/--split-part to reproduce
run_llm_eval.py's real-trial pool bit for bit; get the base_url wrong and every lookup misses.
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402
from matplotlib.colorbar import ColorbarBase  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

UNALT_C, FED_C, HSMM_C = "#d9bda8", "#c1712c", "#2e7d72"
WIN_C = "#f6e3e3"
SEV_C = {"low": "#55a868", "medium": "#dd8452", "high": "#c44e52"}

# Every LLM flag is the same solid color -- a flag is a flag. Line shading is reserved for the
# ones that were WRONG, so "did it fire?" and "was it right?" are two separate reads instead of
# three hues competing with the colored stream rows above.
LLM_C, LLM_EDGE = "#4a6d8c", "#22384a"
LLM_STYLE = {
    "hit":         dict(facecolor=LLM_C, hatch=None,  alpha=1.00),
    "false_alarm": dict(facecolor=LLM_C, hatch="///", alpha=1.00),
    "debris":      dict(facecolor=LLM_C, hatch=None,  alpha=0.38),
}
LLM_STYLE_LABEL = {"hit": "flagged (correct)", "false_alarm": "flagged (false alarm)",
                   "debris": "flagged (debris, excluded, not scored)"}
GUIDE_C = "#b0b0b0"  # segment-boundary lines carried down through the AD lanes

# Flag strength is severity()'s value/threshold ratio, shaded continuously rather than bucketed.
# The scale is FIXED at 1x..3x across every figure so two figures are comparable: 1.0 is the
# threshold itself, and 1.5 / 3.0 are exactly where surprise.severity cuts low|medium|high, so
# the color transitions land on the bucket edges.
STRENGTH_LO, STRENGTH_HI = 1.0, 3.0
STRENGTH_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "hsmm_strength", [(0.0, SEV_C["low"]), (0.25, SEV_C["medium"]), (1.0, SEV_C["high"])])
ROW_H = 1.0
# The "observations" block: a lettered bar up top (FED_LETTER_H tall) with the detector lanes
# packed directly beneath it (LANE_H each, LANE_GAP between) -- one tall block, not four rows a
# full ROW_H apart. GAP_UNALT_OBS is the only "row gap" left, between unaltered and that block.
GAP_UNALT_OBS = 0.20
FED_LETTER_H = 0.62
LANE_H = 0.42
LANE_GAP = 0.05
# matplotlib stacks multiline text at linespacing 1.2, so n lines of CARD_FS occupy
# (1.2n - 0.2) * CARD_FS points. Reserving more than that is just white space at the bottom.
LINE_H = 7.5 * 1.2 / 72.0
BASE_W, MAX_W = 15.5, 32.0     # figure width in inches: floor, and the ceiling for long trials
IN_PER_TICK = 0.030            # a bar only has to fit a LETTER, so the axis can stay tight
LEFT_IN, RIGHT_IN = 2.1, 0.3   # margins in INCHES, not fractions -- the row-label gutter needs the
                               # same width at 15in and at 50in, not a third of the figure


def _runs(verb_ids, noun_ids, lexicon):
    from cook_ad.llm import textify
    return textify.steps_from_ids(verb_ids, noun_ids, lexicon)


# monospace at CARD_FS: cards are never wrapped -- a broken quote is harder to read than a wide
# figure -- so the page widens to hold the longest line instead.
CARD_FS = 7.5
CARD_CHAR_W = CARD_FS / 72.0 * 0.601


def _fig_width(t_max, cards):
    """Width is whichever needs more room: the time axis or the widest card line.

    Short trials keep the familiar 15.5in layout. MAX_W caps only the TICK term -- a long quote
    still gets the width it needs, since clipping a detector's own words defeats the figure.
    """
    from_ticks = min(max(BASE_W, t_max * IN_PER_TICK), MAX_W)
    from_cards = 0.06 + max((len(c) for c in cards), default=0) * CARD_CHAR_W + 0.3
    return float(max(from_ticks, from_cards))


def _llm_tick_runs(verdicts, gt_steps, debris):
    """RLE over consecutive flagged ticks that share a kind (hit / false_alarm / debris) -- the
    tick-unit analogue of a llm/textify Step, and what makes a tick-unit LLM lane readable.

    Kind is computed exactly as the draw loop below computes it per verdict (v.step_index in
    gt_steps -> hit, in debris -> debris, else false_alarm); a run breaks whenever the kind
    changes, a tick is unflagged, or the next flagged tick isn't adjacent to the last one.

    Returns a list of dicts: start, end (inclusive tick indices), kind, verdicts (the run's own
    Verdict objects, in order) -- everything the card text and the numbered marker need, without
    recomputing kind twice.
    """
    gt_steps = set(gt_steps)
    runs = []
    for v in verdicts:
        if not v.is_anomaly:
            continue
        if v.step_index in gt_steps:
            kind = "hit"
        elif v.step_index in debris:
            kind = "debris"
        else:
            kind = "false_alarm"
        if runs and runs[-1]["kind"] == kind and v.step_index == runs[-1]["end"] + 1:
            runs[-1]["end"] = v.step_index
            runs[-1]["verdicts"].append(v)
        else:
            runs.append({"start": v.step_index, "end": v.step_index, "kind": kind,
                        "verdicts": [v]})
    return runs


def _letter(i):
    """0->A, 25->Z, 26->AA."""
    out = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        out = chr(65 + r) + out
    return out


def _step_letters(*step_lists):
    """One letter per distinct (verb, noun), in order of first appearance across every row.

    Shared across rows on purpose: the same action gets the same letter wherever it occurs, so a
    transposition reads as reordered letters and a repetition as a doubled one. Writing the letter
    in the bar instead of the full name is what lets the x axis stay tight -- a bar only has to be
    wide enough for one character.
    """
    by_action, key = {}, []
    for steps in step_lists:
        for st in steps:
            name = f"{st.verb} {st.noun}"
            if name not in by_action:
                by_action[name] = _letter(len(by_action))
                key.append((by_action[name], name))
    return by_action, key


def _bar(ax, y, steps, color, letters, min_label_ticks, height=0.52, fontsize=8.0):
    """Every step gets its letter. Slivers too narrow to hold one carry it just above the bar
    instead of dropping it -- an unlabelled sliver is indistinguishable from a rendering gap.
    """
    half = height / 2
    for st in steps:
        ax.add_patch(Rectangle((st.tick_start, y - half), st.duration, height,
                               facecolor=color, edgecolor="white", lw=1.1, zorder=2))
        inside = st.duration >= min_label_ticks
        ax.text(st.tick_start + st.duration / 2, y if inside else y + half + 0.14,
                letters[f"{st.verb} {st.noun}"], ha="center",
                va="center" if inside else "bottom", fontsize=fontsize,
                color="white" if inside else "#333", fontweight="bold", zorder=3)


def plot(unaltered, fed, gt_ticks, hsmm, llm_arms, title, out_path, unit="step"):
    """hsmm: ({tick: strength_ratio}, queries)
    llm_arms: [(label, steps, verdicts, gt_steps, debris_steps), ...]
    unit: "step" (default, unchanged rendering) or "tick" -- see the module docstring for what
    changes. `steps` inside each llm_arms tuple must already be at this unit (llm/textify's
    steps_from_* vs ticks_from_*); this function does not re-derive it.

    `debris_steps` (textify.injection_touched_steps) are steps the injection itself created or
    reshaped but which are not the ground-truth anomaly -- element_metrics excludes them from
    false-positive scoring entirely (docs/llm.md, eval/element_metrics.py). Drawn as a distinct
    grey "debris (excluded)" rather than red FALSE ALARM, so this figure and the corrected score
    agree: a flag here is neither a hit nor a false alarm, it's not scored at all.

    Layout: "observations" is ONE tall block, not four stacked rows. The lettered fed-bar sits
    at its top; the HSMM and LLM verdict lanes sit directly beneath, inside the SAME tinted
    region, each its own thin strip. That is what makes the detectors read as overlaying the
    observations rather than living in a separate panel a full row-height away -- there is no
    gap left to close, because there is no separate row.
    """
    t_max = fed[-1].tick_end
    letters, letter_key = _step_letters(unaltered, fed)
    lane_names = ["HSMM (joint)"] + [lab for lab, _, _, _, _ in llm_arms]
    n_lanes = len(lane_names)

    cards = []
    for i, q in enumerate(hsmm[1], 1):
        cards.append(f"{i}. [HSMM]  t={q['tick']}  {q['channel']}  {q['severity']}")
        cards.append("     " + q["text"])
    # Only verdicts the LLM actually raised get a card AND a bar. A step it passed on has nothing
    # to quote, and drawing a "missed" marker for it just repeats what the empty row inside the
    # injected window already says.
    k = len(hsmm[1])
    card_no = {}
    # unit="tick" only: computed once here, reused by the draw loop below so kind/numbering agree.
    tick_runs = {}
    for lab, steps, verdicts, gt_steps, debris in llm_arms:
        if unit == "tick":
            runs = _llm_tick_runs(verdicts, gt_steps, debris)
            tick_runs[lab] = runs
            for run in runs:
                k += 1
                card_no[(lab, run["start"])] = k
                st = steps[run["start"]]
                span = run["end"] - run["start"] + 1
                mark = {"hit": "HIT", "debris": "DEBRIS (excluded, not scored)",
                        "false_alarm": "FALSE ALARM"}[run["kind"]]
                extent = (f"tick {run['start'] + 1}" if span == 1 else
                          f"ticks {run['start'] + 1}-{run['end'] + 1} ({span}s)")
                cards.append(f"{k}. [{lab}]  {extent}  ({st.verb} {st.noun})  {mark}")
                more = f"   (+{span - 1} more tick{'s' if span > 2 else ''}, same verdict)" \
                    if span > 1 else ""
                cards.append("     " + run["verdicts"][0].raw.strip().replace("\n", " ") + more)
            continue
        for v in verdicts:
            if not v.is_anomaly:
                continue
            k += 1
            card_no[(lab, v.step_index)] = k
            st = steps[v.step_index]
            if v.step_index in gt_steps:
                mark = "HIT"
            elif v.step_index in debris:
                mark = "DEBRIS (excluded, not scored)"
            else:
                mark = "FALSE ALARM"
            cards.append(f"{k}. [{lab}]  step {v.step_index + 1} "
                         f"({st.verb} {st.noun}, {st.duration}s)  {mark}")
            cards.append("     " + v.raw.strip().replace("\n", " "))
    cards = cards or ["(nothing to report)"]
    cards = ["steps:  " + "   ".join(f"{ltr} = {name}" for ltr, name in letter_key), ""] + cards

    fig_w = _fig_width(t_max, cards)
    ax_w_in = fig_w - LEFT_IN - RIGHT_IN
    # a bar earns its letter once it is wide enough in INCHES to hold one character
    min_label_ticks = max(1.0, 0.13 * t_max / ax_w_in)

    # ---- vertical layout: a cursor walked top-down, in y-axis data units ------------------
    y = 0.0
    y_unalt = y - ROW_H / 2
    y -= ROW_H + GAP_UNALT_OBS
    obs_top = y
    y_fed = y - FED_LETTER_H / 2
    y -= FED_LETTER_H
    lane_ys = []
    for i in range(n_lanes):
        if i:
            y -= LANE_GAP
        lane_ys.append(y - LANE_H / 2)
        y -= LANE_H
    obs_bottom = y
    obs_block_h = obs_top - obs_bottom
    pad_top, pad_bot = 0.35, 0.30
    ax_data_h = -obs_bottom + pad_top + pad_bot

    # Headroom above the axes is budgeted in inches, top-down: colorbar, legend, title. Fixed
    # inches rather than axes fractions because the figure width -- and so the legend's wrapped
    # height -- varies a lot between a 30-tick and a 560-tick trial.
    head_in = 1.55
    chart_h = 0.55 + head_in + 0.55 + ax_data_h
    card_h = 0.22 + LINE_H * len(cards)
    fig_h = chart_h + card_h
    ax_h_in = chart_h - 0.55 - head_in
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([LEFT_IN / fig_w, (card_h + 0.55) / fig_h, ax_w_in / fig_w,
                       ax_h_in / fig_h])
    cax = fig.add_axes([0.02, 0.01, 0.96, (card_h - 0.06) / fig_h])
    cax.axis("off")
    # colorbar centred at the top, reading as one more legend entry rather than a stray widget
    cb_w = 2.2 / fig_w
    bar_ax = fig.add_axes([0.5 - cb_w / 2, (fig_h - 0.40) / fig_h, cb_w, 0.12 / fig_h])

    if gt_ticks:
        ax.add_patch(Rectangle((gt_ticks[0], obs_bottom - pad_bot),
                               gt_ticks[1] - gt_ticks[0] + 1, ax_data_h,
                               color=WIN_C, zorder=0))
        ax.axvline(gt_ticks[0], color="#c44e52", ls="--", lw=1.2, zorder=1)
        ax.text(gt_ticks[0] + 1, pad_top - 0.10, "injected", color="#c44e52", fontsize=8,
                zorder=4)

    _bar(ax, y_unalt, unaltered, UNALT_C, letters, min_label_ticks)

    # The observations block: a faint FED_C wash spans the whole block first -- the visual cue
    # that "observations" extends down through the lanes -- then the opaque lettered bar is
    # drawn on top of it at the block's top, at higher zorder than the lanes below it.
    ax.add_patch(Rectangle((0, obs_bottom), t_max, obs_block_h, facecolor=FED_C, alpha=0.14,
                           edgecolor="none", zorder=1))
    _bar(ax, y_fed, fed, FED_C, letters, min_label_ticks, height=FED_LETTER_H)

    # Each fed-bar segment boundary continues straight down through every lane below it, so a
    # detector's marker can be read against the observation it lines up with instead of just
    # its raw tick number.
    for st in fed[1:]:
        ax.plot([st.tick_start, st.tick_start], [obs_bottom, obs_top], color=GUIDE_C, lw=0.7,
                zorder=2.5)

    # HSMM lane: one marker vocabulary, not two. Every flagged tick is a triangle shaded by its
    # strength; the ticks that became a narrated query are the same triangle drawn large and
    # numbered, so "which flag is card 3" is read off size, not a second symbol.
    yh = lane_ys[0]
    ax.add_patch(Rectangle((0, yh - LANE_H / 2), t_max, LANE_H, facecolor="white", alpha=0.55,
                           edgecolor="#cfcfcf", lw=0.5, zorder=2))
    strength = hsmm[0]
    norm = mcolors.Normalize(STRENGTH_LO, STRENGTH_HI)
    q_ticks = {q["tick"] for q in hsmm[1]}

    def _strength_color(tick, fallback_sev=None):
        r = strength.get(tick)
        if r is None:
            r = {"low": 1.2, "medium": 2.0, "high": 3.0}.get(fallback_sev, STRENGTH_HI)
        return STRENGTH_CMAP(norm(min(r, STRENGTH_HI)))

    for tick in sorted(strength):
        if tick in q_ticks:
            continue
        ax.plot([tick], [yh], marker="v", ls="none", ms=5,
                color=_strength_color(tick), zorder=3)
    for i, q in enumerate(hsmm[1], 1):
        ax.plot([q["tick"]], [yh], marker="v", ls="none", ms=12,
                markeredgecolor="#333", markeredgewidth=0.7,
                color=_strength_color(q["tick"], q["severity"]), zorder=4)
        # Above the marker, not beside it: at tick resolution flagged markers can sit only 1-2
        # ticks apart, and a label offset sideways runs straight into the next one. Centering it
        # vertically over the marker's apex uses space no neighboring tick is competing for.
        ax.text(q["tick"], yh + LANE_H * 0.32, str(i), ha="center", va="bottom", fontsize=7,
                fontweight="bold", zorder=5,
                bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none", alpha=0.85))

    # LLM lanes: step-resolution verdicts, drawn in the same marker language as the HSMM lane
    # above -- a triangle, shaded and hatched exactly as the old verdict boxes were -- so the
    # three detectors read as one visual family instead of two different chart types stacked
    # together. The line from the step's start tick to the triangle is what a plain point marker
    # would lose: which step, and how long it ran, is still readable at a glance.
    for li, (lab, steps, verdicts, gt_steps, debris) in enumerate(llm_arms, 1):
        y = lane_ys[li]
        ax.add_patch(Rectangle((0, y - LANE_H / 2), t_max, LANE_H, facecolor="white", alpha=0.4,
                               edgecolor="#cfcfcf", lw=0.5, zorder=2))

        if unit == "tick":
            # Small unlabeled triangle per flagged tick -- the same marker weight as the HSMM
            # lane's baseline ms=5 (scatter's `s` is area, so s=20 ~= a 5pt-diameter marker: the
            # two lanes read as one visual family at their common resolution).
            for v in verdicts:
                if not v.is_anomaly:
                    continue
                if v.step_index in gt_steps:
                    kind = "hit"
                elif v.step_index in debris:
                    kind = "debris"
                else:
                    kind = "false_alarm"
                ax.scatter([v.step_index], [y], marker="v", s=20, edgecolors=LLM_EDGE,
                          linewidths=0.5, zorder=3, **LLM_STYLE[kind])
            # One bigger numbered triangle per RUN (s=110, matching the HSMM lane's ms=12 query
            # marker) -- the tick-unit analogue of "this is the flag that got narrated": a
            # bounded, meaningful highlight instead of one per tick.
            for run in tick_runs[lab]:
                ax.scatter([run["start"]], [y], marker="v", s=110, edgecolors="#333",
                          linewidths=0.8, zorder=4, **LLM_STYLE[run["kind"]])
                num = card_no.get((lab, run["start"]))
                if num is None:
                    continue
                # Above the marker, not beside it -- see the matching HSMM comment above; a run's
                # start can sit 1-2 ticks from the previous run's, and a sideways label collides.
                ax.text(run["start"], y + LANE_H * 0.32, str(num), ha="center", va="bottom",
                        fontsize=7, fontweight="bold", zorder=5,
                        bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none", alpha=0.85))
            continue

        for v in verdicts:
            if not v.is_anomaly:
                continue
            st = steps[v.step_index]
            if v.step_index in gt_steps:
                kind = "hit"
            elif v.step_index in debris:
                kind = "debris"
            else:
                kind = "false_alarm"
            x0, x1 = st.tick_start, st.tick_start + st.duration
            tick_h = LANE_H * 0.30
            ax.plot([x0, x0], [y - tick_h, y + tick_h], color=LLM_EDGE, lw=1.1, zorder=3)
            ax.plot([x0, x1], [y, y], color=LLM_EDGE, lw=1.2, zorder=3)
            ax.scatter([x1], [y], marker="v", s=110, edgecolors=LLM_EDGE, linewidths=0.8,
                      zorder=4, **LLM_STYLE[kind])
            num = card_no.get((lab, v.step_index))
            if num is None:
                continue
            ax.text(x1 + t_max * 0.008, y, str(num), ha="left", va="center", fontsize=7.5,
                    fontweight="bold", color="#222", zorder=5,
                    bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none", alpha=0.85))

    ax.text(-t_max * 0.012, y_unalt, "unaltered\n(what happened)", ha="right", va="center",
            fontsize=8.5)
    ax.text(-t_max * 0.012, y_fed, "observations\n(fed to detector)", ha="right", va="center",
            fontsize=8.5)
    for lab, y in zip(lane_names, lane_ys):
        ax.text(-t_max * 0.010, y, lab, ha="right", va="center", fontsize=7.3,
                fontweight="bold" if "HSMM" in lab or "gemma" in lab else "normal")

    ax.set_xlim(0, t_max)
    ax.set_ylim(obs_bottom - pad_bot, pad_top)
    ax.set_yticks([])
    ax.set_xlabel("tick (seconds)")
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    handles = [Rectangle((0, 0), 1, 1, color=WIN_C)]
    labels = ["injected window"]
    for kind in ("hit", "false_alarm", "debris"):
        handles.append(ax.scatter([], [], marker="v", s=90, edgecolors=LLM_EDGE,
                                  linewidths=0.8, **LLM_STYLE[kind]))
        labels.append(f"LLM {LLM_STYLE_LABEL[kind]}")
    handles += [plt.Line2D([], [], marker="v", ls="none", color="#888", ms=5),
                plt.Line2D([], [], marker="v", ls="none", color="#888", ms=11,
                           markeredgecolor="#333")]
    labels += ["HSMM flagged tick", "HSMM flag it narrated (numbered)"]
    ax.legend(handles, labels, loc="lower center",
              bbox_to_anchor=(0.5, 1.0 + 0.50 / ax_h_in), ncol=3, fontsize=7.6, frameon=False)

    cb = ColorbarBase(bar_ax, cmap=STRENGTH_CMAP, norm=norm, orientation="horizontal",
                      extend="max")
    cb.set_ticks([1.0, 1.5, 2.0, 2.5, 3.0])
    cb.set_ticklabels(["1x", "1.5x", "2x", "2.5x", "3x+"])
    cb.ax.tick_params(labelsize=6.5, length=2, pad=1.5)
    cb.set_label("HSMM flag strength (surprise / its threshold)", fontsize=7.6, labelpad=3)
    cb.ax.xaxis.set_label_position("top")

    ax.set_title(title, fontsize=11.5, pad=11)
    cax.text(0, 1, "\n".join(cards), va="top", ha="left", family="monospace",
             fontsize=CARD_FS)
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
    ratios = surprise.flagged_tick_severity(trace, flags, tables, with_ratio=True)
    # A tick can be flagged on several channels at once; the marker shows the strongest, which is
    # the one that would drive its severity label anyway.
    strength = {}
    for _ch, ticks in ratios.items():
        for t, r in ticks.items():
            strength[int(t)] = max(strength.get(int(t), 0.0), float(r))
    qs = [{"tick": int(q.tick), "channel": q.channel, "severity": q.severity, "text": q.text}
          for q in queries]
    return r_hat, rho, strength, qs




def _hsmm_score(strength, queries, window, latency_tol):
    """export_anomaly.py's criterion for a good HSMM example: it must have narrated at least one
    query inside the injected window (+ the same latency tolerance metrics.score_trial allows),
    and among those we prefer the trial with the fewest flagged ticks OUTSIDE the window.

    The out-of-window term counts flagged TICKS, not queries, as export_anomaly._out_of_window_count
    does. On short trials the HSMM usually narrates exactly one query, so a query-only score is
    degenerate -- nearly every eligible trial ties at (1, 0) and "best" degrades into a coin flip.
    Flagged ticks are the channel with real spread.
    """
    t0, hi = window[0], window[1] + latency_tol
    inw_q = sum(t0 <= q["tick"] <= hi for q in queries)
    ticks = set(strength)
    out_ticks = sum(not (t0 <= t <= hi) for t in ticks)
    return (1 if inw_q else 0, -out_ticks, inw_q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/breakfast.yaml")
    ap.add_argument("--joint-params", default="dataset/processed/breakfast/joint_params.npz")
    ap.add_argument("--sequences", default="dataset/processed/breakfast/sequences.json")
    ap.add_argument("--labels", default="dataset/processed/breakfast/labels.json")
    ap.add_argument("--vocab", default="dataset/processed/breakfast/vocab.json")
    ap.add_argument("--cache-dir", default="dataset/processed/breakfast/llm_cache")
    ap.add_argument("--model", default="gemma3:27b")
    ap.add_argument("--base-url", default="http://localhost:11434/v1",
                    help="must match what the run being rendered actually used -- it is part of "
                         "the response cache key, so the wrong port makes every lookup miss")
    ap.add_argument("--protocol", default="conversational",
                    help="must match the run being rendered (e.g. 'incremental' for a tick run)")
    ap.add_argument("--unit", choices=("step", "tick"), default="step",
                    help="'step': original behaviour, LLM lane drawn as a line spanning each "
                         "step. 'tick': LLM answered once per second -- lane drawn in the HSMM "
                         "lane's own marker language, sharing its time scale. See module "
                         "docstring")
    ap.add_argument("--variant", choices=("no-recipes", "with-recipes", "both"), default="both")
    ap.add_argument("--split-file", default=None,
                    help="a split.json from split_dataset.py. When given, the trial pool is the "
                         "REAL run_llm_eval.py pool for --split-part (filtered, then capped at "
                         "--max-real) instead of the first --max-real rows of --sequences -- use "
                         "this to render a real test-split run rather than a synthetic sample")
    ap.add_argument("--split-part", choices=("train", "test"), default=None)
    ap.add_argument("--chunk-size", type=int, default=16,
                    help="batched-inference chunk size for the --split-file real-trial path "
                         "(run_llm_eval.py's own default)")
    ap.add_argument("--max-real", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0,
                    help="drop trials with more than this many steps; 0 (default) keeps all. "
                         "The figure widens with trial length instead of squashing, so the old "
                         "cap of 8 mostly just biased the pool toward few-step trials.")
    ap.add_argument("--random-seed", type=int, default=0,
                    help="seed for picking the random example; independent of --seed, which must "
                         "stay pinned to the run whose LLM replies are cached")
    ap.add_argument("--max-hsmm-scan", type=int, default=60,
                    help="cap on HSMM inference passes per error type; the scanned subset is "
                         "drawn in shuffled order so it stays unbiased")
    ap.add_argument("--only", default=None,
                    help="comma-separated subset of error types to render, e.g. 'repetition' -- "
                         "for iterating on layout without re-running the other four")
    ap.add_argument("--out-dir", default="dataset/processed/breakfast/figures_conv100")
    args = ap.parse_args()

    import os
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    jax.config.update("jax_enable_x64", True)
    from cook_ad.anomaly import narrate
    from cook_ad.data.config import load_config
    from cook_ad.data import split as split_mod
    from cook_ad.eval import metrics
    from cook_ad.hsmm import joint_params
    from cook_ad.llm import client as llm_client
    from cook_ad.llm import detect, prompts
    from cook_ad.synthetic import error_injection, generate
    import run_llm_eval as R

    if args.split_file and not args.split_part:
        ap.error("--split-part is required when --split-file is given")

    d_max = load_config(args.config)["duration"]["d_max_ticks"]
    vocab = json.load(open(args.vocab))
    labels = json.load(open(args.labels))
    jp = joint_params.load_params(args.joint_params)
    marg = joint_params.collapse_to_marginal(jp)
    lex = narrate.Lexicon(vocab, marg)
    seqs = json.load(open(args.sequences))
    if args.split_file:
        # Same order run_llm_eval.py's real-source path uses: filter to the split part FIRST,
        # cap at --max-real second. Reversing that order picks a different set of trials than
        # the run actually scored.
        split = split_mod.load_split(args.split_file)
        seqs = split_mod.filter_sequences(seqs, split, args.split_part)
    seqs = seqs[: args.max_real]
    if args.split_file:
        # Batched, matching run_llm_eval.py exactly (a per-trial loop recompiles the recipe and
        # Viterbi kernels once per distinct trial length).
        traj = generate.trajectories_from_real_joint(jp, seqs, d_max, chunk_size=args.chunk_size)
    else:
        traj = [generate.trajectory_from_real_joint(jp, s["verb_ids"], s["noun_ids"], d_max)
                for s in seqs]
    pool = R.build_pool(traj, np.random.default_rng(args.seed), marg)
    # build_pool drops trajectories under MIN_SEGMENTS, so pool position != sequence position.
    src = R.usable_indices(traj)
    label_of = {e["trial_id"]: e for e in labels}
    print(f"{len(pool)} usable trials of {len(seqs)}", flush=True)

    # A SEPARATE stream from the injection rng above: perturbing that one would change the
    # injections and they would no longer match the cached LLM replies.
    rng_fig = np.random.default_rng(args.random_seed)

    all_variants = (("no-recipes", "gemma3 (no recipes)"), ("with-recipes", "gemma3 (+ recipes)"))
    wanted = all_variants if args.variant == "both" else \
        tuple(vl for vl in all_variants if vl[0] == args.variant)
    clients = {}
    for variant, lab in wanted:
        clients[lab] = (llm_client.ChatClient(
            model=args.model, base_url=args.base_url,
            cache_dir=Path(args.cache_dir) / args.model.replace("/", "_"),
            rpm=0, concurrency=1, max_requests=0),
            prompts.build_variant(variant, vocab, labels, args.protocol, args.unit))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    error_types = (tuple(args.only.split(",")) if args.only else error_injection.ERROR_TYPES)
    for error_type in error_types:
        shortlist = []
        for idx, (traj_i, degraded) in enumerate(pool):
            deg = degraded[error_type]
            steps, gt, _, debris = R.steps_and_truth(traj_i, deg, lex, args.unit)
            if not gt or (args.max_steps and len(steps) > args.max_steps):
                continue
            arms, ok = [], True
            for lab, (client, sysp) in clients.items():
                try:
                    arms.append((lab, steps, detect.run_trial(client, sysp, steps, vocab,
                                                              args.protocol, args.unit),
                                set(gt), debris))
                except Exception:
                    ok = False
                    break
            if not ok:
                continue
            hits = sum(any(v.is_anomaly and v.step_index in set(gt) for v in vs)
                       for _, _, vs, _, _ in arms)
            shortlist.append(((hits, -len(steps)), idx, traj_i, deg, steps, gt, arms))
        shortlist.sort(key=lambda r: r[0], reverse=True)
        if not shortlist:
            print(f"  {error_type}: no fully-cached readable trial, skipped")
            continue

        hsmm_cache = {}

        def hsmm_of(idx, deg):
            if idx not in hsmm_cache:
                hsmm_cache[idx] = _hsmm_bits(jp, vocab, deg["verb_ids"], deg["noun_ids"], d_max)
            return hsmm_cache[idx]

        # Best for the LLM: the existing ranking, first candidate the HSMM also spoke on so all
        # three detector rows carry something.
        llm_best = None
        for cand in shortlist:
            if hsmm_of(cand[1], cand[3])[3]:
                llm_best = cand
                break
        if llm_best is None:
            llm_best = shortlist[0]

        # Shuffle FIRST, then cap: the HSMM pass is the expensive part, so only --max-hsmm-scan of
        # them run, and drawing that subset at random keeps it unbiased w.r.t. the LLM ranking.
        order = list(rng_fig.permutation(len(shortlist)))
        eligible = []
        for j in order[: args.max_hsmm_scan]:
            cand = shortlist[j]
            if hsmm_of(cand[1], cand[3])[3]:
                eligible.append(cand)
        n_elig = len(eligible)
        n_scanned = len(order[: args.max_hsmm_scan])
        print(f"  {error_type}: {len(shortlist)} cached trials, HSMM narrated nothing on "
              f"{n_scanned - n_elig}/{n_scanned} scanned -- every figure below is drawn from "
              f"the {n_elig} where it spoke")

        # Best for the HSMM: in-window query first, then fewest false alarms, then readability.
        hsmm_best = None
        if eligible:
            hsmm_best = max(eligible, key=lambda c: (
                _hsmm_score(hsmm_of(c[1], c[3])[2], hsmm_of(c[1], c[3])[3], c[3]["window"],
                            metrics.DEFAULT_LATENCY_TOL),
                -len(c[4])))

        # Random: first eligible trial in the shuffled order that is neither of the two bests.
        taken = {llm_best[1]} | ({hsmm_best[1]} if hsmm_best is not None else set())
        rand = next((c for c in eligible if c[1] not in taken), None)

        def render(cand, tag, suffix):
            _, idx, traj_i, deg, steps, gt, arms = cand
            v_ids, n_ids = deg["verb_ids"], deg["noun_ids"]
            r_hat, rho, strength, queries = hsmm_of(idx, deg)
            unaltered = _runs(traj_i["verb_ids"], traj_i["noun_ids"], lex)
            fed = _runs(v_ids, n_ids, lex)
            trial_id = seqs[src[idx]]["trial_id"]
            # The recipe in the title used to be the HSMM's INFERRED cluster only, which reads as
            # a claim about the dish. These are real trials, so name the dataset's own label and
            # keep the inferred cluster beside it as what the detector believed.
            true_recipe = label_of.get(trial_id, {}).get("recipe_label", "?")
            title = (f"{error_type} [{tag}] -- {trial_id} ({len(v_ids)} ticks, "
                     f"recipe {true_recipe}; HSMM inferred r={int(r_hat)} "
                     f"conf={float(rho[r_hat]):.2f})  |  "
                     f"HSMM (tick-level) vs gemma3:27b {args.protocol} ({args.unit}-level)")
            p = out / f"combined_narrate_{error_type}{suffix}.png"
            plot(unaltered, fed, deg["window"], (strength, queries), arms, title, p,
                unit=args.unit)
            det, nout, _ = _hsmm_score(strength, queries, deg["window"],
                                       metrics.DEFAULT_LATENCY_TOL)
            print(f"  {error_type} [{tag}]: {trial_id} "
                  f"llm_hits={cand[0][0]} hsmm_in_window={bool(det)} "
                  f"hsmm_out_of_window_ticks={-nout} -> {p}")

        render(llm_best, f"best for LLM, of {len(shortlist)}", "")
        if hsmm_best is not None:
            render(hsmm_best, f"best for HSMM, of {n_elig}", "_hsmm_best")
        else:
            print(f"  {error_type}: no trial the HSMM narrated, no _hsmm_best figure")
        if rand is not None:
            render(rand, f"random, of {n_elig}", "_random1")
        else:
            print(f"  {error_type}: no eligible trial distinct from the bests, no _random1 figure")


if __name__ == "__main__":
    main()
