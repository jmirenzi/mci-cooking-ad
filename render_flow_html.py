"""Render dataset/processed/breakfast/flow/flow.json into a single self-contained HTML page.

Reads real model output only -- see export_flow.py for how the JSON is produced. No inference
happens here, just layout.
"""
import argparse
import base64
import html
import json
import math
from pathlib import Path

FONT_DIR = Path(__file__).parent / "assets" / "fonts"

CHIP_BASE = 34
CHIP_SQRT_SCALE = 9
LABEL_COL_PX = 168


def _b64(name):
    return base64.b64encode((FONT_DIR / name).read_bytes()).decode("ascii")


def _font_faces():
    faces = [
        ("Zilla Slab", 500, "zilla500.woff2"),
        ("Zilla Slab", 600, "zilla600.woff2"),
        ("Karla", 400, "karla400.woff2"),
        ("Karla", 700, "karla700.woff2"),
        ("IBM Plex Mono", 400, "plexmono400.woff2"),
        ("IBM Plex Mono", 500, "plexmono500.woff2"),
    ]
    rules = []
    for family, weight, fname in faces:
        rules.append(f"""@font-face {{
  font-family: '{family}';
  font-style: normal;
  font-weight: {weight};
  font-display: swap;
  src: url(data:font/woff2;base64,{_b64(fname)}) format('woff2');
}}""")
    return "\n".join(rules)


def _esc(s):
    return html.escape(str(s), quote=True)


def _col_width(duration):
    return round(CHIP_BASE + CHIP_SQRT_SCALE * math.sqrt(duration))


def _rle_true_labels(labels):
    out = []
    if not labels:
        return out
    cur, n = labels[0], 1
    for lab in labels[1:]:
        if lab == cur:
            n += 1
        else:
            out.append((cur, n))
            cur, n = lab, 1
    out.append((cur, n))
    return out


def _trial_card(trial):
    segments = trial["segments"]
    runs = trial["runs"]
    recipe_path = {r["seg"]: r["r"] for r in trial["recipe_path"]}
    n = len(segments)

    widths = [_col_width(s["end"] - s["start"]) for s in segments]
    grid_cols = f"{LABEL_COL_PX}px " + " ".join(f"{w}px" for w in widths)

    def chip(col, cls, top, bottom=None, clipped=False, title=None):
        clip_attr = ' data-clipped="true"' if clipped else ""
        title_attr = f' title="{_esc(title)}"' if title else ""
        bottom_html = f'<span class="chip-sub">{_esc(bottom)}</span>' if bottom else ""
        return (
            f'<div class="chip {cls}" style="grid-column:{col};"{clip_attr}{title_attr}>'
            f'<span class="chip-main">{_esc(top)}</span>{bottom_html}</div>'
        )

    obs_cells = []
    sub_cells = []
    rec_cells = []
    for i, (run, seg) in enumerate(zip(runs, segments)):
        col = i + 2  # column 1 is the row label
        dur = seg["end"] - seg["start"]
        obs_cells.append(chip(
            col, "chip-obs", run["phrase"], f"×{run['n']}",
            title=f"verb={run['verb']}  noun={run['noun']}  ticks {run['start']}–{run['end']}",
        ))
        sub_cells.append(chip(
            col, "chip-sub-tier", f"z{seg['z']}", seg["name"], clipped=seg["clipped"],
            title=f"expected duration ≈{seg['expected_duration']:.0f}t, observed {dur}t"
            + (" -- d_max seam, not a real re-entry" if seg["clipped"] else ""),
        ))
        r = recipe_path.get(i)
        rec_cells.append(chip(col, "chip-rec", f"r{r}" if r is not None else "–"))

    truth_rle = _rle_true_labels(trial["true_subtask_labels"])
    truth_html = "".join(
        f'<span class="truth-pill">{_esc(lab.replace("_", " "))}<span class="truth-n">×{n}</span></span>'
        for lab, n in truth_rle
    )

    return f"""
    <section class="trial-card" id="{_esc(trial['trial_id'])}">
      <div class="card-head">
        <h2>{_esc(trial['true_recipe'])}</h2>
        <div class="card-meta">
          <span class="mono">{_esc(trial['trial_id'])}</span>
          <span class="dot">·</span>
          <span>{trial['T']} ticks</span>
          <span class="dot">·</span>
          <span>{n} segments</span>
        </div>
      </div>
      <div class="track-scroll">
        <div class="tracks-grid" style="grid-template-columns:{grid_cols};">
          <div class="row-label" style="grid-row:1;">observations<span class="row-label-sub">verb / noun stream</span></div>
          {"".join(obs_cells)}
          <div class="row-label" style="grid-row:2;">subtask<span class="row-label-sub">HSMM state (Viterbi)</span></div>
          {"".join(sub_cells)}
          <div class="row-label" style="grid-row:3;">recipe-HMM<span class="row-label-sub">per-segment state</span></div>
          {"".join(rec_cells)}
        </div>
      </div>
      <details class="answer-key">
        <summary>show ground-truth subtask labels</summary>
        <div class="truth-strip">{truth_html}</div>
      </details>
    </section>"""


PAGE_TEMPLATE = """<title>Cook-AD — Pipeline Flow</title>
<style>
{fonts}

:root {{
  --paper: #ECEDE8;
  --surface: #F8F8F5;
  --surface-raised: #FFFFFF;
  --ink: #20241F;
  --ink-soft: #5B615C;
  --ink-faint: #8A8E84;
  --line: #D8D6CC;
  --flame: #B85A17;
  --flame-bg: #F3E3D3;
  --pot: #0E6E63;
  --pot-bg: #DCEBE7;
  --plum: #6B4472;
  --plum-bg: #E9E0EB;
  --shadow: 0 1px 2px rgba(32,36,31,0.06), 0 6px 20px -10px rgba(32,36,31,0.18);
}}

@media (prefers-color-scheme: dark) {{
  :root {{
    --paper: #17181A;
    --surface: #1F2120;
    --surface-raised: #262825;
    --ink: #EDEAE2;
    --ink-soft: #A6A99C;
    --ink-faint: #71756A;
    --line: #34362F;
    --flame: #E2884A;
    --flame-bg: #3A2A1C;
    --pot: #45B0A3;
    --pot-bg: #1B3733;
    --plum: #C79ED1;
    --plum-bg: #362B3B;
    --shadow: 0 1px 2px rgba(0,0,0,0.3), 0 10px 30px -12px rgba(0,0,0,0.55);
  }}
}}
:root[data-theme="dark"] {{
  --paper: #17181A;
  --surface: #1F2120;
  --surface-raised: #262825;
  --ink: #EDEAE2;
  --ink-soft: #A6A99C;
  --ink-faint: #71756A;
  --line: #34362F;
  --flame: #E2884A;
  --flame-bg: #3A2A1C;
  --pot: #45B0A3;
  --pot-bg: #1B3733;
  --plum: #C79ED1;
  --plum-bg: #362B3B;
  --shadow: 0 1px 2px rgba(0,0,0,0.3), 0 10px 30px -12px rgba(0,0,0,0.55);
}}
:root[data-theme="light"] {{
  --paper: #ECEDE8;
  --surface: #F8F8F5;
  --surface-raised: #FFFFFF;
  --ink: #20241F;
  --ink-soft: #5B615C;
  --ink-faint: #8A8E84;
  --line: #D8D6CC;
  --flame: #B85A17;
  --flame-bg: #F3E3D3;
  --pot: #0E6E63;
  --pot-bg: #DCEBE7;
  --plum: #6B4472;
  --plum-bg: #E9E0EB;
  --shadow: 0 1px 2px rgba(32,36,31,0.06), 0 6px 20px -10px rgba(32,36,31,0.18);
}}

* {{ box-sizing: border-box; }}
html {{ background: var(--paper); }}
body {{
  margin: 0;
  background: var(--paper);
  color: var(--ink);
  font-family: 'Karla', ui-sans-serif, system-ui, sans-serif;
  font-size: 15px;
  line-height: 1.5;
  -webkit-font-smoothing: antialiased;
  overflow-x: hidden;
}}
.mono {{ font-family: 'IBM Plex Mono', ui-monospace, monospace; font-variant-numeric: tabular-nums; }}

.page {{
  max-width: 960px;
  margin: 0 auto;
  padding: 56px 28px 96px;
}}

.eyebrow {{
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11.5px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--ink-soft);
  margin: 0 0 14px;
}}
h1 {{
  font-family: 'Zilla Slab', ui-serif, serif;
  font-weight: 600;
  font-size: 34px;
  line-height: 1.15;
  margin: 0 0 16px;
  text-wrap: balance;
  letter-spacing: -0.005em;
}}
.dek {{
  max-width: 62ch;
  color: var(--ink-soft);
  font-size: 15.5px;
  margin: 0 0 30px;
}}
.dek code {{
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.92em;
  background: var(--surface-raised);
  border: 1px solid var(--line);
  border-radius: 4px;
  padding: 0.05em 0.35em;
}}

.legend {{
  display: flex;
  flex-wrap: wrap;
  gap: 10px 22px;
  padding: 16px 18px;
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 8px;
  margin-bottom: 14px;
}}
.legend-item {{ display: flex; align-items: center; gap: 9px; font-size: 13.5px; color: var(--ink-soft); }}
.legend-swatch {{ width: 13px; height: 13px; border-radius: 3px; flex: none; }}
.legend-swatch.obs {{ background: var(--flame); }}
.legend-swatch.sub {{ background: var(--pot); }}
.legend-swatch.rec {{ background: var(--plum); }}

.caveats {{
  border: 1px solid var(--line);
  border-left: 3px solid var(--plum);
  background: var(--surface);
  border-radius: 0 8px 8px 0;
  padding: 14px 18px;
  margin-bottom: 48px;
  font-size: 13.5px;
  color: var(--ink-soft);
}}
.caveats strong {{ color: var(--ink); font-weight: 700; }}
.caveats ul {{ margin: 8px 0 0; padding-left: 1.1em; }}
.caveats li {{ margin: 3px 0; }}

.trial-card {{
  background: var(--surface-raised);
  border: 1px solid var(--line);
  border-radius: 10px;
  box-shadow: var(--shadow);
  padding: 20px 20px 16px;
  margin-bottom: 22px;
}}
.card-head {{
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 4px 16px;
  margin-bottom: 16px;
  padding-bottom: 12px;
  border-bottom: 1px solid var(--line);
}}
.card-head h2 {{
  font-family: 'Zilla Slab', serif;
  font-weight: 600;
  font-size: 20px;
  margin: 0;
  text-transform: capitalize;
}}
.card-meta {{ font-size: 12.5px; color: var(--ink-faint); display: flex; align-items: center; gap: 6px; }}
.card-meta .dot {{ opacity: 0.6; }}

.track-scroll {{ overflow-x: auto; padding-bottom: 4px; }}
.tracks-grid {{
  display: grid;
  grid-template-rows: repeat(3, auto);
  gap: 5px 6px;
  align-items: stretch;
  width: max-content;
  min-width: 100%;
}}

.row-label {{
  align-self: center;
  font-size: 11.5px;
  color: var(--ink-soft);
  font-weight: 700;
  letter-spacing: 0.01em;
  padding-right: 10px;
  line-height: 1.3;
}}
.row-label-sub {{
  display: block;
  font-weight: 400;
  font-size: 10.5px;
  color: var(--ink-faint);
  letter-spacing: 0;
  margin-top: 1px;
}}

.chip {{
  border-radius: 5px;
  padding: 6px 7px 5px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  text-align: center;
  min-height: 40px;
  overflow: hidden;
  cursor: default;
}}
.chip-main {{
  font-size: 11.5px;
  font-weight: 700;
  line-height: 1.15;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 100%;
}}
.chip-sub {{
  font-family: 'IBM Plex Mono', monospace;
  font-size: 9.5px;
  opacity: 0.75;
  margin-top: 1px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 100%;
}}

.chip-obs {{ background: var(--flame-bg); color: var(--flame); grid-row: 1; }}
.chip-sub-tier {{ background: var(--pot-bg); color: var(--pot); grid-row: 2; }}
.chip-sub-tier .chip-main {{ font-family: 'IBM Plex Mono', monospace; font-size: 12.5px; }}
.chip-rec {{ background: var(--plum-bg); color: var(--plum); grid-row: 3; min-height: 30px; }}
.chip-rec .chip-main {{ font-family: 'IBM Plex Mono', monospace; font-size: 11.5px; font-weight: 500; }}

.chip[data-clipped="true"] {{
  background-image: repeating-linear-gradient(
    -55deg, transparent, transparent 5px, currentColor 5px, currentColor 6px
  );
  background-size: 100% 100%;
  background-blend-mode: normal;
  box-shadow: inset 0 0 0 1px currentColor;
  opacity: 0.92;
}}
.chip[data-clipped="true"]::after {{
  content: "d_max seam";
  display: none;
}}

.answer-key {{ margin-top: 14px; }}
.answer-key summary {{
  cursor: pointer;
  font-size: 12px;
  color: var(--ink-faint);
  font-family: 'IBM Plex Mono', monospace;
  letter-spacing: 0.02em;
  user-select: none;
  list-style: none;
}}
.answer-key summary::-webkit-details-marker {{ display: none; }}
.answer-key summary::before {{ content: "\\25B8  "; }}
.answer-key[open] summary::before {{ content: "\\25BE  "; }}
.answer-key summary:hover {{ color: var(--ink-soft); }}
.truth-strip {{
  margin-top: 10px;
  padding-top: 10px;
  border-top: 1px dashed var(--line);
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}}
.truth-pill {{
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10.5px;
  color: var(--ink-soft);
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 4px;
  padding: 3px 7px;
  white-space: nowrap;
}}
.truth-n {{ opacity: 0.6; margin-left: 3px; }}

footer {{
  margin-top: 40px;
  padding-top: 20px;
  border-top: 1px solid var(--line);
  font-size: 12px;
  color: var(--ink-faint);
}}

*:focus-visible {{ outline: 2px solid var(--pot); outline-offset: 2px; }}

@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; }} }}
</style>

<div class="page">
  <p class="eyebrow">Cook-AD · Pipeline Readout · Part 1 of 2</p>
  <h1>From kitchen actions to inferred structure</h1>
  <p class="dek">
    One held-out trial per recipe (10 of 10), run through the fitted cascade model exactly as
    scored in evaluation — <code>surprise.compute_trace</code> → Viterbi segmentation
    → per-segment recipe-HMM state. Every chip below is a direct read of a model output;
    subtask and recipe rows are left as bare ids on purpose — the observation stream above
    them is what should let a human guess the subtask before reading its name.
  </p>

  <div class="legend">
    <div class="legend-item"><span class="legend-swatch obs"></span>observations — verb / noun, as seen</div>
    <div class="legend-item"><span class="legend-swatch sub"></span>subtask — HSMM state, Viterbi-decoded</div>
    <div class="legend-item"><span class="legend-swatch rec"></span>recipe-HMM — per-segment state</div>
  </div>

  <div class="caveats">
    <strong>Reading the recipe-HMM row:</strong> these ids are <em>not</em> a recipe identity.
    Measured directly on this checkpoint, the cascade recipe-HMM's per-segment posterior is
    often ~100% confident yet lands on a <em>different</em> state almost every segment within
    the same trial — the states track something closer to phase-in-sequence than "which
    recipe." Treat the row as a diagnostic of the cascade, not a recipe label.
    <ul>
      <li>Hatched chips mark a <code class="mono">d_max</code> seam: a single action lasted
      longer than the model's max segment length (200 ticks) and got split into two adjacent
      segments of the same state — not a second visit to that subtask.</li>
      <li>This is offline retrodiction: Viterbi runs over the whole trial at once, not tick by
      tick as a live system would.</li>
    </ul>
  </div>

  <main>
{cards}
  </main>

  <footer>
    Source: <span class="mono">dataset/processed/breakfast</span> · cascade HSMM +
    recipe-HMM checkpoint · generated by <span class="mono">export_flow.py</span> +
    <span class="mono">render_flow_html.py</span>
  </footer>
</div>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow-json", default="dataset/processed/breakfast/flow/flow.json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.flow_json) as f:
        data = json.load(f)

    cards = "\n".join(_trial_card(t) for t in data["trials"])
    page = PAGE_TEMPLATE.format(fonts=_font_faces(), cards=cards)

    Path(args.out).write_text(page)
    print(f"wrote {args.out} ({len(page)/1024:.0f} KiB)")


if __name__ == "__main__":
    main()
