"""Tables from compare_s_count[_breakfast].py: metric G per variant (mean over persons) and the
per-session alarm rates. Writes markdown to stdout.

    ./py scripts/eval/summarize_s_count.py > runs/drift_synth/scount/summary.md
"""
import glob
import json
import re

import numpy as np

TYPES = ("substitution", "abandonment", "omission", "transposition", "repetition")


def load():
    out = {}
    for f in sorted(glob.glob("runs/drift_synth/compare_s_count_*p0?.json")):
        d = json.load(open(f))
        pop = re.match(r".*compare_s_count_pop(\d+)_p0\d\.json", f)
        corpus = "synthetic" + (f" +population {pop.group(1)}/recipe" if pop else "")
        for n, modes in d["results"].items():
            for mode, r in modes.items():
                out.setdefault((corpus, f"N={n}", mode), []).append(r)
    for f in glob.glob("runs/drift_synth/scount/compare_s_count_breakfast.json"):
        d = json.load(open(f))
        for mode, r in d["results"].items():
            out.setdefault(("breakfast", "N=402", mode), []).append(r)
    return out, d["levels"] if out else None


def main():
    res, levels = load()
    for (corpus, n, mode), runs in sorted(res.items()):
        variants = list(runs[0]["rows"])
        print(f"\n### {corpus} {n} {mode} ({len(runs)} person{'s' if len(runs) > 1 else ''}) -- metric G\n")
        print("| variant | F1 | ΔF1 vs off | 95% CI (bootstrap) | precision | recall | healthy FPR | alarms/10 min | s_count α | "
              + " | ".join(t[:5] for t in TYPES) + " |")
        print("|---" * (9 + len(TYPES)) + "|")
        off = np.mean([r["rows"]["off"]["f1"] for r in runs])
        for v in variants:
            rows = [r["rows"][v] for r in runs]
            m = lambda k: np.mean([x[k] for x in rows])  # noqa: E731
            b = [x.get("delta_f1_vs_off") for x in rows]
            ci = ("–" if v == "off" or None in b else
                  f"[{b[0]['lo']:+.3f}, {b[0]['hi']:+.3f}]" if len(b) == 1 else
                  f"{sum(x['lo'] > 0 for x in b)} up / {sum(x['hi'] < 0 for x in b)} down of {len(b)}")
            al = "/".join(f"{x.get('count_alpha', 0):g}" for x in rows) if v != "off" else "–"
            print(f"| {v} | {m('f1'):.3f} | {m('f1') - off:+.3f} | {ci} | {m('precision'):.3f} | {m('recall'):.3f} | "
                  f"{m('healthy_trial_fpr'):.2f} | {m('alarms_per_10min'):.2f} | {al} | "
                  + " | ".join(f"{np.mean([x['per_type'][t] for x in rows]):.2f}" for t in TYPES) + " |")
        print("\nSessions with any s_count alarm (test; live at κ = 0.5), by α:\n")
        print("| variant | α | healthy | " + " | ".join(t[:5] for t in TYPES) + " |")
        print("|---" * (3 + len(TYPES)) + "|")
        for v in variants:
            if v == "off":
                continue
            for li, al in enumerate(levels):
                if al == 0 or al not in (0.002, 0.01, 0.05):
                    continue
                s = [r["session_any"][v] for r in runs]
                g = lambda k: np.mean([x[k][li] for x in s])  # noqa: E731
                print(f"| {v} | {al:g} | {g('healthy'):.2f} | " + " | ".join(f"{g(t):.2f}" for t in TYPES) + " |")


if __name__ == "__main__":
    main()
