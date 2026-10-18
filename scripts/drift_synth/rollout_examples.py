"""Roll out every recipe for a few hand-set, very different people, for reading by eye.

Clean events (no recognizer noise, no day noise, speed 1), CC4D-calibrated durations as in
eval_synth_prior.py --variant cal. Writes runs/drift_synth/reports/rollout_examples.md:
per session the event stream with logistics in italics, flavour additions in bold (+knob) and
removals struck through (-knob).

    ./py scripts/drift_synth/rollout_examples.py [--unreviewed-affinity]
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())

PEOPLE = {   # name -> (who, theta); fine_motor below its hinge for all (healthy, no repeats)
    "A": ("loves rich, salty, sweet food; cooks fast and messy, no prep",
          dict(sour=0.9, sweet=0.95, salty=1.0, fatty=1.0, mise_en_place=0.05, tidy=0.05,
               slow=0.4, fine_motor=0.0)),
    "B": ("health-conscious planner: no salt, sugar or fat, lemon instead; preps everything first, "
          "tidies as they go",
          dict(sour=0.9, sweet=0.0, salty=0.0, fatty=0.0, mise_en_place=0.95, tidy=0.95,
               slow=0.6, fine_motor=0.2)),
    "C": ("cooks the recipe as written; average order and tidiness, a little slow and clumsy",
          dict(sour=0.5, sweet=0.5, salty=0.5, fatty=0.5, mise_en_place=0.5, tidy=0.5,
               slow=0.6, fine_motor=0.45)),
}


def load(unreviewed):
    import cook_ad.drift_synth.assets as A
    a = A.load_assets(require_reviewed_affinity=not unreviewed)
    a.gen["durations"] = dict(a.gen["durations"], step_scale="configs/drift_synth/step_time_scale.json",
                              max_event_s=600.0)
    a.step_scale = A.load_step_scale(a.gen)
    return a


def render(session):
    """The clean stream with logistics in italics, additions bold, removals struck through."""
    parts, changes = [], []
    gone = {}
    for step, verb, noun, knob in session.meta["removed"]:
        gone.setdefault(step, []).append(f"~~{verb} {noun}~~ (−{knob})")
        changes.append(f"−{knob} {verb} {noun}")
    shown = set()
    for (t0, t1, verb, noun, step), kind in zip(session.meta["clean_events"], session.meta["kinds"]):
        if step is not None and step not in shown:
            parts += gone.get(step, [])
            shown.add(step)
        txt = f"{verb} {noun} ({t1 - t0:.0f}s)"
        if kind == "logistics":
            parts.append(f"*{txt}*")
        elif kind == "core":
            parts.append(txt)
        else:
            parts.append(f"**{txt} ({kind})**")
            changes.append(f"{kind} {verb} {noun}")
    for step, items in gone.items():       # a step whose every token was removed
        if step not in shown:
            parts += items
    return parts, changes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unreviewed-affinity", action="store_true",
                    help="read affinity.yaml before it is marked reviewed (preview only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/drift_synth/reports/rollout_examples.md")
    a = ap.parse_args()
    from cook_ad.drift_synth.noise import NoiseCfg
    from cook_ad.drift_synth.sampler import rng_for, sample_session
    assets = load(a.unreviewed_affinity)
    noise = NoiseCfg().off()
    lines = ["# Example rollouts: 3 people × every recipe", "",
             "Clean events (no recognizer or day noise), CC4D-calibrated durations. *Italics* = "
             "take/put/wash logistics; **bold (+knob)** = a flavour addition; ~~struck (−knob)~~ = "
             "a removed ingredient. Seconds in brackets.", ""]
    lines += [f"- **{k}** — {who}: " + ", ".join(f"{n} {v:g}" for n, v in th.items())
              for k, (who, th) in PEOPLE.items()]
    for r in sorted(assets.graphs):
        lines += ["", f"## {r}", ""]
        for k, (_, th) in PEOPLE.items():
            s = sample_session(assets.graphs[r], assets.expansions[r], th, 1.0, assets, noise,
                               rng_for(a.seed, 7, ord(k), sorted(assets.graphs).index(r)), k)
            parts, changes = render(s)
            ev = s.meta["clean_events"]
            mins = (ev[-1][1] - ev[0][0]) / 60
            lines += [f"**{k}** — {len(ev)} events, {mins:.1f} min, step order "
                      f"{'-'.join(s.meta['order'])}; changes: {', '.join(changes) or 'none'}", "",
                      " · ".join(parts), ""]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(lines) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
