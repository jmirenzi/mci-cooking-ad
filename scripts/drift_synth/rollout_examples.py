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
    if unreviewed:
        orig = A.load_affinity
        A.load_affinity = lambda p, r, require_reviewed=True: orig(p, r, False)
    a = A.load_assets()
    a.gen["durations"] = dict(a.gen["durations"], step_scale="configs/drift_synth/step_time_scale.json",
                              max_event_s=600.0)
    a.step_scale = A.load_step_scale(a.gen)
    return a


def labelled(exp, used):
    """The step's tokens as apply_variants built them, with display-only removals:
    [(kind, token, knob)], kind in core / sub / ins / removed."""
    replaced, inserts = {}, []
    for i in used:
        v = exp.variants[i]
        if v.kind == "insert":
            inserts += [("ins", t, v.knob) for t in v.tokens]
            continue
        idx = list(v.replaces or [])
        for j in idx:
            replaced[j] = [("removed", exp.tokens[j], v.knob)]
        if v.kind == "substitute" and idx:
            replaced[idx[0]] += [("sub", t, v.knob) for t in v.tokens]
    out = []
    for j, tok in enumerate(exp.tokens):
        out += replaced.get(j, [("core", tok, None)])
    return out + inserts


def render(session, assets):
    exps = assets.expansions[session.recipe_id]
    queues = {s: labelled(exps[s], session.meta["variants_applied"].get(s, []))
              for s in session.meta["order"] if s in exps}
    parts, changes = [], []

    def fmt(kind, tok, knob, dur=None):
        txt = f"{tok[0]} {tok[1]}" + (f" ({dur:.0f}s)" if dur is not None else "")
        if kind == "removed":
            changes.append(f"−{knob} {tok[0]} {tok[1]}")
            return f"~~{tok[0]} {tok[1]}~~ (−{knob})"
        if kind in ("ins", "sub"):
            changes.append(f"+{knob} {tok[0]} {tok[1]}")
            return f"**{txt} (+{knob})**"
        return txt

    for t0, t1, verb, noun, step in session.meta["clean_events"]:
        if step is None:
            parts.append(f"*{verb} {noun} ({t1 - t0:.0f}s)*")
            continue
        q = queues[step]
        while q and q[0][0] == "removed":
            parts.append(fmt(*q.pop(0)))
        kind, tok, knob = q.pop(0)
        assert tok == (verb, noun), (session.recipe_id, step, tok, verb, noun)
        parts.append(fmt(kind, tok, knob, t1 - t0))
    for s, q in queues.items():           # a step whose every token was removed
        parts += [fmt(*x) for x in q if x[0] == "removed"]
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
            parts, changes = render(s, assets)
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
