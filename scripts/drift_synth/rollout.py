"""Print one rollout of a recipe per person, with start and end times, for slides.

A person is a knob vector (theta in [0, 1]; unnamed knobs sit at 0.5 = cooked as written). Clean
events (no recognizer or session noise), speed 1, the generator's durations (CC4D step totals,
actions capped, written timers), as in rollout_examples.py.

    ./py scripts/drift_synth/rollout.py --recipe zoodles \\
        --person "Sam: salty=1 fatty=1 mise_en_place=0.05 tidy=0.05" \\
        --person "Ana: sweet=0 salty=0 fatty=0 mise_en_place=0.95 tidy=0.95 slow=0.6" \\
        --person "Lee: slow=0.8 fine_motor=0.45"

Knobs: sour sweet salty fatty mise_en_place tidy slow fine_motor. `--person` repeats; a bare vector
with no `Name:` is called P1, P2, ... `--list` prints the recipes. A tidy put-back prints as
`return X`; in the data it is EPIC's `put X` (EPIC has no separate verb for it).
"""
import argparse
import os
import sys

sys.path.insert(0, os.getcwd())


def parse_person(spec, i, knobs):
    name, _, body = spec.rpartition(":")
    name = name.strip() or f"P{i}"
    theta = {k: 0.5 for k in knobs}
    for tok in body.replace(",", " ").split():
        k, _, v = tok.partition("=")
        if k not in theta:
            sys.exit(f"unknown knob {k!r}; knobs are {', '.join(knobs)}")
        theta[k] = float(v)
        if not 0 <= theta[k] <= 1:
            sys.exit(f"{k}={v} is outside [0, 1]")
    return name, theta


def clock(s):
    return f"{int(s // 60)}:{int(s % 60):02d}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--recipe")
    ap.add_argument("--person", action="append", default=[], help='"Name: knob=value ..."')
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    import cook_ad.drift_synth.assets as A
    from cook_ad.drift_synth.noise import NoiseCfg
    from cook_ad.drift_synth.sampler import rng_for, sample_session
    assets = A.load_assets()
    if a.list or not a.recipe:
        sys.exit("recipes: " + ", ".join(sorted(assets.graphs)))
    if a.recipe not in assets.graphs:
        sys.exit(f"unknown recipe {a.recipe!r}; recipes: {', '.join(sorted(assets.graphs))}")
    if not a.person:
        sys.exit("give at least one --person")
    noise = NoiseCfg().off()
    r = a.recipe
    for i, spec in enumerate(a.person, 1):
        name, theta = parse_person(spec, i, list(assets.knobs))
        s = sample_session(assets.graphs[r], assets.expansions[r], theta, 1.0, assets, noise,
                           rng_for(a.seed, 7, i, sorted(assets.graphs).index(r)), name)
        ev, kinds = s.meta["clean_events"], s.meta["kinds"]
        gone = [f"-{knob} {v} {n}" for _, v, n, knob in s.meta["removed"]]
        print(f"\n=== {name} on {r} — {len(ev)} runs, {clock(ev[-1][1])} total")
        print("theta: " + (" ".join(f"{k}={v:g}" for k, v in theta.items() if v != 0.5) or "all 0.5"))
        if gone:
            print("left out: " + ", ".join(gone))
        for (t0, t1, verb, noun, _), kind in zip(ev, kinds):
            tag = "" if kind == "core" else "  [logistics]" if kind == "logistics" else f"  [{kind}]"
            shown = "return" if kind == "logistics" and verb == "put" else verb
            print(f"  {clock(t0)}-{clock(t1)}  {t1 - t0:4.0f}s  {shown} {noun}{tag}")


if __name__ == "__main__":
    main()
