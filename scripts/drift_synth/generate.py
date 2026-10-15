"""Stages 4-9 of the drift generator (docs/drift_synth.md): seeded, cheap, no LLM. Writes
dataset/processed/drift_synth/<condition>/seed<k>/<person>/ for every condition and seed.

    ./py scripts/drift_synth/generate.py                       # every condition
    ./py scripts/drift_synth/generate.py --condition null --seed 0
"""
import argparse
from pathlib import Path

import yaml

from cook_ad.drift_synth.assets import load_assets
from cook_ad.drift_synth.export import export
from cook_ad.drift_synth.sampler import ConditionCfg, generate_person
from cook_ad.drift_synth.schedule import DriftRegime


def conditions(path):
    """-> ({name: ConditionCfg}, seeds) from conditions.yaml."""
    cfg = yaml.safe_load(Path(path).read_text())
    d = cfg["defaults"]

    def make(name, regime, moving, mix_shift=False):
        reg = DriftRegime(regime, d["onset_day"] if regime != "none" else 0, d["duration_days"],
                          d["period_days"])
        return ConditionCfg(name, reg, dict(moving), mix_shift, d["n_persons"], d["n_days"])

    out = {}
    for name, c in cfg["extra"].items():
        out[name] = make(name, c["regime"], c["moving"], c.get("mix_shift", False))
    for knob in cfg["knobs"]:
        for regime in cfg["regimes"]:
            out[f"{knob}_{regime}"] = make(f"{knob}_{regime}", regime, {knob: d["delta"]})
    for knob in cfg["sweep"]["knobs"]:
        for delta in cfg["sweep"]["deltas"]:
            name = f"{knob}_sweep_{delta:g}"
            out[name] = make(name, "sudden", {knob: delta})
    return out, d["seeds"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", default="configs/drift_synth/generator.yaml")
    ap.add_argument("--conditions", default="configs/drift_synth/conditions.yaml")
    ap.add_argument("--condition", action="append", help="only these (repeatable)")
    ap.add_argument("--seed", type=int, action="append", help="only these seeds (repeatable)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    assets = load_assets(args.generator)
    conds, seeds = conditions(args.conditions)
    names = args.condition or list(conds)
    unknown = sorted(set(names) - set(conds))
    if unknown:
        raise SystemExit(f"unknown conditions {unknown}; known: {sorted(conds)}")
    out_root = Path(args.out or assets.gen["paths"]["out"])
    for name in names:
        cond = conds[name]
        for seed in args.seed or seeds:
            n = 0
            for i in range(cond.n_persons):
                person, sessions, gt = generate_person(i, cond, seed, assets)
                export(sessions, out_root / name / f"seed{seed}" / person.person_id, assets.vocab,
                       gt, assets.gen["export"])
                n += len(sessions)
            print(f"{name} seed{seed}: {cond.n_persons} persons, {n} sessions")


if __name__ == "__main__":
    main()
