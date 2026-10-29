"""Everything the seeded generator reads, loaded once: phased graphs, step expansions, tags,
knobs, the duration models (EPIC actions, CC4D step totals), the vocabulary and the
generator config."""
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from cook_ad.drift_synth.types import StepExpansion, Variant


@dataclass
class Assets:
    graphs: dict          # recipe_id -> RecipeGraph (with phases)
    expansions: dict      # recipe_id -> {step_id: StepExpansion}
    tags: dict            # {"verbs": {...}, "nouns": {...}}
    knobs: dict
    durations: object     # DurationModel
    vocab: dict           # {"verbs": {name: id}, "nouns": {name: id}, ...}
    gen: dict             # configs/drift_synth/generator.yaml
    step_times: object = None        # durations.StepTimes (CC4D step totals); None = EPIC time only
    affinity: dict | None = None     # recipe -> {flavour knob: affinity in [0, 1]}; None = all 1
    vessels: dict | None = None      # recipe -> {"serves_in": noun|None, "steps": {step: [noun]}}; None = named nouns only
    removals: dict | None = None     # recipe -> {"keep": set, "acts_on": {step: [noun]}}; None = no annotation


# ---- expansions on disk: configs/drift_synth/expansions/<recipe>.json -------------------------

def expansion_to_json(exp):
    return {"tokens": [list(t) for t in exp.tokens],
            "variants": [{"knob": v.knob, "kind": v.kind, "anchor_step": v.anchor_step,
                          "tokens": [list(t) for t in v.tokens], "replaces": v.replaces}
                         for v in exp.variants]}


def expansion_from_json(step_id, d):
    return StepExpansion(
        step_id=step_id, tokens=[tuple(t) for t in d["tokens"]],
        variants=[Variant(v["knob"], v["kind"], v["anchor_step"], [tuple(t) for t in v["tokens"]],
                          v.get("replaces")) for v in d["variants"]])


def save_expansions(path, expansions):
    Path(path).write_text(json.dumps(
        {k: expansion_to_json(e) for k, e in sorted(expansions.items(), key=lambda kv: int(kv[0]))},
        indent=1) + "\n")


def load_expansions(path):
    return {k: expansion_from_json(k, d) for k, d in json.loads(Path(path).read_text()).items()}


def load_assets(gen_path="configs/drift_synth/generator.yaml", require_reviewed_affinity=True,
                require_step_times=True):
    """The real assets. Needs reviewed tags and built expansions (Stages 2-3), a reviewed
    affinity.yaml unless require_reviewed_affinity=False (previews and calibration only), and the
    CC4D step times unless require_step_times=False (calibrate_cc4d.py, which fits them)."""
    from cook_ad.drift_synth.durations import DurationModel, fit_duration_priors
    from cook_ad.drift_synth.graphs import load_phased
    from cook_ad.drift_synth.knobs import load_knobs
    from cook_ad.drift_synth.tagging import load_tags

    gen = yaml.safe_load(Path(gen_path).read_text())
    p = gen["paths"]
    recipes = yaml.safe_load(Path(p["recipes"]).read_text())["recipes"]
    dur_path = Path(p["durations_cache"])
    if dur_path.exists():
        durations = DurationModel.load(dur_path)
    else:
        durations = fit_duration_priors(gen["durations"]["source"], p["epic_annotations"])
        dur_path.parent.mkdir(parents=True, exist_ok=True)
        durations.save(dur_path)
    graphs = load_phased(p["cc4d"], recipes, p["phases"])
    if p.get("graph_edits"):
        from cook_ad.drift_synth.graphs import with_edges
        edits = yaml.safe_load(Path(p["graph_edits"]).read_text())["add_edges"] or {}
        graphs = {r: with_edges(g, edits.get(r, ())) for r, g in graphs.items()}
    return Assets(
        graphs=graphs,
        expansions={r: load_expansions(Path(p["expansions"]) / f"{r}.json") for r in recipes},
        tags=load_tags(p["tags"]),
        knobs=load_knobs(p["knobs"]),
        durations=durations,
        vocab=json.loads(Path(p["vocab"]).read_text()),
        gen=gen,
        step_times=load_step_times(gen, require_step_times),
        affinity=load_affinity(p["affinity"], recipes, require_reviewed_affinity),
        vessels=load_vessels(p.get("vessels"), recipes),
        removals=load_removals(p.get("removals"), recipes),
    )


def load_step_times(gen, required=True):
    """The CC4D step totals (durations.StepTimes) if generator.yaml's durations.step_times names
    a file (`calibrate_cc4d.py --write-step-times` writes it)."""
    from cook_ad.drift_synth.durations import StepTimes
    path = (gen.get("durations") or {}).get("step_times")
    if not path:
        return None
    if not Path(path).exists():
        if required:
            raise FileNotFoundError(f"{path}: run scripts/drift_synth/calibrate_cc4d.py --write-step-times {path}")
        return None
    return StepTimes.load(path)


def load_vessels(path, recipes):
    """configs/drift_synth/vessels.yaml -> {recipe: {"serves_in": noun | None, "steps": {step: [noun]}}}."""
    if not path:
        return None
    d = yaml.safe_load(Path(path).read_text())["vessels"]
    return {r: {"serves_in": d[r].get("serves_in"),
                "steps": {str(s): list(v) for s, v in d[r]["steps"].items()}} for r in recipes}


def load_removals(path, recipes):
    """configs/drift_synth/removals.yaml -> {recipe: {"keep": set, "acts_on": {step: [noun]}}}."""
    if not path:
        return None
    d = yaml.safe_load(Path(path).read_text())["removals"] or {}
    return {r: {"keep": set((d.get(r) or {}).get("keep", ())),
                "acts_on": {str(s): list(v) for s, v in ((d.get(r) or {}).get("acts_on") or {}).items()}}
            for r in recipes}


def load_affinity(path, recipes, require_reviewed=True):
    """configs/drift_synth/affinity.yaml -> {recipe: {knob: affinity}}. How plausible it is that
    a cook who likes the flavour adds it to this dish; scales the flavour knobs' inserts only.
    Raises unless the file says reviewed: true (LLM-proposed, hand-checked)."""
    from cook_ad.drift_synth.knobs import FLAVOUR
    d = yaml.safe_load(Path(path).read_text())
    if require_reviewed and not d.get("reviewed"):
        raise ValueError(f"{path} is not reviewed (set reviewed: true after checking it)")
    out = {}
    for r in recipes:
        row = d["affinity"][r]
        if set(row) != set(FLAVOUR) or not all(0.0 <= float(v) <= 1.0 for v in row.values()):
            raise ValueError(f"{path}: {r} needs {list(FLAVOUR)} in [0, 1], got {row}")
        out[r] = {k: float(v) for k, v in row.items()}
    return out
