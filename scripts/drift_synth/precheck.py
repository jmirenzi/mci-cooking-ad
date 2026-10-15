"""Pre-checks for the synthetic drift generator (docs/drift_synth.md): what CaptainCook4D gives,
what the closed vocabulary is, what stream format the HSMM reads, and a proposed recipe set.

Reads only annotation files: dataset/captaincook4d (github.com/CaptainCook4D/annotations),
dataset/epic_kitchens/annotations (github.com/epic-kitchens/epic-kitchens-100-annotations) and
dataset/processed/epic. Writes runs/drift_synth/reports/precheck.md.
"""
import argparse
import csv
import json
import re
import statistics
from collections import Counter
from pathlib import Path

from cook_ad.drift_synth import lexicon
from cook_ad.drift_synth.graphs import (TooManyIdeals, ancestors, count_linear_extensions,
                                        load_task_graphs, recipe_key)

# Knob coverage is ESTIMATED here from step text, because tags (Stage 2) do not exist yet.
SOUR_WORDS = ("lime", "lemon", "vinegar", "curd", "yogurt", "salsa", "sweet and sour",
              "sweet-and-sour", "tamarind", "pickle")
FAT_WORDS = ("butter", "oil", "cheese", "mozzarella", "cheddar", "parmesan", "mayonnaise",
             "cooking spray", "chocolate", "milk", "avocado", "cream")
# A savory dish where adding lemon, lime, vinegar or yoghurt to an existing mix/season/garnish
# step is ordinary cooking (judgment, not data).
SOUR_INSERT_PLAUSIBLE = {
    "broccolistirfry", "capresebruschetta", "cucumberraita", "herbomeletwithfriedtomatoes",
    "panfriedtofu", "ramen", "scrambledeggs", "spicytunaavocadowraps", "tomatochutney",
    "tomatomozzarellasalad", "zoodles",
}
COOK_VERBS = {"cook", "heat", "microwave", "melt", "saute", "simmer", "boil", "toast", "fry",
              "continue cooking", "blitz", "return", "lower"}
FINE_MOTOR_VERBS = {"chop", "slice", "cut", "mince", "peel", "grate", "crack", "whisk", "beat",
                    "spread", "roll", "prepare", "trim", "spiralize", "secure", "slicing",
                    "cut or tear", "chop or grate", "pull", "seal", "cross", "slide"}
TOOL_NOUNS = {"blender", "bowl", "can", "cup", "board:chopping", "filter", "fork", "pan", "jar",
              "kettle", "knife", "lid", "microwave", "towel:kitchen", "towel", "plate", "scissors",
              "colander", "spatula", "spoon", "tongs", "cocktail", "bag", "package"}

# The proposal (judgment, checked below): everyday dishes, few unnameable items, each knob >= 4.
PROPOSED = [
    "scrambledeggs", "herbomeletwithfriedtomatoes", "tomatomozzarellasalad", "capresebruschetta",
    "cucumberraita", "buttercorncup", "sautedmushrooms", "broccolistirfry", "zoodles",
    "panfriedtofu", "tomatochutney", "microwaveeggsandwich", "microwavefrenchtoast",
    "breakfastburritos", "spicedhotchocolate", "cheesepimiento", "blenderbananapancakes", "coffee",
]
EXCLUDED_WHY = {
    "pinwheels": "dental-floss slicing has no EPIC noun and only 9 valid step orders",
    "mugcake": "frosting, piping bag and cupcake liner have no EPIC nouns",
    "dressedupmeatballs": "fine, but convenience food (pre-made meatballs); a reserve",
    "microwavemugpizza": "fine; a reserve",
    "ramen": "instant noodles; a reserve",
    "spicytunaavocadowraps": "fine; a reserve (2.5M orders)",
}


def _size(path):
    return f"{path.stat().st_size / 1024:.0f} KB"


def _sample(obj, width=600):
    text = json.dumps(obj)
    return text if len(text) <= width else text[:width] + " ..."


def inventory(cc4d):
    lines = ["| file | size |", "|---|---|"]
    for p in sorted(cc4d.rglob("*")):
        if p.is_file() and ".git" not in p.parts:
            lines.append(f"| `{p.relative_to(cc4d)}` | {_size(p)} |")
    return "\n".join(lines)


def schemas(cc4d):
    j = cc4d / "annotation_json"
    graph = json.loads((cc4d / "task_graphs" / "coffee.json").read_text())
    steps = json.loads((j / "step_annotations.json").read_text())
    complete = json.loads((j / "complete_step_annotations.json").read_text())
    errors = json.loads((j / "error_annotations.json").read_text())
    err = next(e for e in errors if e["is_error"])
    err_step = next(s for s in err["step_annotations"] if "errors" in s)
    rec_steps = json.loads((j / "recording_id_step_idx.json").read_text())
    act_steps = json.loads((j / "activity_idx_step_idx.json").read_text())
    step_desc = json.loads((j / "step_idx_description.json").read_text())
    cats = json.loads((j / "error_category_idx.json").read_text())
    durations = [s["end_time"] - s["start_time"] for r in steps.values() for s in r["steps"]
                 if s["start_time"] >= 0]
    return f"""**Task graphs** (`task_graphs/<recipe>.json`, 24 files). Ids are local to the graph
(not the global step ids below). `START` and `END` are ordinary entries in `steps` (START is id 0,
END the largest id); an edge `[a, b]` means a must precede b. Every real step has a parent and a
child, so START is the unique source and END the unique sink.

```
{_sample(graph, 900)}
```

**Step annotations** (`annotation_json/step_annotations.json`, {len(steps)} recordings keyed
`<activity_id>_<n>`): per recording, steps in performed order with global `step_id`, `start_time`,
`end_time` (seconds; -1 for a skipped step), `description` (the graph's text, sometimes with a space
before the hyphen) and `has_errors`. Step durations: median {statistics.median(durations):.1f} s,
p10 {statistics.quantiles(durations, n=10)[0]:.1f} s, p90 {statistics.quantiles(durations, n=10)[-1]:.1f} s.

```
{_sample(steps["1_7"])}
```

`complete_step_annotations.json` is the same plus `activity_id`, `activity_name`, `person_id`,
`environment` (keys: {sorted(complete["1_7"])}).

**Error annotations** (`error_annotations.json`, a list of {len(errors)}): per recording
`is_error` ({sum(not e["is_error"] for e in errors)} normal, {sum(e["is_error"] for e in errors)}
with errors) and `step_annotations`; an erroneous step carries `errors: [{{tag, description}}]` and
`modified_description`. Categories: {", ".join(cats)}.

```
{_sample(err_step)}
```

**Id mappings.** `recording_id_step_idx.json` {{recording_id: [step_id, ...]}} ({len(rec_steps)});
`activity_idx_step_idx.json` {{activity_id: [step_id, ...]}} ({len(act_steps)}, with repeats: it
concatenates recordings); `step_idx_description.json` {{step_id: description}} ({len(step_desc)}
global steps). `metadata/video_information.csv`: recording_id, environment_id, person_id, duration
(min, sec). `metadata/average_segment_length.csv`: per activity. `data_splits/recipes_*.json` are
empty; the other splits list recording ids.
"""


def fine_grained_verdict(cc4d):
    keys = set()
    for p in (cc4d / "annotation_json").glob("*.json"):
        def walk(x):
            if isinstance(x, dict):
                keys.update(x)
                for v in x.values():
                    walk(v)
            elif isinstance(x, list):
                for v in x[:50]:
                    walk(v)
        walk(json.loads(p.read_text()))
    actiony = sorted(k for k in keys if "action" in k.lower())
    return f"""**No.** The CaptainCook4D project page advertises "10K fine-grained action annotations",
but they are not in the annotations repo, the other 9 repos of the CaptainCook4D GitHub org, or
the downloader's link list (which covers video only). No key in any annotation JSON mentions
actions ({actiony or "none found"}). The finest public CC4D unit is the **step**: "Chop-Chop 1/4
medium onion", median ~30-40 s, which is above logistics level (one step is several (verb, noun)
actions, and fetch/put-back is never annotated).

Consequence (decided with Jack): durations and few-shot exemplars come from **EPIC-100** (spec
source 2), whose narrations are exactly logistics level ("take bowl", "open fridge"). CC4D step
durations are used to calibrate and check session length. `durations.py` keeps a `cc4d_fine` source
slot in case the fine-grained annotations are obtained from the authors later."""


def vocab_section(vocab_path):
    vocab = json.loads(vocab_path.read_text())
    return f"""Path `{vocab_path}` (built by `src/cook_ad/data/parse_epic.py`, config `configs/epic.yaml`).
JSON `{{"verbs": {{name: id}}, "nouns": {{name: id}}, "recipes": {{...}}}}`.

- **Verbs and nouns are separate classes**: {len(vocab["verbs"])} verbs (97 EPIC-100 verb classes + SIL
  `stall`), {len(vocab["nouns"])} nouns (305 EPIC-100 noun classes v2 + SIL `idle`). Ids are sorted
  name order. A token is a (verb, noun) pair; any pair is representable.
- The class *instances* in `EPIC_100_{{verb,noun}}_classes*.csv` say what each class covers, and
  that is how CC4D words map: `wipe`, `clean`, `rinse` -> `wash`; `whisk`, `stir`, `beat` -> `mix`;
  `chop`, `slice`, `trim` -> `cut`; `crack` -> `break`; `spread` -> `apply`; `tortilla` -> `wrap`;
  `mug` -> `cup`; `ramekin` -> `bowl`; `toothpick` -> `cocktail`; `paper towel` -> `towel:kitchen`.
- **No `wipe` verb**: the spec's wash/wipe logistics map to `wash` (and `dry`).
- Verbs are coarse: every cutting style is `cut`, every stirring style `mix`. The `fine_motor` knob
  therefore acts on `cut`, `peel`, `grate`, `break`, `mix`, `apply`, `roll`, `fold`.
- EPIC's `curd` instance is bean curd (`tofu`); CC4D's curd (raita) is yoghurt, mapped by hand."""


def stream_section(seq_path):
    seq = json.loads(seq_path.read_text())[0]
    head = {k: (v[:12] if isinstance(v, list) else v) for k, v in seq.items()}
    return f"""`dataset/processed/epic/sequences.json`, a list ordered by sorted `trial_id`:

```
{json.dumps(head)}   (lists truncated)
```

One integer (verb, noun) pair per **0.5 s tick** (`configs/epic.yaml: tick_seconds`). Gaps between
narrations are the SIL pair (`stall`, `idle`). A tick's value is the plurality pair over its frames
(`data/tick_expansion.py:expand_verb_noun_to_ticks`); the trailing idle is trimmed and counted in
`terminal_idle_ticks`. Alongside: `labels.json` (validation only: `trial_id`, `recipe_label`, ...),
`vocab.json`, `split.json`. The loaders read these files directly (`data/split.py`,
`hsmm/batch.pad`), so the generator writes them per person, plus one timestamped event file per
session for anything that wants seconds rather than ticks."""


def recipe_rows(cc4d, graphs):
    j = cc4d / "annotation_json"
    errors = json.loads((j / "error_annotations.json").read_text())
    complete = json.loads((j / "complete_step_annotations.json").read_text())
    with open(cc4d / "metadata" / "video_information.csv", newline="") as f:
        video_sec = {r["recording_id"]: float(r["duration(sec)"]) for r in csv.DictReader(f)}
    normal = {e["recording_id"] for e in errors if not e["is_error"]}
    by_recipe = {}
    for rid, rec in complete.items():
        by_recipe.setdefault(recipe_key(rec["activity_name"]), []).append(rid)

    rows = {}
    for rid_, g in graphs.items():
        recs = by_recipe.get(rid_, [])
        norm = [r for r in recs if r in normal]
        norm_desc = {_norm(s.description) for s in g.steps.values()}
        n_ann = n_hit = 0
        orders = set()
        for r in norm:
            steps = complete[r]["steps"]
            orders.add(tuple(s["step_id"] for s in steps))
            for s in steps:
                n_ann += 1
                n_hit += _norm(s["description"]) in norm_desc
        try:
            n_orders = count_linear_extensions(g)
        except TooManyIdeals:
            n_orders = None
        text = " ".join(s.description for s in g.steps.values()).lower()
        named, unnamed = [], []
        for s in g.steps.values():
            f, u = lexicon.nouns_in(s.description)
            named += f
            unnamed += u
        verbs = [lexicon.step_verb(s.description) for s in g.steps.values()]
        # Prep-capable: a non-cooking step with no cooking step upstream of it, i.e. one that
        # mise en place could pull forward. A free pair is (prep, cook) with the prep step NOT
        # forced before the cook step, so a messy cook can do it after.
        anc = ancestors(g)
        cooks = [k for k, s in g.steps.items() if lexicon.step_verb(s.description) in COOK_VERBS]
        preps = [k for k in g.steps if k not in cooks and not any(c in anc[k] for c in cooks)]
        free_pairs = sum(1 for p in preps for c in cooks if p not in anc[c])
        rows[rid_] = {
            "name": complete[recs[0]]["activity_name"] if recs else rid_,
            "nodes": len(g.steps), "edges": len(g.edges), "orders": n_orders,
            "recordings": len(recs), "normal": len(norm),
            "mean_min": statistics.mean(video_sec[r] for r in norm) / 60 if norm else float("nan"),
            "distinct_orders": len(orders), "desc_match": n_hit / n_ann if n_ann else float("nan"),
            "epic_nouns": sorted(set(named)), "unnamed": sorted(set(unnamed)),
            "sour": sorted({w for w in SOUR_WORDS if re.search(rf"\b{w}", text)}),
            "sour_plausible": rid_ in SOUR_INSERT_PLAUSIBLE,
            "fat": sorted({w for w in FAT_WORDS if re.search(rf"\b{w}", text)}),
            "prep": len(preps), "cook": len(cooks), "free_pairs": free_pairs,
            "fine_motor": sum(v in FINE_MOTOR_VERBS for v in verbs),
            "tools": len(set(named) & TOOL_NOUNS),
        }
    return rows


def _norm(desc):
    return " ".join(desc.replace(" -", "-").lower().split())


def covers(row):
    """Knob -> does this recipe give the knob something to act on (estimate)."""
    return {
        "sour": bool(row["sour"]) or row["sour_plausible"],
        "calorie_conscious": bool(row["fat"]),
        "mise_en_place": row["free_pairs"] >= 2,
        "tidy": row["tools"] >= 2,
        "slow": True,
        "fine_motor": row["fine_motor"] >= 2,
    }


def recipe_table(rows):
    out = ["| recipe | steps | edges | valid orders | recordings | normal | mean normal min | "
           "distinct normal orders | text match | EPIC nouns | no EPIC noun |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in sorted(rows.items()):
        orders = f"{r['orders']:.3g}" if r["orders"] else "> cap"
        out.append(f"| {k} | {r['nodes']} | {r['edges']} | {orders} | {r['recordings']} | "
                   f"{r['normal']} | {r['mean_min']:.1f} | {r['distinct_orders']} | "
                   f"{r['desc_match']:.0%} | {len(r['epic_nouns'])} | "
                   f"{', '.join(r['unnamed']) or '-'} |")
    return "\n".join(out)


def coverage_table(rows, recipes):
    knobs = list(covers(next(iter(rows.values()))))
    out = ["| recipe | sour words (or insert plausible) | fat words | prep / cook steps "
           "(free pairs) | fine-motor steps | tool nouns | " + " | ".join(knobs) + " |",
           "|---|---|---|---|---|---|" + "---|" * len(knobs)]
    totals = Counter()
    for k in recipes:
        r = rows[k]
        c = covers(r)
        totals.update(n for n, v in c.items() if v)
        sour = ", ".join(r["sour"]) or ("(plausible)" if r["sour_plausible"] else "-")
        out.append(f"| {k} | {sour} | {', '.join(r['fat']) or '-'} | {r['prep']} / {r['cook']} "
                   f"({r['free_pairs']}) | {r['fine_motor']} | {r['tools']} | "
                   + " | ".join("x" if c[n] else "" for n in knobs) + " |")
    out.append(f"| **recipes covering** | | | | | | " + " | ".join(str(totals[n]) for n in knobs) + " |")
    return "\n".join(out), totals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cc4d", default="dataset/captaincook4d")
    ap.add_argument("--epic-processed", default="dataset/processed/epic")
    ap.add_argument("--out", default="runs/drift_synth/reports/precheck.md")
    args = ap.parse_args()
    cc4d = Path(args.cc4d)
    epic = Path(args.epic_processed)

    graphs = load_task_graphs(cc4d)
    rows = recipe_rows(cc4d, graphs)
    missing = [k for k in PROPOSED if k not in rows]
    if missing:
        raise SystemExit(f"proposed recipes not in CC4D: {missing}")
    all_table, _ = coverage_table(rows, sorted(rows))
    prop_table, totals = coverage_table(rows, PROPOSED)
    short = [n for n, v in totals.items() if v < 4]

    report = f"""# Pre-check: synthetic drift data from CaptainCook4D

Written by `scripts/drift_synth/precheck.py`. CC4D annotations cloned from
github.com/CaptainCook4D/annotations into `{cc4d}`; EPIC-100 annotations from
github.com/epic-kitchens/epic-kitchens-100-annotations into `dataset/epic_kitchens/annotations`.

## 1. CaptainCook4D inventory and schemas

{inventory(cc4d)}

{schemas(cc4d)}

## 2. Are there logistics-level fine-grained actions?

{fine_grained_verdict(cc4d)}

## 3. The closed vocabulary

{vocab_section(epic / "vocab.json")}

## 4. The stream the HSMM consumes

{stream_section(epic / "sequences.json")}

## 5. Recipes

All 24 task graphs load and pass validation (acyclic, connected, START the only source and END the
only sink). "Steps" and "edges" exclude START/END. "Valid orders" is the exact number of
topological orders (DP over order ideals, the same counter the ordering sampler will use). "Normal"
is `is_error == false`; durations are the video lengths of normal recordings. "Text match" is the
share of normal recordings' annotated steps whose description equals a graph step's (how step
annotations will be joined to graph nodes). "No EPIC noun" lists items in the step text that
EPIC-100 has no class for (`src/cook_ad/drift_synth/lexicon.py`, hand-made).

{recipe_table(rows)}

## 6. Proposed recipes (Jack approves)

Knob coverage is an **estimate from step text**: tags do not exist until Stage 2. A recipe covers

- `sour` if its text names a sour ingredient, or adding lemon, lime, vinegar or yoghurt to an existing step is ordinary (a judgment list in the script);
- `calorie_conscious` if it names a fat ingredient;
- `mise_en_place` if at least 2 (prep, cook) step pairs are unordered by the graph, so the knob can move them. "Cook" is guessed from the step verb (cook, heat, microwave, melt, ...); "prep" is any other step with no cooking step upstream;
- `tidy` if it uses at least 2 tools or containers;
- `slow` always;
- `fine_motor` if at least 2 steps use a fine-motor verb.

**Proposal: {len(PROPOSED)} recipes.** Every knob is covered by
{min(totals.values())} or more ({", ".join(f"{n} {v}" for n, v in totals.items())}){"; SHORT: " + ", ".join(short) if short else ""}.

{prop_table}

Left out:

{chr(10).join(f"- `{k}`: {v}" for k, v in EXCLUDED_WHY.items())}

All 24, for reference:

{all_table}

## 7. Findings that bend the spec

- **Fine-grained actions are unavailable** (§2), so the duration and exemplar source is EPIC.
- **No `wipe`** in the vocab: the cleaning logistics are `wash` / `dry`.
- **Coarse EPIC verbs**: `fine_motor`'s duration and repetition effects land on a handful of
  verb classes (§3), and `mise_en_place` ordering is mostly visible through nouns.
- **`pepper` is one class** for bell pepper and black pepper, and **`sauce`** covers soy, salsa,
  sriracha, mayonnaise and marinara. Sour or fat variants built on a sauce are invisible in noun
  space; tagging must not tag `sauce` sour or fat on the strength of one member.
- **`mise_en_place` has little room in CC4D.** The task graphs already force almost every prep
  step before the first cooking step: 18 of 24 recipes have *no* (prep, cook) pair the knob could
  reorder, including every stovetop dish (scrambled eggs, stir fry, chutney, sauteed mushrooms,
  tofu). Six recipes have room (banana pancakes, butter corn cup, caprese bruschetta, cheese
  pimiento, coffee, microwave egg sandwich), and all six are in the proposal. Even there the knob
  moves 1-4 steps, typically a garnish or a side component, so its effect will be small.
- **The ordering sampler**: a softmax over available steps is not uniform over topological
  orders, which test 8 requires at mise_en_place = 0.5. Candidates will be weighted by the number
  of completions (the §5 counter); every graph counts in milliseconds.
"""
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    print(f"wrote {out}; proposal coverage {dict(totals)}")


if __name__ == "__main__":
    main()
