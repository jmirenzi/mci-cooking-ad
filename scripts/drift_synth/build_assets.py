"""Stages 1-3 of the drift generator (docs/drift_synth.md): the offline LLM assets. Run once;
every result is cached and the parsed outputs are committed under configs/drift_synth/.

    ./py scripts/drift_synth/build_assets.py --stage phases
    ./py scripts/drift_synth/build_assets.py --stage tags      # then review tags.yaml by hand
    ./py scripts/drift_synth/build_assets.py --stage expansions [--recipe R]   # then spot-check
"""
import argparse
import json
import random
from pathlib import Path

import yaml

from cook_ad.drift_synth import lexicon, tagging
from cook_ad.drift_synth.expand import ExemplarIndex, build_expansions
from cook_ad.drift_synth.graphs import _id_key, assign_phase, load_phased, load_task_graphs


def load_cfg(path):
    return yaml.safe_load(Path(path).read_text())


def stage_phases(cfg, recipes, report_dir):
    graphs = load_task_graphs(cfg["paths"]["cc4d"])
    lines = ["# Step phases (LLM, Stage 1)", ""]
    for r in recipes:
        g = assign_phase(graphs[r], cfg["llm"], cfg["paths"]["phases"])
        lines += [f"## {r}", ""] + [f"- `{s.phase}` {k}: {s.description}"
                                     for k, s in sorted(g.steps.items(), key=lambda kv: int(kv[0]))]
        lines.append("")
    (report_dir / "phases.md").write_text("\n".join(lines))
    print(f"phases for {len(recipes)} recipes -> {cfg['paths']['phases']}, {report_dir / 'phases.md'}")


def stage_tags(cfg, report_dir):
    path = Path(cfg["paths"]["tags"])
    if path.exists() and tagging.load_tags(path, require_reviewed=False) and \
            yaml.safe_load(path.read_text()).get("reviewed"):
        raise SystemExit(f"{path} is already reviewed; refusing to overwrite it")
    vocab = json.loads(Path(cfg["paths"]["vocab"]).read_text())
    instances = tagging.class_instances(cfg["paths"]["epic_annotations"])
    tags = tagging.propose_tags(vocab, instances, cfg["llm"], cfg["tag_batch_size"])
    tagging.write_tags(path, tags, cfg["tag_set"])
    (report_dir / "tags_summary.md").write_text("# Proposed tags (LLM, Stage 2)\n\n" + tagging.summary(tags) + "\n")
    print(f"wrote {path} (reviewed: false) and {report_dir / 'tags_summary.md'}")


def step_nouns(description):
    """EPIC nouns a step names (the pre-check lexicon), for exemplar lookup only."""
    return sorted(set(lexicon.nouns_in(description)[0]))


def stage_expansions(cfg, recipes, report_dir, review_frac=0.1, seed=0):
    p = cfg["paths"]
    vocab = json.loads(Path(p["vocab"]).read_text())
    tags = tagging.load_tags(p["tags"])
    graphs = load_phased(p["cc4d"], recipes, p["phases"])
    print("indexing EPIC exemplars ...")
    index = ExemplarIndex.from_epic(p["epic_annotations"])
    all_exp, all_fail = {}, {}
    for r in recipes:
        exps, fails = build_expansions(graphs[r], vocab, tags, index, cfg["llm"], p["expansions"],
                                       step_nouns)
        all_exp[r], all_fail[r] = exps, fails
        n_var = sum(len(e.variants) for e in exps.values())
        skipped = len(graphs[r].steps) - len(exps)
        print(f"{r}: {len(exps)}/{len(graphs[r].steps)} steps, {n_var} variants, "
              f"{skipped} skipped, {len(fails)} with a failed pass")

    lines = ["# Expansion failures (Stage 3)", "",
             "Passes whose 3 attempts all failed validation. A failed `core` pass skips the step "
             "in the sampler; a failed variant pass leaves that variant out.", ""]
    for r, fails in all_fail.items():
        for sid, log in sorted(fails.items(), key=lambda kv: _id_key(kv[0])):
            for what, hist in log.items():
                if hist and hist[-1]:
                    desc = graphs[r].steps[sid].description if sid in graphs[r].steps else "(whole recipe)"
                    lines.append(f"- **{r} {sid}** `{what}` {desc}: "
                                 + " / ".join("; ".join(e) for e in hist))
    (report_dir / "expansion_failures.md").write_text("\n".join(lines) + "\n")

    pool = [(r, sid) for r in recipes for sid in sorted(all_exp[r], key=_id_key)]
    picks = sorted(random.Random(seed).sample(pool, max(1, round(review_frac * len(pool)))))
    lines = [f"# Expansion review (Stage 3): a seeded random {review_frac:.0%} of steps", "",
             f"{len(pool)} steps expanded across {len(recipes)} recipes; "
             f"{sum(len(graphs[r].steps) - len(all_exp[r]) for r in recipes)} skipped (expansion_failures.md).", ""]
    for r, sid in picks:
        e = all_exp[r][sid]
        lines += [f"## {r} {sid} ({graphs[r].steps[sid].phase})", "",
                  f"> {graphs[r].steps[sid].description}", "",
                  "tokens: " + ", ".join(f"`{v} {n}`" for v, n in e.tokens)]
        for var in e.variants:
            what = ", ".join(f"`{v} {n}`" for v, n in var.tokens) or "-"
            lines.append(f"- variant {var.knob} {var.kind} replaces={var.replaces}: {what}")
        lines.append("")
    (report_dir / "expansion_review.md").write_text("\n".join(lines))
    print(f"wrote {report_dir / 'expansion_review.md'} ({len(picks)} steps) and expansion_failures.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["phases", "tags", "expansions"], required=True)
    ap.add_argument("--recipe", action="append", help="only these recipes (repeatable)")
    ap.add_argument("--config", default="configs/drift_synth/assets.yaml")
    ap.add_argument("--recipes", default="configs/drift_synth/recipes.yaml")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    report_dir = Path(cfg["paths"]["reports"])
    report_dir.mkdir(parents=True, exist_ok=True)
    recipes = args.recipe or load_cfg(args.recipes)["recipes"]
    if args.stage == "phases":
        stage_phases(cfg, recipes, report_dir)
    elif args.stage == "tags":
        stage_tags(cfg, report_dir)
    else:
        stage_expansions(cfg, recipes, report_dir)


if __name__ == "__main__":
    main()
