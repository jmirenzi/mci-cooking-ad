"""Stages 1-3 of the drift generator (docs/drift_synth.md): the offline LLM assets. Run once;
every result is cached and the parsed outputs are committed under configs/drift_synth/.

    ./py scripts/drift_synth/build_assets.py --stage phases
    ./py scripts/drift_synth/build_assets.py --stage tags      # then review tags.yaml by hand
"""
import argparse
import json
from pathlib import Path

import yaml

from cook_ad.drift_synth import tagging
from cook_ad.drift_synth.graphs import assign_phase, load_task_graphs


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["phases", "tags"], required=True)
    ap.add_argument("--config", default="configs/drift_synth/assets.yaml")
    ap.add_argument("--recipes", default="configs/drift_synth/recipes.yaml")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    report_dir = Path(cfg["paths"]["reports"])
    report_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "phases":
        stage_phases(cfg, load_cfg(args.recipes)["recipes"], report_dir)
    else:
        stage_tags(cfg, report_dir)


if __name__ == "__main__":
    main()
