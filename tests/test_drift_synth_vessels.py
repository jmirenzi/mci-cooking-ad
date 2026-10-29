"""configs/drift_synth/vessels.yaml names real steps and real containers or tools."""
import json
from pathlib import Path

import yaml

ROOT = Path("configs/drift_synth")


def test_vessels_yaml_names_real_steps_and_vessel_nouns():
    ves = yaml.safe_load((ROOT / "vessels.yaml").read_text())["vessels"]
    nouns = yaml.safe_load((ROOT / "tags.yaml").read_text())["nouns"]
    assert set(ves) == set(yaml.safe_load((ROOT / "recipes.yaml").read_text())["recipes"])
    for r, d in ves.items():
        steps = json.loads((ROOT / "expansions" / f"{r}.json").read_text())
        used = {n for v in d["steps"].values() for n in v}
        assert all(str(s) in steps for s in d["steps"]), r
        assert all({"tool", "container"} & set(nouns[n]) for n in used), r
        assert d.get("serves_in") in (None, *used), r


MICROWAVE = ["open microwave", "insert {x}", "close microwave", "turn-on microwave", "wait microwave",
             "open microwave", "remove {x}"]


def test_every_microwave_use_reads_the_same():
    """Any step that touches the microwave runs the one sequence, for whatever vessel goes in."""
    for p in sorted((ROOT / "expansions").glob("*.json")):
        for s, d in json.loads(p.read_text()).items():
            toks = [" ".join(t) for t in d["tokens"]]
            if not any("microwave" in t for t in toks):
                continue
            i = next(j for j, t in enumerate(toks) if t == "open microwave")
            x = toks[i + 1].split()[1]
            assert toks[i:] == [m.format(x=x) for m in MICROWAVE], (p.stem, s, toks)
