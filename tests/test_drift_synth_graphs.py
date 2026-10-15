import json
import subprocess
import sys

import pytest

from cook_ad.drift_synth import lexicon
from cook_ad.drift_synth.graphs import (count_linear_extensions, is_topological, load_task_graph,
                                        topological_order, validate_graph)
from cook_ad.drift_synth.types import RecipeGraph, Step


def _graph(n, edges, recipe_id="toy"):
    return RecipeGraph(recipe_id, {str(i): Step(str(i), f"Do-step {i}") for i in range(1, n + 1)},
                       [(str(p), str(c)) for p, c in edges])


def _write(tmp_path, steps, edges, name="toy"):
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps({"steps": steps, "edges": edges}))
    return path


@pytest.mark.parametrize("n, edges, expected", [
    (3, [(1, 2), (2, 3)], 1),                  # chain
    (3, [(1, 3), (2, 3)], 2),                  # 1 and 2 free, both before 3
    (4, [(1, 2), (1, 3), (2, 4), (3, 4)], 2),  # diamond
    (5, [(1, 2), (3, 4), (2, 5), (4, 5)], 6),  # two chains of 2 interleaved: C(4, 2)
])
def test_count_linear_extensions(n, edges, expected):
    g = _graph(n, edges)
    validate_graph(g)
    assert count_linear_extensions(g) == expected


def test_disconnected_raises():
    with pytest.raises(ValueError, match="not connected"):
        validate_graph(_graph(3, [(1, 3)]))


def test_topological_order_and_check():
    g = _graph(4, [(1, 2), (1, 3), (2, 4), (3, 4)])
    order = topological_order(g)
    assert is_topological(g, order)
    assert not is_topological(g, ["2", "1", "3", "4"])
    assert not is_topological(g, ["1", "2", "4"])


def test_cycle_raises():
    with pytest.raises(ValueError, match="cycle"):
        validate_graph(_graph(3, [(1, 2), (2, 3), (3, 1)]))


def test_load_strips_sentinels(tmp_path):
    path = _write(tmp_path, {"0": "START", "1": "Chop-Chop 1 onion", "2": "Cook-Cook it", "3": "END"},
                  [[0, 1], [1, 2], [2, 3]])
    g = load_task_graph(path)
    assert set(g.steps) == {"1", "2"} and g.edges == [("1", "2")]


def test_step_without_parent_raises(tmp_path):
    path = _write(tmp_path, {"0": "START", "1": "Chop-a", "2": "Cook-b", "3": "END"},
                  [[0, 1], [1, 3], [2, 3]])
    with pytest.raises(ValueError, match="only source"):
        load_task_graph(path)


def test_lexicon_matches_longest_phrase_first():
    found, unmapped = lexicon.nouns_in("Heat 2 tablespoons olive oil in a skillet; slice with floss")
    assert sorted(found) == ["oil", "pan"] and unmapped == ["floss"]
    assert lexicon.step_verb("Chop or grate-chop or grate the cucumber") == "chop or grate"


def test_drift_synth_imports_no_model_code():
    """Ground truth lives in observable space: the generator must not touch the HSMM."""
    code = ("import sys, importlib, pkgutil, cook_ad.drift_synth as p\n"
            "for m in pkgutil.iter_modules(p.__path__): importlib.import_module(f'{p.__name__}.{m.name}')\n"
            "bad = [m for m in sys.modules if m.startswith(('cook_ad.hsmm', 'cook_ad.detectors', 'cook_ad.eval'))]\n"
            "print(bad); sys.exit(1 if bad else 0)")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
