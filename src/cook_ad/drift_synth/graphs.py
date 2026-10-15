"""Stage 1: CaptainCook4D task graphs.

A task graph file is {"steps": {id: "Verb-description"}, "edges": [[before, after], ...]} with a
START and an END sentinel step. Loading strips the sentinels; every real step must be reachable
from START and reach END, which is what makes the sentinels the graph's unique source and sink.
"""
import json
from functools import lru_cache
from pathlib import Path

from cook_ad.drift_synth.types import RecipeGraph, Step

SENTINELS = ("START", "END")


def recipe_key(activity_name):
    """CC4D activity name -> task graph file stem ("Microwave Egg Sandwich" -> microwaveeggsandwich)."""
    return "".join(ch for ch in activity_name.lower() if ch.isalnum())


def _check_sentinels(recipe_id, steps, edges):
    by_name = {v: k for k, v in steps.items() if v in SENTINELS}
    missing = [s for s in SENTINELS if s not in by_name]
    if missing:
        raise ValueError(f"{recipe_id}: no {missing} step")
    start, end = by_name["START"], by_name["END"]
    has_parent = {c for _, c in edges}
    has_child = {p for p, _ in edges}
    for k in steps:
        if k != start and k not in has_parent:
            raise ValueError(f"{recipe_id}: step {k} has no parent, so START is not the only source")
        if k != end and k not in has_child:
            raise ValueError(f"{recipe_id}: step {k} has no child, so END is not the only sink")
    return start, end


def load_task_graph(path):
    raw = json.loads(Path(path).read_text())
    recipe_id = Path(path).stem
    steps = {str(k): v for k, v in raw["steps"].items()}
    edges = [(str(p), str(c)) for p, c in raw["edges"]]
    start, end = _check_sentinels(recipe_id, steps, edges)
    graph = RecipeGraph(
        recipe_id=recipe_id,
        steps={k: Step(step_id=k, description=v) for k, v in steps.items() if k not in (start, end)},
        edges=[(p, c) for p, c in edges if start not in (p, c) and end not in (p, c)],
    )
    validate_graph(graph)
    return graph


def load_task_graphs(root):
    """{recipe_id: RecipeGraph} for every task_graphs/*.json under the CC4D annotations root."""
    paths = sorted((Path(root) / "task_graphs").glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"no task graphs under {root}/task_graphs -- clone "
                                f"github.com/CaptainCook4D/annotations there")
    return {g.recipe_id: g for g in map(load_task_graph, paths)}


def validate_graph(graph):
    """Acyclic, weakly connected, and every edge between real steps."""
    nodes = set(graph.steps)
    for p, c in graph.edges:
        if p not in nodes or c not in nodes:
            raise ValueError(f"{graph.recipe_id}: edge {p}->{c} names an unknown step")
    topological_order(graph)  # raises on a cycle
    undirected = {k: set() for k in nodes}
    for p, c in graph.edges:
        undirected[p].add(c)
        undirected[c].add(p)
    seen, stack = set(), [next(iter(nodes))]
    while stack:
        k = stack.pop()
        if k not in seen:
            seen.add(k)
            stack.extend(undirected[k] - seen)
    if seen != nodes:
        raise ValueError(f"{graph.recipe_id}: not connected ({len(nodes - seen)} steps unreachable)")


def parents(graph):
    out = {k: set() for k in graph.steps}
    for p, c in graph.edges:
        out[c].add(p)
    return out


def topological_order(graph):
    """One deterministic topological order (Kahn, smallest id first); raises on a cycle."""
    par = parents(graph)
    remaining = {k: set(v) for k, v in par.items()}
    order = []
    while remaining:
        ready = sorted((k for k, v in remaining.items() if not v), key=_id_key)
        if not ready:
            raise ValueError(f"{graph.recipe_id}: cycle among {sorted(remaining)}")
        k = ready[0]
        order.append(k)
        del remaining[k]
        for v in remaining.values():
            v.discard(k)
    return order


def ancestors(graph):
    """{step: every step that must precede it} (transitive closure of the edges)."""
    par = parents(graph)
    out = {}
    for k in topological_order(graph):
        out[k] = set(par[k]).union(*(out[p] for p in par[k])) if par[k] else set()
    return out


def is_topological(graph, order):
    pos = {k: i for i, k in enumerate(order)}
    return set(order) == set(graph.steps) and len(order) == len(graph.steps) and all(
        pos[p] < pos[c] for p, c in graph.edges)


class TooManyIdeals(RuntimeError):
    pass


def linear_extension_counter(graph, cap=2_000_000):
    """-> (index, count): `count(mask)` is the number of topological orders of the steps still in
    `mask` (bit i = graph step index[i]). A DP over order ideals; raises TooManyIdeals past `cap`
    memoised states, so a graph too free to count is reported rather than hanging."""
    index = sorted(graph.steps, key=_id_key)
    bit = {k: 1 << i for i, k in enumerate(index)}
    par_mask = [sum(bit[p] for p in parents(graph)[k]) for k in index]
    n = len(index)

    @lru_cache(maxsize=None)
    def count(mask):
        if mask == 0:
            return 1
        if count.cache_info().currsize > cap:
            raise TooManyIdeals(f"{graph.recipe_id}: more than {cap} order ideals")
        total = 0
        for i in range(n):
            # i is available iff it is still remaining and none of its parents is
            if mask >> i & 1 and not par_mask[i] & mask:
                total += count(mask & ~(1 << i))
        return total

    return index, count


def count_linear_extensions(graph, cap=2_000_000):
    index, count = linear_extension_counter(graph, cap)
    return count((1 << len(index)) - 1)


def _id_key(k):
    return (0, int(k)) if k.isdigit() else (1, k)
