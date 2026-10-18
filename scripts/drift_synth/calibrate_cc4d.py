"""Calibrate the drift generator against CaptainCook4D: real people on the same task graphs
(docs/drift_synth.md §9.3). Writes runs/drift_synth/reports/cc4d_calibration.{md,json}.

1. Orders. Per approved recipe, real recordings with no Order Error or Missing Step against
   rollouts: distinct orders, mean normalized Kendall distance to the medoid order, and the
   prep-before-cook share over (prep, cook) step pairs the graph leaves unordered.
2. Habit. Within-person vs between-person Kendall distance on the participants' repeat
   recordings of a recipe, against simulated people. Two parameters: `habit` (how tightly a person
   repeats their routine) and `routine_concentration` (how close people's routines are to a shared
   typical order, the real medoid; 0 = routines uniform over legal orders, the generator today).
   Each is fitted by grid search to the real distance.
3. Durations. Per step, the CC4D step duration (normal recordings) against the generator's time
   for that step's tokens plus their logistics and gaps, noise off, theta = 0.5, speed 1.

    ./py scripts/drift_synth/calibrate_cc4d.py [--sims 200]
"""
import argparse
import json
import statistics
from itertools import combinations
from pathlib import Path

import numpy as np
import yaml

from cook_ad.drift_synth.assets import load_assets
from cook_ad.drift_synth.graphs import ancestors, recipe_key
from cook_ad.drift_synth.knobs import sample_topological_order
from cook_ad.drift_synth.noise import NoiseCfg
from cook_ad.drift_synth.sampler import rng_for, sample_session, step_owners

BAD_TAGS = {"Order Error", "Missing Step"}
HABITS = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0)
CONCENTRATIONS = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0)
WAIT_VERBS = {"wait"}


def _norm(desc):
    return " ".join(desc.replace(" -", "-").lower().split())


def kendall(a, b):
    """Normalized Kendall distance over the steps both orders contain (0 = same, 1 = reversed)."""
    common = [s for s in a if s in set(b)]
    pos = {s: i for i, s in enumerate(b)}
    n = len(common)
    if n < 2:
        return None
    disc = sum(1 for i in range(n) for j in range(i + 1, n) if pos[common[i]] > pos[common[j]])
    return disc / (n * (n - 1) / 2)


def medoid(orders):
    best = min(orders, key=lambda o: sum(kendall(o, p) or 0.0 for p in orders))
    return list(best)


def real_recordings(cc4d, graphs):
    """{recipe: [(person, recording_id, order of local step ids, {step: seconds})]}."""
    j = Path(cc4d) / "annotation_json"
    complete = json.loads((j / "complete_step_annotations.json").read_text())
    errors = json.loads((j / "error_annotations.json").read_text())
    tags = {e["recording_id"]: {er["tag"] for s in e["step_annotations"] for er in s.get("errors", [])}
            for e in errors}
    normal = {e["recording_id"] for e in errors if not e["is_error"]}
    out = {}
    for rid, rec in complete.items():
        r = recipe_key(rec["activity_name"])
        if r not in graphs or tags.get(rid, set()) & BAD_TAGS:
            continue
        by_desc = {_norm(s.description): k for k, s in graphs[r].steps.items()}
        steps = [s for s in rec["steps"] if s["start_time"] >= 0 and _norm(s["description"]) in by_desc]
        order = [by_desc[_norm(s["description"])] for s in sorted(steps, key=lambda s: s["start_time"])]
        durs = {by_desc[_norm(s["description"])]: s["end_time"] - s["start_time"] for s in steps}
        out.setdefault(r, []).append({"person": rec["person_id"], "rid": rid, "order": order,
                                      "durations": durs, "normal": rid in normal})
    return out


def complete_order(graph, partial):
    """A full topological order following `partial` wherever the graph allows: repeatedly take
    the available step earliest in `partial` (steps it lacks go when nothing else is free)."""
    from cook_ad.drift_synth.graphs import parents
    pos = {k: i for i, k in enumerate(partial)}
    par = {k: set(v) for k, v in parents(graph).items()}
    done, out = set(), []
    while len(out) < len(graph.steps):
        free = [k for k in graph.steps if k not in done and par[k] <= done]
        k = min(free, key=lambda x: (pos.get(x, len(pos)), int(x)))
        out.append(k)
        done.add(k)
    return out


def free_prep_cook_pairs(graph):
    anc = ancestors(graph)
    prep = [k for k, s in graph.steps.items() if s.phase == "prep"]
    cook = [k for k, s in graph.steps.items() if s.phase == "cook"]
    return [(p, c) for p in prep for c in cook if p not in anc[c] and c not in anc[p]]


def order_stats(orders, graph, med):
    pairs = free_prep_cook_pairs(graph)
    pc = []
    for o in orders:
        pos = {s: i for i, s in enumerate(o)}
        pc += [pos[p] < pos[c] for p, c in pairs if p in pos and c in pos]
    d = [kendall(o, med) for o in orders]
    return {"n": len(orders), "distinct": len({tuple(o) for o in orders}) / len(orders),
            "distinct_at": None,
            "to_medoid": float(np.mean([x for x in d if x is not None])),
            "prep_first": float(np.mean(pc)) if pc else None, "free_pairs": len(pairs)}


def sim_people(graph, canon, habit, conc, n_people, per_person, rng, beta):
    """[[orders of person p]]: each person's routine ~ Mallows(canon, conc), sessions ~
    Mallows(routine, habit), all through the generator's own sampler (mise_en_place = 0.5)."""
    people = []
    for _ in range(n_people):
        routine = sample_topological_order(graph, 0.5, beta, rng, canon, conc)
        people.append([sample_topological_order(graph, 0.5, beta, rng, routine, habit)
                       for _ in range(per_person)])
    return people


def within_between(people):
    w = [kendall(a, b) for p in people for a, b in combinations(p, 2)]
    b = [kendall(a, c) for p, q in combinations(people, 2) for a in p for c in q]
    return float(np.mean([x for x in w if x is not None])), float(np.mean([x for x in b if x is not None]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=200, help="rollouts per recipe for order/duration stats")
    ap.add_argument("--out", default="runs/drift_synth/reports/cc4d_calibration")
    ap.add_argument("--write-scales", default=None,
                    help="write per-step time scales (CC4D median / generator median) here, e.g. "
                         "configs/drift_synth/step_time_scale.json")
    args = ap.parse_args()
    assets = load_assets(require_reviewed_affinity=False)   # theta 0.5 below: no flavour inserts
    assets.step_scale = None   # measure the UNcalibrated generator
    graphs = assets.graphs
    beta = assets.knobs["mise_en_place"]["beta"]
    real = real_recordings(assets.gen["paths"]["cc4d"], graphs)
    rng = np.random.default_rng(0)
    report = {"orders": {}, "habit": {}, "durations": {}}

    # 1. orders, real vs uniform rollouts (the generator at habit 0)
    for r in sorted(graphs):
        recs = real.get(r, [])
        if len(recs) < 2:
            continue
        orders = [x["order"] for x in recs]
        med = medoid(orders)
        sims = [sample_topological_order(graphs[r], 0.5, beta, rng) for _ in range(args.sims)]
        sim_stats = order_stats(sims, graphs[r], med)
        # distinct share depends on n: compare at the real sample size
        sim_stats["distinct_at"] = float(np.mean([
            len({tuple(sims[i]) for i in rng.choice(len(sims), len(orders), replace=False)}) / len(orders)
            for _ in range(200)]))
        report["orders"][r] = {"real": order_stats(orders, graphs[r], med),
                               "sim_uniform": sim_stats,
                               "real_valid_topological": float(np.mean([
                                   all(o.index(p) < o.index(c) for p, c in graphs[r].edges
                                       if p in o and c in o) for o in orders])),
                               "medoid": complete_order(graphs[r], med)}

    # 2. habit: real within/between person distances vs simulated people
    w_real, b_real, weights = [], [], {}
    for r, recs in real.items():
        by_p = {}
        for x in recs:
            by_p.setdefault(x["person"], []).append(x["order"])
        w = [kendall(a, b) for os in by_p.values() for a, b in combinations(os, 2)]
        b = [kendall(a, c) for p, q in combinations(list(by_p.values()), 2) for a in p for c in q]
        w = [x for x in w if x is not None]
        b = [x for x in b if x is not None]
        if w and b:
            w_real += w
            b_real += b
            weights[r] = (len(w), len(b), len(by_p), max(len(v) for v in by_p.values()))
    real_w, real_b = float(np.mean(w_real)), float(np.mean(b_real))
    grid = {}
    for h in HABITS:
        for c in CONCENTRATIONS:
            ws, bs = [], []
            for r, (nw, nb, n_people, _) in weights.items():
                canon = report["orders"][r]["medoid"]
                w, b = within_between(sim_people(graphs[r], canon, h, c, max(n_people, 4), 3,
                                                 rng, beta))
                ws.append((w, nw))
                bs.append((b, nb))
            grid[(h, c)] = (sum(w * n for w, n in ws) / sum(n for _, n in ws),
                            sum(b * n for b, n in bs) / sum(n for _, n in bs))
    best = min(grid, key=lambda k: (grid[k][0] - real_w) ** 2 + (grid[k][1] - real_b) ** 2)
    report["habit"] = {"real_within": real_w, "real_between": real_b, "n_within_pairs": len(w_real),
                       "n_between_pairs": len(b_real), "recipes": {r: list(v) for r, v in weights.items()},
                       "grid": {f"{h:g},{c:g}": v for (h, c), v in grid.items()},
                       "best": {"habit": best[0], "routine_concentration": best[1],
                                "sim_within": grid[best][0], "sim_between": grid[best][1]},
                       "today": {"habit": 2.0, "routine_concentration": 0.0,
                                 "sim_within": grid[(2.0, 0.0)][0], "sim_between": grid[(2.0, 0.0)][1]}}

    # 3. durations per step, CC4D normal recordings vs generator (noise off, theta 0.5)
    theta = {k: 0.5 for k in assets.knobs}
    off = NoiseCfg().off()
    rows, ratio_wait, ratio_core, sess_real, sess_sim = [], [], [], [], []
    for r in sorted(graphs):
        recs = [x for x in real.get(r, []) if x["normal"]]
        if not recs:
            continue
        step_sim, step_core, step_log = {}, {}, {}
        tot_sim = []
        for i in range(args.sims):
            s = sample_session(graphs[r], assets.expansions[r], theta, 1.0, assets, off, rng_for(7, 77, i))
            ev = s.meta["clean_events"]
            tot_sim.append(ev[-1][1] - ev[0][0] if ev else 0.0)
            # each event's time (to the next event's start) goes to its step, by the sampler's own
            # ownership rule, so the scales written below are the ones the generator applies
            # ownership rule; take/put/wash (e[4] None) are not scaled, so they are kept apart
            owner = step_owners([e[4] for e in ev])
            core, logi = {}, {}
            for k, e in enumerate(ev):
                end = ev[k + 1][0] if k + 1 < len(ev) else e[1]
                acc = logi if e[4] is None else core
                acc[owner[k]] = acc.get(owner[k], 0.0) + (end - e[0])
            for st in set(core) | set(logi):
                step_core.setdefault(st, []).append(core.get(st, 0.0))
                step_log.setdefault(st, []).append(logi.get(st, 0.0))
                step_sim.setdefault(st, []).append(core.get(st, 0.0) + logi.get(st, 0.0))
        sess_sim.append(float(np.median(tot_sim)))
        sess_real.append(float(np.median([sum(x["durations"].values()) for x in recs])))
        has_wait = {k: any(v in WAIT_VERBS for v, _ in e.tokens) for k, e in assets.expansions[r].items()}
        for k, step in graphs[r].steps.items():
            real_d = [x["durations"][k] for x in recs if k in x["durations"]]
            if not real_d or k not in step_sim:
                continue
            rd, sd = statistics.median(real_d), statistics.median(step_sim[k])
            sc, sl = statistics.mean(step_core[k]), statistics.mean(step_log[k])
            rows.append({"recipe": r, "step": k, "description": step.description[:60], "real_s": rd,
                         "sim_s": sd, "ratio": rd / sd if sd else None, "wait": has_wait.get(k, False),
                         # the scale on the step's own actions that brings its total to rd, with
                         # its take/put/wash left at EPIC time
                         "sim_core_s": sc, "sim_logistics_s": sl,
                         "core_scale": (rd - sl) / sc if sc else None})
            (ratio_wait if has_wait.get(k) else ratio_core).append(rd / sd)
    report["durations"] = {
        "steps": rows,
        "median_ratio_wait_steps": float(np.median(ratio_wait)) if ratio_wait else None,
        "median_ratio_other_steps": float(np.median(ratio_core)),
        "n_wait_steps": len(ratio_wait), "n_other_steps": len(ratio_core),
        "median_step_sum_real_s": float(np.median(sess_real)),
        "median_session_sim_s": float(np.median(sess_sim)),
    }

    if args.write_scales:
        write_scales(report["durations"]["steps"], sorted(graphs), args.write_scales)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(report, indent=1, default=float))
    write_md(report, out.with_suffix(".md"))
    print(f"wrote {out}.md / .json")


SCALE_CLIP = (0.5, 60.0)


def write_scales(rows, recipes, path):
    """{recipe: {step: scale, "_recipe": the recipe's median scale}}, global median for recipes
    with no CC4D normal recording. Clipped to SCALE_CLIP."""
    clip = lambda x: float(min(max(x, SCALE_CLIP[0]), SCALE_CLIP[1]))  # noqa: E731
    by_r = {}
    for row in rows:
        if row.get("core_scale"):
            by_r.setdefault(row["recipe"], {})[row["step"]] = clip(row["core_scale"])
    glob = float(np.median([v for d in by_r.values() for v in d.values()]))
    scales = {}
    for r in recipes:
        d = by_r.get(r, {})
        scales[r] = dict(d, _recipe=float(np.median(list(d.values()))) if d else glob)
    Path(path).write_text(json.dumps(
        {"source": "scripts/drift_synth/calibrate_cc4d.py --write-scales: CC4D normal-recording "
                   "median step time minus the step's mean take/put/wash time, over the mean time "
                   "of its own actions and their gaps, in the uncalibrated generator (noise off, "
                   "theta 0.5, speed 1); applied to those actions and gaps only", "clip": SCALE_CLIP,
         "global_median": glob, "scales": scales}, indent=1, sort_keys=True) + "\n")
    print(f"wrote {path} (global median scale {glob:.2f})")


def write_md(rep, path):
    L = ["# Calibration against CaptainCook4D", "",
         "Real recordings: no Order Error or Missing Step tag. Distances are normalized Kendall "
         "(share of step pairs in opposite order).", "", "## 1. Orders: real vs the generator (habit 0)", "",
         "| recipe | real n | real distinct | sim distinct (same n) | real to medoid | sim to medoid | "
         "real prep-first | sim prep-first | free prep/cook pairs | real orders legal |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r, o in rep["orders"].items():
        a, b = o["real"], o["sim_uniform"]
        f = lambda x: "-" if x is None else f"{x:.2f}"  # noqa: E731
        L.append(f"| {r} | {a['n']} | {a['distinct']:.2f} | {b['distinct_at']:.2f} | {a['to_medoid']:.3f} | "
                 f"{b['to_medoid']:.3f} | {f(a['prep_first'])} | {f(b['prep_first'])} | {a['free_pairs']} | "
                 f"{o['real_valid_topological']:.2f} |")
    h = rep["habit"]
    L += ["", "## 2. Habit", "",
          f"Real: within-person {h['real_within']:.3f} ({h['n_within_pairs']} pairs), between-person "
          f"{h['real_between']:.3f} ({h['n_between_pairs']} pairs).", "",
          f"Generator today (habit 2, routines uniform): within {h['today']['sim_within']:.3f}, "
          f"between {h['today']['sim_between']:.3f}.", "",
          f"Best fit: habit {h['best']['habit']:g}, routine concentration "
          f"{h['best']['routine_concentration']:g}: within {h['best']['sim_within']:.3f}, between "
          f"{h['best']['sim_between']:.3f}.", "",
          "| habit \\ concentration | " + " | ".join(f"{c:g}" for c in CONCENTRATIONS) + " |",
          "|---|" + "---|" * len(CONCENTRATIONS)]
    for hb in HABITS:
        L.append(f"| {hb:g} | " + " | ".join(
            "{:.2f} / {:.2f}".format(*h["grid"][f"{hb:g},{c:g}"]) for c in CONCENTRATIONS) + " |")
    d = rep["durations"]
    L += ["", "(cells: within / between)", "", "## 3. Durations", "",
          f"Median real step-time sum per recording {d['median_step_sum_real_s']:.0f} s; median "
          f"synthetic session {d['median_session_sim_s']:.0f} s.", "",
          f"Median real/synthetic ratio: steps with a `wait` token {d['median_ratio_wait_steps']:.1f}x "
          f"({d['n_wait_steps']} steps), other steps {d['median_ratio_other_steps']:.1f}x "
          f"({d['n_other_steps']} steps).", "",
          "| recipe | step | real s | synthetic s | ratio | wait |", "|---|---|---|---|---|---|"]
    for row in sorted(d["steps"], key=lambda x: -(x["ratio"] or 0))[:25]:
        L.append(f"| {row['recipe']} | {row['description']} | {row['real_s']:.0f} | {row['sim_s']:.1f} | "
                 f"{row['ratio']:.1f} | {'yes' if row['wait'] else ''} |")
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
