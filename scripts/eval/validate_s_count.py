# ruff: noqa: E402  (imports follow the sys.path setup and the variant switch)
"""Validate the s_count channel on the scrubbed synthetic persons (docs/detectors.md §3.6a, §5.6).

For each person's N = 250 model (prior lambda = 1, stage-4b durations), visit counts are fitted on
its own training decodes. Then, on the held-out TEST sessions and the step-level pools:

  1. calibration: the fraction of healthy visits / missing-state checks with p <= alpha, against
     alpha (smoothed frequencies should give about alpha or less);
  2. how often s_count ALONE alarms on a healthy session, hindsight (p <= alpha) and live
     (bad mass > kappa = 0.5), per alpha;
  3. per error type, the spans s_count catches (alarm in [start, end + 10]), and the spans it
     catches that no other channel does at a strict reference operating point (alpha 1e-5, the
     neighbourhood validation picks for them): its unique contribution.

    ./py scripts/eval/validate_s_count.py [--person 0 ...] [--sessions 250]
"""
import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "scripts/eval")
import numpy as np

import eval_synth_prior as E

E.set_variant(os.environ.get("SYNTH_VARIANT", "sess"))
E.SCRUB = True
from cook_ad.xla_env import disable_gpu_autotuning

disable_gpu_autotuning()
import jax

jax.config.update("jax_enable_x64", True)
from cook_ad.core import OperatingPoint, Trial
from cook_ad.data.config import load_config
from cook_ad.detectors.hsmm_hindsight import CHANNELS, COUNT, HindsightHSMMDetector
from cook_ad.detectors.hsmm_live import LiveHSMMDetector
from cook_ad.detectors.thresholds import BadSets
from cook_ad.eval import harness, metric_g
from cook_ad.eval.synth_oracle import person_support
from cook_ad.hsmm import params as hparams
from cook_ad.hsmm.model import HSMM, durations_path
from cook_ad.hsmm.train.global_prior import apply_global_prior
from cook_ad.hsmm.train.lexical_init import recipe_label_assignment
from cook_ad.hsmm.train.np_hazard import load_tables
from cook_ad.hsmm.visit_counts import VisitCounts

ALPHAS = [0.1, 0.05, 0.02, 0.01, 0.005]
REF = 1e-5                     # the other channels' reference alpha
GRID = sorted(set(harness.ALPHAS))


def hits(span, ticks, deadline=metric_g.DEADLINE):
    return any(span[0] <= t <= span[1] + deadline for t in ticks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--person", type=int, action="append")
    ap.add_argument("--sessions", type=int, default=250)
    ap.add_argument("--out", default="runs/drift_synth/validate_s_count.json")
    a = ap.parse_args()
    d_max = load_config(E.CONFIG)["duration"]["d_max_ticks"]
    gc = json.loads((E.ROOT / "global_counts.json").read_text())
    report = {}
    for i in a.person or list(E.PERSONS):
        pid = f"p{i:02d}"
        d = E.ROOT / "persons" / pid
        labels = json.loads((d / "null" / "labels.json").read_text())
        seqs = E._in_order(json.loads((d / "null" / "sequences.json").read_text()), labels)
        _, names = recipe_label_assignment(seqs, labels)
        model = d / f"model_s{a.sessions}" / "model.npz"
        p = apply_global_prior(hparams.load(str(model)), gc, names, 1.0)
        dp = durations_path(model)
        hsmm = HSMM(p, d_max, duration_tables=load_tables(dp)) if os.path.exists(dp) else HSMM(p, d_max)
        bad = BadSets(hsmm, GRID)
        base = HindsightHSMMDetector(hsmm, bad, OperatingPoint(0.01), chunk_size=8)
        train = seqs[:a.sessions]
        tr_dish = recipe_label_assignment(train, labels)[0]
        tr = base.traces([Trial(s["trial_id"], np.asarray(s["verb_ids"]), np.asarray(s["noun_ids"]),
                                int(r), (), "healthy") for s, r in zip(train, tr_dish)])
        vc = VisitCounts.fit([(t.r_hat, [k for k, _ in t.segments]) for t in tr])

        vocab = json.loads((d / "null" / "vocab.json").read_text())
        sup = person_support(i, E.SEED, E._assets({"theta0_by_knob": E.CAPABILITY, "habit": 2.0}), vocab,
                             E.N_SESSIONS, E.ORACLE_SESSIONS, E.TICK, cache=d / f"oracle_support_v{E.ORACLE_VERSION}.pkl")
        trials = E.gen_pool(i, seqs[E.TEST_START:E.TEST_END], labels, sup)

        hind = HindsightHSMMDetector(hsmm, bad, OperatingPoint(0.01), chunk_size=8, counts=vc)
        groups = defaultdict(list)
        for g, t in trials:
            groups[g].append(t)
        traces = {g: hind.traces(ts) for g, ts in groups.items()}

        # 1. calibration of the hindsight p-values on healthy test sessions
        pv = np.array([p_ for tr_ in traces["healthy"] for _, p_ in vc.path_pvalues(int(tr_.r_hat), tr_.segments)])
        calib = {str(al): float(np.mean(pv <= al)) for al in ALPHAS}

        # 2 + 3, hindsight; the other channels at the strict reference
        opr = OperatingPoint({c: REF for c in CHANNELS} | {COUNT: 0.0})
        ref = {g: [np.flatnonzero(np.any(np.stack([f[c] for c in CHANNELS]), axis=0)).tolist()
                   for f in (hind.flags(t_, opr) for t_ in trs)] for g, trs in traces.items()}
        res = {"n_pvalues": int(len(pv)), "calibration": calib, "hindsight": {}, "live": {}}
        for al in ALPHAS:
            op = OperatingPoint({c: al for c in CHANNELS} | {COUNT: al})
            row = {}
            for g, trs in traces.items():
                fl = [hind.flags(tr_, op) for tr_ in trs]
                cnt = [np.flatnonzero(f[COUNT]).tolist() for f in fl]
                oth = ref[g]
                if g == "healthy":
                    row["healthy_sessions_with_s_count"] = float(np.mean([bool(c) for c in cnt]))
                    row["healthy_sessions_with_others"] = float(np.mean([bool(o) for o in oth]))
                    continue
                sp = [(s, c, o) for t, c, o in zip(groups[g], cnt, oth) for s in t.spans]
                row[g] = {"s_count": float(np.mean([hits(s, c) for s, c, _ in sp])),
                          "others": float(np.mean([hits(s, o) for s, _, o in sp])),
                          "s_count_only": float(np.mean([hits(s, c) and not hits(s, o) for s, c, o in sp]))}
            res["hindsight"][str(al)] = row

        # 2 + 3, live (L = 0, kappa = 0.5)
        live = LiveHSMMDetector(hsmm, bad, OperatingPoint(0.01, 0.5), counts=vc)
        ltr = {g: [live.trace(t) for t in ts] for g, ts in groups.items()}
        oplr = OperatingPoint({c: REF for c in CHANNELS} | {COUNT: 0.0}, 0.5)
        ref = {g: [np.flatnonzero(np.any(np.stack([f[c] for c in CHANNELS]), axis=0)).tolist()
                   for f in (live.flags(t_, oplr) for t_ in trs)] for g, trs in ltr.items()}
        for al in ALPHAS:
            op = OperatingPoint({c: al for c in CHANNELS} | {COUNT: al}, 0.5)
            row = {}
            for g, trs in ltr.items():
                fl = [live.flags(tr_, op) for tr_ in trs]
                cnt = [np.flatnonzero(f[COUNT]).tolist() for f in fl]
                oth = ref[g]
                if g == "healthy":
                    row["healthy_sessions_with_s_count"] = float(np.mean([bool(c) for c in cnt]))
                    row["healthy_sessions_with_others"] = float(np.mean([bool(o) for o in oth]))
                    continue
                sp = [(s, c, o) for t, c, o in zip(groups[g], cnt, oth) for s in t.spans]
                row[g] = {"s_count": float(np.mean([hits(s, c) for s, c, _ in sp])),
                          "others": float(np.mean([hits(s, o) for s, _, o in sp])),
                          "s_count_only": float(np.mean([hits(s, c) and not hits(s, o) for s, c, o in sp]))}
            res["live"][str(al)] = row
        report[pid] = res
        print(f"{pid}: {len(pv)} healthy p-values; calibration P(p <= a): "
              + ", ".join(f"{k}: {v:.3f}" for k, v in calib.items()), flush=True)
        for mode in ("hindsight", "live"):
            for al in ("0.05", "0.01"):
                r = res[mode][al]
                print(f"  {mode:9s} a={al:5s} healthy sessions with s_count {r['healthy_sessions_with_s_count']:.2f} "
                      f"(others {r['healthy_sessions_with_others']:.2f}) | catches s_count/others/only: "
                      + " ".join(f"{g[:5]} {r[g]['s_count']:.2f}/{r[g]['others']:.2f}/{r[g]['s_count_only']:.2f}"
                                 for g in r if isinstance(r[g], dict)), flush=True)
    harness.dump(a.out, report)


if __name__ == "__main__":
    main()
