# ruff: noqa: E402  (imports follow the sys.path setup and the variant switch)
"""(docs/drift_synth.md §9.5) Why do structural injected errors go undetected on the synthetic data?

For one person (calibrated variant, N = 250 sessions, prior lambda = 1):
  1. rebuild the exact test pool (same placement model, seed, sessions)
  2. classify each injection's target segment: idle gap / logistics (take/put/wash) / recipe action
  3. oracles from a big healthy sample of the SAME person (3000 days): would an infinite-data
     run-bigram model, a run-count check, or a run-duration check see anything?
  4. HSMM per-trial hit (alarm within [span start, span end + 10])
"""
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "scripts/eval")
import numpy as np

PID = int(sys.argv[1]) if len(sys.argv) > 1 else 0   # ./py scripts/eval/diagnose_synth_injections.py 0
N = 250

import eval_synth_prior as E
E.set_variant("cal")
from cook_ad.xla_env import disable_gpu_autotuning
disable_gpu_autotuning()
import jax
jax.config.update("jax_enable_x64", True)
from cook_ad.core import OperatingPoint
from cook_ad.data.config import load_config
from cook_ad.detectors.hsmm_hindsight import HindsightHSMMDetector
from cook_ad.detectors.thresholds import BadSets
from cook_ad.drift_synth.export import to_ticks
from cook_ad.drift_synth.sampler import ConditionCfg, generate_person
from cook_ad.eval import harness, pools
from cook_ad.hsmm import params as hparams
from cook_ad.hsmm.model import HSMM
from cook_ad.hsmm.train.global_prior import apply_global_prior
from cook_ad.hsmm.train.lexical_init import recipe_label_assignment
from cook_ad.synthetic.error_injection import _seg_bounds

pid = f"p{PID:02d}"
d = E.ROOT / "persons" / pid
labels = json.loads((d / "null" / "labels.json").read_text())
seqs = E._in_order(json.loads((d / "null" / "sequences.json").read_text()), labels)
rec_of = {e["trial_id"]: e["recipe_label"] for e in labels}
vocab = json.loads((d / "null" / "vocab.json").read_text())
iv = {i: v for v, i in vocab["verbs"].items()}
inn = {i: n for n, i in vocab["nouns"].items()}
d_max = load_config(E.CONFIG)["duration"]["d_max_ticks"]
placement = hparams.load(str(d / f"model_s{max(E.TRAIN_SESSIONS)}" / "model.npz"))
part = seqs[E.TEST_START:E.TEST_END]
entries, keep = pools.injected_per_trial(placement, part, d_max, seed=E.SEED, chunk_size=8)
recipes = [rec_of[part[k]["trial_id"]] for k in keep]
IDLE = (vocab["verbs"]["stall"], vocab["nouns"]["idle"])
LOG_VERBS = {vocab["verbs"][v] for v in ("take", "put", "wash")}


def runs(v, n):
    out = []
    for a, b in zip(v, n):
        if out and out[-1][0] == (a, b):
            out[-1][1] += 1
        else:
            out.append([(a, b), 1])
    return [(t, k) for t, k in out]


# ---- a big healthy sample of the same person --------------------------------------------------
assets = E._assets({"theta0_by_knob": E.CAPABILITY, "habit": 2.0})
_, sessions, _ = generate_person(PID, ConditionCfg("null", n_persons=1, n_days=3000), E.SEED, assets)
sessions = [s for s in sessions if s.day >= E.N_DAYS]          # days the test pool never saw
big = defaultdict(list)
for s in sessions:
    v, n = to_ticks(s.events, vocab, assets.gen["export"]["sil"], assets.gen["export"]["fps"], E.TICK)
    big[s.recipe_id].append(runs(v, n))
bigram, counts, durs = defaultdict(set), defaultdict(lambda: defaultdict(list)), defaultdict(list)
for r, rs in big.items():
    for rr in rs:
        toks = ["BOS"] + [t for t, _ in rr] + ["EOS"]
        bigram[r].update(zip(toks, toks[1:]))
        c = Counter(t for t, _ in rr if t != IDLE)
        for t in set(c) | set(counts[r]):
            counts[r][t].append(c.get(t, 0))
        for t, k in rr:
            durs[t].append(k)
lo_q = {t: np.quantile(x, 0.002) for t, x in durs.items()}
hi_q = {t: np.quantile(x, 0.998) for t, x in durs.items()}
print(f"{pid}: healthy oracle sample {sum(len(x) for x in big.values())} sessions")


def oracle(r, v, n):
    rr = runs(v, n)
    toks = ["BOS"] + [t for t, _ in rr] + ["EOS"]
    o_bi = any(b not in bigram[r] for b in zip(toks, toks[1:]))
    c = Counter(t for t, _ in rr if t != IDLE)
    o_cnt = any(not (min(counts[r].get(t, [0])) <= c.get(t, 0) <= max(counts[r].get(t, [0])))
                for t in set(c) | set(counts[r]))
    o_dur = any(t in lo_q and not (lo_q[t] <= k <= hi_q[t]) for t, k in rr)
    return o_bi, o_cnt, o_dur


def kind(tok):
    if tok == IDLE:
        return "idle gap"
    return "take/put/wash" if tok[0] in LOG_VERBS else "recipe action"


def target(h, inj, et):
    """Token(s) of the healthy segment(s) the injection hit."""
    b = _seg_bounds(h["segments"])
    v, n = np.asarray(h["verb_ids"]), np.asarray(h["noun_ids"])
    tm = np.asarray(inj["tick_map"])
    if et == "transposition":
        ticks = [int(tm[s[0]]) for s in inj["spans"]]
    elif et == "omission":
        ticks = [int(inj["window"][0])]
    else:
        ticks = [int(tm[inj["window"][0]])]
    out = []
    for t in ticks:
        seg = next(x for x in b if x[0] <= t < x[1])
        out.append((int(v[seg[0]]), int(n[seg[0]])))
    return out


# ---- HSMM at the operating point the experiment picked -----------------------------------------
res = json.loads(open(f"runs/drift_synth/eval_synth_prior_split_cal_{pid}.json").read())["results"][str(N)]["prior lambda=1"]
gcounts = json.loads((E.ROOT / "global_counts.json").read_text())
_, names = recipe_label_assignment(seqs, labels)
p = apply_global_prior(hparams.load(str(d / f"model_s{N}" / "model.npz")), gcounts, names, 1.0)
hsmm = HSMM(p, d_max)
bad = BadSets(hsmm, harness.ALPHAS)
op = OperatingPoint({c: (res["alpha_transition"] if c in harness.TRANSITION_CHANNELS else res["alpha_rest"])
                     for c in harness.CHANNELS})
det = HindsightHSMMDetector(hsmm, bad, op, chunk_size=8)
dish = recipe_label_assignment([part[k] for k in keep], labels)[0]
streams = pools.per_trial_to_streams(entries)
trials = harness.trials_from_streams(streams, dish)
dets = {}
for group in streams:
    ts = [t for g, t in trials if g == group]
    dets[group] = det.detect_many(ts)

print(f"operating point: alpha_trans {res['alpha_transition']:g}, alpha_rest {res['alpha_rest']:g}; "
      f"reported F1 {res['f1']:.3f}")
hfa = np.mean([len(x.alarms) > 0 for x in dets["healthy"]])
print(f"healthy trials with any alarm: {hfa:.2f}")
rows = []
for et in [g for g in streams if g != "healthy"]:
    for i, ((h, injs), r, dd) in enumerate(zip(entries, recipes, dets[et])):
        inj = injs[et]
        ob, oc, od = oracle(r, inj["verb_ids"], inj["noun_ids"])
        same = (len(inj["verb_ids"]) == len(h["verb_ids"]) and runs(inj["verb_ids"], inj["noun_ids"])
                == runs(h["verb_ids"], h["noun_ids"])) if et != "abandonment" else False
        hit = any(a <= x.tick <= b + 10 for x in dd.alarms for a, b in inj["spans"])
        k = kind(target(h, inj, et)[0]) if et != "transposition" else \
            "+".join(sorted(kind(t) for t in target(h, inj, et)))
        rows.append(dict(et=et, kind=k, bi=ob, cnt=oc, dur=od, any=ob or oc or od, hit=hit,
                         runs_unchanged=same))
json.dump(rows, open(f"runs/drift_synth/diagnose_injections_{pid}.json", "w"))


def show(sel, label):
    if not sel:
        return
    m = lambda k: np.mean([x[k] for x in sel])  # noqa: E731
    print(f"  {label:34s} n={len(sel):3d}  oracle: bigram {m('bi'):.2f} count {m('cnt'):.2f} "
          f"dur {m('dur'):.2f} any {m('any'):.2f} | HSMM hit {m('hit'):.2f}")


for et in dict.fromkeys(x["et"] for x in rows):
    sel = [x for x in rows if x["et"] == et]
    print(et)
    show(sel, "all")
    for k in sorted({x["kind"] for x in sel}):
        show([x for x in sel if x["kind"] == k], k)
    det_any = [x for x in sel if x["any"]]
    show(det_any, "oracle-detectable only")
