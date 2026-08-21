"""Does the Viterbi decode launder injected structural anomalies?

`tools_transition_ceiling.py` counts the junctions an injection creates that the data never
contains. This follows those through the decode to the flag, localising the loss:

    created & novel  ->  what the data makes flaggable at all
    still in decode  ->  what Viterbi did not re-explain through another state path
    flagged          ->  what the threshold then admitted
"""
import argparse
import json
from collections import defaultdict

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from cook_ad.anomaly import quantile, surprise
from cook_ad.data import split as split_mod
from cook_ad.data.config import load_config
from cook_ad.eval import batch
from cook_ad.hsmm import joint_params
from cook_ad.synthetic import error_injection, generate

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--joint-params", default="dataset/processed/breakfast/joint_params_train.npz")
ap.add_argument("--split-part", default="train")
ap.add_argument("--alpha", type=float, default=2e-2)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

d_max = load_config("configs/breakfast.yaml")["duration"]["d_max_ticks"]
root = "dataset/processed/breakfast"
seqs = json.load(open(f"{root}/sequences.json"))
seqs = split_mod.filter_sequences(seqs, split_mod.load_split(f"{root}/split.json"), a.split_part)
jp = joint_params.load_params(a.joint_params)
marg = joint_params.collapse_to_marginal(jp)
traj = generate.trajectories_from_real_joint(jp, seqs, d_max, chunk_size=32)
usable = [t for t in traj if len(t["segments"]) >= error_injection.MIN_SEGMENTS]

global_bigrams, recipe_bigrams = defaultdict(int), defaultdict(int)
for t in usable:
    st = [s for s, _ in t["segments"]]
    for u, v in zip(st[:-1], st[1:]):
        global_bigrams[(u, v)] += 1
        recipe_bigrams[(t["recipe_id"], u, v)] += 1

rng = np.random.default_rng(a.seed)
for et in ("omission", "transposition", "repetition", "abandonment"):
    deg = [error_injection.inject(et, t, rng, marg) for t in usable]
    traces, lp, r_hat, ltm = batch.compute_traces_joint(jp, deg, d_max, chunk_size=32)
    cache = quantile.JointThresholdCache(lp, ltm)

    n_trial = novel_trial = survive_trial = flag_trial = 0
    recipe_flip = 0
    for t, d, tr, rh in zip(usable, deg, traces, r_hat):
        n_trial += 1
        recipe_flip += int(rh) != t["recipe_id"]
        seg_of_tick = np.concatenate([np.full(dd, s) for s, dd in t["segments"]])
        mapped = seg_of_tick[d["tick_map"]]
        runs = [mapped[0]]
        for x in mapped[1:]:
            if x != runs[-1]:
                runs.append(x)
        healthy_pairs = set(zip([s for s, _ in t["segments"]][:-1], [s for s, _ in t["segments"]][1:]))
        created = [(u, v) for u, v in zip(runs[:-1], runs[1:]) if (u, v) not in healthy_pairs]
        novel = [(u, v) for u, v in created if (u, v) not in global_bigrams]
        if not novel:
            continue
        novel_trial += 1

        # what the decode actually produced, as (from_state, to_state) at segment starts
        decoded = set()
        segs = []
        z = tr.z_star
        pos = 0
        while pos < len(z):
            k = z[pos]
            end = pos
            while end < len(z) and z[end] == k:
                end += 1
            segs.append(int(k))
            pos = end
        decoded = set(zip(segs[:-1], segs[1:]))
        if not any(p in decoded for p in novel):
            continue
        survive_trial += 1

        flags = surprise.flag_joint_cached(cache, tr, lp, int(rh), ltm, alpha=a.alpha)
        fs = np.asarray(tr.from_state)
        fired = False
        boundaries = np.flatnonzero(fs != -1)
        for b in boundaries:
            if (int(fs[b]), int(z[b])) in novel and (flags["s_transition"][b] or flags["s_recipe_transition"][b]):
                fired = True
                break
        flag_trial += fired

    print(f"\n{et} (n={n_trial}, alpha={a.alpha:.0e})")
    print(f"  trials with a never-seen created junction : {novel_trial/n_trial:.3f}")
    print(f"  ...that junction survives into the decode : {survive_trial/n_trial:.3f} "
          f"({survive_trial/max(1,novel_trial):.3f} of them)")
    print(f"  ...and the transition channel fires there : {flag_trial/n_trial:.3f} "
          f"({flag_trial/max(1,survive_trial):.3f} of survivors)")
    print(f"  MAP recipe flipped vs healthy decode      : {recipe_flip/n_trial:.3f}")
