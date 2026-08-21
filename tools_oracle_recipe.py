"""How much recall is lost to the MAP recipe being re-inferred from the degraded stream?

On a transposition the assignment moves to a different cluster 40-60% of the time -- and the
cluster it moves to is, by selection, one whose transition matrix finds the new ordering
ordinary. Re-scores every degraded trial with r_hat pinned to the healthy decode's value. An
oracle, not a detector.
"""
import argparse
import json

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from cook_ad.anomaly import narrate, quantile, surprise
from cook_ad.data import split as split_mod
from cook_ad.data.config import load_config
from cook_ad.eval import batch
from cook_ad.hsmm import joint_params
from cook_ad.llm import textify
from cook_ad.synthetic import error_injection, generate

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--joint-params", default="runs/joint_la_s05t1.npz")
ap.add_argument("--split-part", default="train")
ap.add_argument("--alpha", type=float, default=2e-2)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

d_max = load_config("configs/breakfast.yaml")["duration"]["d_max_ticks"]
root = "dataset/processed/breakfast"
vocab = json.load(open(f"{root}/vocab.json"))
seqs = json.load(open(f"{root}/sequences.json"))
seqs = split_mod.filter_sequences(seqs, split_mod.load_split(f"{root}/split.json"), a.split_part)
jp = joint_params.load_params(a.joint_params)
marg = joint_params.collapse_to_marginal(jp)
lex = narrate.Lexicon(vocab, marg)
traj = generate.trajectories_from_real_joint(jp, seqs, d_max, chunk_size=32)
usable = [t for t in traj if len(t["segments"]) >= error_injection.MIN_SEGMENTS]
healthy_r = np.array([t["recipe_id"] for t in usable])

rng = np.random.default_rng(a.seed)
print(f"{len(usable)} trials, alpha={a.alpha:.0e}\n")
print(f"{'type':14s} {'recall (inferred r)':>20s} {'recall (pinned r)':>19s} {'flip rate':>10s}")
for et in error_injection.ERROR_TYPES:
    deg = [error_injection.inject(et, t, rng, marg) for t in usable]

    def score(r_override):
        tr, lp, rh, ltm = batch.compute_traces_joint(jp, deg, d_max, chunk_size=32, r_hat=r_override)
        cache = quantile.JointThresholdCache(lp, ltm)
        hits = 0
        for t, d, rr in zip(tr, deg, rh):
            steps = textify.steps_from_ids(d["verb_ids"], d["noun_ids"], lex)
            gt = textify.gt_steps_for_window(steps, d["window"])
            debris = textify.injection_touched_steps(steps, d["tick_map"], d["edited_ticks"], gt)
            pos = np.zeros(len(d["verb_ids"]), dtype=bool)
            for si in set(gt) | debris:
                pos[steps[si].tick_start : steps[si].tick_end] = True
            f = surprise.flag_joint_cached(cache, t, lp, int(rr), ltm, alpha=a.alpha)
            m = np.zeros(len(pos), dtype=bool)
            for c in surprise.CHANNELS:
                m |= np.asarray(f[c])
            hits += bool((pos & m).any())
        return hits / len(deg), np.asarray(rh)

    r_inferred, rh = score(None)
    r_pinned, _ = score(healthy_r)
    print(f"{et:14s} {r_inferred:20.3f} {r_pinned:19.3f} {np.mean(rh != healthy_r):10.3f}")
