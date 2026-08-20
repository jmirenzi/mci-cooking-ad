"""Is a per-trial pace factor worth adding to the duration model?

The duration channels' power is set by how tight P(D | state, recipe) is; measured on the
current fit its coefficient of variation is ~0.63, which is why doubling a segment (what a
repetition collapses to, since banned self-transitions force Viterbi to merge the duplicated
run) clears the alpha tail only ~28% of the time.

A large share of that spread may be BETWEEN trials rather than within them: some participants
simply cook slowly. If so, the same NB family conditioned on a one-parameter per-trial pace
would be materially tighter, and every duration-driven detection would get sharper -- without
changing the model family, only what it conditions on.

This decomposes the observed log-duration residual into a between-trial part (the pace) and a
within-trial part (what is left), and reports the CV before and after removing the pace.
"""
import argparse
import json

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from cook_ad.data import split as split_mod
from cook_ad.data.config import load_config
from cook_ad.hsmm import joint_params
from cook_ad.synthetic import generate

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--joint-params", default="runs/joint_lex0.npz")
ap.add_argument("--split-part", default="train")
a = ap.parse_args()

d_max = load_config("configs/breakfast.yaml")["duration"]["d_max_ticks"]
root = "dataset/processed/breakfast"
seqs = json.load(open(f"{root}/sequences.json"))
seqs = split_mod.filter_sequences(seqs, split_mod.load_split(f"{root}/split.json"), a.split_part)
jp = joint_params.load_params(a.joint_params)
traj = generate.trajectories_from_real_joint(jp, seqs, d_max, chunk_size=32)

# per-state (pooled over recipes) empirical mean log-duration, then per-trial offset
segs = []
for i, t in enumerate(traj):
    for k, d in t["segments"][:-1]:
        segs.append((i, int(k), float(d)))
segs = np.array(segs)
trial_id = segs[:, 0].astype(int)
state = segs[:, 1].astype(int)
logd = np.log(segs[:, 2])

K = int(state.max()) + 1
for _ in range(30):  # alternating least squares on logd ~ mu[state] + pace[trial]
    mu = np.zeros(K)
    pace = np.zeros(len(traj)) if _ == 0 else pace
    for k in range(K):
        m = state == k
        if m.any():
            mu[k] = np.mean(logd[m] - pace[trial_id[m]])
    pace = np.zeros(len(traj))
    for i in range(len(traj)):
        m = trial_id == i
        if m.any():
            pace[i] = np.mean(logd[m] - mu[state[m]])
    pace -= pace.mean()

resid_no_pace = logd - mu[state]
resid_pace = logd - mu[state] - pace[trial_id]
print(f"{len(segs)} segments, {len(traj)} trials, {K} states")
print(f"log-duration residual sd: state-only = {resid_no_pace.std():.3f}, "
      f"state+trial pace = {resid_pace.std():.3f}  "
      f"(variance explained by pace: {1 - resid_pace.var()/resid_no_pace.var():.3f})")
print(f"per-trial pace sd = {pace.std():.3f} log-units "
      f"(x{np.exp(pace.std()):.2f} spread between a 1-sd fast and average trial)")
# lognormal CV from residual sd
for name, r in (("state only", resid_no_pace), ("state + pace", resid_pace)):
    cv = np.sqrt(np.exp(r.var()) - 1)
    print(f"  implied duration CV, {name}: {cv:.3f}")
# what a 2x segment looks like in residual units
print(f"\na repetition doubles a duration: log(2) = {np.log(2):.3f}")
print(f"  z-score of that under state-only spread : {np.log(2)/resid_no_pace.std():.2f}")
print(f"  z-score under state + pace              : {np.log(2)/resid_pace.std():.2f}")
