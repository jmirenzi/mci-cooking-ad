"""Where the transition channel loses: usage-weighted transition entropy, and the actual
surprise / threshold values at real vs perturbed junctions."""
import json, itertools, argparse
import numpy as np, jax.numpy as jnp, jax
jax.config.update("jax_enable_x64", True)
from cook_ad.hsmm import joint_params, joint_em, em
from cook_ad.recipe import segmentize
from cook_ad.anomaly import quantile
from cook_ad.data import split as split_mod
from cook_ad.data.config import load_config

ap = argparse.ArgumentParser()
ap.add_argument("--joint-params", default="dataset/processed/breakfast/joint_params_train.npz")
ap.add_argument("--part", default="train")
a = ap.parse_args()
d_max = load_config("configs/breakfast.yaml")["duration"]["d_max_ticks"]
root = "dataset/processed/breakfast"
seqs = json.load(open(f"{root}/sequences.json"))
split = split_mod.load_split(f"{root}/split.json")
seqs = split_mod.filter_sequences(seqs, split, a.part)

jp = joint_params.load_params(a.joint_params)
lp = joint_params.to_log_probs_joint(jp, d_max)
K = jp.verb_counts.shape[0]; KR = jp.pi_counts.shape[0]
v_ids, n_ids, mask = em.pad_batch(seqs)
r_hat, _, _ = joint_em.infer_recipe(jp, v_ids, n_ids, mask, d_max, chunk_size=8)
r_hat = np.asarray(r_hat)
segres = segmentize.segment_all_conditioned(lp, jnp.asarray(r_hat), v_ids, n_ids, mask, d_max)
lt = np.asarray(lp.log_trans)  # (KR,K,K)

# usage-weighted transition row entropy: only rows actually traversed
rows, surprises = [], []
for i, sr in enumerate(segres):
    st = [s for s, _ in sr["segments"]]
    r = int(r_hat[i])
    for u, v in zip(st[:-1], st[1:]):
        rows.append((r, int(u)))
        surprises.append(-lt[r, int(u), int(v)])
rows = np.array(rows); surprises = np.array(surprises)
A = np.exp(lt)
with np.errstate(invalid="ignore"):
    ent = -np.nansum(np.where(A > 1e-300, A * lt, 0.0), axis=-1)
used_ent = np.array([ent[r, u] for r, u in rows])
print(f"traversed transitions: n={len(rows)}")
print(f"usage-weighted row entropy: mean={used_ent.mean():.3f} nats -> effective out-degree exp(H) mean={np.exp(used_ent).mean():.1f}, median={np.exp(np.median(used_ent)):.1f}")
print(f"  quantiles of exp(H): {np.round(np.exp(np.quantile(used_ent,[.1,.25,.5,.75,.9])),1)}")
print(f"observed transition surprise -logP: mean={surprises.mean():.3f} median={np.median(surprises):.3f} q90={np.quantile(surprises,.9):.3f}")

# what does an "impossible" transition cost?  sample random (u,v) not observed
rng = np.random.default_rng(0)
obs = set()
for i, sr in enumerate(segres):
    st = [s for s, _ in sr["segments"]]; r = int(r_hat[i])
    for u, v in zip(st[:-1], st[1:]): obs.add((r, int(u), int(v)))
rand_s = []
for _ in range(20000):
    r, u = rows[rng.integers(len(rows))]
    v = int(rng.integers(K))
    if v == u or (r, u, v) in obs: continue
    rand_s.append(-lt[r, u, v])
rand_s = np.array(rand_s)
print(f"unobserved-transition surprise: mean={rand_s.mean():.3f} median={np.median(rand_s):.3f} q10={np.quantile(rand_s,.1):.3f}")

for alpha in (0.05, 5e-3, 1e-4):
    tab = np.asarray(quantile.transition_quantile_threshold(jnp.asarray(lt[0]), alpha))  # per-recipe API takes (K,K)
    ths = []
    for r, u in rows:
        t = np.asarray(quantile.transition_quantile_threshold(jnp.asarray(lt[r]), alpha))
        ths.append(t[u])
    ths = np.array(ths)
    print(f"alpha={alpha:.0e}: threshold mean={ths.mean():.3f} | observed exceed rate={np.mean(surprises>ths):.4f}")
    break
