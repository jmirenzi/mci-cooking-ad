"""What the fitted duration model can deliver, without running the detector.

Re-scores every healthy segment's `s_dur_two` at three durations: its own d (any flag is a false
alarm), 2d (what a repetition collapses to, since banned self-transitions force Viterbi to merge
the duplicated run) and 1 (what an abandonment truncates to). Predicts THAT CHANNEL's recall, not
the error type's -- `s_temporal` is not modelled here and carries most of repetition.
"""
import argparse
import json

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from cook_ad.data import split as split_mod
from cook_ad.data.config import load_config
from cook_ad.hsmm import durations, joint_em, joint_params
from cook_ad.recipe import segmentize
from cook_ad.synthetic import generate

LOG2 = float(np.log(2.0))


def s_dur_two(d, r, p):
    ls = durations.nb_log_survival_np(d, r, p)
    lc = durations.nb_log_cdf_np(d, r, p)
    return np.maximum(0.0, -(LOG2 + np.minimum(ls, lc)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--joint-params", required=True)
    ap.add_argument("--split-part", default="train")
    ap.add_argument("--config", default="configs/breakfast.yaml")
    a = ap.parse_args()

    d_max = load_config(a.config)["duration"]["d_max_ticks"]
    root = "dataset/processed/breakfast"
    seqs = json.load(open(f"{root}/sequences.json"))
    seqs = split_mod.filter_sequences(seqs, split_mod.load_split(f"{root}/split.json"), a.split_part)

    jp = joint_params.load_params(a.joint_params)
    traj = generate.trajectories_from_real_joint(jp, seqs, d_max, chunk_size=32)
    dur_r = np.asarray(jp.dur_r)
    dur_p = np.asarray(jp.dur_p)

    rows = []
    for t in traj:
        r = t["recipe_id"]
        for i, (k, d) in enumerate(t["segments"][:-1]):  # last is right-censored, never scored
            rows.append((r, k, d))
    rows = np.array(rows)
    rr = dur_r[rows[:, 0], rows[:, 1]]
    pp = dur_p[rows[:, 0], rows[:, 1]]
    d = rows[:, 2].astype(float)

    print(f"{len(rows)} scored healthy segments; mean d={d.mean():.1f}")
    fitted_mean = 1.0 + rr * (1 - pp) / pp
    fitted_sd = np.sqrt(rr * (1 - pp) / pp**2)
    print(f"fitted NB mean: median={np.median(fitted_mean):.1f}  fitted sd: median={np.median(fitted_sd):.1f}  "
          f"median CV={np.median(fitted_sd / fitted_mean):.2f}")
    print(f"empirical: per-segment |d - fitted_mean| median={np.median(np.abs(d - fitted_mean)):.1f}")
    print(f"median dispersion r: {np.median(rr):.2f}  (r -> inf is Poisson, r -> 0 is very over-dispersed)")

    healthy = s_dur_two(d, rr, pp)
    doubled = s_dur_two(2.0 * d, rr, pp)
    truncated = s_dur_two(np.ones_like(d), rr, pp)
    print(f"\n{'alpha':>8} {'healthy':>9} {'2x (repetition)':>16} {'1 tick (abandon)':>17}")
    for alpha in (0.05, 0.02, 0.01, 5e-3, 1e-3, 1e-4):
        th = -np.log(alpha)
        print(f"{alpha:8.0e} {np.mean(healthy > th):9.3f} {np.mean(doubled > th):16.3f} "
              f"{np.mean(truncated > th):17.3f}")

    # PIT: for a well-fit duration model the healthy PIT values are ~Uniform[0,1]
    from cook_ad.anomaly import temporal
    pits = []
    for t in traj:
        segs = t["segments"][:-1]
        if not segs:
            continue
        r = t["recipe_id"]
        pits.append(temporal.pit_coordinate(segs, dur_r[r], dur_p[r]))
    pit = np.concatenate(pits)
    print(f"\nPIT mean={pit.mean():.3f} (0.5 if calibrated); "
          f"deciles={np.round(np.histogram(pit, bins=10, range=(0, 1))[0] / len(pit), 3)}")


if __name__ == "__main__":
    main()
