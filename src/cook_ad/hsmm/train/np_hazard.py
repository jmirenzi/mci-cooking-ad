"""A nonparametric duration hazard: training stage 4b (docs/training.md §4b).

h_ij = P(D = j | D >= j, z = i) estimated directly from decoded segment lengths, as a discrete
Kaplan-Meier hazard: of the segments of i still running at j ticks, the fraction that ended at j.
The censored last segment of a trial is at risk up to its length and never an event, so it needs
no imputation. Thin cells borrow strength in two steps:

    h_pool(k, j) = (E_kj + kappa0 * h_nb(r, k, j)) / (R_kj + kappa0)     pooled over recipes
    h(r, k, j)   = (e_rkj + kappa * h_pool) / (n_rkj + kappa)             the cell

e = events, n = at risk, E / R their sums over recipes, h_nb the model's negative-binomial hazard
(the fallback where nothing is observed). `bandwidth` > 0 first smooths events and at-risk counts
over j with a triangular kernel of that half-width (in ticks), so the hazard is not spiky at the
lengths that happened to occur. A cell with fewer than `min_segments` training segments keeps
its negative-binomial hazard outright: on few segments the empirical shape is noise, and at cold
start that cost F1 (next_steps.md §3). Returns (log_pmf, log_surv) in the layout of
hsmm/durations.duration_tables: the last column of log_pmf holds P(D >= D_max).
"""
import numpy as np

from cook_ad.hsmm.train.refit import duration_histograms

H_CLIP = 1e-9


def _smooth(x, bandwidth):
    if bandwidth <= 0:
        return x
    w = np.concatenate([np.arange(1, bandwidth + 2), np.arange(bandwidth, 0, -1)]).astype(float)
    w /= w.sum()
    pad = [(0, 0)] * (x.ndim - 1) + [(bandwidth, bandwidth)]
    xp = np.pad(x, pad, mode="edge")
    return np.apply_along_axis(lambda v: np.convolve(v, w, mode="valid"), -1, xp)


def at_risk(xi, cens):
    """n[..., j] = segments (closed or censored) still running at j + 1 ticks."""
    tot = xi + cens
    return np.flip(np.cumsum(np.flip(tot, -1), -1), -1)


def tables_from_hazard(h):
    """(..., D) hazard (last column forced to 1) -> (log_pmf, log_surv)."""
    h = np.clip(np.asarray(h, dtype=float), H_CLIP, 1.0 - H_CLIP)
    h[..., -1] = 1.0
    log1m = np.log1p(-np.minimum(h, 1.0 - H_CLIP))
    log_surv = np.concatenate([np.zeros(h.shape[:-1] + (1,)), np.cumsum(log1m, -1)[..., :-1]], -1)
    log_pmf = log_surv + np.log(h)
    log_pmf[..., -1] = log_surv[..., -1]
    return log_pmf, log_surv


def fit_np_hazard(trajectories, k_recipe, k_subtask, d_max, log_h_nb, kappa=5.0, kappa0=2.0,
                  bandwidth=0, min_segments=0, nb_tables=None):
    """trajectories: [{"recipe_id", "segments": [(state, d)]}] (decoded training sessions);
    log_h_nb: (K_R, K, D) the model's NB log hazard (hsmm.tables.log_h); nb_tables: its exact
    (log_pmf, log_surv), copied into the cells the gate keeps on the NB (rebuilding them from the
    hazard would lose the NB's underflowed, exactly-zero tail)."""
    xi, cens = duration_histograms(trajectories, k_recipe, k_subtask, d_max)
    ev, risk = _smooth(xi, bandwidth), _smooth(at_risk(xi, cens), bandwidth)
    h_nb = np.exp(np.asarray(log_h_nb, dtype=float))
    h_pool = (ev.sum(0)[None] + kappa0 * h_nb) / (risk.sum(0)[None] + kappa0)
    h = (ev + kappa * h_pool) / (risk + kappa)
    thin = at_risk(xi, cens)[..., 0] < min_segments          # segments observed per cell
    h = np.where(thin[..., None], h_nb, h)
    log_pmf, log_surv = tables_from_hazard(h)
    if nb_tables is not None:
        log_pmf = np.where(thin[..., None], np.asarray(nb_tables[0], dtype=float), log_pmf)
        log_surv = np.where(thin[..., None], np.asarray(nb_tables[1], dtype=float), log_surv)
    return log_pmf, log_surv


def np_hsmm(params, d_max, sequences, recipes=None, kappa=5.0, kappa0=2.0, bandwidth=1, chunk_size=16,
            min_segments=0):
    """An HSMM whose duration tables are the nonparametric hazard fitted on `params`' own Viterbi
    decode of `sequences` (the training split), under `recipes` (given dish per sequence) or the
    inferred one."""
    from cook_ad.hsmm.model import HSMM
    from cook_ad.synthetic import generate
    base = HSMM(params, d_max)
    traj = generate.trajectories_from_real_joint(params, sequences, d_max, chunk_size=chunk_size,
                                                 r_hat=recipes)
    k_r, k = np.asarray(params.init_counts).shape
    return HSMM(params, d_max, duration_tables=fit_np_hazard(
        traj, k_r, k, d_max, np.asarray(base.tables.log_h), kappa=kappa, kappa0=kappa0,
        bandwidth=int(bandwidth), min_segments=min_segments,
        nb_tables=(base.tables.log_pmf, base.tables.log_surv)))


def save_tables(path, log_pmf, log_surv, meta=None):
    """The stage-4b sidecar, `<model>.durations.npz` (hsmm/model.py:durations_path)."""
    np.savez(path, log_pmf=np.asarray(log_pmf), log_surv=np.asarray(log_surv),
             meta=np.asarray(__import__("json").dumps(meta or {})))


def load_tables(path):
    z = np.load(path)
    return z["log_pmf"], z["log_surv"]
