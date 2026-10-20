# ruff: noqa: E402  (imports follow the sys.path setup and the variant switch)
"""Which s_count rule, if any? Every candidate on the scrubbed synthetic persons (metric G and per session).

One pass per trial computes the hindsight decode and the live entry posterior m_t(r, i); every rule
reads those same inputs, so the variants differ only in the rule. Hindsight is the live code on
the decoded path: entries one-hot at the segment starts, the decoded recipe certain.

Rules (each "+" adds its alarms; the extra-visit half is the same everywhere):
  extra     c-th visit of k with P(C_k >= c | r) <= alpha (c >= 2)
  prec      + at j's first visit, some k not yet with P(k not before j | r, j) <= alpha
            (the committed rule; fires again at every later j that k should precede)
  prec1     + the same evidence as a state: k is missing once min_{j touched} P(k not before j)
            <= alpha, flagged once per k
  rank      + k untouched after m distinct subtasks with P(rank_k >= m | r) <= alpha, once per k
            (progress only: how far into the recipe, not which subtasks)
  set<l>    + k untouched given the touched set S: a ridge-logistic P(k untouched | r, S, m / M_r)
            per (recipe, k), fitted on training snapshots (one per new subtask), lambda = l,
            floored at 0.5 / (N_r + 1); once per k
  +end      + at the last tick, k untouched with P(C_k = 0 | r) <= alpha
"Touched" is the expected visit count c_k >= 1/2 (live) or a decoded visit (hindsight). Each rule
gets its own alpha axis (level 0 = off) next to the other channels' split (hindsight) or
alpha x kappa (live) grid; everything is picked on validation by G-F1 and reported on test. Per
session: the fraction of sessions with any s_count alarm, by group, at each level.

    ./py scripts/eval/compare_s_count.py [--person 0 ...] [--sessions 30 --sessions 250]
"""
import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "scripts/eval")
import numpy as np

import eval_synth_prior as E

E.set_variant("cal_scrub")
E.SCRUB = True
from cook_ad.xla_env import disable_gpu_autotuning

disable_gpu_autotuning()
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
from cook_ad.core import OperatingPoint, Trial
from cook_ad.data.config import load_config
from cook_ad.detectors.hsmm_hindsight import HindsightHSMMDetector
from cook_ad.detectors.hsmm_live import EVENTS, LiveHSMMDetector, entry_table, event_table
from cook_ad.detectors.thresholds import BadSets
from cook_ad.eval import harness, metric_g
from cook_ad.eval.synth_oracle import person_support
from cook_ad.hsmm import params as hparams
from cook_ad.hsmm.model import HSMM, durations_path
from cook_ad.hsmm.train.global_prior import apply_global_prior
from cook_ad.hsmm.train.lexical_init import recipe_label_assignment
from cook_ad.hsmm.train.np_hazard import load_tables

LEVELS = (0.0, 0.002, 0.005, 0.01, 0.02, 0.05)          # s_count alpha axis; 0 = off
VARIANTS = ("extra", "prec", "prec_noextra", "prec1", "rank", "set1", "set10",
            "extra+end", "prec+end", "rank+end")
ALPHAS, KAPPAS = harness.ALPHAS, harness.KAPPAS


# ---- statistics from the training decodes ---------------------------------------------------


def _logistic(X, y, lam, iters=30):
    """Ridge logistic regression by Newton; the intercept is not penalised."""
    X1 = np.hstack([X, np.ones((len(X), 1))])
    w = np.zeros(X1.shape[1])
    reg = lam * np.eye(X1.shape[1])
    reg[-1, -1] = 1e-6
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(X1 @ w, -30, 30)))
        g = X1.T @ (p - y) + reg @ w
        H = (X1 * (p * (1 - p))[:, None]).T @ X1 + reg
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-7:
            break
    return w


class Stats:
    """Per recipe: N, visit tails, precedence, rank tails, P(C_k = 0), and the touched-set logits."""

    def __init__(self, decodes, k, ignore, lams=(1.0, 10.0)):
        by = defaultdict(list)
        for r, ks in decodes:
            by[int(r)].append([int(x) for x in ks if int(x) not in ignore])
        self.k, self.ignore = k, set(ignore)
        self.r = {}
        for r, sess in by.items():
            n = len(sess)
            floor = 0.5 / (n + 1)
            counts = np.array([np.bincount(s, minlength=k) for s in sess])          # (n, K)
            vmax = int(counts.max()) + 1
            ge = np.array([[(np.sum(counts[:, kk] >= v) + 0.5) / (n + 1) for v in range(vmax + 2)]
                           for kk in range(k)])                                      # ge[k, v]
            ge[:, 0] = 1.0
            p0 = (np.sum(counts == 0, axis=0) + 0.5) / (n + 1)
            first = [list(dict.fromkeys(s)) for s in sess]
            n_with = np.zeros(k)
            before = np.zeros((k, k))
            for f in first:
                for pos, j in enumerate(f):
                    n_with[j] += 1
                    before[j, f[:pos]] += 1
            prec = np.where(n_with[:, None] > 0, (n_with[:, None] - before + 0.5) / (n_with[:, None] + 1), 1.0)
            mmax = max(len(f) for f in first) + 1
            rank = np.full((k, mmax + 1), np.inf)
            rk = np.array([[f.index(kk) if kk in f else np.inf for kk in range(k)] for f in first])  # (n, K)
            for m in range(mmax + 1):
                rank[:, m] = (np.sum(rk >= m, axis=0) + 0.5) / (n + 1)
            vocab = np.flatnonzero(counts.sum(0) > 0)
            mbar = float(np.mean([len(f) for f in first]))
            # touched-set snapshots: after each new subtask
            snaps = []
            for f in first:
                t = np.zeros(k)
                for j in f:
                    t[j] = 1
                    snaps.append(t.copy())
            S = np.array(snaps)                                                      # (n_snap, K)
            logit = {}
            for lam in lams:
                W = np.zeros((k, k))
                wm = np.zeros(k)
                b = np.full(k, 30.0)                                                 # p = 1 off-vocab
                for kk in vocab:
                    y = 1.0 - S[:, kk]
                    if y.all():
                        continue
                    X = S[:, vocab].copy()
                    X[:, np.flatnonzero(vocab == kk)] = 0.0
                    X = np.hstack([X, S.sum(1, keepdims=True) / mbar])
                    w = _logistic(X, y, lam)
                    W[kk, vocab] = w[:len(vocab)]
                    W[kk, kk] = 0.0
                    wm[kk], b[kk] = w[len(vocab)], w[-1]
                logit[lam] = (W, wm, b)
            self.r[r] = dict(n=n, floor=floor, ge=ge, p0=p0, prec=prec, rank=rank, vocab=vocab,
                             mbar=mbar, logit=logit)
        for st in self.r.values():          # ignored subtasks never count or go missing
            for key in ("prec",):
                st[key][:, list(self.ignore)] = 1.0
            st["p0"][list(self.ignore)] = 1.0
            st["rank"][list(self.ignore)] = 1.0
            st["ge"][list(self.ignore), 1:] = 1.0


# ---- the rules ------------------------------------------------------------------------------


def count_masses(stats, recipes, ent, pr, levels=LEVELS):
    """{variant: (T, L) alarm mass} from entries (T, R, K) and recipe posteriors (T, R)."""
    T, R, K = ent.shape
    al = np.asarray(levels)[None, :]
    on = al > 0
    cond = ent / np.maximum(pr[:, :, None], 1e-300)
    c_after = np.cumsum(cond, axis=0)
    c_before = c_after - cond
    parts = {k_: np.zeros((T, len(levels))) for k_ in ("extra", "prec", "prec1", "rank", "set1", "set10", "end")}
    ign = np.zeros(K, dtype=bool)
    ign[list(stats.ignore)] = True
    for a, r in enumerate(recipes):
        st = stats.r.get(int(r))
        if st is None:
            continue
        # beginnings (entry-weighted): extra visits and the committed precedence rule
        tt, ii = np.nonzero((ent[:, a] > 1e-12) & ~ign[None, :])
        visit = np.floor(c_before[tt, a, ii] + 0.5).astype(int) + 1
        w = ent[tt, a, ii]
        ge = st["ge"]
        rep = visit >= 2
        p_rep = ge[ii[rep], np.minimum(visit[rep], ge.shape[1] - 1)]
        np.add.at(parts["extra"], tt[rep], w[rep, None] * ((p_rep[:, None] <= al) & on))
        fi = ~rep
        not_yet = (c_before[tt[fi], a] < 0.5) & ~ign[None, :]                    # (n, K)
        pk = np.where(not_yet, st["prec"][ii[fi]], 1.0).min(axis=1)
        p_first = np.minimum(ge[ii[fi], 1], pk)
        np.add.at(parts["prec"], tt[fi], w[fi, None] * ((p_first[:, None] <= al) & on))

        # missing as a state of the touched set, flagged once per k
        touched = (c_after[:, a] >= 0.5) & ~ign[None, :]                           # (T, K)
        untouched = ~touched & ~ign[None, :]
        m = touched.sum(1)
        change = np.r_[True, np.any(touched[1:] != touched[:-1], axis=1)]
        p = {}
        pm = np.ones((T, K))
        last = None
        for t in range(T):
            if change[t]:
                rows = np.flatnonzero(touched[t])
                last = st["prec"][rows].min(axis=0) if len(rows) else np.ones(K)
            pm[t] = last
        p["prec1"] = pm
        p["rank"] = st["rank"][:, np.minimum(m, st["rank"].shape[1] - 1)].T            # (T, K)
        for lam, name in ((1.0, "set1"), (10.0, "set10")):
            W, wm, b = st["logit"][lam]
            z = touched.astype(float) @ W.T + (m / st["mbar"])[:, None] * wm[None, :] + b[None, :]
            p[name] = np.maximum(1.0 / (1.0 + np.exp(-np.clip(z, -30, 30))), st["floor"])
        for name, pv in p.items():
            F = untouched[:, :, None] & (pv[:, :, None] <= al[None]) & on[None]         # (T, K, L)
            seen = np.cumsum(F, axis=0) - F
            newly = (F & (seen == 0)).any(axis=1)                                      # (T, L)
            parts[name] += pr[:, a, None] * newly
        miss = untouched[-1] & (st["p0"] <= np.max(levels))
        if miss.any():
            parts["end"][-1] += pr[-1, a] * ((st["p0"][miss][:, None] <= al[0]) & on[0]).any(axis=0)
    P = parts
    out = {"extra": P["extra"], "prec": P["extra"] + P["prec"], "prec_noextra": P["prec"],
           "prec1": P["extra"] + P["prec1"], "rank": P["extra"] + P["rank"],
           "set1": P["extra"] + P["set1"], "set10": P["extra"] + P["set10"],
           "extra+end": P["extra"] + P["end"], "prec+end": P["extra"] + P["prec"] + P["end"],
           "rank+end": P["extra"] + P["rank"] + P["end"]}
    return {k_: np.minimum(v, 1.0) for k_, v in out.items()}


# ---- inputs per trial -----------------------------------------------------------------------


def live_inputs(live, trial):
    """(live grid (T, A, K), entries (T, R, K), recipe posteriors (T, R), recipes) of one trial."""
    live.reset(trial.recipe)
    fr = live.hsmm.filter(trial.verbs, trial.nouns, recipe=trial.recipe)
    verb, noun, _ = live.hsmm.pad(trial.verbs, trial.nouns)
    p = live.emission_pvalues(fr.log_stay, fr.log_switch, verb, noun)
    dyn = live._dynamics()
    ev = event_table(dyn, live.bins, jnp.asarray(live.bad.alphas), fr.log_b, live.beta(fr), fr.loglik, fr.mask)
    n = fr.n
    from cook_ad.detectors.hsmm_live import LiveTrace
    tr = LiveTrace({c: np.asarray(x)[:n] for c, x in p.items()},
                   {c: np.asarray(x)[:n] for c, x in zip(EVENTS, ev)},
                   live.bad.alphas, np.asarray(fr.log_norm)[:n], fr.occupancy)
    ent, pr = entry_table(dyn, fr.log_b, jnp.zeros_like(fr.log_b), fr.loglik, fr.mask)
    recipes = live.hsmm.current.recipes
    if isinstance(recipes, slice):
        recipes = range(live.hsmm.k_recipe)[recipes]
    return (harness.live_grid(live, tr, ALPHAS, KAPPAS), np.asarray(ent)[:n], np.asarray(pr)[:n],
            [int(x) for x in np.atleast_1d(recipes)])


def hindsight_inputs(trace, k):
    T = len(trace.z_star)
    ent = np.zeros((T, 1, k))
    t = 0
    for s, d in trace.segments:
        ent[t, 0, int(s)] = 1.0
        t += int(d)
    return ent, np.ones((T, 1)), [int(trace.r_hat)]


# ---- scoring --------------------------------------------------------------------------------


class TrialScore(metric_g.GridScore):
    """GridScore that also keeps each trial's hits and false alarms over the grid, for a
    bootstrap at the picked operating point."""

    def __init__(self, tick_seconds=1.0):
        super().__init__(tick_seconds)
        self.per = []                                   # (group, hits (P,), n_spans, fp (P,))

    def add(self, group, flags, spans, verbs, nouns):
        super().add(group, flags, spans, verbs, nouns)
        f = np.asarray(flags, dtype=bool)
        r = metric_g.score_dense(f.reshape(f.shape[0], -1), spans, metric_g.correct_runs(verbs, nouns, spans),
                                 metric_g.run_starts(verbs, nouns), self.deadline)
        self.per.append((group, r["hits"].sum(axis=0).reshape(f.shape[1:]), len(spans),
                         r["fp"].reshape(f.shape[1:])))


def bootstrap_delta(score, idx, idx0, reps=2000, seed=0):
    """Paired, group-stratified bootstrap of F1(idx) - F1(idx0) over test trials (healthy
    trials carry no TP/FP in G-F1, so they drop out)."""
    rng = np.random.default_rng(seed)
    per = [x for x in score.per if x[0] != "healthy"]
    tp = np.array([[h[idx], h[idx0]] for _, h, _, _ in per], dtype=float)
    fp = np.array([[f[idx], f[idx0]] for _, _, _, f in per], dtype=float)
    ns = np.array([n for _, _, n, _ in per], dtype=float)
    groups = defaultdict(list)
    for j, (g, *_ ) in enumerate(per):
        groups[g].append(j)

    def f1(sel):
        t, f, n = tp[sel].sum(0), fp[sel].sum(0), ns[sel].sum()
        pr = np.where(t + f > 0, t / np.maximum(t + f, 1e-12), 0.0)
        rc = t / max(n, 1)
        return np.where(pr + rc > 0, 2 * pr * rc / np.maximum(pr + rc, 1e-12), 0.0)
    d = []
    for _ in range(reps):
        sel = np.concatenate([rng.choice(ix, len(ix)) for ix in groups.values()])
        a, b = f1(sel)
        d.append(a - b)
    d = np.asarray(d)
    return {"mean": float(d.mean()), "lo": float(np.quantile(d, 0.025)), "hi": float(np.quantile(d, 0.975)),
            "p_le_0": float(np.mean(d <= 0))}


def _slice0(res):
    sl = lambda x: np.asarray(x)[..., 0]  # noqa: E731
    return {k: ({t: sl(x) for t, x in v.items()} if k == "per_type_recall" else sl(v)) for k, v in res.items()}


def evaluate(person, n, mode, val, test, model_parts, tick=None, hind_grid=harness.hindsight_split_grid):
    """`stats` is a Stats, or (stats for validation trials by trial id, stats for test) when
    validation trials were in training (cross-fitted, Breakfast)."""
    hsmm, bad, stats, k = model_parts
    tick = E.TICK if tick is None else tick
    if isinstance(stats, Stats):
        stats = (lambda t, s=stats: s, stats)
    det = HindsightHSMMDetector(hsmm, bad, OperatingPoint(0.01), chunk_size=8)
    live = LiveHSMMDetector(hsmm, bad, OperatingPoint(0.01, 0.5))
    kap = np.asarray(KAPPAS)

    def run(trials, part):
        scores = {v: TrialScore(tick) for v in VARIANTS}
        sess = {v: defaultdict(list) for v in VARIANTS}
        groups = list(dict.fromkeys(g for g, _ in trials))
        for g in groups:
            ts = [t for g_, t in trials if g_ == g]
            if mode == "hindsight":
                traces = det.traces(ts)
            for i, t in enumerate(ts):
                if mode == "hindsight":
                    tr = traces[i]
                    base = hind_grid(det, tr, ALPHAS)                               # (T, A, A) or (T, A)
                    ent, pr, rec = hindsight_inputs(tr, k)
                else:
                    base, ent, pr, rec = live_inputs(live, t)                         # (T, A, K)
                masses = count_masses(stats[0](t) if part == "val" else stats[1], rec, ent, pr)
                for v, ms in masses.items():
                    if mode == "hindsight":
                        cnt = ms > 0.5                                                 # (T, L)
                        grid = base[..., None] | cnt.reshape((len(cnt),) + (1,) * (base.ndim - 1) + (-1,))
                        sess[v][g].append(cnt.any(axis=0))
                    else:
                        cnt = ms[:, None, :] > kap[None, :, None]                      # (T, K, L)
                        grid = base[..., None] | cnt[:, None, :, :]
                        sess[v][g].append(cnt[:, KAPPAS.index(0.5)].any(axis=0))
                    scores[v].add(g, grid, list(t.spans), t.verbs, t.nouns)
        run.scores = scores
        return {v: s.result() for v, s in scores.items()}, {
            v: {g: np.mean(x, axis=0).tolist() for g, x in d.items()} for v, d in sess.items()}

    t0 = time.time()
    gv, _ = run(val, "val")
    gt, st = run(test, "test")
    rows = {}
    off_v, off_t = _slice0(gv[VARIANTS[0]]), _slice0(gt[VARIANTS[0]])
    i0 = harness.pick(off_v)
    rows["off"] = harness.at(off_t, i0)
    test_scores = run.scores
    for v in VARIANTS:
        idx = harness.pick(gv[v])
        row = harness.at(gt[v], idx)
        row["count_alpha"] = LEVELS[idx[-1]]
        row["index"] = [int(x) for x in idx]
        row["delta_f1_vs_off"] = bootstrap_delta(test_scores[v], tuple(idx), tuple(i0) + (0,))
        rows[v] = row
    print(f"{person} N={n:3d} {mode:9s} ({time.time() - t0:.0f}s) off F1={rows['off']['f1']:.3f} "
          f"FPR={rows['off']['healthy_trial_fpr']:.2f} | "
          + " ".join(f"{v}={rows[v]['f1']:.3f}@{rows[v]['count_alpha']:g}" for v in VARIANTS), flush=True)
    return {"rows": rows, "session_any": st}


def population_sessions(names, m, seed=0):
    """M sessions per recipe from the prior's population (eval_synth_prior._population), as
    (sequence, recipe index); a fixed random subset."""
    seqs, recs = E._population()
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for s, r in zip(seqs, recs):
        by[r].append(s)
    out = []
    for r, ss in sorted(by.items()):
        if r not in names:
            continue
        for j in rng.permutation(len(ss))[:m]:
            out.append((ss[j], names.index(r)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--person", type=int, action="append")
    ap.add_argument("--sessions", type=int, action="append")
    ap.add_argument("--mode", action="append", choices=["hindsight", "live"])
    ap.add_argument("--limit", type=int, default=None, help="sessions per group (smoke test)")
    ap.add_argument("--out", default="runs/drift_synth/compare_s_count")
    ap.add_argument("--pop", type=int, default=0,
                    help="add M population sessions per recipe (decoded by the person's model) to the s_count statistics")
    a = ap.parse_args()
    if a.pop:
        a.out += f"_pop{a.pop}"
    d_max = load_config(E.CONFIG)["duration"]["d_max_ticks"]
    gc = json.loads((E.ROOT / "global_counts.json").read_text())
    for i in a.person or list(E.PERSONS):
        pid = f"p{i:02d}"
        d = E.ROOT / "persons" / pid
        labels = json.loads((d / "null" / "labels.json").read_text())
        seqs = E._in_order(json.loads((d / "null" / "sequences.json").read_text()), labels)
        _, names = recipe_label_assignment(seqs, labels)
        vocab = json.loads((d / "null" / "vocab.json").read_text())
        sup = person_support(i, E.SEED, E._assets({"theta0_by_knob": E.CAPABILITY, "habit": 2.0}), vocab,
                             E.N_DAYS, E.ORACLE_DAYS, E.TICK, cache=d / f"oracle_support_v{E.ORACLE_VERSION}.pkl")
        val = E.gen_pool(i, seqs[E.VAL_START:E.TEST_START], labels, sup)
        test = E.gen_pool(i, seqs[E.TEST_START:E.TEST_END], labels, sup)
        if a.limit:
            def cut(tr):
                c = Counter()
                out = []
                for g, t in tr:
                    c[g] += 1
                    if c[g] <= a.limit:
                        out.append((g, t))
                return out
            val, test = cut(val), cut(test)
        report = {"metric": "G", "person": pid, "levels": LEVELS, "variants": VARIANTS, "results": {}}
        for n in a.sessions or (30, 250):
            model = d / f"model_s{n}" / "model.npz"
            p = apply_global_prior(hparams.load(str(model)), gc, names, 1.0)
            hsmm = HSMM(p, d_max, duration_tables=load_tables(durations_path(model)))
            bad = BadSets(hsmm, ALPHAS)
            k = int(np.asarray(hsmm.tables.log_emit_v).shape[0])
            idle = set(np.flatnonzero(np.argmax(np.asarray(hsmm.tables.log_emit_v), axis=-1)
                                      == vocab["verbs"]["stall"]).tolist())
            det = HindsightHSMMDetector(hsmm, bad, OperatingPoint(0.01), chunk_size=8)
            train = seqs[:n]
            td = recipe_label_assignment(train, labels)[0]
            trs = det.traces([Trial(s["trial_id"], np.asarray(s["verb_ids"]), np.asarray(s["noun_ids"]),
                                    int(r), (), "healthy") for s, r in zip(train, td)])
            dec = [(t.r_hat, [s for s, _ in t.segments]) for t in trs]
            if a.pop:
                pop = population_sessions(names, a.pop)
                ptr = det.traces([Trial(f"pop{j}", np.asarray(s["verb_ids"]), np.asarray(s["noun_ids"]), r, (), "healthy")
                                  for j, (s, r) in enumerate(pop)])
                dec += [(t.r_hat, [s for s, _ in t.segments]) for t in ptr]
            stats = Stats(dec, k, idle)
            report["results"][n] = {}
            for mode in a.mode or ("hindsight", "live"):
                report["results"][n][mode] = evaluate(pid, n, mode, val, test, (hsmm, bad, stats, k))
                harness.dump(f"{a.out}_{pid}.json", report)


if __name__ == "__main__":
    main()
