"""The anomaly detector on the synthetic persons: a global prior from a simulated population,
fine-tuned per person, scored on step-level injected errors (docs/drift_synth.md §7).

Population: 60 people (seed 1000, never test subjects), 40 sessions each, preference knobs
theta0 ~ U(0.05, 0.95), slow ~ U(0.4, 0.6), fine_motor ~ U(0, 0.4) (no repeats), habit ~ U(0, 1).
Their tick streams are run-length counted per recipe (hsmm/train/global_prior.count_global): ONE
global prior for every person model.

Test persons: seed 0, persons 0-3, habit 2, slow ~ U(0.4, 0.6), fine_motor ~ U(0, 0.4), 400
sessions each. Train on the first N in {10, 30, 250}; validation sessions 250-324, test 325-399.
Every stream is idle-scrubbed (data/scrub.py). Each person model is the training pipeline
(docs/training.md, stages 1-4b; configs/drift_synth/hsmm.yaml), then the global prior is added
post-fit at strength lambda (apply_global_prior): the arms are lambda = 0 (person only) and 1.

Pools: each held-out session with one real error edited into its clean events
(drift_synth/inject.py), re-exported and scrubbed. Hindsight picks the transition channels' alpha
apart from the rest (harness.hindsight_split_grid), Live picks alpha x kappa, both on validation
by best F1, reported on test. The metric is the harness's (H at delay 2; COOK_METRIC=G for G).
--core-count adds the s_count channel (hsmm/visit_counts.py) fitted on the training decodes: in the transition group for Hindsight, on its own alpha axis for Live.

    ./py scripts/eval/eval_synth_prior.py --stage population
    ./py scripts/eval/eval_synth_prior.py --stage persons
    ./py scripts/eval/eval_synth_prior.py --stage train [--person 0 --sessions 250]
    ./py scripts/eval/eval_synth_prior.py --stage eval [--person 0]      # Hindsight
    ./py scripts/eval/eval_synth_prior.py --stage eval-live [--person 0]
    ./py scripts/eval/eval_synth_prior.py --stage summary
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())

ROOT = Path("dataset/processed/drift_synth_prior_sess")
RESULTS = Path("runs/drift_synth/prior")
CONFIG = "configs/drift_synth/hsmm.yaml"
POP_SEED, POP_N, POP_SESSIONS = 1000, 60, 40
SEED, PERSONS = 0, (0, 1, 2, 3)
N_SESSIONS = 400                                     # generated per test person
VAL_START, TEST_START, TEST_END = 250, 325, 400      # session indices
TRAIN_SESSIONS = (10, 30, 250)
SOFT_ITERS = 20             # the soft-EM objective moves < 0.2% after iteration 20
ORACLE_SESSIONS = 4000      # healthy-support sample: sessions 400..3999
ORACLE_VERSION = 3          # synth_oracle.VERSION the cache was built with
CAPABILITY = {"slow": [0.4, 0.6], "fine_motor": [0.0, 0.4]}   # fine_motor below its hinge: no repeats
TICK = 1.0
ARMS = {"person only": 0.0, "prior lambda=1": 1.0}
COUNT_ALPHAS = (0.0, 0.005, 0.01, 0.02, 0.05)       # Live's s_count axis; 0 = off
CORE_COUNT = False          # --core-count


def _assets(person_cfg):
    from cook_ad.drift_synth.assets import load_assets
    a = load_assets()
    a.gen["recognizer"] = {k: (v if k == "min_event_s" else 0.0) for k, v in a.gen["recognizer"].items()}
    a.gen["person"] = dict(a.gen["person"], **person_cfg)
    a.gen["export"] = dict(a.gen["export"], tick_seconds=TICK)
    return a


def _scrub_export(out, vocab, sil_names):
    """Rewrite an export's sequences.json / labels.json without idle ticks (data/scrub.py); the
    originals stay as *_raw.json and scrub_kept.json maps each kept tick to its original tick."""
    from cook_ad.data.scrub import scrub_sequence
    sil = (vocab["verbs"][sil_names[0]], vocab["nouns"][sil_names[1]])
    seqs = json.loads((out / "sequences.json").read_text())
    labels = json.loads((out / "labels.json").read_text())
    lab = {e["trial_id"]: e for e in labels}
    (out / "sequences_raw.json").write_text(json.dumps(seqs))
    (out / "labels_raw.json").write_text(json.dumps(labels))
    new_s, new_l, kept = [], {}, {}
    for s in seqs:
        s2, l2, k = scrub_sequence(s, sil, lab[s["trial_id"]])
        new_s.append(s2)
        new_l[s["trial_id"]] = l2
        kept[s["trial_id"]] = k.tolist()
    (out / "sequences.json").write_text(json.dumps(new_s))
    (out / "labels.json").write_text(json.dumps([new_l[e["trial_id"]] for e in labels]))
    (out / "scrub_kept.json").write_text(json.dumps(kept))


def _in_order(seqs, labels):
    """The person's sessions in cooking order (the session index)."""
    key = {e["trial_id"]: e["session"] for e in labels}
    return sorted(seqs, key=lambda s: key[s["trial_id"]])


def _person(i):
    """(dir, sessions in order, labels, recipe names) of test person i."""
    from cook_ad.hsmm.train.lexical_init import recipe_label_assignment
    d = ROOT / "persons" / f"p{i:02d}"
    labels = json.loads((d / "null" / "labels.json").read_text())
    seqs = _in_order(json.loads((d / "null" / "sequences.json").read_text()), labels)
    return d, seqs, labels, recipe_label_assignment(seqs, labels)[1]


# ---- data and training -----------------------------------------------------------------------


def stage_population():
    from cook_ad.drift_synth.export import export
    from cook_ad.drift_synth.sampler import ConditionCfg, generate_person
    from cook_ad.hsmm.train.global_prior import count_global

    assets = _assets({"theta0": [0.05, 0.95], "theta0_by_knob": CAPABILITY, "habit": [0.0, 1.0]})
    cond = ConditionCfg("null", n_persons=POP_N, n_sessions=POP_SESSIONS)
    seqs, recipes, n = [], [], 0
    for i in range(POP_N):
        person, sessions, gt = generate_person(i, cond, POP_SEED, assets)
        out = ROOT / "population" / person.person_id
        export(sessions, out, assets.vocab, gt, assets.gen["export"])
        _scrub_export(out, assets.vocab, assets.gen["export"]["sil"])
        labels = {e["trial_id"]: e["recipe_label"] for e in json.loads((out / "labels.json").read_text())}
        for s in json.loads((out / "sequences.json").read_text()):
            seqs.append(s)
            recipes.append(labels[s["trial_id"]])
        n += len(sessions)
    counts = count_global(seqs, recipes)
    (ROOT / "global_counts.json").write_text(json.dumps(counts, sort_keys=True))
    print(f"population: {POP_N} people, {n} sessions; sessions per recipe "
          f"{dict(sorted(counts['n_trials'].items()))}")


def stage_persons():
    from cook_ad.drift_synth.export import export
    from cook_ad.drift_synth.sampler import ConditionCfg, generate_person

    assets = _assets({"theta0_by_knob": CAPABILITY, "habit": 2.0})
    cond = ConditionCfg("null", n_persons=1, n_sessions=N_SESSIONS)
    for i in PERSONS:
        person, sessions, gt = generate_person(i, cond, SEED, assets)
        d = ROOT / "persons" / person.person_id
        export(sessions, d / "null", assets.vocab, gt, assets.gen["export"])
        _scrub_export(d / "null", assets.vocab, assets.gen["export"]["sil"])
        _, seqs, _, _ = _person(i)
        assert len(seqs) >= TEST_END, (person.person_id, len(seqs))
        for n in TRAIN_SESSIONS:
            (d / "null" / f"split_s{n}.json").write_text(json.dumps(
                {"seed": SEED, "test_frac": None, "train_trial_ids": [s["trial_id"] for s in seqs[:n]],
                 "test_trial_ids": [s["trial_id"] for s in seqs[VAL_START:TEST_END]],
                 "rule": f"train = first {n} sessions; held out = sessions {VAL_START}-{TEST_END - 1}"},
                indent=1))
        print(f"{person.person_id}: {len(sessions)} sessions, repertoire {list(person.repertoire)}, "
              f"habit {person.habit}")


def stage_train(persons, sizes):
    for i in persons:
        d = ROOT / "persons" / f"p{i:02d}"
        for n in sizes:
            out = d / f"model_s{n}"
            if (out / "model.npz").exists():
                print(f"p{i:02d} N={n}: exists")
                continue
            out.mkdir(parents=True, exist_ok=True)
            common = ["--config", CONFIG, "--sequences", str(d / "null" / "sequences.json"),
                      "--labels", str(d / "null" / "labels.json"),
                      "--split-file", str(d / "null" / f"split_s{n}.json"), "--recipe-labels"]
            steps = [
                ["scripts/train/fit_lexical.py", *common, "--init-prior-scale", "0.0", "--anchor",
                 "50", "--max-iters", str(SOFT_ITERS), "--out", str(out / "soft.npz")],
                ["scripts/train/fit_hard_em.py", *common, "--init-from", str(out / "soft.npz"),
                 "--keep-init-emissions", "--iters", "5", "--out", str(out / "hard.npz")],
                ["scripts/train/smooth_params.py", "--in", str(out / "hard.npz"),
                 "--out", str(out / "smooth.npz"), "--strength", "0.7", "--backoff-tau", "30"],
                ["scripts/train/refit_durations.py", *common, "--in", str(out / "smooth.npz"),
                 "--out", str(out / "model.npz"), "--kappa", "0.001"],
                ["scripts/train/fit_np_hazard.py", *common, "--in", str(out / "model.npz"),
                 "--kappa", "5", "--bandwidth", "1", "--min-segments", "30"],
            ]
            with open(out / "train.log", "w") as log:
                for cmd in steps:
                    subprocess.run(["./py", *cmd], check=True, stdout=log, stderr=subprocess.STDOUT)
            print(f"p{i:02d} N={n}: trained", flush=True)


# ---- pools -----------------------------------------------------------------------------------


_GRAPHS = {}


def support(i):
    """The healthy-support oracle of person i (eval/synth_oracle.py), cached next to the data."""
    from cook_ad.eval.synth_oracle import person_support
    d = ROOT / "persons" / f"p{i:02d}"
    vocab = json.loads((d / "null" / "vocab.json").read_text())
    return person_support(i, SEED, _assets({"theta0_by_knob": CAPABILITY, "habit": 2.0}), vocab,
                          N_SESSIONS, ORACLE_SESSIONS, TICK, cache=d / f"oracle_support_v{ORACLE_VERSION}.pkl")


def gen_pool(person_idx, part, labels, sup):
    """[(group, Trial)] from the step-level injectors (drift_synth/inject.py): each held-out
    session with one real error edited into its clean events, re-exported to ticks and scrubbed.
    Prints the fraction the healthy-support oracle confirms."""
    import numpy as np
    from cook_ad.data.scrub import map_spans, scrub_idle
    from cook_ad.drift_synth.export import to_ticks
    from cook_ad.drift_synth.inject import ERROR_TYPES, event_spans, inject
    from cook_ad.eval import harness
    from cook_ad.hsmm.train.lexical_init import recipe_label_assignment
    if not _GRAPHS:
        _GRAPHS.update(_assets({}).graphs)
    d = ROOT / "persons" / f"p{person_idx:02d}"
    vocab = json.loads((d / "null" / "vocab.json").read_text())
    ex = _assets({}).gen["export"]
    rec = {e["trial_id"]: e["recipe_label"] for e in labels}
    dish = recipe_label_assignment(part, labels)[0]
    iv = {i: k for k, i in vocab["verbs"].items()}
    inn = {i: k for k, i in vocab["nouns"].items()}
    sil = set(ex["sil"])
    sil_ids = (vocab["verbs"][ex["sil"][0]], vocab["nouns"][ex["sil"][1]])
    seen = {}
    for r, rng_ in sup.count_range.items():
        toks = [t for t in rng_ if rng_[t][1] > 0]
        seen[r] = ({iv[v] for v, _ in toks}, {inn[n] for _, n in toks})
    rng = np.random.default_rng([SEED, person_idx, int(part[0]["trial_id"].rsplit("_s", 1)[1]), len(part)])
    groups = {"healthy": [(s, r) for s, r in zip(part, dish)]}
    confirmed = {}
    for et in ERROR_TYPES:
        groups[et] = []
        for s, r in zip(part, dish):
            sess = json.loads((d / "null" / "sessions" / f"{s['trial_id']}.json").read_text())
            name = rec[s["trial_id"]]
            sv, sn = seen.get(name, (set(), set()))
            res = inject(et, sess, _GRAPHS[name], rng,
                         unseen_verbs=sorted(set(vocab["verbs"]) - sv - sil),
                         unseen_nouns=sorted(set(vocab["nouns"]) - sn - sil))
            if res is None:
                continue
            events, marked = res
            v, n = to_ticks(events, vocab, ex["sil"], ex["fps"], TICK)
            spans = [(a, min(b, len(v) - 1)) for a, b in event_spans(events, marked, TICK) if a < len(v)]
            c = sup.check(name, v, n)              # the oracle reads the stream before scrubbing
            v, n, kept = scrub_idle(v, n, sil_ids)
            spans = map_spans(spans, kept)
            if not spans:
                continue
            groups[et].append(({"verb_ids": v, "noun_ids": n, "spans": spans}, r))
            confirmed.setdefault(et, []).append((c["anomalous"], c["anomalous_or_order"]))
    print("  injections confirmed by the oracle (local checks / + order): "
          + ", ".join(f"{et} {np.mean([x[0] for x in c]):.2f} / {np.mean([x[1] for x in c]):.2f} (n={len(c)})"
                      for et, c in confirmed.items()), flush=True)
    return harness.trials_from_groups(groups)


# ---- models ----------------------------------------------------------------------------------


def _jax():
    from cook_ad.xla_env import disable_gpu_autotuning
    disable_gpu_autotuning()
    import jax
    jax.config.update("jax_enable_x64", True)


def load_model(d, n, names, lam, gcounts, d_max):
    """Person model at N = n with the global prior at lambda, and its stage-4b duration tables."""
    from cook_ad.hsmm import params as hparams
    from cook_ad.hsmm.model import HSMM, durations_path
    from cook_ad.hsmm.train.global_prior import apply_global_prior
    from cook_ad.hsmm.train.np_hazard import load_tables
    model = d / f"model_s{n}" / "model.npz"
    p = apply_global_prior(hparams.load(str(model)), gcounts, names, lam)
    dp = durations_path(model)
    return HSMM(p, d_max, duration_tables=load_tables(dp) if os.path.exists(dp) else None)


def fit_counts(hsmm, bad, seqs, labels, n, vocab, chunk_size):
    """s_count statistics from the model's own decodes of its training sessions (idle states ignored)."""
    import numpy as np
    from cook_ad.core import OperatingPoint, Trial
    from cook_ad.detectors.hsmm_hindsight import HindsightHSMMDetector
    from cook_ad.hsmm.train.lexical_init import recipe_label_assignment
    from cook_ad.hsmm.visit_counts import VisitCounts
    train = seqs[:n]
    dish = recipe_label_assignment(train, labels)[0]
    det = HindsightHSMMDetector(hsmm, bad, OperatingPoint(0.005), chunk_size=chunk_size)
    traces = det.traces([Trial(s["trial_id"], np.asarray(s["verb_ids"]), np.asarray(s["noun_ids"]),
                               int(r), (), "healthy") for s, r in zip(train, dish)])
    idle = np.flatnonzero(np.argmax(np.asarray(hsmm.tables.log_emit_v), axis=-1) == vocab["verbs"]["stall"])
    return VisitCounts.fit([(t.r_hat, [k for k, _ in t.segments]) for t in traces], ignore=idle.tolist())


def _suffix():
    from cook_ad.eval import harness
    tag = "" if (harness.METRIC, harness.DELAY) == ("H", 2) else f"_{harness.METRIC}" + (
        f"{harness.DELAY}" if harness.METRIC == "H" else "")
    return ("_scount" if CORE_COUNT else "") + tag


def _print(pid, n, label, row, extra=""):
    print(f"{pid} N={n:3d} {label:28s} F1={row['f1']:.3f} P={row['precision']:.3f} R={row['recall']:.3f} "
          f"FPR={row['healthy_trial_fpr']:.2f} alarms/10min={row['alarms_per_10min']:.2f} {extra}| "
          + " ".join(f"{k[:5]} {v:.2f}" for k, v in row["per_type"].items()), flush=True)


# ---- evaluation ------------------------------------------------------------------------------


def stage_eval(persons, chunk_size=8):
    """Hindsight with a split alpha (transition channels apart from the rest), both arms; the
    single-alpha pick (the grid's diagonal) is reported beside it."""
    _jax()
    import numpy as np
    from cook_ad.core import OperatingPoint
    from cook_ad.data.config import load_config
    from cook_ad.detectors.hsmm_hindsight import HindsightHSMMDetector
    from cook_ad.detectors.thresholds import BadSets
    from cook_ad.eval import harness

    d_max = load_config(CONFIG)["duration"]["d_max_ticks"]
    gcounts = json.loads((ROOT / "global_counts.json").read_text())
    for i in persons:
        d, seqs, labels, names = _person(i)
        pid = d.name
        vocab = json.loads((d / "null" / "vocab.json").read_text())
        sup = support(i)
        val = gen_pool(i, seqs[VAL_START:TEST_START], labels, sup)
        test = gen_pool(i, seqs[TEST_START:TEST_END], labels, sup)
        out = {}
        for n in TRAIN_SESSIONS:
            out[n] = {}
            for arm, lam in ARMS.items():
                hsmm = load_model(d, n, names, lam, gcounts, d_max)
                bad = BadSets(hsmm, harness.ALPHAS)
                vc = fit_counts(hsmm, bad, seqs, labels, n, vocab, chunk_size) if CORE_COUNT else None
                det = HindsightHSMMDetector(hsmm, bad, OperatingPoint(0.005), chunk_size=chunk_size, counts=vc)

                def grid(trials):
                    score = harness.SCORER.GridScore(TICK, harness.DELAY)   # per trial: grids get large
                    for group in dict.fromkeys(g for g, _ in trials):
                        ts = [t for g, t in trials if g == group]
                        for t_, tr in zip(ts, det.traces(ts)):
                            score.add(group, harness.hindsight_split_grid(det, tr, harness.ALPHAS),
                                      list(t_.spans), t_.verbs, t_.nouns)
                    return score.result()
                gv, gt = grid(val), grid(test)
                idx = harness.pick(gv)
                row = harness.at(gt, idx)
                row.update(alpha_transition=harness.ALPHAS[idx[0]], alpha_rest=harness.ALPHAS[idx[1]])
                out[n][arm] = row
                diag = [float(np.asarray(gv["f1"])[j, j]) for j in range(len(harness.ALPHAS))]
                j = int(np.argmax(diag))
                single = harness.at(gt, (j, j))
                single["alpha_transition"] = single["alpha_rest"] = harness.ALPHAS[j]
                out[n][arm + " (single alpha)"] = single
                _print(pid, n, arm, row, f"a_trans={row['alpha_transition']:g} a_rest={row['alpha_rest']:g} ")
        RESULTS.mkdir(parents=True, exist_ok=True)
        harness.dump(RESULTS / f"hindsight{_suffix()}_{pid}.json",
                     {"metric": harness.LABEL, "person": pid, "unit": "sessions", "results": out})


def stage_eval_live(persons):
    """The causal detector (LiveHSMMDetector, L = 0): alpha x kappa picked on validation, plus an
    s_count alpha axis with --core-count (level 0 = off, so the same grid gives the row without)."""
    _jax()
    import numpy as np
    from cook_ad.core import OperatingPoint
    from cook_ad.data.config import load_config
    from cook_ad.detectors.hsmm_live import COUNT, LiveHSMMDetector
    from cook_ad.detectors.thresholds import BadSets
    from cook_ad.eval import harness

    d_max = load_config(CONFIG)["duration"]["d_max_ticks"]
    gcounts = json.loads((ROOT / "global_counts.json").read_text())
    kap = np.asarray(harness.KAPPAS)
    for i in persons:
        d, seqs, labels, names = _person(i)
        pid = d.name
        vocab = json.loads((d / "null" / "vocab.json").read_text())
        sup = support(i)
        val = gen_pool(i, seqs[VAL_START:TEST_START], labels, sup)
        test = gen_pool(i, seqs[TEST_START:TEST_END], labels, sup)
        out = {}
        for n in TRAIN_SESSIONS:
            out[n] = {}
            for arm, lam in ARMS.items():
                hsmm = load_model(d, n, names, lam, gcounts, d_max)
                bad = BadSets(hsmm, harness.ALPHAS)
                vc = fit_counts(hsmm, bad, seqs, labels, n, vocab, 8) if CORE_COUNT else None
                live = LiveHSMMDetector(hsmm, bad, OperatingPoint(0.01, 0.5), counts=vc)

                def grid(trials):
                    score = harness.SCORER.GridScore(TICK, harness.DELAY)
                    for group, t in trials:
                        tr = live.trace(t)
                        g = harness.live_grid(live, tr, harness.ALPHAS, harness.KAPPAS)    # (T, A, K)
                        if vc is not None:
                            ev = tr.events[COUNT]
                            cnt = np.stack([np.zeros((len(ev), len(kap)), dtype=bool) if a_ == 0 else
                                            ev[:, bad.alpha_index(a_)][:, None] > kap[None, :]
                                            for a_ in COUNT_ALPHAS], axis=-1)              # (T, K, C)
                            g = g[..., None] | cnt[:, None, :, :]
                        score.add(group, g, list(t.spans), t.verbs, t.nouns)
                    return score.result()
                gv, gt = grid(val), grid(test)
                idx = harness.pick(gv)
                row = harness.at(gt, idx)
                row.update(alpha=harness.ALPHAS[idx[0]], kappa=harness.KAPPAS[idx[1]])
                if vc is not None:
                    row["count_alpha"] = COUNT_ALPHAS[idx[2]]
                out[n][arm] = row
                _print(pid, n, arm + " live", row, f"a={row['alpha']:g} k={row['kappa']:g} ")
        RESULTS.mkdir(parents=True, exist_ok=True)
        harness.dump(RESULTS / f"live{_suffix()}_{pid}.json",
                     {"metric": harness.LABEL, "person": pid, "unit": "sessions", "results": out})


def summarize():
    """Mean over persons of every results file of this metric (and --core-count setting)."""
    import numpy as np
    from cook_ad.eval import harness
    types = ("substitution", "abandonment", "omission", "transposition", "repetition")
    for kind in ("hindsight", "live"):
        rows = {}
        for f in sorted(RESULTS.glob(f"{kind}{_suffix()}_p*.json")):
            for n, arms in json.loads(f.read_text())["results"].items():
                for arm, r in arms.items():
                    rows.setdefault((int(n), arm), []).append(r)
        if not rows:
            continue
        print(f"\n### {kind}, mean over persons, test sessions, metric {harness.LABEL}\n")
        print("| train sessions | arm | persons | F1 | precision | recall | alarms / 10 min | healthy-trial FPR | "
              + " | ".join(t[:5] + "." for t in types) + " |")
        print("|---" * (8 + len(types)) + "|")
        for (n, arm), rs in sorted(rows.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            m = lambda k: np.mean([r[k] for r in rs])  # noqa: E731
            print(f"| {n} | {arm} | {len(rs)} | {m('f1'):.3f} | {m('precision'):.3f} | {m('recall'):.3f} | "
                  f"{m('alarms_per_10min'):.2f} | {m('healthy_trial_fpr'):.2f} | "
                  + " | ".join(f"{np.mean([r['per_type'][t] for r in rs]):.2f}" for t in types) + " |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True,
                    choices=["population", "persons", "train", "eval", "eval-live", "summary"])
    ap.add_argument("--person", type=int, action="append")
    ap.add_argument("--sessions", type=int, action="append", help="training sizes (default 10, 30, 250)")
    ap.add_argument("--core-count", action="store_true", help="add the s_count channel")
    a = ap.parse_args()
    global CORE_COUNT, TRAIN_SESSIONS
    CORE_COUNT = a.core_count
    if a.sessions and a.stage != "train":
        TRAIN_SESSIONS = tuple(a.sessions)
    persons = a.person or PERSONS
    if a.stage == "population":
        stage_population()
    elif a.stage == "persons":
        stage_persons()
    elif a.stage == "train":
        stage_train(persons, a.sessions or TRAIN_SESSIONS)
    elif a.stage == "eval":
        stage_eval(persons)
        summarize()
    elif a.stage == "eval-live":
        stage_eval_live(persons)
        summarize()
    else:
        summarize()


if __name__ == "__main__":
    main()
