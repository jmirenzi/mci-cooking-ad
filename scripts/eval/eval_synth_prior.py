"""A global prior from a simulated population, fine-tuned per person (docs/drift_synth.md §9.2).

Population: 60 people (seed 1000, never test subjects), 30 days each, preference knobs
theta0 ~ U(0.05, 0.95), slow ~ U(0.4, 0.6), fine_motor ~ U(0, 0.4) (no repeats), habit ~ U(0, 1). Their tick streams are
run-length counted per recipe (hsmm/train/global_prior.count_global): ONE global prior for every
person model.

Test persons: seed 0, persons 0-3, habit 2, slow ~ U(0.4, 0.6), fine_motor ~ U(0, 0.4), 320 days (402-429
sessions). Sizes are counted in SESSIONS, in cooking order (day, then session of the day): train
on the first N in {10, 30, 100, 250}, validation sessions 250-324, test sessions 325-399, so every
person has the same 75 + 75 held-out sessions. Each person model is
the four-stage dish fit (configs/drift_synth/hsmm.yaml), then the global prior is added post-fit
at strength lambda (apply_global_prior; lambda = 0 is the person-only model, "global only" drops
the person's own init/transition counts). Injected errors are placed once per person (by its
largest-N person-only model), so every arm and N is scored on the same trials. Hindsight's alpha
is picked per arm on validation by best G-F1 and reported on test. Recognizer noise is off.

    ./py scripts/eval/eval_synth_prior.py --stage population
    ./py scripts/eval/eval_synth_prior.py --stage persons
    ./py scripts/eval/eval_synth_prior.py --stage train [--person 0 --sessions 250]
    ./py scripts/eval/eval_synth_prior.py --stage eval [--person 0]

--variant cal is the CC4D-calibrated generator (configs/drift_synth/step_time_scale.json, events
capped at 600 s): 1 s ticks, d_max 300 (configs/drift_synth/hsmm_cal.yaml), training sizes 10 /
30 / 250 sessions and 20 soft-EM iterations (the objective moves < 0.2% after iteration 20 in 13 of
16 uncalibrated fits), to fit in a night. Everything goes to drift_synth_prior_cal/ and
runs/drift_synth/*_cal_*.json.
"""
import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.getcwd())

ROOT = Path("dataset/processed/drift_synth_prior")
CONFIG = "configs/drift_synth/hsmm.yaml"
POP_SEED, POP_N, POP_DAYS = 1000, 60, 30
SEED, PERSONS = 0, (0, 1, 2, 3)
N_DAYS = 320                                         # generated; every person has >= 402 sessions
VAL_START, TEST_START, TEST_END = 250, 325, 400      # session indices in cooking order
TRAIN_SESSIONS = (10, 30, 100, 250)
ORACLE_DAYS = 3000                                   # healthy-support sample: days N_DAYS..2999
ORACLE_VERSION = 2                                   # synth_oracle.VERSION the cache was built with
LAMBDAS = (0.0, 1.0, 3.0, 10.0, 30.0, 100.0)
GLOBAL_ONLY_LAMBDA = 30.0
CAPABILITY = {"slow": [0.4, 0.6], "fine_motor": [0.0, 0.4]}   # fine_motor below its hinge: no repeats
TICK = 0.5
SOFT_ITERS = 60
TAG = ""          # results-file suffix
VARIANT = "base"
POOL = "all"      # eval-split pools: all | real (oracle-confirmed only) | gen (step-level injectors)
POOL_ARMS = ("person only", "prior lambda=1")   # the arms run on the real / gen pools
STRUCT = set()    # --count-channel / --skip-idle: structural channels on one more grid axis
COUNT_LEVELS = (0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1)   # their shared p levels; 0 = off
DUR_SPLIT = False  # --dur-split: duration channels on their own alpha axis (harness.DUR_ALPHAS)
EMP_DUR = False    # --emp-dur: detectors/empirical_duration.py on its own axis
NP_HAZARD = None    # --np-hazard KAPPA BW: nonparametric duration hazard (hsmm/train/np_hazard.py)
CORE_COUNT = False  # --core-count: the detectors' s_count channel, in the transition alpha group
STRUCT_PRIOR = 0.0  # --struct-prior W: the prior arms' skip-idle channel gets W population pseudo-counts per row
EMP_LEVELS = (0.0, 0.002, 0.005, 0.01, 0.02)


def set_variant(name):
    global ROOT, CONFIG, TICK, TRAIN_SESSIONS, SOFT_ITERS, TAG, VARIANT
    VARIANT = name
    if name in ("cal", "cal_scrub"):
        ROOT = Path("dataset/processed/drift_synth_prior_cal")
        CONFIG = "configs/drift_synth/hsmm_cal.yaml"
        TICK, TRAIN_SESSIONS, SOFT_ITERS, TAG = 1.0, (10, 30, 250), 20, "_cal"
    if name == "cal_scrub":          # idle ticks dropped before the model (data/scrub.py, §9.8)
        ROOT = Path("dataset/processed/drift_synth_prior_cal_scrub")
        TAG = "_calscrub"


def _assets(person_cfg):
    from cook_ad.drift_synth.assets import load_assets
    a = load_assets()
    a.gen["recognizer"] = {k: (v if k == "min_event_s" else 0.0) for k, v in a.gen["recognizer"].items()}
    a.gen["person"] = dict(a.gen["person"], **person_cfg)
    if VARIANT in ("cal", "cal_scrub"):
        from cook_ad.drift_synth.assets import load_step_scale
        a.gen["durations"] = dict(a.gen["durations"], step_scale="configs/drift_synth/step_time_scale.json",
                                  max_event_s=600.0)
        a.step_scale = load_step_scale(a.gen)
        a.gen["export"] = dict(a.gen["export"], tick_seconds=TICK)
    return a


SCRUB = False      # set by the cal_scrub variant


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
    """The person's sessions in cooking order (day, then session of the day)."""
    key = {e["trial_id"]: (e["day"], e["session_idx"]) for e in labels}
    return sorted(seqs, key=lambda s: key[s["trial_id"]])


def stage_population():
    from cook_ad.drift_synth.export import export
    from cook_ad.drift_synth.sampler import ConditionCfg, generate_person
    from cook_ad.hsmm.train.global_prior import count_global

    assets = _assets({"theta0": [0.05, 0.95], "theta0_by_knob": CAPABILITY, "habit": [0.0, 1.0]})
    cond = ConditionCfg("null", n_persons=POP_N, n_days=POP_DAYS)
    seqs, recipes, n = [], [], 0
    for i in range(POP_N):
        person, sessions, gt = generate_person(i, cond, POP_SEED, assets)
        out = ROOT / "population" / person.person_id
        export(sessions, out, assets.vocab, gt, assets.gen["export"])
        if SCRUB:
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
    cond = ConditionCfg("null", n_persons=1, n_days=N_DAYS)
    for i in PERSONS:
        person, sessions, gt = generate_person(i, cond, SEED, assets)
        d = ROOT / "persons" / person.person_id
        export(sessions, d / "null", assets.vocab, gt, assets.gen["export"])
        if SCRUB:
            _scrub_export(d / "null", assets.vocab, assets.gen["export"]["sil"])
        seqs = _in_order(json.loads((d / "null" / "sequences.json").read_text()),
                         json.loads((d / "null" / "labels.json").read_text()))
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
            ]
            with open(out / "train.log", "w") as log:
                for cmd in steps:
                    subprocess.run(["./py", *cmd], check=True, stdout=log, stderr=subprocess.STDOUT)
            print(f"p{i:02d} N={n}: trained", flush=True)


def stage_eval(persons, chunk_size=8):
    from cook_ad.xla_env import disable_gpu_autotuning
    disable_gpu_autotuning()
    import jax
    import numpy as np
    jax.config.update("jax_enable_x64", True)
    from cook_ad.core import OperatingPoint
    from cook_ad.data.config import load_config
    from cook_ad.detectors.hsmm_hindsight import HindsightHSMMDetector
    from cook_ad.detectors.thresholds import BadSets
    from cook_ad.eval import harness, pools
    from cook_ad.hsmm import params as hparams
    from cook_ad.hsmm.model import HSMM
    from cook_ad.hsmm.train.global_prior import apply_global_prior
    from cook_ad.hsmm.train.lexical_init import recipe_label_assignment

    d_max = load_config(CONFIG)["duration"]["d_max_ticks"]
    gcounts = json.loads((ROOT / "global_counts.json").read_text())
    results = {}
    for i in persons:
        pid = f"p{i:02d}"
        d = ROOT / "persons" / pid
        labels = json.loads((d / "null" / "labels.json").read_text())
        seqs = _in_order(json.loads((d / "null" / "sequences.json").read_text()), labels)
        _, names = recipe_label_assignment(seqs, labels)
        placement = hparams.load(str(d / f"model_s{max(TRAIN_SESSIONS)}" / "model.npz"))

        def pool(lo, hi):
            part = seqs[lo:hi]
            entries, keep = pools.injected_per_trial(placement, part, d_max, seed=SEED,
                                                     chunk_size=chunk_size)
            streams = pools.per_trial_to_streams(entries)
            dish = recipe_label_assignment([part[k] for k in keep], labels)[0]
            return harness.trials_from_streams(streams, dish)

        val, test = pool(VAL_START, TEST_START), pool(TEST_START, TEST_END)

        def masks(det, trials, grid):
            out = []
            for group in dict.fromkeys(g for g, _ in trials):
                ts = [t for g, t in trials if g == group]
                if grid:
                    out += [harness.hindsight_grid(det, tr, harness.ALPHAS) for tr in det.traces(ts)]
                else:
                    out += [harness.alarm_mask(x, len(t.verbs)) for x, t in zip(det.detect_many(ts), ts)]
            return out

        results[pid] = {}
        for n in TRAIN_SESSIONS:
            base = hparams.load(str(d / f"model_s{n}" / "model.npz"))
            arms = {f"lambda={lam:g}": apply_global_prior(base, gcounts, names, lam) for lam in LAMBDAS}
            arms["global only"] = apply_global_prior(base, gcounts, names, GLOBAL_ONLY_LAMBDA, replace=True)
            results[pid][n] = {}
            for arm, p in arms.items():
                hsmm = HSMM(p, d_max)
                bad = BadSets(hsmm, harness.ALPHAS)
                det = HindsightHSMMDetector(hsmm, bad, OperatingPoint(0.005), chunk_size=chunk_size)
                grid = harness.sweep(masks(det, val, True), val, TICK)
                idx = harness.pick(grid)
                alpha = harness.ALPHAS[idx[0]]
                det = HindsightHSMMDetector(hsmm, bad, OperatingPoint(alpha), chunk_size=chunk_size)
                row = harness.score_masks(masks(det, test, False), test, TICK)
                row["alpha"] = alpha
                results[pid][n][arm] = row
                print(f"{pid} N={n:3d} {arm:12s} alpha={alpha:<7g} F1={row['f1']:.3f} "
                      f"P={row['precision']:.3f} R={row['recall']:.3f} "
                      f"healthy-trial FPR={row['healthy_trial_fpr']:.2f} "
                      f"alarms/10min={row['alarms_per_10min']:.2f}", flush=True)
        harness.dump(f"runs/drift_synth/eval_synth_prior{TAG}_{pid}.json",
                     {"metric": "G", "person": pid, "unit": "sessions", "results": results[pid]})
    return results


FADE_ARMS = {  # arm -> (lam, n0); every arm picks the transition alpha separately
    "person only": (0.0, None),
    "prior lambda=1": (1.0, None),
    "fade lambda0=10 n0=3": (10.0, 3.0),
    "fade lambda0=10 n0=10": (10.0, 10.0),
    "fade lambda0=30 n0=3": (30.0, 3.0),
}


_GRAPHS = {}
GEN_CONFIRMED = {}
_POP = []


def _population():
    """(sequences, recipe names) of the prior's population (stage_population's export)."""
    if not _POP:
        seqs, recs = [], []
        for d in sorted((ROOT / "population").iterdir()):
            lab = {e["trial_id"]: e["recipe_label"] for e in json.loads((d / "labels.json").read_text())}
            for s in json.loads((d / "sequences.json").read_text()):
                seqs.append(s)
                recs.append(lab[s["trial_id"]])
        _POP.extend([seqs, recs])
    return _POP[0], _POP[1]


def gen_pool(person_idx, part, labels, support):
    """Groups of (trajectory, dish) from the step-level injectors (drift_synth/inject.py): each
    held-out session with one real error edited into its clean events, re-exported to ticks.
    Prints the fraction the healthy-support oracle confirms."""
    import numpy as np
    from cook_ad.drift_synth.export import to_ticks
    from cook_ad.data.scrub import map_spans, scrub_idle
    from cook_ad.drift_synth.inject import ERROR_TYPES as GEN_TYPES, event_spans, inject
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
    seen = {}
    for r, rng_ in support.count_range.items():
        toks = [t for t in rng_ if rng_[t][1] > 0]
        seen[r] = ({iv[v] for v, _ in toks}, {inn[n] for _, n in toks})
    rng = np.random.default_rng([SEED, person_idx, int(part[0]["day"]) if "day" in part[0] else 0, len(part)])
    groups = {"healthy": [(s, r) for s, r in zip(part, dish)]}
    confirmed = {}
    for et in GEN_TYPES:
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
            c = support.check(name, v, n)              # the oracle reads the stream before scrubbing
            if SCRUB:
                v, n, kept = scrub_idle(v, n, (vocab["verbs"][ex["sil"][0]], vocab["nouns"][ex["sil"][1]]))
                spans = map_spans(spans, kept)
            if not spans:
                continue
            groups[et].append(({"verb_ids": v, "noun_ids": n, "spans": spans}, r))
            confirmed.setdefault(et, []).append((c["anomalous"], c["anomalous_or_order"]))
    print("  step-level injections confirmed by the oracle (local checks / + order): "
          + ", ".join(f"{et} {np.mean([x[0] for x in c]):.2f} / {np.mean([x[1] for x in c]):.2f} (n={len(c)})"
                      for et, c in confirmed.items()), flush=True)
    GEN_CONFIRMED[person_idx] = {et: [float(np.mean([x[0] for x in c])), float(np.mean([x[1] for x in c]))]
                                 for et, c in confirmed.items()}
    return harness_trials(groups)


def harness_trials(groups):
    from cook_ad.eval import harness
    return harness.trials_from_groups(groups)


def stage_eval_split(persons, chunk_size=8):
    """Every arm with a split operating point: the transition channels' alpha picked separately
    from the other channels' (harness.hindsight_split_grid), both on validation by G-F1."""
    from cook_ad.xla_env import disable_gpu_autotuning
    disable_gpu_autotuning()
    import jax
    jax.config.update("jax_enable_x64", True)
    from cook_ad.core import OperatingPoint
    from cook_ad.data.config import load_config
    from cook_ad.detectors.hsmm_hindsight import HindsightHSMMDetector
    from cook_ad.detectors.thresholds import BadSets
    from cook_ad.eval import harness, pools
    from cook_ad.hsmm import params as hparams
    from cook_ad.hsmm.model import HSMM
    from cook_ad.hsmm.train.global_prior import apply_global_prior, count_global
    from cook_ad.hsmm.train.lexical_init import recipe_label_assignment
    import numpy as np

    d_max = load_config(CONFIG)["duration"]["d_max_ticks"]
    gcounts = json.loads((ROOT / "global_counts.json").read_text())
    for i in persons:
        pid = f"p{i:02d}"
        d = ROOT / "persons" / pid
        labels = json.loads((d / "null" / "labels.json").read_text())
        seqs = _in_order(json.loads((d / "null" / "sequences.json").read_text()), labels)
        rec = {e["trial_id"]: e["recipe_label"] for e in labels}
        _, names = recipe_label_assignment(seqs, labels)
        placement = hparams.load(str(d / f"model_s{max(TRAIN_SESSIONS)}" / "model.npz"))

        support = None
        if POOL in ("real", "gen"):
            from cook_ad.eval.synth_oracle import person_support
            vocab = json.loads((d / "null" / "vocab.json").read_text())
            support = person_support(i, SEED, _assets({"theta0_by_knob": CAPABILITY, "habit": 2.0}), vocab,
                                     N_DAYS, ORACLE_DAYS, TICK, cache=d / f"oracle_support_v{ORACLE_VERSION}.pkl")

        def pool(lo, hi):
            part = seqs[lo:hi]
            if POOL == "gen":
                return gen_pool(i, part, labels, support)
            entries, keep = pools.injected_per_trial(placement, part, d_max, seed=SEED,
                                                     chunk_size=chunk_size)
            dish = recipe_label_assignment([part[k] for k in keep], labels)[0]
            names_k = [rec[part[k]["trial_id"]] for k in keep]
            groups = {"healthy": [(h, r) for (h, _), r in zip(entries, dish)]}
            for et in pools.ERROR_TYPES:
                items = [(inj[et], r, nm) for (_, inj), r, nm in zip(entries, dish, names_k)]
                if POOL == "real":
                    items = [x for x in items
                             if support.check(x[2], x[0]["verb_ids"], x[0]["noun_ids"])["anomalous"]]
                groups[et] = [(t, r) for t, r, _ in items]
            return harness.trials_from_groups(groups)

        val, test = pool(VAL_START, TEST_START), pool(TEST_START, TEST_END)
        sizes = {part: dict(Counter(g for g, _ in tr)) for part, tr in (("val", val), ("test", test))}
        print(f"{pid} pool sizes: {sizes}", flush=True)
        arms = FADE_ARMS if POOL == "all" else {k: FADE_ARMS[k] for k in POOL_ARMS}
        out = {}
        for n in TRAIN_SESSIONS:
            base = hparams.load(str(d / f"model_s{n}" / "model.npz"))
            train = seqs[:n]
            person = count_global(train, [rec[s["trial_id"]] for s in train])
            out[n] = {}
            for arm, (lam, n0) in arms.items():
                p = apply_global_prior(base, gcounts, names, lam, person_counts=person if n0 else None,
                                       n0=n0)
                hsmm = HSMM(p, d_max)
                if NP_HAZARD:       # nonparametric hazard from this model's decode of its training
                    from cook_ad.hsmm.train.np_hazard import fit_np_hazard
                    from cook_ad.synthetic import generate
                    tr_traj = generate.trajectories_from_real_joint(
                        p, train, d_max, chunk_size=chunk_size, r_hat=recipe_label_assignment(train, labels)[0])
                    k_r, k_s = np.asarray(p.init_counts).shape
                    hsmm = HSMM(p, d_max, duration_tables=fit_np_hazard(
                        tr_traj, k_r, k_s, d_max, np.asarray(hsmm.tables.log_h), kappa=NP_HAZARD[0],
                        kappa0=2.0, bandwidth=int(NP_HAZARD[1])))
                bad = BadSets(hsmm, harness.ALPHAS)
                det = HindsightHSMMDetector(hsmm, bad, OperatingPoint(0.005), chunk_size=chunk_size)

                axes = []           # [(name, [channels], levels)]: one more grid axis each
                if STRUCT or EMP_DUR or CORE_COUNT:
                    from cook_ad.core import Trial
                    from cook_ad.detectors.count_channel import CountChannel
                    from cook_ad.detectors.empirical_duration import EmpiricalShortDuration
                    from cook_ad.detectors.skip_idle import SkipIdleTransitions
                    vocab_ = json.loads((d / "null" / "vocab.json").read_text())
                    idle = set(np.flatnonzero(np.argmax(np.asarray(hsmm.tables.log_emit_v), axis=-1)
                                              == vocab_["verbs"]["stall"]).tolist())
                    tr_dish = recipe_label_assignment(train, labels)[0]
                    tr_trials = [Trial(s_["trial_id"], np.asarray(s_["verb_ids"]), np.asarray(s_["noun_ids"]),
                                       int(r_), (), "healthy") for s_, r_ in zip(train, tr_dish)]
                    tr_traces = det.traces(tr_trials)
                    if CORE_COUNT:      # the detector's own s_count channel (hsmm/visit_counts.py)
                        from cook_ad.hsmm.visit_counts import VisitCounts
                        vc = VisitCounts.fit([(t.r_hat, [k for k, _ in t.segments]) for t in tr_traces],
                                             ignore=idle)
                        det = HindsightHSMMDetector(hsmm, bad, OperatingPoint(0.005), chunk_size=chunk_size,
                                                    counts=vc)
                    if STRUCT:
                        chans = []
                        for c in sorted(STRUCT):
                            if c == "skip" and STRUCT_PRIOR and lam > 0:
                                from cook_ad.detectors.skip_idle import population_prior
                                from cook_ad.hsmm.train.global_prior import state_pairs
                                pop_s, pop_r = _population()
                                pri = population_prior(pop_s, pop_r, state_pairs(p), names,
                                                       (vocab_["verbs"]["stall"], vocab_["nouns"]["idle"]))
                                chans.append(SkipIdleTransitions.fit(tr_traces, ignore=idle, prior=pri,
                                                                     prior_weight=STRUCT_PRIOR))
                            else:
                                chans.append({"count": CountChannel, "skip": SkipIdleTransitions}[c].fit(
                                    tr_traces, ignore=idle))
                        axes.append(("struct", chans, COUNT_LEVELS))
                    if EMP_DUR:
                        axes.append(("empdur", [EmpiricalShortDuration.fit(tr_traces, ignore=idle)], EMP_LEVELS))

                def grid(trials):
                    from cook_ad.eval import metric_g
                    score = metric_g.GridScore(TICK)     # accumulated per trial: grids get large
                    for group in dict.fromkeys(g for g, _ in trials):
                        ts = [t for g, t in trials if g == group]
                        for t_, tr in zip(ts, det.traces(ts)):
                            if DUR_SPLIT:
                                g2 = harness.hindsight_split3_grid(det, tr, harness.ALPHAS)
                            else:
                                g2 = harness.hindsight_split_grid(det, tr, harness.ALPHAS)
                            for _, chans, lv in axes:
                                c_ = np.any(np.stack([c.flags(tr, lv) for c in chans]), axis=0)
                                g2 = g2[..., None] | c_.reshape((c_.shape[0],) + (1,) * (g2.ndim - 1) + (c_.shape[1],))
                            score.add(group, g2, list(t_.spans), t_.verbs, t_.nouns)
                    return score.result()
                gv, gt = grid(val), grid(test)
                if DUR_SPLIT or axes:
                    # one pick over every axis; the plain rows come from runs without these options
                    idx = harness.pick(gv)
                    row = harness.at(gt, idx)
                    row.update(alpha_transition=harness.ALPHAS[idx[0]], alpha_rest=harness.ALPHAS[idx[1]])
                    k = 2
                    if DUR_SPLIT:
                        row["alpha_duration"] = harness.DUR_ALPHAS[idx[2]]
                        k = 3
                    for name, _, lv in axes:
                        row[f"{name}_level"] = lv[idx[k]]
                        k += 1
                    label = arm + (" +dur" if DUR_SPLIT else "") + "".join(f" +{c}" for c in sorted(STRUCT)) \
                        + (" +empdur" if EMP_DUR else "")
                    out[n][label] = row
                    print(f"{pid} N={n:3d} {label:34s} F1={row['f1']:.3f} P={row['precision']:.3f} "
                          f"R={row['recall']:.3f} FPR={row['healthy_trial_fpr']:.2f} levels "
                          + " ".join(f"{k_}={v}" for k_, v in row.items() if k_.endswith(("_level", "alpha_duration")))
                          + " | per type " + " ".join(f"{k_[:5]} {v:.2f}" for k_, v in row["per_type"].items()),
                          flush=True)
                    continue
                idx = harness.pick(gv)
                row = harness.at(gt, idx)
                row["alpha_transition"] = harness.ALPHAS[idx[0]]
                row["alpha_rest"] = harness.ALPHAS[idx[1]]
                cc = (" +s_count" if CORE_COUNT else "") + (f" +np(k={NP_HAZARD[0]:g},bw={NP_HAZARD[1]:g})" if NP_HAZARD else "")
                out[n][arm + cc] = row
                # the single-alpha pick is the grid's diagonal (same alpha for every channel)
                diag = [float(np.asarray(gv["f1"])[j, j]) for j in range(len(harness.ALPHAS))]
                j = int(np.argmax(diag))
                single = harness.at(gt, (j, j))
                single["alpha_transition"] = single["alpha_rest"] = harness.ALPHAS[j]
                out[n][arm + cc + " (single alpha)"] = single
                print(f"{pid} N={n:3d} {arm + cc:22s} a_trans={row['alpha_transition']:<7g} "
                      f"a_rest={row['alpha_rest']:<7g} F1={row['f1']:.3f} P={row['precision']:.3f} "
                      f"R={row['recall']:.3f} FPR={row['healthy_trial_fpr']:.2f} "
                      f"alarms/10min={row['alarms_per_10min']:.2f}", flush=True)
        harness.dump(f"runs/drift_synth/eval_synth_prior_split{TAG}_{pid}.json",
                     {"metric": "G", "person": pid, "unit": "sessions", "pool": POOL,
                      "pool_sizes": sizes, "arms": {k: list(v) for k, v in arms.items()},
                      "results": out})


def _unit(pattern):
    """Training-size unit of the results files: "sessions", or "days" for files written before
    the switch to session counts (the base variant at f590c55)."""
    files = sorted(Path("runs/drift_synth").glob(pattern))
    return json.loads(files[0].read_text()).get("unit", "days") if files else "sessions"


def summarize_split():
    import numpy as np
    unit = _unit(f"eval_synth_prior_split{TAG}_p*.json")
    rows = {}
    for f in sorted(Path("runs/drift_synth").glob(f"eval_synth_prior_split{TAG}_p*.json")):
        for n, arms in json.loads(f.read_text())["results"].items():
            for arm, r in arms.items():
                rows.setdefault((int(n), arm), []).append(r)
    print("\n### Split alpha (transition channels picked separately), mean over persons, test "
          f"{unit}, metric G (Hindsight)\n")
    print(f"| train {unit} | arm | persons | F1 | precision | recall | alarms / 10 min | healthy-trial FPR "
          "| subst. | aband. | omission | transp. | repet. |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    order = [x for a in FADE_ARMS for x in (a + " (single alpha)", a)]
    for (n, arm), rs in sorted(rows.items(), key=lambda kv: (kv[0][0], order.index(kv[0][1])
                                                              if kv[0][1] in order else 99)):
        m = lambda k: np.mean([r[k] for r in rs])  # noqa: E731
        t = lambda k: np.mean([r["per_type"][k] for r in rs])  # noqa: E731
        print(f"| {n} | {arm} | {len(rs)} | {m('f1'):.3f} | {m('precision'):.3f} | {m('recall'):.3f} | "
              f"{m('alarms_per_10min'):.2f} | {m('healthy_trial_fpr'):.2f} | {t('substitution'):.2f} | "
              f"{t('abandonment'):.2f} | {t('omission'):.2f} | {t('transposition'):.2f} | "
              f"{t('repetition'):.2f} |")


def summarize():
    import numpy as np
    unit = _unit(f"eval_synth_prior{TAG}_p*.json")
    rows = {}
    for f in sorted(Path("runs/drift_synth").glob(f"eval_synth_prior{TAG}_p*.json")):
        res = json.loads(f.read_text())["results"]
        for n, arms in res.items():
            for arm, r in arms.items():
                rows.setdefault((int(n), arm), []).append(r)
    print(f"\n### Global prior + fine-tune, mean over persons, test {unit}, metric G (Hindsight)\n")
    print(f"| train {unit} | arm | persons | F1 | precision | recall | alarms / 10 min | healthy-trial FPR "
          "| subst. | aband. | omission | transp. | repet. |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for (n, arm), rs in sorted(rows.items(), key=lambda kv: (kv[0][0], list(rows).index(kv[0]))):
        m = lambda k: np.mean([r[k] for r in rs])  # noqa: E731
        t = lambda k: np.mean([r["per_type"][k] for r in rs])  # noqa: E731
        print(f"| {n} | {arm} | {len(rs)} | {m('f1'):.3f} | {m('precision'):.3f} | {m('recall'):.3f} | "
              f"{m('alarms_per_10min'):.2f} | {m('healthy_trial_fpr'):.2f} | {t('substitution'):.2f} | "
              f"{t('abandonment'):.2f} | {t('omission'):.2f} | {t('transposition'):.2f} | "
              f"{t('repetition'):.2f} |")


def plot(path=None):
    """Cold-start curves: F1, healthy-trial FPR and recall vs training sessions, per person (thin)
    and mean (thick), for person-only, + prior (lambda = 1) and global-only."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    arms = [("lambda=0", "person only", "#2a78d6"), ("lambda=1", "+ global prior (λ = 1)", "#eb6834"),
            ("global only", "global only", "#1baf7a")]
    path = path or f"runs/drift_synth/reports/prior_curves{TAG}.png"
    unit = _unit(f"eval_synth_prior{TAG}_p*.json")
    res = {f.stem.split("_")[-1]: json.loads(f.read_text())["results"]
           for f in sorted(Path("runs/drift_synth").glob(f"eval_synth_prior{TAG}_p*.json"))}
    sizes = sorted(int(n) for n in next(iter(res.values())))
    ink, ink2, grid = "#1a1a19", "#5f5e58", "#e4e3dd"
    panels = [("f1", "F1 (injected errors)"), ("healthy_trial_fpr", "healthy-trial FPR"),
              ("recall", "recall")]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.4), constrained_layout=True)
    for ax, (key, title) in zip(axes[:3], panels):
        ax.grid(True, color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for arm, label, color in arms:
            per = np.array([[res[p][str(n)][arm][key] for n in sizes] for p in sorted(res)])
            for row in per:
                ax.plot(sizes, row, color=color, linewidth=0.8, alpha=0.35)
            ax.plot(sizes, per.mean(axis=0), color=color, linewidth=2.2, marker="o", markersize=5,
                    label=label)
        ax.set_xscale("log")
        ax.set_xticks(sizes, [str(n) for n in sizes])
        ax.tick_params(colors=ink2, labelsize=8)
        ax.set_title(title, color=ink, fontsize=10, loc="left")
        ax.set_xlabel(f"person's own training {unit}", color=ink2, fontsize=9)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=ink)
    # per person: F1(+prior) - F1(person only); hollow = the prior arm's validation pick left
    # the strict end of the alpha grid (alpha > 1e-7)
    ax = axes[3]
    ax.grid(True, color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.axhline(0, color=ink2, linewidth=1)
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    for color, pid in zip(colors, sorted(res)):
        delta = [res[pid][str(n)]["lambda=1"]["f1"] - res[pid][str(n)]["lambda=0"]["f1"] for n in sizes]
        loose = [res[pid][str(n)]["lambda=1"]["alpha"] > 1e-7 for n in sizes]
        ax.plot(sizes, delta, color=color, linewidth=1.6)
        for x, y, l in zip(sizes, delta, loose):
            ax.plot(x, y, marker="o", markersize=7, color=color, markerfacecolor="white" if l else color,
                    markeredgewidth=2)
        ax.text(sizes[-1] * 1.08, delta[-1], pid, color=ink2, fontsize=8, va="center")
    ax.plot([], [], marker="o", color=ink2, linestyle="none", markerfacecolor="white",
            label="prior arm picked a loose α (> 1e-7)")
    ax.plot([], [], marker="o", color=ink2, linestyle="none", label="picked α ≤ 1e-7")
    ax.legend(frameon=False, fontsize=8, labelcolor=ink, loc="lower left")
    ax.set_xscale("log")
    ax.set_xticks(sizes, [str(n) for n in sizes])
    ax.set_xlim(sizes[0] * 0.8, sizes[-1] * 1.5)
    ax.tick_params(colors=ink2, labelsize=8)
    ax.set_title("F1 change from the prior, per person", color=ink, fontsize=10, loc="left")
    ax.set_xlabel(f"person's own training {unit}", color=ink2, fontsize=9)
    fig.suptitle(f"Global prior + fine-tune, {len(res)} synthetic persons (thin) and mean (thick); "
                 f"Hindsight, test {unit}, metric G", color=ink, fontsize=11, x=0.01, ha="left")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


def plot_split(path=None):
    """Mean over persons vs training sessions for the split-alpha arms (and person-only, single alpha)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    path = path or f"runs/drift_synth/reports/prior_split_curves{TAG}.png"
    unit = _unit(f"eval_synth_prior_split{TAG}_p*.json")
    res = {f.stem.split("_")[-1]: json.loads(f.read_text())["results"]
           for f in sorted(Path("runs/drift_synth").glob(f"eval_synth_prior_split{TAG}_p*.json"))}
    if not res:
        raise SystemExit("no split results")
    sizes = sorted(int(n) for n in next(iter(res.values())))
    palette = ["#5f5e58", "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    arms = [a for a in ["person only (single alpha)", *FADE_ARMS]
            if all(a in res[p][str(n)] for p in res for n in sizes)]
    ink, ink2, grid = "#1a1a19", "#5f5e58", "#e4e3dd"
    panels = [("f1", "F1 (injected errors)"), ("healthy_trial_fpr", "healthy-trial FPR"), ("recall", "recall")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)
    for ax, (key, title) in zip(axes, panels):
        ax.grid(True, color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for color, arm in zip(palette, arms):
            m = [np.mean([res[p][str(n)][arm][key] for p in res]) for n in sizes]
            ax.plot(sizes, m, color=color, linewidth=2.2 if "single" not in arm else 1.4,
                    linestyle="--" if "single" in arm else "-", marker="o", markersize=5,
                    label=arm + ("" if "single" in arm else ", split alpha"))
        ax.set_xscale("log")
        ax.set_xticks(sizes, [str(n) for n in sizes])
        ax.tick_params(colors=ink2, labelsize=8)
        ax.set_title(title, color=ink, fontsize=10, loc="left")
        ax.set_xlabel(f"person's own training {unit}", color=ink2, fontsize=9)
    axes[0].legend(frameon=False, fontsize=7.5, labelcolor=ink)
    fig.suptitle(f"Fading prior + split alpha, mean of {len(res)} persons{' (CC4D-calibrated)' if TAG else ''}; "
                 f"Hindsight, test {unit}, metric G", color=ink, fontsize=11, x=0.01, ha="left")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["population", "persons", "train", "eval", "summary", "plot",
                                        "eval-split", "summary-split", "plot-split"],
                    required=True)
    ap.add_argument("--person", type=int, action="append")
    ap.add_argument("--sessions", type=int, action="append")
    ap.add_argument("--variant", choices=["base", "cal", "cal_scrub"], default="base")
    ap.add_argument("--pool", choices=["all", "real", "gen"], default="all",
                    help="eval-split pools: the Breakfast injectors (all), only their oracle-confirmed "
                         "anomalies (real), or the step-level injectors (gen); real/gen run 2 arms")
    ap.add_argument("--dur-split", action="store_true",
                    help="eval-split: duration channels on their own alpha axis")
    ap.add_argument("--np-hazard", type=float, nargs=2, metavar=("KAPPA", "BW"),
                    help="eval-split: nonparametric duration hazard, shrinkage kappa and kernel half-width")
    ap.add_argument("--core-count", action="store_true",
                    help="eval-split: the detectors' own s_count channel (fitted on the training decodes)")
    ap.add_argument("--struct-prior", type=float, default=0.0,
                    help="eval-split: population pseudo-counts per row for the prior arms' skip-idle channel")
    ap.add_argument("--emp-dur", action="store_true",
                    help="eval-split: also the empirical short-duration channel, on its own axis")
    ap.add_argument("--skip-idle", action="store_true",
                    help="eval-split: also the idle-skipping transition channel (detectors/skip_idle.py)")
    ap.add_argument("--count-channel", action="store_true",
                    help="eval-split: also the visit-count channel (detectors/count_channel.py)")
    a = ap.parse_args()
    set_variant(a.variant)
    global POOL, TAG, STRUCT, DUR_SPLIT, EMP_DUR, STRUCT_PRIOR, SCRUB, CORE_COUNT, NP_HAZARD
    SCRUB = a.variant == "cal_scrub"
    if SCRUB and a.pool == "real":
        raise SystemExit("--pool real reads the tick injectors' streams before scrubbing; use gen")
    POOL = a.pool
    TAG += "" if POOL == "all" else f"_{POOL}"
    STRUCT = ({"count"} if a.count_channel else set()) | ({"skip"} if a.skip_idle else set())
    TAG += "".join(f"_{c}" for c in sorted(STRUCT))
    DUR_SPLIT = a.dur_split
    TAG += "_dur" if DUR_SPLIT else ""
    EMP_DUR = a.emp_dur
    TAG += "_empdur" if EMP_DUR else ""
    STRUCT_PRIOR = a.struct_prior
    TAG += f"_sp{STRUCT_PRIOR:g}" if STRUCT_PRIOR else ""
    CORE_COUNT = a.core_count
    TAG += "_scount" if CORE_COUNT else ""
    NP_HAZARD = tuple(a.np_hazard) if a.np_hazard else None
    TAG += f"_np{NP_HAZARD[0]:g}bw{NP_HAZARD[1]:g}" if NP_HAZARD else ""
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
    elif a.stage == "plot":
        plot()
    elif a.stage == "eval-split":
        stage_eval_split(persons)
        summarize_split()
    elif a.stage == "summary-split":
        summarize_split()
    elif a.stage == "plot-split":
        plot_split()
    else:
        summarize()


if __name__ == "__main__":
    main()
