# Evaluation

How detectors are scored: metric H (the headline), metric G (the previous headline, for comparing
with older numbers), metric E (only for comparing with PREGO), how operating points are chosen, and
the scripts that produce every table.

**Every table states its metric letter.** A number without one is not citable.

| Letter | Scorer | Use it for |
|---|---|---|
| **H** | `eval/metric_h.py` | the headline, every corpus, at delay 2 (`H, delay 2`) |
| **G** | `eval/metric_g.py` | numbers before 2026-10-21, and `COOK_METRIC=G` to compare with them |
| **E** | `eval/metric_e.py` | only the comparison with published PREGO / TI-PREGO numbers |

Older letters (A `trial_loc`, B step verdicts, D span-any-flag, F counterfactual, G-gap) appear only
in [`logbook.md`](logbook.md) entries; their code was deleted in the restructure.

---

## 1. Metric H

A **run** is a maximal stretch of ticks with one constant (verb, noun). Every detector observes
runs directly, and they make no claim about the user's steps. Say "run", not "step".

- **TP**: a ground-truth span whose window [start, end + delay] contains any flag. At most one per
  span. The headline delay is 2 ticks (`metric_h.DEFAULT_DELAY`).
- **FN**: a span with no flag in its window.
- **FP (a miss)**: a run that holds a flag outside every span's window, one per run however many
  flags it holds.
- **TN**: undefined.

Precision, recall and F1 are pooled over spans; recall is also reported per error type. A span
already covers every tick the error touched; the short delay admits a retrospective flag raised
just after it (an abandonment is visible only once the run closes).

**Two false-alarm numbers go with every table**, measured on error-free activity:

- **alarms per 10 minutes**: how often the user would be interrupted. An alarm is a contiguous
  flagged stretch inside one run, so a flickering channel is charged for every flicker;
- **run FPR**: the share of runs that overlap no mistake but contain any flag.

Tables also print the healthy-trial FPR (share of healthy trials with any alarm). It is never
folded into F1: it is a different claim on a different denominator.

**Why H.** It credits what the detector did right and charges what it did wrong, separately: a
stray flag early in a run and then a flag on the error are one hit and one miss. Under G the run's
alarm is dated by its first flag, so the same case is a false positive *and* a false negative.
**Nulls:** flag every tick hits every span and misses every run it does not lie wholly inside;
flag every run (one flag at each run's first tick) is printed beside it. Neither can be beaten by
flagging more, because misses are counted per run, not per flag.

**Implementation.** `metric_h.score_trial` and `evaluate` score lists of `Flag(tick, strength,
channel)`; `GridScore` scores dense (T, P) flag matrices, P operating points at once, which is what
sweeps use. `evaluate` takes every span of a trial at once: passing the same flags once per error
type would charge a flag that hit a transposition as a false positive in the substitution entry.
Alarms per 10 minutes need the tick length: `GridScore(tick_seconds)`, default 1.0 (Breakfast,
EPIC-Tent, the synthetic data). Pass 0.5 for EPIC-KITCHENS, or the rate reads half the truth.
`harness.SCORER` is the metric every eval script uses: H unless `COOK_METRIC=G`; `COOK_DELAY` moves
H's delay. The table header says which (`H, delay 2`).

## 1b. Metric G (the headline until 2026-10-21)

Same runs and spans. An **alarm** is the first flag inside a run; later flags in the same run belong
to it. A TP is a span whose window [start, end + 10] contains an alarm *onset*; an FP is an onset
in no span's window. Only onsets count, so flag-every-tick and flag-every-run score identically.
It was adopted over earlier metrics that let one never-ending alarm find every span (D, G-gap) or
assumed one anomaly per trial (A, B, E). Numbers under G and H are close (H at delay 10 is within
0.004 F1 of G; delay 2 is 0.01–0.03 lower), and no comparison measured so far changes rank between
them ([`logbook.md`](logbook.md), 2026-10-19 to 2026-10-21).

---

## 2. Metric E: PREGO's protocol

Used only to compare with PREGO (CVPR'24) and TI-PREGO. It reproduces their scorer
(`step_anticipation/src/models/llm_hf.py`):

```python
correct = matches[:-1]      # every step before the last is a negative
mistake = matches[-1]       # the last step of each trimmed sequence is THE mistake
```

Each EPIC-Tent test session is cut after its first in-group mistake, **every detector is re-run on
the cut stream**, and the collapsed action sequence (idle removed, repeats merged) is scored step by
step. That gives one positive per session. "Step" is their term. Run with `eval_epic_tent.py
--trim`. The LLM's under-run alarm is suppressed on the cut final step, which never closed.

Two precisions are in circulation. PREGO reports *classical* precision (≤ 0.107 on Epic-tent-O).
TI-PREGO reports *balanced* precision, Prec_b = TPR / (TPR + FPR), under which chance scores 0.5
and flagging every step scores Prec_b 0.500, F1_b 0.667: above every Epic-tent-O row TI-PREGO
reports. Their balanced figures cannot be reproduced exactly: at recall 1.0, Prec_b = 0.407 would
need more false positives than negative steps. Both columns are printed.

**What it shows.** Flagging every step scores precision 0.089–0.108 at recall 1.0. PREGO's
published 0.107 / 0.867 implies they flag ~71% of correct steps; it sits on that null. Our sequences
match PREGO's published `gt` exactly on 6 of 15 sessions (most others end on the same mistake 1–3
steps shorter, from first-come-wins overlap resolution); the null is also computed on their own
`gt`.

---

## 3. The harness and operating points

`eval/harness.py` holds what both eval scripts share:

- `trials_from_streams` turns a pool into `(group, Trial)` pairs;
- `hindsight_grid` and `live_grid` threshold one trace at every grid point: α ∈ `ALPHAS` (21
  values from 0.5 to 1e-10, dense between 0.1 and 1e-3), and for Live/Lag also κ ∈ `KAPPAS`
  (0.9 … 0.02). Traces are computed once; thresholding is cheap;
- `sweep` scores the grid under `SCORER`; `pick` chooses an index (best F1, or best recall within an
  alarm budget); `at` reads one row;
- `null_masks` builds the two nulls; `table` prints a metric-labelled markdown table;
  `write_operating_points` merges picks into `configs/operating_points.yaml`.

**Operating points are picked on train and reported on test, never picked on the split they are
reported on.** Breakfast picks each model's α (and κ) by best train F1. Two cautions, both
measured:

- *Train alarm budgets do not hold on test*: ≤ 1 alarm per 10 min on train lands at 1.4–1.8 on test,
  because healthy train trials are the ones the model was fitted to.
- *Regularisation is selected on a nested dev split*, not on train ([`training.md`](training.md)
  §3).

**EPIC-Tent is not picked.** Every training session carries annotated errors, and the HSMM is
fitted on the error-excised fragments, so there is no clean train set to pick on. It runs at the
fixed per-channel default: α = 0.005 for every channel, α = 1e-5 for `s_temporal`, κ = 0.5. On
Breakfast that `s_temporal` setting removed 37% of false-positive ticks for 0.006 recall (metric A,
picked on train and confirmed on test).

---

## 4. The trial figure

`scripts/figures/trial_figure.py` draws one Breakfast trial from the LLM pool with every detector:
the observed runs, the injected span, one lane per detector with a marker at each alarm, and the
Hindsight narration as numbered cards. LLM lanes are read from the cache (`--max-requests 0`
by default), so it costs nothing to redraw.

```bash
./py scripts/figures/trial_figure.py --trial 0 --condition omission --out runs/fig_trial0.png
```

---

## 5. The scripts

Every table in README is produced by these commands. LLM arms replay from the cache: each needs its
original `--base-url` and `--max-requests 0` ([`llm.md`](llm.md) §5).

```bash
# Breakfast: train picks, test grids, the headline table; writes the operating points
./py scripts/eval/eval_breakfast.py --max-requests 0 \
    --out runs/eval_breakfast.json --operating-points configs/operating_points.yaml

# EPIC-Tent, default split, every arm (LLM replay)
./py scripts/eval/eval_epic_tent.py --live --live-lags 0 --variant both --skip-preflight \
    --base-url http://localhost:11434/v1 --max-requests 0 --out runs/et_H_llm.json

# EPIC-Tent, PREGO's split, HSMM arms incl. lag
./py scripts/eval/eval_epic_tent.py --live --live-lags 0,5,inf --skip-llm \
    --split-file dataset/processed/epic_tent/split_prego.json \
    --joint-params dataset/processed/epic_tent/joint_params_clean_prego.npz --out runs/et_live_prego.json

# EPIC-Tent, PREGO's protocol (metric E), every arm
./py scripts/eval/eval_epic_tent.py --live --live-lags 0 --variant both --skip-preflight --trim \
    --split-file dataset/processed/epic_tent/split_prego.json \
    --joint-params dataset/processed/epic_tent/joint_params_clean_prego.npz \
    --base-url http://localhost:11434/v1 --max-requests 0 --out runs/et_E_llm_prego_trimmed.json

# Synthetic persons (drift_synth.md §7): data, models and every table, about 4 h
scripts/drift_synth/prior_experiment.sh
```

`eval_breakfast.py` scores three models (`--models dish-given,dish-inferred,lexhard`) and the
detectors `hsmm-hindsight, hsmm-live, hsmm-lag` (L = 5, ∞), `llm-predictive` (4 variants),
`llm-verdict` (2) and the nulls. On a fresh machine, or to collect new LLM responses, use the
wrappers `scripts/eval/llm_breakfast.sh` and `scripts/eval/llm_epic_tent.sh`, which manage the
ollama server ([`llm.md`](llm.md) §6).

`eval_epic_tent.py` also prints an **injected control**: the same HSMM on the same sessions with
injected errors (`pools.injected_per_trial`). Its healthy rate is not a false-alarm rate: the
"healthy" sessions still carry their real errors.

---

## 6. Reading a table

- **Read F1 against its null.** On EPIC-Tent 29% of test ticks lie inside some annotated span, and
  the nulls' F1 is 0.168–0.244 (metric H). On Breakfast flag-every-run scores 0.337 and
  flag-every-tick 0.301.
- **Read the alarm columns together.** An arm close to the null in F1 that raises ten times the
  alarms is not a detector a person would tolerate.
- **97 trials per group means about ±0.03 of noise** on F1. Differences below that are not
  findings.
- **A per-type recall with n = 3** (EPIC-Tent correction and repeat) is a count, not a rate.
- **Metric E's single-session differences are large.** One session separates Live (2/15) from
  Hindsight (3/15) under PREGO's protocol.

---

## 7. Code

| Concept | Where |
|---|---|
| metric G | `eval/metric_g.py:score_trial`, `evaluate`, `GridScore`, `run_starts`, `alarm_onsets`, `alarm_ticks`, `DEADLINE` |
| metric H | `eval/metric_h.py:score_dense`, `GridScore`, `evaluate`, `stretches`; selected by `harness.METRIC` (`COOK_METRIC`, `COOK_DELAY`) |
| metric E | `eval/metric_e.py:score`, `step_extents`, `drop_truncated_under_run` |
| sweeps, picks, tables | `eval/harness.py:hindsight_grid`, `live_grid`, `sweep`, `pick`, `at`, `null_masks`, `table`, `write_operating_points` |
| alarms → per-tick masks | `eval/harness.py:alarm_mask` |
| the picked operating points | `configs/operating_points.yaml` |
| pools | `eval/pools.py` ([`data.md`](data.md) §5) |
| Breakfast evaluation | `scripts/eval/eval_breakfast.py` |
| EPIC-Tent evaluation | `scripts/eval/eval_epic_tent.py` |
| the trial figure | `scripts/figures/trial_figure.py` |
| tests | `tests/test_metric_h.py`, `test_metric_g.py`, `test_prego_protocol.py`, `test_pools.py` |
