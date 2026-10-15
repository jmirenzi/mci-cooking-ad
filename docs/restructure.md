# Restructure plan

Status: **in progress — Phases A–E done** (tag `pre-restructure`; goldens in `runs/golden/`,
made by `scripts/verify/{dump_golden.py,run_tables.sh}`, checked by `compare_golden.py`; see
"Progress notes" at the end). Written 2026-10-14 with the repo owner. The agent doing the work
reads this file top to bottom before touching code, works on branch `restructure`, and deletes this
file in its last commit. A one-paragraph summary of the restructure goes into `docs/logbook.md`.

---

## 0. The goal and the one rule

The owner wants to understand three things exactly: **the HSMM at inference**, **how anomalies come
out of it**, and (later) **drift detection**. Today those are spread over `hsmm/`, `recipe/`,
`anomaly/`, `eval/batch.py` and five scripts that import each other. The restructure makes them
readable as a small set of classes:

- one `AnomalyDetector` base class;
- HSMM detectors and LLM detectors as subclasses of it;
- every detector returns the same `Alarm` objects.

**The rule: behaviour does not change.** Given the same input, every detector produces the same
alarms, the same log-likelihoods and the same LLM requests as `main` does today. Section 9 says how
that is proven. Changes to the numbers are allowed in one place only: Phase F (§10) switches the
evaluation to canonical pools, after equivalence is proven.

Areas the owner does **not** need to read: data parsing, training, plotting and the eval harness. Optimise
those for agents. Long comments and docstrings are fine there.

Areas the owner **does** read: `core.py`, `hsmm/model.py`, `hsmm/kernels.py`, `hsmm/params.py`
and `detectors/*`. The style rules in §4 apply to these.

---

## 1. Decisions already made

| # | Decision |
|---|---|
| D1 | One `AnomalyDetector` base. Its subclasses are three HSMM detectors (**Live**, **Lag**, **Hindsight**) and two LLM detectors (**Verdict**, **Predictive**). |
| D2 | Live, Lag and Hindsight are separate classes. They share one `HSMM` object, which holds the parameters, the belief and the inference kernels. |
| D3 | Lag keeps `L` as a parameter. `L = inf` (the full smoother) is allowed. It is a reference, not a causal detector. |
| D4 | `narrate()` stays. It works on Hindsight output. Given Live or Lag output, it **raises** `NotImplementedError` with a message saying that live narration is planned. |
| D5 | The LLM verdict arms (incremental / conversational / batch) stay. |
| D6 | Dead code is **deleted outright**, not archived. Tag `pre-restructure` on `main` first. |
| D7 | `lifecycle/` is deleted. `docs/lifecycle.md` becomes a short "next: drift detection" page (§8). |
| D8 | EPIC-KITCHENS stays: `parse_epic.py`, `configs/epic.yaml`, its tests and its section in data.md. |
| D9 | The owner keeps personal notes outside the repo. **The agent maintains `docs/` and updates it in the same commit as any code change it describes.** Write this rule into a new project `CLAUDE.md` (§8). |
| D10 | Metric G is the only headline metric. Metric E is kept only for comparison with PREGO. The code for A, B, D, F and G-gap is deleted, and their old numbers live in the logbook only. |
| D11 | Code says **run** (a maximal stretch of constant (verb, noun)), never "step". The one exception is text that goes into LLM prompts, which must stay byte-identical because the response cache keys on it. |

---

## 2. Target layout

```
CLAUDE.md                      NEW: agent rules (docs-with-every-commit, ./py, cache, metric labels)
py                             unchanged
configs/                       breakfast.yaml, epic.yaml, epic_tent.yaml  (breakfast_mini.yaml deleted)
                               + operating_points.yaml  NEW: α/κ per (corpus, model, detector), picked on train
src/cook_ad/
  core.py                      Trial, Span, Alarm, Detection, OperatingPoint, AnomalyDetector
  hsmm/
    params.py                  JointHSMMParams, LogTables, load/save, to_log_tables
    durations.py               NB pmf / survival / cdf / hazard (inference half only)
    kernels.py                 pure JAX: predict, update, filter scan, backward step, Viterbi
    model.py                   class HSMM
    train/                     everything that FITS parameters (agent-optimised; see §5)
  detectors/
    thresholds.py              class BadSets: the α-tail sets every HSMM channel is tested against
    hsmm_live.py               class LiveHSMMDetector
    hsmm_lag.py                class LagHSMMDetector(LiveHSMMDetector)
    hsmm_hindsight.py          class HindsightHSMMDetector (+ HindsightTrace)
    narrate.py                 narrate(detection) -> list[Query]; Lexicon
    llm_base.py                class LLMDetector (prompt building, fit = exemplars/rationales)
    llm_verdict.py             class LLMVerdictDetector
    llm_predictive.py          class LLMPredictiveDetector
  llm/                         client.py, prompts.py, textify.py   (plumbing, unchanged behaviour)
  data/                        parse_breakfast, parse_epic, parse_epic_tent, split, labels, config,
                               tick_expansion
  synthetic/                   generate.py, error_injection.py
  eval/
    metric_g.py                metric G (from span_metrics, G only) + run_starts
    metric_e.py                PREGO trimmed scorer (from run_epic_tent_eval.score_prego)
    pools.py                   canonical trial pools (§7)
    harness.py                 run detectors over pools, pick operating points on train, print tables
  xla_env.py
scripts/                       see §6
tests/                         see §9.4
docs/                          see §8
```

Folders that disappear: `anomaly/` (its contents move to `detectors/`), `recipe/` (split between
`hsmm/kernels.py` and `hsmm/train/`), and `lifecycle/`.

---

## 3. Core types and classes

The signatures below are the contract. The agent may adjust them when the code needs it, and records
any deviation in this file's successor (the logbook entry).

### 3.1 `core.py`

```python
@dataclass(frozen=True)
class Span:      start: int; end: int; kind: str          # an annotated or injected anomaly

@dataclass(frozen=True)
class Trial:
    trial_id: str
    verbs: np.ndarray            # (T,) int
    nouns: np.ndarray            # (T,) int
    recipe: int | None = None    # known dish, if given (Breakfast dish model); None = infer it
    spans: tuple[Span, ...] = () # ground truth; empty for healthy trials
    group: str = "healthy"       # "healthy" or the error type / error group

@dataclass(frozen=True)
class Alarm:
    tick: int        # the tick the alarm is ABOUT (what metric G scores)
    channel: str     # e.g. "s_emit", "s_temporal", "p_symbol"
    score: float     # p-value, posterior, or surprise, whatever the channel tests
    known_at: int    # the tick at which the detector could first raise it (tick + L for Lag)

@dataclass
class Detection:
    alarms: list[Alarm]
    trace: object | None = None  # detector-specific, e.g. HindsightTrace for narrate()

@dataclass(frozen=True)
class OperatingPoint:
    alpha: float | dict[str, float]   # one α, or per channel (e.g. s_temporal=1e-5)
    kappa: float | None = None        # posterior cutoff for hidden-event channels (Live/Lag only)

class AnomalyDetector(ABC):
    name: str
    causal: bool                                   # True: step() is supported
    def fit(self, train: list[Trial]) -> None      # default: no-op (params loaded at construction)
    def reset(self, recipe: int | None = None) -> None
    def step(self, verb: int, noun: int) -> list[Alarm]       # causal detectors only
    def finish(self) -> list[Alarm]                # flush alarms still pending at end of trial
    def detect_trial(self, trial: Trial) -> Detection
    def detect_many(self, trials: list[Trial]) -> list[Detection]   # default: loop
```

The default `detect_trial` is `reset(trial.recipe)`, then `step()` on every tick, then `finish()`.
Causal detectors may override it with a fast vectorised path. **A test asserts that the fast path
equals the step loop** (§9.4). Hindsight sets `causal = False`, and its `step()` raises.

`Flag` (span_metrics), `Verdict`/`StepVerdict` (LLM) and the HSMM flag dicts all become `Alarm`.
Metric G reads `alarm.tick` only, exactly as it reads `Flag.tick` today, so scores do not move.

### 3.2 `hsmm/model.py`: `class HSMM`

This is the probabilistic model, and the class the owner's "state belief" notes describe. It holds
the parameters, the log tables derived from them, and the current belief
b_t(r, i, j) = P(recipe r, subtask i, elapsed j | o₀…o_t).

```python
class HSMM:
    def __init__(self, params: JointHSMMParams, d_max: int)
    @classmethod
    def load(cls, path, d_max) -> "HSMM"

    # ---- one tick at a time (the live view)
    def reset(self, recipe: int | np.ndarray | None = None)  # None: fitted P(R); int: pinned; array: log prior
    def predict(self) -> Forecast        # b_{t-1} -> predictive table; Forecast has .stay and .switch
    def update(self, verb, noun) -> float  # multiply by P(o_t | i), normalise; returns log P(o_t | o_<t)
    belief, recipe_belief, occupancy     # read-only properties (occupancy = π̃_t)

    # ---- whole trial (the same kernels, scanned)
    def filter(self, verbs, nouns, recipe=None) -> FilterResult    # lax.scan of predict+update
    def smooth(self, verbs, nouns, lag) -> Smoothed                # fixed-lag β (lag=inf: full)
    def infer_recipe(self, verbs, nouns) -> tuple[int, np.ndarray] # r̂ and P(r | o_{0:T})
    def viterbi(self, verbs, nouns, recipe: int) -> list[Segment]  # MAP path under one recipe
```

The pure functions live in `hsmm/kernels.py`: `predict`, `update`, `filter_scan`, `backward_step`
and `viterbi`. They come from `anomaly/live.py` (`_predict`, `run_filter`, `_backstep`,
`full_beta`, `fixed_lag_beta`), `hsmm/messages.py` and `recipe/segmentize.py`. **`step` and `filter`
call the same kernel functions.** That is how live and whole-trial runs are guaranteed to agree.

There are three unifications to attempt. Keep each one only if §9 passes:

1. **One filter.** Hindsight's π̃ comes today from `messages.predictive_occupancy` under r̂. Replace it with `HSMM.filter(recipe=r̂).occupancy`. `tests/test_live.py` already shows they agree to tolerance, so this is likely safe.
2. **r̂ from the filter.** `joint_em.infer_recipe` computes P(r | o) ∝ π_r P(o | r). That is the filter's final recipe belief, and `test_live` asserts the log-likelihoods match. The argmax must be identical on every golden trial.
3. **Viterbi** stays its own kernel. It takes a max over paths, while the filter sums over them.

### 3.3 `detectors/thresholds.py`: `class BadSets`

The thresholds are a property of the **model**, not of the data. For each channel, the "bad set"
is the set of outcomes that together carry at most α of the model's own probability. The data only
chooses α (and κ). `BadSets(hsmm, alphas)` is built once per model and α grid, and holds:

- per-state emission quantiles (`quantile.categorical/joint_quantile_threshold`);
- transition and recipe-excess tables (`transition_quantile_threshold`, `recipe_excess_thresholds`);
- duration tails;
- the binned versions Live uses (today `LiveTables.{stall,early,trans,recipe}_bin`, where a cell's bin is the smallest α at which it enters the set).

Source files: `quantile.py`, `surprise.JointThresholdCache`, and the bin code in `live.build_tables`.

### 3.4 The three HSMM detectors

All three report the same 7 channel names: `s_emit`, `s_verb`, `s_noun`, `s_temporal`,
`s_dur_two`, `s_transition`, `s_recipe_transition`. Each channel is **one named method**, with a
docstring of one or two lines that names the equation's section in `docs/detectors.md`.

**`LiveHSMMDetector(hsmm, bad_sets, op)`.** Causal.
- `step(v, n)` runs `forecast = hsmm.predict()`, then the emission p-values against the forecast, then `hsmm.update(v, n)`, then the hidden-event probabilities.
- Alarms fire where p ≤ α (emission channels) or where P(event in bad set) > κ (event channels).
- Channel methods:
  - `emission_pvalues`: the timing-split version. **`emission="mix"` and `p_mix` are deleted**; the split variant replaced them.
  - `stall_prob`
  - `early_exit_prob`
  - `transition_prob`
  - `recipe_transition_prob`
- Fast path: `hsmm.filter` plus vectorised channels. This is today's `compute_live_trace(lags=(0,))` followed by `flags`.

**`LagHSMMDetector(hsmm, bad_sets, op, lag)`.** Subclasses Live.
- Emission channels are unchanged, because they are forecasts and already final at tick t.
- It overrides **only** the event-probability computation, which conditions on o₀…o_{t+L} through the fixed-lag β.
- Alarms for tick t are emitted at `step()` of tick t+L, with `known_at = t+L`.
- `finish()` flushes the last L ticks.
- `lag=math.inf` is the full smoother (`full_beta`). It sets `causal = False`.

**`HindsightHSMMDetector(hsmm, bad_sets, op)`.** Not causal. `detect_trial` runs:
1. r̂ = `hsmm.infer_recipe` (or `trial.recipe` when it is given);
2. segments = `hsmm.viterbi(r̂)`;
3. π̃ = `hsmm.filter(recipe=r̂).occupancy`;
4. the 7 channels, as one method each:
   - `emission_surprise` (s_emit/s_verb/s_noun, with the dilution-corrected thresholds);
   - `stall_surprise` (s_temporal);
   - `duration_surprise` (s_dur_two);
   - `transition_surprise`;
   - `recipe_transition_surprise`.

   The k_recipe = 1 zeroing and the +inf thresholds stay; see the one-recipe trap in §11.
5. Alarms against `bad_sets`.

It returns `Detection(alarms, trace=HindsightTrace)`. `HindsightTrace` keeps only the fields that
`narrate` and the figure script read, which today are roughly: z*, the segments, r̂, π̃ at z*, the
expected verb/noun/next state, attribution and temporal attribution, and from_state. Delete
`pit`, `s_dur_long`, `s_dur_short`, `belief_concentration`, `from_recipe`, `expected_next_recipe`,
`s_pair` and `pi_all` unless grep shows a kept reader.

`detect_many` keeps today's chunked, length-bucketed batching (`eval/batch.compute_traces_joint`),
because evaluation speed depends on it. A test asserts that it equals `detect_trial` per trial.

**Recipe handling, identical in all three:** `reset(recipe)` / `trial.recipe` takes either a known
dish (int, which pins the recipe), a log prior (array), or None (infer it). This replaces
`--recipe-labels`, `--oracle-recipe`, `pin_recipe`, `with_recipe_prior` and `r_hat=` arguments.

### 3.5 `narrate(detection, hsmm, vocab, op) -> list[Query]`

This is today's `narrate_joint`, reading everything from `HindsightTrace` instead of a list of
seven arguments. If `detection.trace` is not a `HindsightTrace`, it raises
`NotImplementedError("narration needs a Viterbi segmentation; live narration is planned — see docs/detectors.md")`.
Delete the cascade `narrate()`. Keep `Lexicon`, built from the r̂ slice of the joint parameters.

### 3.6 LLM detectors

**`LLMDetector(client, vocab, variant, n_shot, shot_select, shot_cot, labels)`** is the base.
- `fit(train)` builds the system prompt: the recipe text for `with-recipes`, the few-shot exemplars (`prompts.select_exemplars`), and the Auto-CoT rationales (`predict.build_rationales`).
- **The prompt bytes must not change** (§9.2).

**`LLMVerdictDetector(..., protocol, unit)`** comes from `llm/detect.py`.
- `unit` is `"run"` or `"tick"`. The internal name `"step"` becomes `"run"`, but prompt text is untouched.
- `incremental` and `conversational` are causal: `step()` buffers ticks and queries at each element boundary.
- `batch` sees the whole trial and sets `causal = False`.
- A verdict becomes an `Alarm` at the first tick of the element it judges. This is today's `flags_from_verdicts`.
- The metric-B scoring (`element_metrics.from_llm_verdicts`, `evaluate_steps`) is deleted.

**`LLMPredictiveDetector`** comes from `llm/predict.py`. Causal.
- `step()` keeps the current run.
- At each run boundary it closes the run and issues the prediction request for the prefix so far. This is the same message list that `predictive_messages` builds today, so the responses come from the cache.
- Channel `p_dur_long` fires on the first tick past the predicted duration plus tolerance.
- Channels `p_symbol` and `p_dur_short` fire at the boundary.

  Note: `p_dur_short` has `tick` = the last tick of the run and `known_at` = the boundary tick. Today's scoring reads `tick`; keep it that way.

### 3.7 Not built now, but enabled

`Session(live_detector, hindsight_detector)`:
- `step()` streams live alarms during cooking;
- `finish()` runs hindsight over the whole session for a reflection.

List it under "next" in docs/README.md. Do not implement it.

---

## 4. Style rules for the files the owner reads

These cover `core.py`, `hsmm/{model,kernels,params,durations}.py` and `detectors/*`.

- **Comments say what the code doesn't.** No history ("an earlier version…"), no measurements and no defending a choice. History goes in `docs/logbook.md`, rationale in `docs/`.
- Docstrings are 1–3 lines: what the function returns, in the notation of `docs/model.md` / `docs/detectors.md`, plus a section reference for the maths.
- Use the same symbols everywhere: `b` (belief), `pi_tilde` (π̃), `h` (hazard), `A` (transitions), `r_hat`, `alpha`, `kappa`, `L`. Variable names should match the doc equations.
- One method per channel, named after the channel.
- No function longer than about 40 lines in these files. Split by meaning, not by size.
- Module-level constants keep a one-line comment.
- If something non-obvious needs more than 3 lines to explain, it belongs in the doc, not the code.

Everything else (`hsmm/train/`, `data/`, `synthetic/`, `eval/`, `llm/` plumbing, `scripts/`) keeps
its current comment style, history removed where it is plainly stale.

---

## 5. `src/` file map

| Today | Goes to | Notes |
|---|---|---|
| `anomaly/live.py` | `hsmm/kernels.py` (filter, β), `detectors/thresholds.py` (bins), `detectors/hsmm_live.py`, `hsmm_lag.py` | drop `p_mix`, `emission="mix"`, `LiveTrace.switch` (diagnostic) |
| `anomaly/surprise.py` | `detectors/hsmm_hindsight.py` (joint path only) | delete the cascade path: `compute_trace`, `assemble_trace`, `flag`, `compute_pi_all`, `_segment_recipe_path`, `recipe_transition_surprise`; also `s_pair`, `flag_joint_per_channel` (replaced by per-channel α in `OperatingPoint`), `belief_diagnostic` |
| `anomaly/quantile.py` | `detectors/thresholds.py` | delete `pair_quantile_threshold`, `sequence_thresholds`, cascade `threshold_tables` |
| `anomaly/temporal.py` | `detectors/hsmm_hindsight.py` (channel methods) | delete `pit_coordinate` |
| `anomaly/narrate.py` | `detectors/narrate.py` | joint path only |
| `anomaly/sequence.py` | **delete** | |
| `hsmm/messages.py` | `hsmm/kernels.py` (inference); `hsmm/train/` (`combine_sufficient_stats`, `backward_pass` if only EM uses it) | |
| `hsmm/joint_params.py` | `hsmm/params.py` | delete `kernel_v/kernel_n/noun_tilt` and `tilt_terms`: **no shipped model has them** (checked: all five shipped `.npz` hold only the 7 count/duration arrays). Keep `collapse_to_marginal`/`select_recipe` if synthetic or Lexicon need them |
| `hsmm/durations.py` | `hsmm/durations.py` (pmf/survival/cdf/hazard) + `hsmm/train/durations_fit.py` (fitting) | |
| `hsmm/joint_em.py` | `hsmm/train/joint_em.py`; `infer_recipe` → `HSMM.infer_recipe` | delete tilt/kernel branches |
| `hsmm/params.py`, `hsmm/em.py` | **delete** the cascade `HSMMParams`/EM; move shared helpers (`pad_batch`, chunking) to where they are used | `synthetic/` and `Lexicon` use a single-recipe view; give them `params.select_recipe` returning a small `RecipeView` instead of `HSMMParams` |
| `hsmm/kernel.py`, `hsmm/emissions.py` | delete `kernel.py`; `sequence_loglik` → `kernels.py` | |
| `recipe/segmentize.py` | `hsmm/kernels.py` (Viterbi) | |
| `recipe/lexical_init.py` | `hsmm/train/lexical_init.py` | |
| `recipe/recipe_hmm.py`, `recipe/warm_start.py` | **delete** | if a metric helper (`adjusted_rand`, `matched_accuracy`) is still used by training logs, move it to `hsmm/train/` |
| `eval/batch.py` | `HindsightHSMMDetector.detect_many` | |
| `eval/span_metrics.py` | `eval/metric_g.py` | delete D, G-gap and `merge_tol` paths; keep `run_starts`, onset scoring, alarms/10 min, run FPR |
| `eval/element_metrics.py`, `counterfactual.py`, `metrics.py`, `plotting.py` | **delete** | check `textify`/`predict` for imports of `element_metrics` types and inline what they need |
| `lifecycle/*` | **delete** | |
| `llm/detect.py`, `llm/predict.py` | `detectors/llm_verdict.py`, `detectors/llm_predictive.py` (detector logic); parsing helpers may stay in `llm/` | |
| `synthetic/generate.py` | stays | delete cascade-only generators if unused |
| `run_hard_em.py`, `smooth_params.py`, `refit_durations.py` (logic) | `hsmm/train/{hard_em,smoothing,refit}.py` | scripts become thin CLIs (§6) |

---

## 6. Scripts

Fifty-four scripts and three shell files at the root become fourteen, all under `scripts/`. Scripts
never import other scripts. Shared code goes in `src/cook_ad/eval/harness.py` or `pools.py`.

### 6.1 Kept

| New path | From | Notes |
|---|---|---|
| `scripts/data/fetch_epic_tent.py` | `tools_fetch_epic_tent.py` | |
| `scripts/data/split_dataset.py` | `split_dataset.py` | |
| `scripts/data/make_dev_split.py` | `make_dev_split.py` | the dev fold the regularisation was chosen on |
| `scripts/data/epic_tent_splits.py` | `tools_epic_tent_split.py` | PREGO split, clean (excised) split |
| `scripts/data/build_pools.py` | NEW | writes the canonical pools (§7) |
| `scripts/train/fit_lexical.py` | `run_joint_lexical.py` | thin CLI over `hsmm/train` |
| `scripts/train/fit_hard_em.py` | `run_hard_em.py` | |
| `scripts/train/smooth_params.py` | `smooth_params.py` | |
| `scripts/train/refit_durations.py` | `refit_durations.py` | |
| `scripts/train/train_hsmm.sh` | `fit_dish_model.sh` | `PRESET=lexhard\|dish\|epic_tent`; the lexhard and dish presets are the 4-stage pipeline in `joint_params_train_lexhard.npz.meta.json`; epic_tent is `fit_lexical` on the clean split, as in `run_tent_test.sh` §2 |
| `scripts/eval/eval_breakfast.py` | `run_live_eval.py` + `run_detect_eval.py` + Breakfast half of `run_llm_eval.py` + `tools_breakfast_g_table.py` + `tools_live_summary.py` | `--detectors hsmm-live,hsmm-lag,hsmm-hindsight,llm-verdict,llm-predictive,nulls`; picks operating points on train, writes `configs/operating_points.yaml`, prints labelled metric-G tables |
| `scripts/eval/eval_epic_tent.py` | `run_epic_tent_eval.py` | three variants (default, PREGO, PREGO trimmed → metric E) |
| `scripts/eval/llm_breakfast.sh` | `run_tick_test.sh` | ollama GPU placement, keep-alive, context preflight; keep `BASE_URL=http://localhost:11435/v1` |
| `scripts/eval/llm_epic_tent.sh` | `run_tent_test.sh` | |
| `scripts/figures/trial_figure.py` | `render_llm_vs_hsmm_png.py` | one trial, every detector, with narration |
| `scripts/analysis/participant_variability.py` | `explore_participant_variability.py` | kept for the drift work |
| `scripts/demo_trial.py` | NEW, for the owner | load a model, run ONE trial through Live, Lag(5), Hindsight and narrate; print alarms tick by tick. ≤ 80 lines; the readable entry point |

### 6.2 Deleted

Each script's finding, if it had one, goes into the logbook as one line before deletion.

| Group | Scripts |
|---|---|
| Cascade model / `breakfast_mini` | `run_experiment`, `run_recipe`, `run_joint`, `run_anomaly`, `run_evaluation`, `run_rollout_demo`, `run_lifecycle`, `export_flow`, `export_flow_joint`, `export_anomaly`, `render_flow_html`, `render_flow_png`, `render_flow_compare_png`, `render_anomaly_png`, `build_mini_dataset` |
| Metrics A/B/F and α sweeps | `report_final`, `run_step_sweep`, `run_counterfactual`, `run_threshold_sweep`, `run_threshold_sweep_coordinate`, `run_threshold_test_tuned`, `render_threshold_sweep_by_channel`, `render_threshold_sweep_coordinate`, `tools_alarm_load` |
| Removed features | `run_sequence_eval`, `tools_embed_vocab`, `tools_kernel_gradation`, `tools_severity_ranking` |
| Superseded figures | `render_llm_compare_png`, `render_llm_narrate_png` |
| One-off diagnostics | `tools_launder`, `tools_transition_ceiling`, `tools_state_organisation`, `tools_duration_power`, `tools_oracle_recipe`, `tools_run_lengths` |
| Folded (§6.1) | `run_detect_eval`, `run_live_eval`, `run_llm_eval`, `tools_live_summary`, `tools_breakfast_g_table`, `fit_dish_model.sh`, `run_tick_test.sh`, `run_tent_test.sh` |
| Misc | `main.py`, `configs/breakfast_mini.yaml` |

The per-channel α that the sweeps found (`s_temporal = 1e-5`) survives as a value in
`configs/operating_points.yaml`.

### 6.3 Untracked artifacts

`dataset/processed/breakfast` holds 1.2 GB of stale `.npz` files, reports and 15 `figures_*`
directories, and `dataset/processed/union/` is only kernel embeddings. **Do not delete any of it.**
Write `docs/cleanup_candidates.md` listing deletable paths with a reason each, for the owner to
approve.

Never list these:
- `llm_cache/` (both corpora): losing it costs days of GPU time;
- the shipped `.npz` files: Breakfast `joint_params.npz`, `joint_params_train_lexhard.npz` and `joint_params_train_dish.npz`; EPIC-Tent `joint_params_clean.npz` and `joint_params_clean_prego.npz`;
- `sequences*.json`, `labels*.json`, `vocab.json` and `split*.json`.

---

## 7. Evaluation harness and canonical pools

**The problem today.** Breakfast has two pool builders, and they disagree.

- `run_detect_eval` / `run_live_eval` inject **per error type over all trials**, and place the injections using the **model being evaluated**. So the dish model and the cluster model are scored on *different* trials.
- `run_llm_eval.build_pool` injects **per trial over all types**, placed with `joint_params.npz`, over the first 150 test sequences (97 usable).

The LLM pool cannot change: its responses are cached against those exact trials.

**The fix (Phase F, after equivalence is proven):**

- `eval/pools.py: build_pool(corpus, part, seed, placement_params, max_trials=None)` uses the LLM builder's order (per trial, over all types) and a **fixed** placement model (`joint_params.npz`, the one the LLM pool used).
- `scripts/data/build_pools.py` writes `dataset/processed/<corpus>/pools/{train,test}.pkl` once. Every detector is scored on those files.
- Expected property: because the RNG is consumed per trial, the first 97 entries of the full test pool **equal** the LLM pool. Check it bitwise against `runs/golden/streams_pred_random.pkl` (`payload["pools"]["real"]`).
  - If it holds, the LLM is scored on that prefix, and every table also has a "LLM pool (97)" version so all rows are compared on the same trials.
  - If it does not hold, the canonical test pool *is* the LLM pool, and the HSMM's test numbers use those 97 trials.
- EPIC-Tent pools are the real annotations: three variants × error groups, from `run_epic_tent_eval.collect`. Freeze them the same way.

HSMM numbers will move a little in Phase F because the injected trials change. That is expected.
Regenerate every table, and record the shift in the logbook.

`harness.py` provides:
- `run(detectors, pool) -> {name: [Detection]}`;
- `pick_operating_point(detector, train_pool, grid, rule)`, where the rule is G-F1, or best recall under an alarms/10 min budget;
- `table(results, pool)`, which always prints the metric letter, precision, recall, F1, alarms/10 min and run FPR;
- the nulls "flag every run" and "flag every tick" (identical under G; print both).

---

## 8. Docs

Today's docs are 6,489 lines across 10 files, mostly logbook. Rewrite them so each doc explains
**the theory and where it lives in the current code**. Past attempts go to the logbook. Each doc has
a "Code" table mapping concepts to `file:class.method`.

| New doc | Covers | Built from |
|---|---|---|
| `README.md` | what the system is; the class diagram; repo map; how to train / evaluate / run the demo; current headline tables (metric G, labelled); "Next" (drift, live narration, `Session`) | old README §§ layout + pipeline |
| `model.md` | the HSMM: generative story, belief table b_t(r,i,j), predict (hazard, stay, switch into column 0), update (normaliser = P(o_t \| o_<t)), recipe layer, π̃, smoother/fixed lag, Viterbi, r̂. Maps to `HSMM` | hsmm.md, live.md §1 |
| `detectors.md` | the `AnomalyDetector` contract and `Alarm`; bad sets and α; κ; Live channels (timing-split p-values, switch posterior factored per channel); Lag; Hindsight channels (dilution correction, one-recipe rule); narration; worked example | anomaly.md, live.md §§2–6 |
| `llm.md` | the two LLM detectors: prompts, variants, shots, Auto-CoT, the predictive channels, the cache (key includes base_url) | llm.md |
| `training.md` | the pipeline stages and presets, one paragraph each: what it does and its hyperparameters | hsmm.md (EM), recipe.md (lexical init) |
| `data.md` | corpora (Breakfast, EPIC-KITCHENS, EPIC-Tent), ticks, vocab, splits, injection, canonical pools | data.md, synthetic.md |
| `eval.md` | metric G (definition, why, nulls), metric E (PREGO), operating-point selection, the scripts | eval.md §0 |
| `lifecycle.md` | **Next: drift detection.** What it will do (frozen vs live model, preference updates, duration drift), that the old cascade implementation was deleted (see the `pre-restructure` tag), and that it will be built on `HSMM` | lifecycle.md (≤ 40 lines) |
| `logbook.md` | dated entries, newest first, ≤ 5 lines each: what was tried, the number, the verdict, and the script/commit that measured it. Condensed from the old docs' history sections, deleted scripts' findings, and the findings in the agent memory index | everything removed above |

Delete `anomaly.md`, `live.md`, `hsmm.md`, `recipe.md` and `synthetic.md` once their content has a
new home. Before deleting each, grep it for numbers and findings, and confirm each one is either in a
new doc (if current) or in the logbook (if history).

**`CLAUDE.md` (new, repo root):**
- update `docs/` in the same commit as any code it describes;
- every table states its metric letter;
- run Python through `./py`;
- never `uv sync` without `--extra dev --extra gpu`;
- LLM replays need the original `--base-url` and `--max-requests 0`;
- never `pkill -f` / `pgrep -f` a pattern that matches your own shell;
- say "run", not "step".

---

## 9. Verification

Nothing merges until all of §9.1–9.5 pass. Results go in the PR description.

### 9.1 Golden outputs (generated on `main` BEFORE any change)

`scripts/verify/dump_golden.py` (temporary, deleted in the last commit) runs the OLD code and writes
`runs/golden/*.pkl`. `runs/` is gitignored.

| Corpus / pool | Model | Arms × operating points |
|---|---|---|
| Breakfast, `run_live_eval` test pool (seed and args as in `runs/live_dishknownR_test.json`) | `joint_params_train_lexhard.npz`; `joint_params_train_dish.npz` with dish given and with dish inferred | hindsight at α ∈ {0.01, 0.05} and at the per-channel α {default 0.005, s_temporal 1e-5}; live L=0 at (0.02, κ 0.5); lag L ∈ {5, inf} at (0.02, κ 0.5) |
| Breakfast LLM pool (97 trials, `runs/golden/streams_pred_*.pkl`) | `joint_params.npz` (the old LLM-report HSMM arm) | its HSMM stream as dumped |
| EPIC-Tent default / PREGO / PREGO-trimmed | `joint_params_clean.npz`, `joint_params_clean_prego.npz` | hindsight and live exactly as `run_epic_tent_eval.py --live` runs them |

Per trial, store:
- the (T, 7) bool flag matrix per arm;
- r̂;
- the filter's total log-likelihood;
- the Viterbi segments;
- the final printed tables (metric G; metric E for trimmed).

For the LLM arms, store the cache-replayed streams: Breakfast is already in
`runs/golden/streams_pred_{random,cot}.pkl`; regenerate EPIC-Tent's with
`run_epic_tent_eval.py ... --max-requests 0`.

### 9.2 Equivalence (after each phase)

- **HSMM:** flag matrices identical (bitwise bool) on every golden trial and arm. r̂ identical. Log-likelihoods within 1e-9 relative. Segments identical.
- **Tables:** the metric-G and metric-E tables reproduce the golden ones to 3 decimals.
- **LLM:** replay every LLM arm with `--max-requests 0`. **100% cache hits** (any miss means a prompt byte changed), and the resulting alarm ticks equal the golden streams.
- **Contract tests (§9.4)** pass.

### 9.3 Cost

On the same GPU, before and after:
1. hindsight `detect_many` on the Breakfast test pool;
2. Live `detect_trial` on the same pool;
3. the EPIC-Tent eval with `--skip-llm`.

After must be ≤ 1.10× before. Also time `step()` per tick on one trial and report it. There is no
target; it is informational.

**Training** is not re-run in full. Equivalence check: run the old and the new `train_hsmm.sh
PRESET=epic_tent` with `--max-iters 2` on EPIC-Tent (29 sessions, minutes). The output `.npz` must
be bitwise equal. Every shipped `.npz` must load into the new `JointHSMMParams` unchanged.

### 9.4 Tests

- **Keep and port:** test_live, test_messages, test_segmentize, test_surprise, test_quantile, test_temporal, test_durations, test_joint_em, test_lexical_init, test_span_metrics (G only), test_predict, test_llm_detect, test_prompts_domain, test_textify, test_parse_breakfast, test_parse_epic, test_parse_epic_tent, test_split, test_prego_protocol, test_narrate.
- **Delete with their modules:** test_counterfactual, test_element_metrics, test_kernel, test_lifecycle, test_duration_drift, test_recipe_hmm, test_sequence, test_pair_channel, test_rollout_demo_smoke, test_evaluation.
- **New:**
  - `test_detector_contract.py`: for Live, Lag(5) and LLMPredictive (with a fake client), `detect_trial` equals reset + `step()` loop + `finish()`, and the alarms' `known_at ≥ tick`;
  - Hindsight `detect_many` equals per-trial `detect_trial`;
  - Hindsight `step()` raises;
  - `narrate` raises on a Live detection.

### 9.5 Readability check

The owner reviews `scripts/demo_trial.py`, `hsmm/model.py` and `detectors/hsmm_live.py` before merge.

---

## 10. Work order

Make one commit per phase. Each phase ends with §9.2 passing.

| Phase | Work |
|---|---|
| A | Tag `pre-restructure`. Write and run `dump_golden.py` on `main`. Record the timings (§9.3). |
| B | `core.py`, `hsmm/{params,durations,kernels,model}.py`, `detectors/thresholds.py`. The old modules call into them (thin shims), so the old scripts still produce golden output. |
| C | The three HSMM detectors, `narrate`, the contract tests. The old eval scripts are switched to the detectors and must reproduce golden output. |
| D | The LLM detectors. Cache replay must be 100% hits. |
| E | `eval/{metric_g,metric_e,harness}.py`, the `scripts/` tree, `hsmm/train/`. Delete the old scripts and modules (§5, §6). Remove the shims. Re-run all of §9. |
| F | Canonical pools (§7). Regenerate every table and `configs/operating_points.yaml`. |
| G | Docs rewrite (§8), `CLAUDE.md`, `cleanup_candidates.md`. Delete this file. |

---

## 11. Traps (read before starting)

- **One recipe (EPIC-Tent, k_recipe = 1):** `s_recipe_transition` must be zeroed **and** its thresholds set to +inf. A test that fits once hid the second half for a week.
- **LLM cache key** = sha256(base_url + model + messages + temperature). Breakfast runs used `http://localhost:11435/v1`. EPIC-Tent used `http://localhost:11434/v1` (recorded under `reports/<group>/<arm>/client` in its report JSON).
- **`./py`** sets PYTHONPATH and the XLA memory fraction. Use it, never a bare `python`.
- **XLA autotune:** large-K compiles "hang" because XLA is autotuning. `xla_env.disable_gpu_autotuning()` must run before JAX is imported in eval scripts (see the top of `run_epic_tent_eval.py`).
- **jit buckets:** live traces pad to multiples of 64 ticks so they compile once per bucket. Keep this in the fast paths, and do not jit per trial length.
- **`pkill -f` / `pgrep -f`** match your own shell. Kill background jobs by PID.
- **Live "early exit" is causal by one tick.** Scored with metric G's deadline of 10 it is fine. Do not "fix" it by shifting ticks.
- **The tied-emission tolerance** (`EMISSION_FLAG_ATOL = 1e-6`, `TIE_ATOL = 1e-10`) is what makes hindsight flags bitwise reproducible. Keep both values.
- **Run lengths:** metric G's `run_starts` is computed from the stream the detector saw (the degraded one), not the healthy source.

---

## Progress notes (for the logbook entry)

**Phase A (34ff480).** Goldens cover §9.1 plus the Breakfast verdict arms (replayed into
`streams_verdict.pkl`). All LLM replays: 0 uncached requests. Two independent dumps of the old
code are bitwise identical, float diagnostics included. Baseline cost on the old code
(`runs/golden/timings.json`, warm): hindsight 17.7 s, live L=0 100.1 s, EPIC-Tent `--skip-llm`
11.7 s. The historical `epic_tent_report_live*.json` on disk predate metric G and are not
comparable to the regenerated reports (headline pooled numbers agree); the two LLM reports
reproduce exactly. An old-code worktree for re-timing lives at `../mci-cooking-ad-old`.

**Phase B deviations from §3.** Recorded here so Phase G's docs and the logbook pick them up.
- `HSMM.smooth(fr, L)` takes the `FilterResult` instead of `(verbs, nouns)`, so the emission
  log-likelihoods are not recomputed. `L = math.inf` is the full smoother.
- `HSMM.infer_recipe` returns `(r_hat, log P(r | o), log P(o))`; in Phase B it is the per-recipe
  forward pass (bitwise the old `joint_em.infer_recipe` arithmetic). The filter-based r̂
  (unification 2) is tried in Phase C.
- `HSMM.predict()` returns a `Forecast(b_pred, log_end, log_m)` with `pi_tilde`/`stay`/`switch`
  properties; `log_end`/`log_m` are what the Live event channels need.
- `HSMM.view(recipe)` returns the per-trial tables (`View`), replacing `pin_recipe` and
  `with_recipe_prior`; `View.recipes` is the slice detectors apply to their bins.
- The new 7-field `params.JointHSMMParams` coexists with the old tilt-capable
  `joint_params.JointHSMMParams` (same field order, so `params.JointHSMMParams(*old[:7])`
  converts) until Phase E deletes the old EM. The cascade `HSMMParams` section stays at the bottom
  of `hsmm/params.py` until then.
- Docstrings cite the planned `docs/model.md` sections: §1 tables and emissions, §3 the filter
  (§3.1 predict, §3.2 update), §5 backward messages, §6 one-recipe forward pass and r̂, §7 Viterbi;
  and `docs/detectors.md` §1 the detector contract, §2 bad sets (§2.3 the one-recipe rule),
  §3 Live channels (§3.3 stall). Phase G must write them with these numbers.

**Phase B results.** `compare_golden.py runs/golden runs/golden_B`: all 20 files pass, bitwise —
every flag matrix, r̂, z*, log-likelihood and float diagnostic, every table, every LLM stream
(0 uncached requests). Full test suite green (513 tests, including the 14 in the new `tests/test_hsmm_model.py`). Training
(`run_joint_lexical.py`, EPIC-Tent clean split, `--max-iters 2`): output `.npz` bitwise equal.
Cost, warm, new / old: hindsight 18.2 / 17.7 s (1.03×), live L=0 100.2 / 100.1 s (1.00×),
EPIC-Tent `--skip-llm` 11.8 / 11.7 s (1.01×). `HSMM.update` one tick at a time: 1.4 ms/tick
(Breakfast lexhard, K_R=16, K=64, D=200).

**Phase C deviations from §3.**
- The old `anomaly/`, `eval/batch.py` and `anomaly/live.py` code is left intact (not shimmed) as
  the reference until Phase E deletes it. The detectors are proven against it two ways:
  `dump_golden.py --impl new` (every per-trial golden through the detector API) and the
  switched eval scripts — `run_epic_tent_eval.py` (hindsight, injected control, live/lag arms)
  and `run_llm_eval.py`'s HSMM arm now call the detectors. `run_live_eval.py` and
  `run_detect_eval.py` still call the old library (both are deleted in Phase E; the new harness
  replaces them).
- New module `hsmm/batch.py` (not in §2's layout): `infer_recipe`, `occupancy` and `viterbi` for
  many trials at once — the chunked, globally padded batching `detect_many` needs to stay
  bitwise with the old `compute_traces_joint`.
- Live and Lag expose `trace(trial) -> LiveTrace` (scores at every grid alpha) and
  `flags(trace, op)`, so a sweep thresholds one trace many times. `LiveTrace` keeps `pi_tilde`
  instead of the diagnostic `switch`; `p_mix` is gone. Hindsight likewise has
  `traces(trials)` / `flags(trace, op)` / `thresholds(trace, op)`.
- Lag raises emission alarms at tick t (`known_at = t`): they are forecasts and final then. Only
  the event alarms wait until t + L (`known_at = min(t + L, T - 1)`).
- `narrate(detection, detector, vocab, op=None)` takes the detector rather than the HSMM, so a
  question's severity is measured against exactly the threshold that gated its flag. `Query`
  drops the `event` field (it fed `lifecycle/`, which Phase E deletes).
- `HindsightTrace` keeps `r_hat, segments, z_star, verbs, nouns, surprise{channel}, pi_at_zstar,
  from_state, expected_verb/noun/next_state, attribution, temporal_attribution`. Narration
  recomputes π̃ under r̂ (exactly as the old `compute_pi_all_joint` did) rather than storing
  a (T, K) table per trial.
- Docstrings cite `docs/detectors.md` §3.1 emission p-values, §3.2–3.6 the event channels
  (§3.3 stall, §3.4 transition, §3.5 early exit, §3.6 recipe), §4 Lag, §5 Hindsight (§5.1–5.5 its
  channels), §6 narration. Phase G writes them with these numbers.

**Phase C results.** `dump_golden.py --impl new` (every per-trial golden through the detector
classes) and `run_tables.sh` (with the switched EPIC-Tent and LLM-eval scripts) into
`runs/golden_C`: all 20 files pass `compare_golden.py` bitwise, float diagnostics included; 0
uncached LLM requests. `tests/test_detector_contract.py` (12 tests): Live and Lag(2, 5) fast path
== reset + step loop + finish, with and without a pinned recipe; `known_at` in [tick, tick + L];
Hindsight `detect_many` == `detect_trial`; Hindsight and Lag(inf) refuse `step()`; Lag(inf) ==
Lag(T); `narrate` raises on a Live detection and reproduces the old `narrate_joint` word for word
on real EPIC-Tent sessions. Full suite: 525 passed. Cost, warm, new / old: Hindsight
`detect_many` 17.7 / 17.7 s (1.00×), Live `detect_trial` 100.1 / 100.1 s (1.00×), EPIC-Tent
`--skip-llm` 11.5 / 11.7 s. Informational: `HSMM.update` 1.6 ms/tick, `LiveHSMMDetector.step`
5.6 ms/tick (Breakfast lexhard).

Unifications (§3.2): (1) π̃ from `HSMM.filter(recipe=r̂)` instead of the one-recipe predictive
occupancy changes the hindsight flags on 1 of 582 Breakfast lexhard golden trials — **rejected**,
the predictive occupancy stays. (2) r̂ from the filter's final recipe belief equals the batched
r̂ on 582/582 trials but is **not adopted**: the full (r, i, j) filter costs several times the
batched per-recipe forward pass, which would break the 1.10× budget. `test_hsmm_model` asserts
the two agree. (3) Viterbi stays its own kernel, as planned.

**Phase D deviations from §3.6.**
- `LLMDetector(client, vocab, lexicon, variant, labels, n_shot, shot_select, shot_cot, seed,
  rationale_client, prompt_unit, **prompt_kwargs)`. `fit(train)` draws the exemplars (and
  Auto-CoT rationales, through `rationale_client`, which Breakfast gives its own token cap) and
  calls `build_prompt()`; a harness that shares one exemplar draw across variants sets
  `exemplars`/`rationales` and calls `build_prompt()` itself. `prompt_kwargs` carries EPIC-Tent's
  domain, SIL tokens, canonical order and interleaving. `prompt_unit` is the unit the prompt is
  built with (Breakfast's predictive arm passes "tick", as before).
- `LLMVerdictDetector(unit="run" | "tick")`; "run" maps to the "step" that the prompt text and
  `llm/` use, which are untouched. `known_at`: a run's verdict is known at the next run's first
  tick (T - 1 for the last run), a tick's at that tick, and every batch verdict at T - 1. An
  alarm's channel is the verdict's error type ("" when none parsed), as `Flag.channel` was.
- `LLMPredictiveDetector`: `p_dur_short` has `known_at = min(tick + 1, T - 1)`, so the last run
  of a trial (closed by the trial's end) is known at T - 1.
- `Detection.trace` is the list of `llm.detect.Verdict`s (verdict) or the parse statistics
  (predictive). The response parsers and the request builders stay in `llm/detect.py` and
  `llm/predict.py`, unchanged; `run_llm_eval.py` still feeds the verdicts to metric B until
  Phase E deletes it.
- The step loop issues exactly the requests the fast path does (`test_detector_contract`
  asserts the request sets are equal), so it replays from the same cache entries.

**Phase D results.** `dump_golden.py --impl new` (EPIC-Tent predictive arms replayed through
`LLMPredictiveDetector`'s step loop) and `run_tables.sh` (every Breakfast and EPIC-Tent LLM arm
through the detectors' fast paths) into `runs/golden_D`: all 20 files pass bitwise. Cache
replay: 158,234 requests served from cache, 0 uncached (Breakfast verdict 2 × 70,486, predictive
4 × 3,790; EPIC-Tent predictive 2 × (561 + 258 + 232); the EPIC-Tent report runs abort on any miss
and completed). `test_detector_contract.py` now has 20 tests: LLMPredictive and LLMVerdict
(incremental and conversational, run and tick units) fast path == step loop with identical
request sets, batch refuses `step()`, prompt bytes equal `prompts.build_variant`, and a real-cache
step-loop replay of two Breakfast trials (tick unit) with 0 uncached requests. Full suite: 533
passed.

**Phase E.** Built: `hsmm/train/` (joint EM without the tilt/kernel branches, lexical init,
hard EM, smoothing, duration refit, duration fitting, E-step messages, clustering metrics),
`scripts/train/` with `train_hsmm.sh` (`PRESET=lexhard|dish|epic_tent|epic_tent_prego`),
`eval/{metric_g,metric_e,pools,harness}.py`, `scripts/eval/{eval_breakfast,eval_epic_tent}.py`
and the two LLM runner shells, `scripts/data/*`, `scripts/figures/trial_figure.py` (rewritten,
not ported), `scripts/demo_trial.py`. Deleted: the 43 root scripts (findings in the logbook),
`anomaly/`, `recipe/`, `lifecycle/`, the cascade halves of `hsmm/`, `eval/{batch,span_metrics,
element_metrics,counterfactual,metrics,plotting}.py`, `configs/breakfast_mini.yaml`, the shims.
`llm/textify.py`'s ground-truth helpers stay (llm/ plumbing is unchanged; the injector tests use them).

Deviations: `eval/pools.py` exists already in E, holding today's two Breakfast builders
(per-type, placed by the scored model; per-trial, the LLM pool) so behaviour is unchanged until
Phase F. The EPIC-Tent `control/hsmm-injected` row was scored under G-gap, which is deleted; it is
now scored under G (one alarm per run), so its numbers move. The `@merge_tol=0` control row is
gone. `eval_breakfast.py` also scores the two LLM verdict arms (tick unit) on the LLM pool, which the
old headline table did not carry.

Results.
- Per-trial goldens (`dump_golden.py`, now new-API only) into `runs/golden_E`: all 6 configs pass
  bitwise, float diagnostics included, LLM replays 0 uncached.
- Tables (`run_tables.sh`, `compare_tables.py`): all five EPIC-Tent reports reproduce to 3
  decimals (metric G and E, LLM arms included; the two rows above skipped). Breakfast: every test
  grid, all train-picked operating points, and every row of the headline LLM-pool table
  reproduce. The train-split grids of lexhard and dish-given differ by at most 0.004 in some
  cells; the OLD code differs from its own rerun on the same table by up to 0.001 (exactly tied
  Viterbi paths, broken differently run to run), so this is not attributable to the restructure.
  Two runs of the NEW code on the same table are identical in every cell.
- Training: `train_hsmm.sh` presets epic_tent and lexhard (shortened) reproduce the old 4-stage
  pipeline bitwise at every stage; dish matches through smoothing and its duration refit differs by
  1.6e-10 from one exactly tied Viterbi path that the old code itself breaks differently between
  runs (shown with two old runs).
- Tests: 352 passed.
- Cost, warm, new / old: Hindsight `detect_many` 17.7 / 17.7 s (1.00x), Live `detect_trial`
  99.9 / 100.1 s (1.00x), EPIC-Tent `--skip-llm` 10.8 / 11.7 s (0.93x).
