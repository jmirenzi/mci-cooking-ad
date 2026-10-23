# Training the HSMM

How the parameters of [`model.md`](model.md) are fitted. Everything is unsupervised on the
integer (verb, noun) streams of the **train** split. `labels.json` is read only by the `dish`
preset, to group training trials by their labelled dish.

The pipeline is four stages, each a script under `scripts/train/` writing one `.npz`.
`scripts/train/train_hsmm.sh` runs them for a preset:

```bash
PRESET=lexhard scripts/train/train_hsmm.sh                   # Breakfast, 16 latent recipe clusters
PRESET=dish scripts/train/train_hsmm.sh                      # Breakfast, recipe = labelled dish
PRESET=epic_tent scripts/train/train_hsmm.sh                 # EPIC-Tent, error-excised train split
PRESET=epic_tent_prego scripts/train/train_hsmm.sh           # the same on PREGO's split
PRESET=dish OUT=runs/x scripts/train/train_hsmm.sh --max-iters 2 --iters 1   # a quick check
```

| Preset | Stages | Output | Notes |
|---|---|---|---|
| `lexhard` | 1 → 2 → 3 → 4 | `breakfast/joint_params_train_lexhard.npz` | K = 64, K_R = 16 latent clusters |
| `dish` | 1 → 2 → 3 → 4, with `--recipe-labels` | `breakfast/joint_params_train_dish.npz` | K_R = the 10 dishes |
| `epic_tent` | 1 | `epic_tent/joint_params_clean.npz` | K = 16, K_R = 1, `sequences_clean.json` |
| `epic_tent_prego` | 1 | `epic_tent/joint_params_clean_prego.npz` | PREGO's 14 train sessions |

`breakfast/joint_params.npz` is an older full-corpus fit. It is kept because it places the
injections of the canonical pools ([`data.md`](data.md) §5) and is the HSMM arm the cached LLM
reports were compared against.

---

## 1. Stage 1: the lexical warm start and soft joint EM

`scripts/train/fit_lexical.py` → `hsmm/train/lexical_init.py:lexical_to_joint`, then
`hsmm/train/joint_em.py:run_joint_em`.

**Why a warm start.** The joint objective Σ_i log Σ_r π_r P(o_i | r) is invariant to permuting
recipes. From a symmetric start EM has no gradient to break the symmetry and stalls with K_R
near-identical recipes. The warm start starts EM inside one basin.

**Why lexical.** The tick stream is coarse action labels, constant within an action, so the
distinct (verb, noun) pairs *are* the action inventory (48 on Breakfast). The warm start reads it
off directly:

1. `observed_pairs`: every pair covering at least `MIN_PAIR_TICKS` = 5 ticks gets a state. State k
   *is* pair k.
2. `anchored_emission_counts`: `ANCHOR_MASS` = 50 on the state's own two tokens and
   `BACKGROUND_MASS` = 1 over the rest. The background keeps an off-pair token's surprise finite;
   the MAP mode then removes it ([`model.md`](model.md) §1). The anchors are also passed to EM as
   per-state emission priors, so EM cannot reassign a state to a different action.
3. `hard_segments`: run-length-encode every trial over those pairs.
4. `cluster_recipes`: spherical k-means (20 restarts) on per-trial bag-of-pairs histograms. A
   recipe is characterised by which actions it contains. On EPIC-KITCHENS the options
   `features="nouns"` and `idf=True` must move together (pair histograms are too sparse at 3,806
   distinct pairs, and equipment nouns drown the dish): dish ARI −0.009 → 0.538. Breakfast and EPIC-Tent keep the
   defaults.
5. Per-recipe hard init/transition counts and per-(recipe, state) duration histograms from 3 and
   4, with the duration fit of §5.

`--recipe-labels` (the `dish` preset) replaces step 4 with the labelled dish
(`recipe_label_assignment`) and, during EM, holds every trial's responsibility on its own dish
with a log one-hot prior at λ = 1 (`pinned_recipe_log_prior`). Decoding a *new* trial still infers
the dish or takes it as given; labels are read only for training.

**Soft joint EM.** Per trial and recipe, the E-step runs forward-backward
(`hsmm/train/messages.py:combine_sufficient_stats`) to get log P(o_i | r), and

$$
\log\rho_{ir} = \log\pi_r + \log P(o_i \mid r) + \lambda \log \text{prior}_{ir} - \operatorname{logsumexp}_{r'}(\cdot).
$$

Statistics are ρ-weighted. For the shared emissions the recipe axis is collapsed first. The
M-step is Dirichlet MAP per recipe for π, init and transitions, shared MAP for emissions, and the
shrunk NB fit of §5 for durations. Hyperparameters:

| Setting | Value | Why |
|---|---|---|
| `--alpha-init`, `--alpha-trans` | 0.5 (α/K per cell) | sparsity: unneeded transitions fall to the floor |
| `alpha_pi` | 1.0 | |
| `--init-prior-scale` | **0.0** in the presets (default 1.0) | scales the Dirichlet prior on the *iteration-0* counts. 1.0 is coherent, but 0.0 fits a better detector: metric-A accuracy 0.556 vs 0.518, at a lower objective and worse subtask ARI |
| `--max-iters` | 60 | |
| `kappa` (duration shrinkage) | 5.0 (config) | §5 |
| `global_damping` | 0.7 (config) | §5 |
| `--lam`, `--lam-schedule` | 0 | the anchoring dial; `make_lam_schedule` forms `const`/`geom`/`linear`/`freeze` |

The E-step sorts trials by length and trims each chunk to its own longest trial (`_length_order`,
`_chunk_ticks`), so a chunk of short trials does not pay for the longest one in the corpus. The
reordering is verified bitwise. Any per-trial prior must be permuted the same way. Expected
transition counts are scanned over t: the dense (T, K, K) form allocated > 10 GB at K = 64,
T = 650 over 503 sequences.

**Convergence.** On Breakfast and EPIC-KITCHENS joint EM lands in a period-3 limit cycle and never
meets `tol`; iterating past ~100 changes nothing downstream. `converged` is true only when the
tolerance actually fired. EPIC-Tent (K_R = 1) converges in 9 iterations: the limit cycle is a
recipe-layer effect.

## 2. Stage 2: hard (Viterbi) EM

`scripts/train/fit_hard_em.py` → `hsmm/train/hard_em.py:hard_em_step`, 5 iterations,
`--keep-init-emissions`. Each iteration decodes every training trial under its r̂
(`hsmm/batch.py:viterbi`), counts hard init/transition counts and duration histograms
(`hard_counts`), and applies the same MAP and duration M-step as soft EM:

$$
\max_\theta \max_z \log P_\theta(o, z) \quad\text{instead of}\quad \max_\theta \log \textstyle\sum_z P_\theta(o, z).
$$

It is the wrong estimator for calibrated posterior uncertainty and the right one here: the
Hindsight detector scores against z*, so the counts the model is fitted from and the decode the
detector reads become the same object. Emissions stay at the lexical anchor. It holds per-tick
subtask ARI at 0.999, where soft EM from the same start drifts to 0.94.

## 3. Stage 3: transition smoothing

`scripts/train/smooth_params.py` → `hsmm/train/smoothing.py`, post-fit only.

- **`shifted_mode_counts(strength=0.7)`** stores c + s, so the MAP numerator becomes c − 1 + s. A
  singleton transition (c = 1) clears the floor; a never-seen one stays on it. s = 1, the
  posterior mean, is worse: it lifts never-seen transitions from ~32 nats to ~9 and the
  transition channel loses its top end.
- **`backoff_counts(tau=30)`** mixes each recipe's rows toward the pooled-over-recipes row. A
  transition merely unseen *in this recipe* inherits the pooled mass; one unseen everywhere stays
  maximally surprising.

Both are regularisation, chosen on a nested dev split (`scripts/data/make_dev_split.py`: 322 fit /
80 dev of the 402 outer-train trials). In-sample, τ = 2 looked best; on dev, τ = 2 was the worst of
the grid and τ = 30 the best. Without backoff the transition table is a lookup of the training
bigrams: flagged healthy runs were 0.8% on train against 4.3% on test.

**A population prior across people** (`hsmm/train/global_prior.py`, post-fit) is the same move
as backoff, across people rather than recipes:
- Run-length counts of first runs and run-to-run transitions per recipe, from many people's streams, keyed by (verb, noun).
- Each of a person's rows gets λ × the population's row, after lifting every cell to count 1, so λ = 0 is the identity.
- Used only on the synthetic drift data so far ([`drift_synth.md`](drift_synth.md) §9.2): it helps at cold start, and with a single α it can erase the transition channel's catches.

## 4. Stage 4: duration refit

`scripts/train/refit_durations.py` → `hsmm/train/refit.py:refit_durations`. Decode every training
trial with the smoothed model (`synthetic/generate.trajectories_from_real_joint`, under r̂ or the
labelled dish), histogram the segment lengths per (recipe, state), and refit (ρ, p) with the §5
fit at κ = 0.001. Each trial's last segment is imputed as censored, as in EM.

### Stage 4b: the nonparametric duration hazard

`scripts/train/fit_np_hazard.py` → `hsmm/train/np_hazard.py:fit_np_hazard`. The final model decodes
the training split, as in stage 4. Each (recipe, subtask) cell's hazard is then the discrete
Kaplan–Meier one, shrunk toward the state's hazard pooled over recipes:

$$
\hat h_{ij} = \frac{n_{ij} + \kappa\, g_{ij}}{N_{i,\ge j} + \kappa},
$$

with n_ij the cell's training segments that ended at exactly j ticks, N_{i,≥j} those still running
at j (the censored last segment is at risk up to its length and never an event), and g_ij the
pooled hazard, itself shrunk with κ₀ = 2 toward the negative binomial's. Counts are smoothed over
j (triangular kernel, half-width 1) and κ = 5. The tables follow as pmf_j = h_j ∏_{j'<j}(1 − h_j')
and surv_j = ∏_{j'<j}(1 − h_j'), and everything downstream reads (log_pmf, log_surv).

A cell with fewer than 30 training segments keeps the negative binomial outright: on few segments
the empirical shape is noise, and ungated it cost 0.02–0.035 F1 at 10–30 sessions. The tables
are written next to the model as `<model>.durations.npz` (`hsmm/model.py:durations_path`).
`HSMM.load` reads them automatically (`durations=None` keeps the NB), and `s_dur_two` and the
early-exit bins then read the tables (`HSMM.table_durations`). Why: the negative binomial has the
right spread but puts ~10% of mass below where real durations start, so a step cut short never
looks short (measurements: logbook.md, 2026-10-20).

### Stage 5 (optional): visit counts

`scripts/train/fit_visit_counts.py` → `hsmm/visit_counts.py:VisitCounts.fit`. The final model decodes
the training split (as stage 4), and per recipe which subtasks each session visits, and which came
before each one's first visit, are written as JSON next to the model. Only the optional `s_count`
(missing-step) channel reads them (detectors.md §3.6a, §5.6); `train_hsmm.sh` does not run this stage.

## 5. The duration fit

`hsmm/train/durations_fit.py:fit_durations_shrunk`, shared by every stage.

1. **Impute the censored segments** (`impute_censored_histogram`). Each trial's last segment is
   only known to satisfy D ≥ d_c. Its count is spread over d ≥ d_c in proportion to the current
   fit's P(D = d | D ≥ d_c). This is an ECM step, rerun every M-step, and mass-conserving.
   Skipping it biases every duration short, because the longest segment of a trial is the one cut
   off.
2. **Pool over recipes** for one global NB per state, then **shrink** each (r, k) cell toward it:
   n̂_{rk}(d) + κ P_global(D = d | k). κ is a pseudocount in units of segments: a starved cell is
   lifted into the stable regime, a well-populated one barely moves. At full Breakfast scale ~931
   of the 1,024 (recipe, state) cells are near-empty.
3. **Fit NB per cell.** p has a closed form given ρ: p̂ = ρN / (ρN + S), with N the count and S the
   sum of (d − 1). ρ has none, so it takes 5 Newton steps on the profile score, **seeded by the
   method of moments** (fallback ρ = 5 for under-dispersed histograms). Seeded from the old ρ,
   Newton can start past the score's minimum and diverge up its flat tail (ρ = 10 against data
   from ρ = 4 went to ~38,000).

**Damping.** The pooled global fit is refit from scratch each M-step, and for a near-empty state
it is noisy: measured, ρ_global swung 471 → 38 in one step, perturbing all 16 recipe copies of that
state at once. `global_damping` = 0.7 makes it an EMA (70% previous, 30% fresh). The EMA state
lives in `run_joint_em`, not in the parameters.

The order (impute per cell → pool → shrink → fit) matters: shrinking before imputing would charge
censored mass against the wrong (ρ, p).

## 6. Selecting a model

**Not by likelihood.** On Breakfast the training objective disagrees with detection every time it
was checked: soft EM from the lexical warm start lowers subtask ARI from 0.999 to 0.94; the highest-likelihood fit available was beaten on
detection by fits a thousand nats below it; and a random recipe partition beats the true one by
175 nats (~65k per-recipe transition parameters against ~10⁴ observed segment transitions).

**Not by recipe ARI either.** Tripling it (0.31 → 0.86) did not move detection. EM roughly halves
recipe ARI relative to its own warm start (Breakfast 0.88 → 0.31).

Select on detection, on held-out data: `scripts/eval/eval_breakfast.py` picks operating points on
train and reports test ([`eval.md`](eval.md) §3). Regularisation (§3) is selected on the nested
dev split.

## 7. Code

| Concept | Where |
|---|---|
| the pipeline | `scripts/train/train_hsmm.sh` |
| population prior (post-fit) | `hsmm/train/global_prior.py:count_global`, `apply_global_prior`, `state_pairs` |
| lexical warm start | `hsmm/train/lexical_init.py:lexical_to_joint`, `observed_pairs`, `anchored_emission_counts`, `hard_segments`, `cluster_recipes` |
| dish labels as recipes | `lexical_init.py:recipe_label_assignment`, `pinned_recipe_log_prior` |
| soft joint EM | `hsmm/train/joint_em.py:run_joint_em`, `e_step`, `m_step`, `make_lam_schedule`, `infer_recipe` |
| forward-backward statistics | `hsmm/train/messages.py:combine_sufficient_stats`, `backward_pass` |
| hard EM | `hsmm/train/hard_em.py:hard_em_step`, `hard_counts` |
| smoothing | `hsmm/train/smoothing.py:shifted_mode_counts`, `backoff_counts` |
| duration refit | `hsmm/train/refit.py:refit_durations`, `duration_histograms` |
| nonparametric hazard (stage 4b) | `scripts/train/fit_np_hazard.py`; `hsmm/train/np_hazard.py:fit_np_hazard`, `tables_from_hazard`, `save_tables`, `load_tables`; `hsmm/model.py:HSMM(duration_tables=)`, `HSMM.load(durations=)`, `durations_path` |
| visit counts (stage 5) | `scripts/train/fit_visit_counts.py`; `hsmm/visit_counts.py:VisitCounts` |
| NB fit, censoring, shrinkage | `hsmm/train/durations_fit.py:fit_durations_shrunk`, `impute_censored_histogram`, `newton_update_r`, `method_of_moments_r`, `update_p_given_r` |
| clustering metrics | `hsmm/train/metrics.py:adjusted_rand`, `matched_accuracy`, `effective_k` |
| XLA autotuning off (EPIC-scale compiles) | `xla_env.py:disable_gpu_autotuning` |
| tests | `tests/test_joint_em.py`, `test_lexical_init.py`, `test_messages.py`, `test_durations.py`, `test_np_hazard.py` |
