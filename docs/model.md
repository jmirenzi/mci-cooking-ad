# The model: a joint explicit-duration HSMM

This doc covers the probabilistic model every HSMM detector reads: the generative story, the
tables it is stored as, the belief b_t(r, i, j) and the filter that updates it one tick at a time,
the backward messages that smooth it, and the whole-trial readouts r̂ and Viterbi.

All of it lives in `hsmm/`. `class HSMM` (`hsmm/model.py`) is the one object detectors hold. How
the parameters are fitted is in [`training.md`](training.md). How the detectors turn beliefs into
alarms is in [`detectors.md`](detectors.md).

Notation: t is a tick, r a recipe, i or k a subtask, and j the column of the elapsed-time axis
(column j means "j + 1 ticks into the current segment"). o_t = (v_t, n_t) is the observed
(verb, noun) pair. K_R, K and D are the numbers of recipes, subtasks and duration columns
(D = D_max).

---

## 1. The generative story and its tables

A trial picks one recipe, then a sequence of segments. Each segment has a subtask and a duration,
and emits one (verb, noun) pair per tick:

$$
\begin{aligned}
R &\sim \operatorname{Cat}(\pi) && \text{once per trial}\\
z_1 \mid R=r &\sim \operatorname{Cat}(\pi^{\text{init}}_r)\\
d_m \mid z_m=k, r &\sim \operatorname{NB}_{\ge 1}(\rho_{rk}, p_{rk}) && \text{segment } m \text{'s length}\\
v_t \mid z_m=k &\sim B^v_{k\cdot},\quad n_t \mid z_m=k \sim B^n_{k\cdot} && \text{i.i.d. over the segment's ticks}\\
z_{m+1} \mid z_m=k, r &\sim A^{(r)}_{k\cdot}, \qquad A^{(r)}_{kk} = 0 .
\end{aligned}
$$

Three choices carry through the rest of the repo:

- **Explicit durations.** A plain HMM gives geometric durations, whose mode is always one tick.
  The negative binomial (on d ≥ 1) has a usable hazard, so "stuck" and "left early" are
  questions the model can answer.
- **No self-transitions** (A_kk = 0). A run of constant z is exactly one segment, so a Viterbi
  path run-length-encodes losslessly.
- **Product emissions**, P(v, n | i) = P(v | i) P(n | i), shared across recipes. The product is
  what lets a detector say "wrong object" apart from "wrong action". Sharing means a recipe
  differs only in *when* things happen and *how long* they take, never in what an action looks
  like.

**Stored form.** `params.JointHSMMParams` holds seven arrays: Dirichlet counts
`init_counts (K_R, K)`, `trans_counts (K_R, K, K)`, `verb_counts (K, V)`, `noun_counts (K, N)`,
`pi_counts (K_R,)`, and NB point estimates `dur_r, dur_p (K_R, K)`. `params.to_log_tables` turns
them into the `LogTables` everything else reads:

| Table | Shape | Definition |
|---|---|---|
| `log_pi`, `log_init`, `log_A`, `log_emit_v`, `log_emit_n` | as above | clipped Dirichlet MAP: θ_w ∝ max(c_w − 1, FLOOR), FLOOR = 1e-12; `log_A`'s diagonal is masked before the row sum |
| `log_pmf` | (K_R, K, D) | log P(D = j + 1); the last column holds P(D ≥ D) so the infinite tail is absorbed, not cut |
| `log_surv` | (K_R, K, D) | log P(D ≥ j + 1) |
| `log_h` | (K_R, K, D) | hazard log P(D = j+1 \| D ≥ j+1); the last column is 0, so a segment always ends at D |
| `log_stay` | (K_R, K, D) | log(1 − h) |
| `log_A_marginal` | (K, K) | log Σ_r π_r A^(r): "how ordinary is this transition in general" |

The NB tails use the regularised incomplete beta: P(D ≤ d) = I_p(ρ, d), P(D ≥ d) = 1 − I_p(ρ,
d − 1) (`hsmm/durations.py`). Its mean is 1 + ρ(1 − p)/p; forgetting the +1 is a known bug source.

**The MAP mode erases thin evidence.** Subtracting one count is nothing for a cell with hundreds
and everything for a cell with one: a bigram seen once lands on FLOOR, the same as one never seen.
On the fitted emissions this is near-total. The lexical warm start gives each state a background
mass of exactly 1.0, the mode removes it, and 97% of emission cells sit on FLOOR. Each emission
row is therefore close to a delta on its own (verb, noun) pair. Two consequences recur in
[`detectors.md`](detectors.md): no other state "accidentally" explains a stray token, and a near
substitution and a far one score identically. Training's transition smoothing
([`training.md`](training.md) §3) exists to undo this for the transition rows.

**Emissions per tick.** `kernels.sequence_loglik` gives e_t(i) = log B^v_{i,v_t} + log
B^n_{i,n_t} as a (T, K) array; padded ticks are 0.

---

## 2. The belief table

The live state of the model is one table over (recipe, subtask, elapsed time):

$$
b_t(r, i, j) = P\bigl(R = r,\ z_t = i,\ e_t = j{+}1 \mid o_{0:t}\bigr), \qquad \text{shape } (K_R, K, D),
$$

and its prediction before o_t is seen, b'_t (the same with o_{0:t−1}). Both are stored in log
space. Because the recipe is one variable per trial, adding it as an axis is exact: this is the
joint model's forward pass, read out every tick. A pinned recipe is K_R = 1.

**Before the first tick**, `kernels.initial_belief` puts log P(r) + log π^init_r(i) in column 0
and −∞ elsewhere.

**Which prior over r.** `HSMM.view(recipe)` builds the per-trial tables:

| `recipe` | Meaning | Belief axis |
|---|---|---|
| `None` | infer the recipe: the fitted P(r) | all K_R recipes |
| an `int` | the recipe is known (e.g. the labelled dish) | that one recipe |
| a (K_R,) array | a log prior over recipes | all K_R, renormalised |

`View.recipes` is the slice of the recipe axis the trial uses; detectors apply it to their own
per-recipe tables (the bad-set bins in [`detectors.md`](detectors.md) §2).

---

## 3. The filter

One tick is `predict` then `update`. `kernels.filter_scan` scans exactly those two kernels over a
padded trial, so a live run (`HSMM.predict` / `HSMM.update` per tick) and a whole-trial run
(`HSMM.filter`) agree by construction. `tests/test_hsmm_model.py` checks that bitwise on real
trials. Cost per tick: O(K_R K (K + D)).

### 3.1 Predict

Mass in cell (r, k, j) either ends now, with hazard h, or ages one column:

$$
\begin{aligned}
\text{end}_t(r,k,j) &= b_{t-1}(r,k,j)\,h_{rkj}, \qquad m_t(r,k) = \textstyle\sum_j \text{end}_t(r,k,j)\\
b'_t(r,i,0) &= \textstyle\sum_k m_t(r,k)\,A^{(r)}_{ki} && \text{a new segment starts: column 0}\\
b'_t(r,i,j) &= b_{t-1}(r,i,j{-}1)\,(1 - h_{r,i,j-1}), \quad j \ge 1 && \text{the segment continues}
\end{aligned}
$$

A segment that reaches column D − 1 ends there with certainty (h = 1). `kernels.predict` returns
`(b_pred, log_end, log_m)`; the Live event channels need `log_end` and `log_m`
([`detectors.md`](detectors.md) §3.2), and `HSMM.predict()` wraps them as a `Forecast`.

### 3.2 Update

Multiply by the emission and normalise over **all** of (r, i, j):

$$
b_t(r,i,j) = \frac{b'_t(r,i,j)\,P(o_t \mid i)}{P(o_t \mid o_{0:t-1})}, \qquad
P(o_t \mid o_{0:t-1}) = \sum_{r,i,j} b'_t(r,i,j)\,P(o_t \mid i).
$$

The normaliser is the one-step predictive likelihood. `HSMM.update` returns its log.

**Exactness check.** Σ_t log P(o_t | o_<t) must equal log Σ_r P(r) P(o | r), the marginal the
segment-form forward pass returns (§6). If they disagree, the filter is wrong.
`tests/test_hsmm_model.py` checks it.

### 3.3 Readouts

| Readout | Definition | Used by |
|---|---|---|
| π̃_t(i) | Σ_{r,j} b'_t(r,i,j) = P(z_t = i \| o_<t) | the occupancy before o_t; emission scoring |
| stay_t(i) | Σ_r Σ_{j ≥ 1} b'_t(r,i,j) | "the segment continued at t" |
| switch_t(i) | Σ_r b'_t(r,i,0) | "a segment began at t" |
| w_t(r) | Σ_{i,j} b_t(r,i,j) = P(r \| o_{0:t}) | `HSMM.recipe_belief` |

`kernels.occupancy` returns all three of π̃, stay and switch. stay + switch = π̃. The Live
emission test uses the split ([`detectors.md`](detectors.md) §3.1).

---

## 4. One tick at a time: `class HSMM`

```python
hsmm = HSMM.load("dataset/processed/breakfast/joint_params_train_lexhard.npz", d_max=200)
hsmm.reset(recipe=None)          # or an int, or a (K_R,) log prior
for v, n in zip(verbs, nouns):
    forecast = hsmm.predict()    # Forecast(b_pred, log_end, log_m); .pi_tilde .stay .switch
    log_p = hsmm.update(v, n)    # log P(o_t | o_<t)
    hsmm.belief, hsmm.recipe_belief, hsmm.occupancy
```

`HSMM.filter(verbs, nouns, recipe)` returns the same quantities for every tick at once as a
`FilterResult` (`log_b`, `log_norm`, `pi_tilde`, `log_stay`, `log_switch`, and the padded `loglik`
and `mask` the backward pass reuses). Trials are padded to a multiple of `BUCKET` = 64 ticks, so
jit compiles once per bucket rather than once per trial length.

---

## 5. Backward messages: lag and the smoother

To condition the belief at t on L more ticks, multiply it by a backward message

$$
\beta_t(r,i,j) = \log P\bigl(o_{t+1:\,\min(t+L,\,T-1)} \mid R=r,\ z_t=i,\ e_t=j{+}1\bigr).
$$

One backward step (`kernels.backward_step`) mirrors predict: either the segment continues into
column j + 1 and emits o_{t+1}, or it ends (hazard h) and a new segment starts in column 0 of some
k via A:

$$
\beta_t(r,i,j) = \log\Bigl[(1-h_{rij})\,P(o_{t+1}\mid i)\,e^{\beta_{t+1}(r,i,j+1)}
+ h_{rij} \textstyle\sum_k A^{(r)}_{ik}\,P(o_{t+1}\mid k)\,e^{\beta_{t+1}(r,k,0)}\Bigr].
$$

`HSMM.smooth(fr, L)` takes a `FilterResult` (so emissions are not recomputed). L = `math.inf` is
the full smoother (`kernels.full_beta`, one reverse scan); a finite L runs L backward steps for
every t at once (`kernels.fixed_lag_beta`); L = 0 is all zeros. The full-lag smoother reproduces
training's E-step occupancies and expected transition counts (`tests/test_live.py`).

---

## 6. One recipe in segment form, π̃ under r̂, and r̂

Training and the Hindsight detector use the classic explicit-duration recursions (Yu 2010) for
one recipe at a time, in segment form. Let F(u, k) be the probability that a segment in state k
starts at u, together with o_{0:u−1}. With L(a, b, k) = Π_{t=a..b} P(o_t | k), computed in O(1)
from a cumulative sum (`kernels.padded_cumsum`):

$$
\begin{aligned}
A^{*}(t,k) &= \textstyle\sum_d F(t{-}d{+}1,k)\,P(D{=}d\mid k)\,L(t{-}d{+}1,t,k) && \text{a segment ends exactly at } t\\
A^{\text{occ}}(t,k) &= \textstyle\sum_d F(t{-}d{+}1,k)\,P(D{\ge}d\mid k)\,L(t{-}d{+}1,t,k) && \text{a segment covers } t\\
F(t{+}1,k') &= \textstyle\sum_k A^{*}(t,k)\,A_{kk'}, \qquad P(o_{0:t}) = \textstyle\sum_k A^{\text{occ}}(t,k).
\end{aligned}
$$

**Right-censoring costs nothing here.** The normaliser uses the survival-weighted A^occ, so a
trial that stops mid-segment is scored correctly with no special case. Every trial's last
segment is censored: recording stopped, not the activity.

`kernels.forward_pass` returns the per-tick log normaliser; its sum is log P(o | r).
`kernels.predictive_occupancy` runs the same recursion and reads out π̃_t(i) under that one
recipe, using L(·, t − 1, ·) so o_t is not seen. Hindsight scores emissions against this π̃ under
r̂ ([`detectors.md`](detectors.md) §5.1).

**The recipe.**

$$
\hat r = \arg\max_r\ \log \pi_r + \log P(o \mid r).
$$

`HSMM.infer_recipe` returns (r̂, log P(r | o), log P(o)); `hsmm/batch.infer_recipe` does the
same for many trials at once. It is one forward pass per recipe and needs no backward pass.

Two quantities here could also be read off the filter, and both choices were measured:

- **r̂.** argmax_r w_{T−1}(r) agrees with the per-recipe forward pass on 582/582 Breakfast
  trials. The forward pass is kept because the full (r, i, j) filter costs several times more.
- **π̃ under r̂.** π̃ from `HSMM.filter(recipe=r̂)` is the same quantity mathematically.
  Numerically it flips one hindsight flag on 1/582 trials, so `predictive_occupancy` is kept.

---

## 7. Viterbi

`kernels.viterbi_decode` is the forward pass with log-sum replaced by max, plus backpointers:
the best duration for each (t, k) under the pmf (`dur_bp`), the same under the survival function
(`dur_bp_surv`), and the best predecessor state (`prev_bp`). The last tick uses the survival
version, because the final segment is censored:

$$
\hat k^{*} = \arg\max_k A^{\text{surv}}(T{-}1, k).
$$

`kernels.traceback` (numpy) pops the final segment with `dur_bp_surv` and every earlier one with
`dur_bp`. The result is a list of `(subtask, duration)` segments; z*_t is its run-length
expansion. The MAP path chooses its own number of segments.

**Exact ties.** Some trials have two segmentations with exactly equal scores. The old code broke
those ties differently from run to run; the current kernels are deterministic. This is why the
restructure's train-split grids moved by up to 0.004 ([`logbook.md`](logbook.md), 2026-10-15).

`HSMM.viterbi(verbs, nouns, recipe)` decodes one trial under one recipe; `hsmm/batch.viterbi`
decodes many, each under its own r̂.

---

## 8. Code

| Concept | Where |
|---|---|
| parameters, MAP normalisation, FLOOR | `hsmm/params.py:JointHSMMParams`, `_row_normalize`, `to_log_tables` |
| hazard and stay tables | `hsmm/params.py:hazard_tables` |
| recipe-marginal transitions | `hsmm/params.py:marginal_log_trans` |
| one recipe's parameters | `hsmm/params.py:select_recipe` → `RecipeView` |
| load / save (raises on unknown keys) | `hsmm/params.py:load`, `save` |
| NB pmf, survival, cdf, D-column tables | `hsmm/durations.py:nb_log_*`, `duration_tables` |
| e_t(i) | `hsmm/kernels.py:sequence_loglik` |
| initial belief, predict, update, readouts | `hsmm/kernels.py:initial_belief`, `predict`, `update`, `occupancy` |
| the whole-trial filter | `hsmm/kernels.py:filter_scan`; `hsmm/model.py:HSMM.filter` → `FilterResult` |
| one tick at a time | `hsmm/model.py:HSMM.reset`, `HSMM.predict` → `Forecast`, `HSMM.update` |
| per-trial recipe prior | `hsmm/model.py:HSMM.view` → `View` |
| backward messages, lag, smoother | `hsmm/kernels.py:backward_step`, `fixed_lag_beta`, `full_beta`; `hsmm/model.py:HSMM.smooth` |
| segment-form forward pass, π̃ under one recipe | `hsmm/kernels.py:forward_pass`, `predictive_occupancy` |
| r̂ | `hsmm/model.py:HSMM.infer_recipe`; `hsmm/batch.py:infer_recipe` |
| Viterbi | `hsmm/kernels.py:viterbi_decode`, `traceback`; `hsmm/model.py:HSMM.viterbi`; `hsmm/batch.py:viterbi` |
| many trials, padded once | `hsmm/batch.py:pad`, `occupancy` |
| tests | `tests/test_hsmm_model.py`, `test_viterbi.py`, `test_messages.py`, `test_durations.py` |
