# `src/cook_ad/synthetic/` — ground truth by construction

| File | Role |
|---|---|
| `generate.py` | ancestral sampling from a fitted model; adapters for real trials |
| `error_injection.py` | the five canonical error types, each with a ground-truth window |

The evaluation problem: real Breakfast trials come with no anomaly labels. Nobody annotated
"here the participant forgot to add sugar". So the harness **manufactures** anomalies whose
location is known exactly, and measures whether the detector finds them.

---

## 1. `generate.py` — sampling healthy trajectories

### Ancestral sampling

`sample_trajectory` walks the generative story of [`hsmm.md`](hsmm.md) forward:

$$
z_1 \sim \boldsymbol\pi^{\text{init}}; \qquad
d_j \sim \operatorname{NB}_{\ge1}(r_{z_j}, p_{z_j}); \qquad
v_t \sim B^v_{z_j,\cdot},\ n_t \sim B^n_{z_j,\cdot} \ \text{i.i.d.}; \qquad
z_{j+1} \sim A_{z_j,\cdot} .
$$

The duration draw is `scipy.stats.nbinom.rvs(r, p) + 1`, matching the $d' = d - 1$ shift the model
uses, then clamped to $[1, D_{\max}]$. The transition draw needs no rejection sampling for the
banned diagonal — `normalize_categoricals` has already made $A_{kk} = 0$ exactly, so
`rng.choice(p=trans_p[state])` can never return `state`.

The final segment is trimmed (`min(..., max_ticks - total)`) so every trajectory is exactly
`max_ticks` long — which incidentally reproduces the right-censoring the real corpus has.

Output:

```python
{"verb_ids", "noun_ids", "segments": [(state, d), ...], "subtask_per_tick"}
```

**Why this matters: the ground truth is exact.** We know every segment boundary and every state
because we *sampled* them. There is no decoding step to be wrong. This isolates detector
performance from model misfit — if the detector fails on data drawn from the model's own
distribution, the problem is the detector.

### The joint sampler

`sample_trajectory_joint` draws $R \sim \boldsymbol\pi$ **once per trial**, then samples from that
recipe's own init/transition/duration tables and the **shared** emissions. Synthetic healthy trials
are therefore genuinely recipe-structured, with the sampled recipe id returned as extra ground
truth. The trajectory dict gains a `"recipe_id"` key; consumers that only read
`verb_ids`/`noun_ids`/`segments` (i.e. all of `error_injection`) are unaffected — an additive
schema change by design.

### The real-data adapters

`trajectory_from_real` Viterbi-segments a real Breakfast trial with the fitted model and packages
it in the **same dict shape**. That is the whole trick: injectors and metrics then run unchanged on
both healthy sources.

The honest caveat is in the docstring — here the "ground-truth" segment boundaries are *decoded*,
not sampled, so they are fuzzier. A boundary the model placed two ticks late shifts the injection
window by two ticks and shows up as apparent latency. That is a real limitation of the real-data
arm of the evaluation, stated rather than hidden.

`trajectory_from_real_joint` does the same via `infer_recipe` → `segment_all_conditioned`.

---

## 2. `error_injection.py` — the five error types

These are the canonical failure modes of a person with MCI following a familiar procedure. Each
injector returns

```python
{"verb_ids", "noun_ids", "window": (t0, t1), "error_type"}
```

where `window` is the ground-truth extent used by `eval.metrics.score_trial`.

| Type | Perturbation | Window | Channel it should light up |
|---|---|---|---|
| **substitution** | one tick's noun → that state's **least likely** noun | the single tick | $s_{\text{noun}}$ (with attribution `"item"`) |
| **abandonment** | truncate an interior segment to ~1 tick | premature end tick | retrospective $s_{\text{dur2}}$, left tail |
| **omission** | delete an interior segment entirely | the new boundary | $s_{\text{trans}}$ |
| **transposition** | swap two adjacent interior segments | the swapped pair | $s_{\text{trans}}$ (twice) |
| **repetition** | duplicate an interior segment in place | the inserted copy | $s_{\text{trans}}$, or $s_{\text{temporal}}$ once Viterbi merges |

**Substitution** is the only injector that needs the model — it reads $B^n$ to pick
$\arg\min_n P(n \mid Z{=}k)$, so the perturbation is *genuinely* anomalous rather than merely
different (with a fallback to the second-least-likely if that already equals the observed noun).
Only the noun is touched, isolating the item channel from the action channel. The other four are
purely structural index surgery on `verb_ids`/`noun_ids`.

**Abandonment exists to test the left tail.** As explained in [`anomaly.md`](anomaly.md), the live
survival channel structurally *cannot* catch a step that ends early: short durations always have
high survival probability. Only $-\log P(D \le d)$ can. Abandonment is the injection that proves
that channel is doing work.

**Repetition is genuinely ambiguous**, and the docstring says so. Duplicating a segment either
creates an impossible $k \to k$ re-entry (caught by $s_{\text{trans}}$, since $A_{kk} = 0$) *or*
gets merged by Viterbi into one over-long segment (caught by $s_{\text{temporal}}$). Which one
happens depends on the decode. Both are correct detections; the attribution matrix simply shows the
split.

### Interior constraints

Each injector restricts its segment choice, and the constraints are not arbitrary:

```python
substitution   lo=0, hi=len            # anywhere
abandonment    lo=1, hi=len            # non-first: needs preceding context
omission       lo=1, hi=len-1          # interior: needs both neighbours
transposition  lo=1, hi=len-2          # i and i+1 must both be interior
repetition     lo=1, hi=len-1          # interior
```

`MIN_SEGMENTS = 4` gates the driver: trajectories with fewer segments are skipped entirely, since
an out-of-order step is only out-of-order *relative to context*.

### Segment selection modes

`_pick_segment(rng, n, lo, hi, select)`:

- `"random"` — uniform over the valid range. This is the **honest recall** number: it measures
  detection on a typical perturbation, not a cherry-picked one.
- `"hardest"` — currently the leftmost valid index, an explicitly labelled **deterministic
  placeholder**. A truly adversarial pick would score each candidate's *induced surprise* and
  choose the minimum; that is noted as future work rather than pretended to be implemented.

`export_anomaly.py` exploits this honestly: it sweeps `[("hardest", 0)] + [("random", s) for s in SEEDS]`
and explains in a comment that `"hardest"` ignores the rng entirely (every remaining rng use is
deterministic once the segment is picked), so it is tried with only one seed instead of repeating
identical work.

### One error type that is *not* here

`run_rollout_demo.py` defines its own **stall** injector — stretch a segment by repeating its final
`(verb, noun)` well past its observed duration. It lives there rather than in this module because
the five canonical error types don't include a pure stall, but it deliberately matches
`_result`'s dict shape so it slots into the same downstream code path (`score_trial`, `narrate`)
with no special-casing.

---

## 3. How this composes into an evaluation

```
fitted params
   ├── generate_healthy(n)  ─────────────────┐   exact ground truth
   └── trajectory_from_real(each trial) ─────┤   decoded ground truth
                                             ▼
                          for each error_type ∈ ERROR_TYPES:
                              inject(...) → (perturbed stream, window)
                                             ▼
                            eval.batch.compute_traces → surprise.flag
                                             ▼
                          eval.metrics.evaluate → recall / precision /
                                                  latency / attribution
```

The healthy trials are run through the *same* trace-and-flag path with no injection, and any
persistent flag they produce is a false positive. That shared pool is what makes precision
meaningful — see [`eval.md`](eval.md) for why it is also reported with the healthy pool excluded.
