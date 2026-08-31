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
{"verb_ids", "noun_ids", "window": (t0, t1), "error_type", "tick_map", "edited_ticks"}
```

where `window` is the ground-truth extent used by `eval.metrics.score_trial`, and the last two
fields record *what the injector actually changed* (§2.1). Substitution adds three more —
`channel` (`"verb"` or `"noun"`), `orig_id` and `new_id` — so a caller can partition its
injections afterwards, e.g. separating genuine near substitutions (`milk` → `water`) from
annotation variants (`egg` → `eggs`). All additive; consumers reading only the six above are
unaffected.

| Type | Perturbation | Window | Channel it should light up |
|---|---|---|---|
| **substitution** | one whole segment's verb **or** noun (channel picked at random) → a token with no training evidence for that state, or its nearest embedding neighbour under `select="near"` (§2.2) | the whole segment | $s_{\text{noun}}$ or $s_{\text{verb}}$ (attribution `"item"`/`"action"` to match) |
| **abandonment** | truncate an interior segment to a random 5-20% of its own duration | premature end tick | retrospective $s_{\text{dur2}}$, left tail |
| **omission** | delete an interior segment entirely | the new boundary | $s_{\text{trans}}$ |
| **transposition** | swap two adjacent interior segments | the swapped pair | $s_{\text{trans}}$ (twice) |
| **repetition** | duplicate an interior segment in place | the inserted copy | $s_{\text{trans}}$, or $s_{\text{temporal}}$ once Viterbi merges |

**Substitution** is the only injector that touches `hsmm_params`, but it no longer takes an
argmin over the fitted probability row for its `"random"` case. It reads the state's raw
accumulated E-step pseudocounts (`hsmm_params.verb_counts`/`noun_counts` — already computed once,
during fitting) and draws uniformly from whichever tokens sit within `SUBSTITUTION_UNSEEN_BAND`
of that row's own floor: essentially no training evidence links them to this state, as opposed to
the single point the fitted distribution ranks worst. That decouples the `"random"` replacement
from the exact distribution that later scores it — the perturbation is no longer engineered to be
the worst possible case under the very model being evaluated — with a fallback to the single
least-observed token (excluding the current one) if the band is empty. Exactly one channel
changes, chosen at random per injection and applied uniformly across the whole segment rather
than a single mid-step tick; the other channel is untouched, which is what still isolates the
item channel from the action channel. `"hardest"` deliberately reintroduces the deterministic
worst case for both the channel choice (whichever channel has the single least-observed cell for
this state) and the token (`_argmin_candidate`, the argmin over that channel's raw counts) — see
§2.4 for why the two modes now diverge on more than just which segment gets picked. The other four injectors remain purely structural index surgery on
`verb_ids`/`noun_ids`, needing no model at all.

**Abandonment exists to test the left tail.** As explained in [`anomaly.md`](anomaly.md), the live
survival channel structurally *cannot* catch a step that ends early: short durations always have
high survival probability. Only $-\log P(D \le d)$ can. Abandonment is the injection that proves
that channel is doing work.

**Repetition is genuinely ambiguous**, and the docstring says so. Duplicating a segment either
creates an impossible $k \to k$ re-entry (caught by $s_{\text{trans}}$, since $A_{kk} = 0$) *or*
gets merged by Viterbi into one over-long segment (caught by $s_{\text{temporal}}$). Which one
happens depends on the decode. Both are correct detections; the attribution matrix simply shows the
split.

### 2.1 `tick_map` and `edited_ticks` — provenance for every degraded tick

`window` says *where* the anomaly is. It does not say which degraded ticks came from which original
ticks, and after a deletion, insertion or swap the two index spaces no longer line up. Two
additional fields carry that correspondence.

**`tick_map`** is `int[T_degraded]`, mapping each degraded tick back to the original tick it came
from. It is exactly the concatenation each injector already applies to `verb_ids`/`noun_ids`, run
instead on `arange(T)`:

| Injector | `tick_map` |
|---|---|
| substitution | `arange(T)` — identity; content changes, order does not |
| abandonment | `concat(arange(0, cut_lo), arange(cut_hi, T))` |
| omission | `concat(arange(0, start), arange(end, T))` |
| transposition | `concat(arange(0,a_start), arange(b_start,b_end), arange(a_start,a_end), arange(b_end,T))` |
| repetition | `concat(arange(0,end), arange(start,end), arange(end,T))` — many-to-one: the copy's ticks map to their originals |

**`edited_ticks`** lists degraded indices whose *content* differs from the mapped original. Only
substitution is non-empty (`arange(start, end)`, the whole retagged segment); the other four
reorder, drop or duplicate ticks without rewriting any surviving one. Both fields are needed
because neither implies the other: `tick_map` alone would assert substitution's retagged span is
unchanged, and `edited_ticks` alone says nothing about the four structural injectors.

Two consumers depend on this:

- `llm.textify.injection_touched_steps` uses it to identify **injection debris** — steps that exist,
  or take the tick range they take, only because of the injection, and which are excluded from
  false-positive scoring ([`llm.md`](llm.md) §5).
- `eval.counterfactual` uses it to project a healthy trial's flags into the degraded trial's tick
  space, so the two can be differenced ([`eval.md`](eval.md) §5).

The schema change is additive, matching the precedent of `sample_trajectory_joint` adding
`recipe_id`: consumers reading only `verb_ids`/`noun_ids`/`window` are unaffected.

### 2.2 The near substitution — `select="near"`

`"random"` and `"hardest"` both replace with a token that has **no training evidence for that
state** — under any semantic kernel, the most *distant* token available. Neither can therefore
produce a near miss, and a near miss is the interesting case: `milk` → `water` is what a person
with MCI actually does, and it is the one the detector should treat as less severe rather than
merely different.

`select="near"` replaces with the token's **nearest embedding neighbour**, taken from
`neighbours`, a `{channel: (W,) int array}` table built by `hsmm.kernel.nearest_neighbours`. Only
the channels present in that dict are eligible, which is how a noun-only near-substitution
benchmark is requested (the verb embedding space does not pass `tools_embed_vocab.py`'s
neighbour gate — [`hsmm.md`](hsmm.md) §8.6).

**The replacement comes from the embeddings alone and never from `hsmm_params`.** That is the one
property that makes the comparison the mode exists for possible: two models pointed at a single
`--traj-params` source are graded on a byte-identical degraded stream, so nothing but the scoring
model differs. It also means `near` is the only substitution mode that does not consult the fitted
counts at all.

Segment choice is uniform, exactly as in `"random"` — `near` changes *what* replaces the token,
not *where*. `run_detect_eval.py --near-subs` scores it as a separate `substitution_near` group
rather than folding it into `substitution`, since mixing near and far replacements in one recall
number would average away the distinction being measured ([`eval.md`](eval.md) §8).

### 2.3 Interior constraints

Each injector restricts its segment choice to an explicit list of valid indices (built once per
call, not just a bare range), and the constraints are not arbitrary:

```python
substitution   1 <= i < len-1                    # interior: skip the leading/trailing idle
abandonment    1 <= i < len-1  and  d[i] >= 2     # interior AND long enough to truncate at all
omission       1 <= i < len-1                     # interior: needs both neighbours
transposition  1 <= i < len-2                     # i and i+1 must both be interior
repetition     1 <= i < len-1                     # interior
```

Breakfast trials open and close on idle (`stall kitchen`), which carries no task semantics, so
every injector is now interior-only. Substitution and abandonment did not always restrict this
way: substitution originally allowed `i` anywhere (including the first/last segment) and
abandonment allowed the last segment specifically, which together with `keep_ticks` being a fixed
1-tick constant (not the current duration-relative fraction) meant a 1-tick segment could be
"truncated" to itself — a degraded stream byte-identical to the healthy one, still carrying a
ground-truth window and scored as a missed anomaly. Measured on 419 real trials before the fix:
12% of substitutions landed on the first segment and 15% on the last (28% total), 21% of
abandonments landed on the trailing idle, and 3.2% of abandonments were exact no-ops. All three
are gated out now (`d[i] >= 2` on top of the interior constraint eliminates the no-op case, since
the new percent-based `keep` is always strictly less than `d` whenever `d >= 2`).

`MIN_SEGMENTS = 4` gates the driver: trajectories with fewer segments are skipped entirely, since
an out-of-order step is only out-of-order *relative to context*. Abandonment's extra `d[i] >= 2`
filter can still leave zero valid segments on a short, unlucky trajectory even past that gate;
`_pick_segment` raises `ValueError` in that case, the same failure mode every other injector
already has for a too-short trajectory, and every caller already handles it the same way.

### 2.4 Segment selection modes

`_pick_segment(rng, valid_indices, select)`:

- `"random"` — uniform over the valid range. This is the **honest recall** number: it measures
  detection on a typical perturbation, not a cherry-picked one.
- `"hardest"` — currently the leftmost valid index, an explicitly labelled **deterministic
  placeholder**. A truly adversarial pick would score each candidate's *induced surprise* and
  choose the minimum; that is noted as future work rather than pretended to be implemented.
- `"near"` — segment picked uniformly, as in `"random"`; what differs is the **replacement**
  (§2.2). Only substitution implements it; the other four are structural and have no token to
  replace.

`export_anomaly.py` exploits this honestly: it sweeps `[("hardest", 0)] + [("random", s) for s in SEEDS]`
and explains in a comment that `"hardest"` ignores the rng entirely (every remaining rng use is
deterministic once the segment is picked), so it is tried with only one seed instead of repeating
identical work.

This held briefly only for omission/transposition/repetition after substitution and abandonment
gained their own post-segment rng draws (channel, replacement token, keep fraction) — under
`select="hardest"` those draws are now themselves deterministic rather than random: substitution
takes whichever channel has the single least-observed cell for this state and uses that exact
token (`_argmin_candidate`, an argmin over the raw counts, not a random member of the near-zero-
evidence band), and abandonment always keeps the bottom of `ABANDON_KEEP_FRAC` (5%, floored at
1 tick) instead of a random draw from the range. So the comment's property is restored for all
five: `select="hardest"` never touches `rng` at all, and `export_anomaly.py`'s "try `hardest`
with only one seed" optimisation is lossless again.

### 2.5 One error type that is *not* here

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

The healthy trials are run through the *same* trace-and-flag path with no injection, and any flag
they produce is a false positive. That shared pool is what makes precision meaningful — see
[`eval.md`](eval.md) for why it is also reported with the healthy pool excluded.
