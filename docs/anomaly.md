# `src/cook_ad/anomaly/` — surprise, calibration, and narration

This is where the fitted model becomes a *detector*. The chain is:

```
(verb_ids, noun_ids)
   → compute_trace / compute_trace_joint   →  SurpriseTrace   (7 real-valued channels per tick)
   → flag / flag_joint                     →  boolean flags   (per-channel, per-tick)
   → narrate / narrate_joint               →  Query cards     (templated English + a PreferenceEvent)
```

| File | Role |
|---|---|
| `surprise.py` | the seven channels, trace assembly, flagging, attribution, severity |
| `temporal.py` | the duration channels: live stall, retrospective two-sided, PIT calibration |
| `quantile.py` | exact per-state $\alpha$-quantile thresholds over discrete supports |
| `narrate.py` | template renderers turning flags into auditable questions |

---

## 1. The design commitment

Every number a user ever sees traces back to a single argmax over a fitted parameter row. The
`narrate.py` module docstring states it plainly: *"routing any of this through a language model
would throw that away."* Categorical emissions were chosen over embeddings for exactly this
reason. When the system says "I expected jelly", you can point at $B^n_{k,\cdot}$ and see why.

---

## 2. The seven channels

All are **surprisal** $-\log P(\cdot)$ measured in nats, so they share a scale and are additive in
the way log-probabilities are.

| Channel | Formula | Fires at | Catches |
|---|---|---|---|
| $s_{\text{emit}}$ | $-\log \sum_k \tilde\pi_t(k) B^v_{k v_t} B^n_{k n_t}$ | every tick | any unexpected observation |
| $s_{\text{verb}}$ | $-\log \sum_k \tilde\pi_t(k) B^v_{k v_t}$ | every tick | wrong action |
| $s_{\text{noun}}$ | $-\log \sum_k \tilde\pi_t(k) B^n_{k n_t}$ | every tick | wrong object |
| $s_{\text{temporal}}$ | $-\log P(D \ge d_{\text{elapsed}} \mid z^*)$ | every tick (live) | being stuck, **while it happens** |
| $s_{\text{dur2}}$ | two-sided duration $p$-value surprise | segment end | too long **or** too short, retrospectively |
| $s_{\text{trans}}$ | $-\log A_{z_{j-1} z_j}$ | segment start | out-of-order / skipped step |
| $s_{\text{recipe}}$ | cascade: $-\log A^R_{\rho_{j-1}\rho_j}$; joint: signed excess | segment start | recipe-level incoherence |

Non-firing ticks are filled with $0$ (or $-1$ for id fields, `NaN` for the PIT diagnostic).

### 2.1 Emission channels and the predictive weighting

All three emission channels are scored against $\tilde\pi_t$, the **predictive** occupancy
$P(Z_t = k \mid o_{0:t-1})$ from `messages.predictive_occupancy` — *not* the smoothed $\gamma_t$
and not a distribution conditioned on $o_t$. Scoring $o_t$ against a belief that already saw $o_t$
would be circular; scoring against $\gamma_t$ would use the future, which a live system does not
have.

Using the **same** weighting for all three is what puts $s_{\text{verb}}$ and $s_{\text{noun}}$ on a
shared scale, which is what makes their difference meaningful:

$$
\texttt{attribution}(t) = \begin{cases}
\text{"item"} & s_{\text{noun}} - s_{\text{verb}} > m\\
\text{"action"} & s_{\text{verb}} - s_{\text{noun}} > m\\
\text{"none"} & |s_{\text{verb}} - s_{\text{noun}}| \le m
\end{cases}
\qquad m = 2.0\ \text{nats}.
$$

Because $P(v,n\mid Z) = P(v\mid Z)P(n\mid Z)$, "you're doing the right action with the wrong
object" is a statement the model can actually make. That is the product model earning its keep.

### 2.2 What "I expected X instead" means — `conditional_expected` / `joint_expected`

The naive answer is $\arg\max_n B^n_{z^*_t, n}$ — what state $z^*$ usually emits. `SurpriseTrace`
carries that as `expected_noun`/`expected_verb`, but **narration deliberately does not use it**,
for two documented reasons.

*Problem 1 — hindsight contradiction.* $z^*$ comes from Viterbi (which sees the whole trial);
$\tilde\pi_t$ is causal. Right at a boundary the two disagree, producing a self-contradictory
*"that's cup, normally you use cup."*

*Problem 2 — incoherent pairs.* Marginalising the live mixture over just the flagged word can pick
something individually plausible but nonsensical with the *other*, held-constant word — the
docstring's example is **"pour kitchen"**.

The fix is to condition on the trustworthy token:

$$
\hat n \;=\; \arg\max_{n}\ \operatorname*{logsumexp}_k\Bigl[\log\tilde\pi_t(k) + \underbrace{\log B^v_{k, v_t}}_{\text{observed verb, held fixed}} + \log B^n_{k,n}\Bigr].
$$

Down-weighting states incompatible with the fixed token *before* the argmax makes the result
compatible with it by construction.

When **both** channels are independently flagged at the same tick, neither token can anchor the
other, so `joint_expected` picks the single best pair jointly:

$$
(\hat v, \hat n) \;=\; \arg\max_{v,n}\ \operatorname*{logsumexp}_k\Bigl[\log\tilde\pi_t(k) + \log B^v_{kv} + \log B^n_{kn}\Bigr],
$$

which can never invent a combination the model's own joint mixture doesn't support.

### 2.3 Temporal channels — `temporal.py`

**Live stall.** Within a segment, with $d_{\text{elapsed}}$ resetting to 1 at each segment start:

$$
s_{\text{temporal}}(t) \;=\; -\log P\bigl(D \ge d_{\text{elapsed}} \mid z^*_t\bigr).
$$

Two properties make this the right object:

- It is **monotonically non-decreasing within a segment** — the longer you persist, the deeper into
  the upper tail you are.
- It **is** a tail probability, so a single global threshold $-\log\alpha$ is *automatically
  per-state calibrated*: it fires exactly when the elapsed duration passes that state's $\alpha$
  upper-tail quantile, whether that state normally takes 3 ticks or 90.
- Survival, not pmf, is the statistically correct object for an in-progress (right-censored)
  segment.

Values past $D_{\max}$ clamp to $P(D \ge D_{\max})$.

**Retrospective, two-sided.** Once a segment closes at observed duration $d$:

$$
s_{\text{long}} = -\log P(D \ge d), \qquad
s_{\text{short}} = -\log P(D \le d),
$$
$$
s_{\text{dur2}} \;=\; -\log\Bigl(\min\bigl\{1,\ 2\min\bigl(P(D{\ge}d),\, P(D{\le}d)\bigr)\bigr\}\Bigr),
$$

the standard two-sided $p$-value, clipped at 0. Attribution is `"stuck"` if the right tail is the
smaller one, else `"left_early"`.

**The left tail exists because the live channel structurally cannot catch abandonment.** A short
duration always has *high* survival probability, hence *low* $s_{\text{temporal}}$ — leaving a step
early is invisible to a monotone survival ramp, by construction. It needs its own retrospective
channel.

**PIT — a model check, not a detector.** The mid-probability integral transform per segment:

$$
u \;=\; P(D \le d-1) \;+\; \tfrac12\, P(D = d).
$$

If the duration model fits, $u$ for healthy segments is approximately $\mathrm{Uniform}[0,1]$ —
flat histogram, mean $\approx 0.5$. Systematic deviation indicates **duration-model misfit**, not
user anomaly. The $\tfrac12 P(D{=}d)$ term is the standard discrete correction (exact PIT
uniformity for a discrete variable needs the randomised transform).

### 2.4 Transition channels

$$
s_{\text{trans}} \;=\; -\log A_{z_{j-1},\, z_j} \quad\text{at segment } j\text{'s first tick.}
$$

Cascade recipe channel: the same thing one level up, evaluated at **every** segment boundary
(recipe self-transitions are legal and expected, so there is no boundary to skip):

$$
s_{\text{recipe}} \;=\; -\log A^{R}_{\rho_{j-1},\, \rho_j}.
$$

**The joint model repurposes this channel entirely.** With per-recipe transition matrices, a more
useful question is *"is this transition ordinary in general but wrong for **this** recipe?"* So:

$$
\boxed{\;s_{\text{recipe}} \;=\; \log \bar A_{ji} \;-\; \log A^{(\hat r)}_{ji}\;}
\qquad \bar A = \sum_r \pi_r A^{(r)} .
$$

This is a **signed excess**, not a neg-log-probability. It can be negative, which has real
consequences downstream (§3.3).

---

## 3. `quantile.py` — where the thresholds come from

### 3.1 The idea

The two duration channels are already tail probabilities, so $-\log\alpha$ is a per-state
calibrated threshold for free. The five categorical/transition channels are **not** — a state
whose noun distribution is nearly deterministic produces enormous surprisal for any deviation,
while a diffuse state produces middling surprisal even for genuinely wrong tokens. A single
hand-tuned nat threshold is badly miscalibrated across states.

So those five get the *same* $\alpha$-tail treatment, computed **exactly** rather than
parametrically, because their support is a finite known vocabulary:

$$
\tau_k \;=\; \max\Bigl\{\tau : \ P_k\bigl(s > \tau\bigr) \le \alpha \Bigr\},
\qquad \alpha = 0.05,
$$

where $P_k$ is the model's **own** distribution for state $k$ — the null hypothesis being
"this observation came from state $k$ as the model believes it".

### 3.2 `_tail_threshold` — the mechanics

Given aligned `(scores, probs)` over one state's discrete support:

1. Drop zero-probability entries (e.g. the masked self-transition diagonal) — they carry no null
   mass.
2. Sort by descending score.
3. **Collapse ties.** Under strict `>`, a threshold set at a tied score excludes the *whole* tie
   group, so achievable tail mass is indexed by distinct score values, not by array position.
4. Cumulate group masses and take the largest number of leading groups whose total is $\le \alpha$.

This is a **step function**: $\alpha$ cannot be hit exactly, only the nearest achievable point at
or below it. The docstring is explicit that the achieved mass should be reported and exact
$\alpha$ coverage never claimed.

### 3.3 The four table builders

| Function | Support scored | Used by |
|---|---|---|
| `categorical_quantile_threshold` | one emission row, $V$ or $N$ entries | $s_{\text{verb}}$, $s_{\text{noun}}$ |
| `joint_quantile_threshold` | the $V \times N$ outer product per state | $s_{\text{emit}}$ |
| `transition_quantile_threshold` | one transition row | $s_{\text{trans}}$, cascade $s_{\text{recipe}}$ |
| `excess_quantile_threshold` | signed excess, null $i \sim A^{(\hat r)}(\cdot\mid j)$ | joint $s_{\text{recipe}}$ |

`joint_quantile_threshold` builds the full joint per state — using conditional independence,
$-\log P(v,n\mid k) = -\log B^v_{kv} - \log B^n_{kn}$ — because the joint tail is *not* recoverable
from the two marginal tails.

`excess_quantile_threshold` scores its null under $A^{(\hat r)}$, since that is the distribution
the observed transition is actually drawn from. **Its threshold can be negative** (when the
recipe-conditioned row is flatter than the marginal, most transitions score below zero), which is
why `flag_joint` must mask non-boundary ticks explicitly rather than relying on `0 > τ` being
trivially false.

### 3.4 The mixture-dilution correction

Here is the subtlest piece in the package, and it is a genuine correctness argument rather than a
tuning knob.

The emission channels are computed against the **mixture** $\tilde\pi_t$, but calibrated against
$z^*$'s **single-state** distribution. Since

$$
\sum_k \tilde\pi_t(k) P(o_t \mid k) \;\ge\; \tilde\pi_t(z^*)\, P(o_t \mid z^*),
$$

taking $-\log$ of both sides gives

$$
\boxed{\;s_{\text{emit}}(t) \;\le\; -\log \tilde\pi_t(z^*_t) \;+\; s_{\text{pure}}(t)\;},
\qquad s_{\text{pure}} = -\log P(o_t \mid z^*).
$$

So the mixture can inflate surprise above $z^*$'s own view by **at most** $-\log\tilde\pi_t(z^*)$.
Adding that same per-tick offset to the threshold cancels the dilution exactly:

$$
\tau^{\text{emit}}_t \;=\; \tau_{z^*_t} \;-\; \log \tilde\pi_t(z^*_t).
$$

An observation inside $z^*$'s own $\alpha$-quantile can then **never** be flagged by dilution
alone. That is one-sided and provable, not heuristic — and it is tight only by the amount of
*actual* dilution present at that tick.

An earlier version used a flat floor (`max(threshold, 6.0/4.0/4.0)` nats). It was measured to sit
**above** the maximum achievable quantile threshold on every state at both mini ($K{=}20$, max
2.60/2.52 nats) and full scale ($K{=}64$, max 3.30/2.87), making the entire quantile calibration
dead code. The commentary in `surprise.py` preserves that finding.

### 3.5 The $z^*$-indexing approximation, and its diagnostic

Thresholds are indexed by the hard Viterbi state $z^*_t$, which is only exact when the filtered
belief is concentrated there. `SurpriseTrace` therefore carries

$$
\texttt{belief\_concentration}(t) = \max_k \tilde\pi_t(k),
$$

and `belief_diagnostic(traces, cutoff=0.8)` reports what fraction of ticks fall below the cutoff,
pooled and per-trial. The docstring names the eventual fix (per-tick mixture-quantile scoring) and
declares it out of scope — the limitation is measured and surfaced, not silently absorbed.

---

## 4. `surprise.py` — assembly and flagging

### Trace assembly

`assemble_trace` (cascade) and `assemble_trace_joint` are pure numpy over one trial and share all
per-channel logic. The joint version differs only in *which tables it slices*: transitions and
durations come from $\hat r$'s slice of the $K_R$ axis, emissions stay shared, and $s_{\text{recipe}}$
becomes the signed excess. Two placement helpers do the bookkeeping:

- `_scatter_segment_end` — one value per segment at its **last** tick (completion events: the
  retrospective duration channels, PIT, temporal attribution).
- `_scatter_from_previous` — at each segment's **first** tick, the id of the *previous* segment.
  This is what lets thresholds be indexed by the **FROM** side of a transition, which is the side
  whose row defines the null.

### Flag rules

```python
flags[ch][t]  =  value[t] > threshold[t]
```

with `_base_flags` covering the six shared channels and each of `flag` / `flag_joint` adding its
own $s_{\text{recipe}}$ variant (cascade indexes by `from_recipe`; joint by `from_state` under
$\hat r$).

Only the two duration channels accept a **scalar** override, via `DEFAULT_THRESHOLDS`. Passing a
scalar for any of the five per-state channels raises `KeyError` with an explanatory message —
because doing so would reintroduce precisely the miscalibration §3 exists to fix. Tuning is done
by moving $\alpha$.

### Severity

$$
\text{ratio} = \frac{\text{value}}{\text{threshold}}, \qquad
\text{label} = \begin{cases}
\text{low} & \text{ratio} < 1.5\\
\text{medium} & 1.5 \le \text{ratio} < 3\\
\text{high} & \text{ratio} \ge 3
\end{cases}
$$

`flagged_tick_severity` recomputes this from the **same** value/threshold pairing `_base_flags`
gated on, so a tick's rendered marker colour always matches what its query card would say.

*Known gap, documented in `render_anomaly_png.py`:* a near-deterministic transition row can have a
tiny-but-positive quantile threshold that the `threshold <= 0` guard doesn't catch, making ratios
blow up to ~1e13. The renderer clamps the display to `×999+`; the underlying calibration gap is
labelled real and pre-existing rather than papered over.

### Drivers

| Function | Scope |
|---|---|
| `compute_trace` / `compute_trace_joint` | one trial, unpadded |
| `eval.batch.compute_traces{,_joint}` | many trials, batched — use this for more than a few |

Each returns not just the trace but the tables it built internally (`log_probs`,
`recipe_log_trans`, `r_hat`, `log_trans_marginal`), because `flag()` and `narrate()` need exactly
those — handing them back avoids a second `to_log_probs` and, more importantly, guarantees all
three stages score against *identical* tables.

`pi_all` is the one thing **not** returned; fetch it with `compute_pi_all` /
`compute_pi_all_joint` on the same inputs.

---

## 5. `narrate.py` — from flags to questions

### `Lexicon`

Maps ids to English. `subtask(k)` names a state by its **modal (verb, noun) pair**
$\bigl(\arg\max_v B^v_{kv},\ \arg\max_n B^n_{kn}\bigr)$ — e.g. `"pour milk"`.

`phrase(v, n)` handles the `SIL` sentinels asymmetrically, and the reasoning is worth reading:

- both SIL → `"idle"`;
- SIL verb + real noun → just the noun (`"bowl"`, not `"stall bowl"`) — reads naturally as
  "no specific action, but this object is present";
- real verb + SIL noun → left as `"{verb} kitchen"`, accepting a literal sentinel leak as the
  lesser evil next to a dangling, objectless sentence.

`expected_duration(k)` returns $1 + r_k(1-p_k)/p_k$ — **with** the shift.

### `Query`

```python
Query(tick, segment_index, channel, kind, severity, ratio, text, event)
```

`event` is a `PreferenceEvent | None` that routes directly into
`lifecycle.state_manager.handle_confirmation` — the detector and the adaptation loop are wired
together at this one type. Severity picks the hedge: `low → "I think"`,
`medium → "I noticed"`, `high → "Wait --"`.

### The four renderers

| Renderer | Channel | Output shape |
|---|---|---|
| `_emission_queries` | $s_{\text{noun}}$ / $s_{\text{verb}}$ (gated on attribution) | *"that's X — I expected Y"* |
| `_stall_queries` | $s_{\text{temporal}}$ | *"are you stuck on X? …{elapsed} ticks; usually {E[D]}"* |
| `_completed_duration_queries` | $s_{\text{dur2}}$ | *"only took d ticks — did you skip part?"* / *"took d ticks — everything okay?"* |
| `_order_queries` | $s_{\text{trans}}$ | *"did you skip B?"* or *"after A you normally do B, not C"* |

**Stall queries fire at the FIRST threshold crossing, not the peak.** $s_{\text{temporal}}$ is
monotone within a segment so its peak is always the last tick — which is not when a live system
would speak up.

**Missing-step bridging.** `_order_queries` first tries to explain an odd transition $a \to c$ as
a *skipped* step:

$$
b^{*} = \arg\max_{b \notin \{a,c\}} \bigl[\log A_{ab} + \log A_{bc}\bigr],
\qquad
\text{gain} = \log A_{ab^*} + \log A_{b^*c} - \log A_{ac} .
$$

If the two-hop path beats the direct jump by $\ge 2$ nats, render *"did you skip B?"*; otherwise
fall back to the one-step *"after A you normally do B"*. $O(K)$ per call.

### Two things narration deliberately does not do

1. **No `s_recipe_transition` renderer.** $K_R$ is a weak-limit nominal and there is no learned
   cluster → name map anywhere in the repo (`recipe_hmm`'s Hungarian alignment exists only for
   scoring). Naming a cluster would print a word corresponding to nothing.
2. **No recipe conditioning on emission expectations in the cascade.** $B^v, B^n$ are
   recipe-agnostic there, so *"I expected jelly"* means "at this step, **across all recipes**, you
   usually use jelly" — not "in a PB sandwich you use jelly". The docstring instructs that this
   caveat be stated whenever a query is presented.

### `narrate_joint`

The four renderers are already model-agnostic (they take a lexicon, thresholds, and a transition
matrix as plain arguments), so the joint variant differs in exactly three places: thresholds from
`threshold_tables_joint`, bridging over $A^{(\hat r)}$, and a `Lexicon` built from
`joint_params.select_recipe(params, r̂)` so that quoted expected durations match the same
per-recipe distribution the surprise was computed against — not a cross-recipe average.
