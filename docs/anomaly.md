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
| `surprise.py` | the eight channels, trace assembly, flagging, attribution, severity |
| `temporal.py` | the duration channels: live stall, retrospective two-sided, PIT calibration |
| `quantile.py` | exact per-state $\alpha$-quantile thresholds over discrete supports |
| `narrate.py` | template renderers turning flags into auditable questions |
| `sequence.py` | retrospective segment-sequence detector: one verdict per junction, names the error type |

`sequence.py` sits **beside** the tick pipeline rather than inside it (§6). It reads the decoded
segment sequence, not the tick stream, so it shares no state with the eight channels.

---

## 1. The design commitment

Every number a user ever sees traces back to a single argmax over a fitted parameter row. The
`narrate.py` module docstring states it plainly: *"routing any of this through a language model
would throw that away."* Categorical emissions were chosen over a learned embedding space for
exactly this reason. When the system says "I expected jelly", you can point at $B^n_{k,\cdot}$ and
see why.

That commitment is what shapes the open-vocabulary work rather than being overturned by it. The
similarity kernel ([`hsmm.md`](hsmm.md) §8) leaves $B$ a categorical over the same fixed
vocabulary and composes it with a **fixed, inspectable** table $S$; the emission stays a
normalised distribution over an enumerable support, so §3's exact discrete tail still applies and
"I expected jelly" still resolves to one row you can print. What the kernel adds is a *second*
auditable object, not a learned opaque one. Its cost is measured, not waived — see
[`hsmm.md`](hsmm.md) §8.7.

---

## 2. The eight channels

All are **surprisal** $-\log P(\cdot)$ measured in nats, so they share a scale and are additive in
the way log-probabilities are.

Seven of the eight are the **default set** (`surprise.CHANNELS`) that every scorecard ORs and every
recorded result was measured on. The eighth, $s_{\text{pair}}$, is opt-in for the reason §2.6
gives; where these docs say "the seven channels" unqualified, they mean the default set.

| Channel | Formula | Fires at | Catches |
|---|---|---|---|
| $s_{\text{emit}}$ | $-\log \sum_k \tilde\pi_t(k) B^v_{k v_t} B^n_{k n_t}$ | every tick | any unexpected observation |
| $s_{\text{verb}}$ | $-\log \sum_k \tilde\pi_t(k) B^v_{k v_t}$ | every tick | wrong action |
| $s_{\text{noun}}$ | $-\log \sum_k \tilde\pi_t(k) B^n_{k n_t}$ | every tick | wrong object |
| $s_{\text{temporal}}$ | $-\log P(D \ge d_{\text{elapsed}} \mid z^*)$ | every tick (live) | being stuck, **while it happens** |
| $s_{\text{dur2}}$ | two-sided duration $p$-value surprise | segment end | too long **or** too short, retrospectively |
| $s_{\text{trans}}$ | $-\log A_{z_{j-1} z_j}$ | segment start | out-of-order / skipped step |
| $s_{\text{recipe}}$ | cascade: $-\log A^R_{\rho_{j-1}\rho_j}$; joint: signed excess | segment start | recipe-level incoherence |
| $s_{\text{pair}}$ | $s_{\text{emit}} - s_{\text{verb}} - s_{\text{noun}} = -\text{PMI}(v,n)$ | every tick (opt-in) | both words ordinary, the **pair** wrong |

Non-firing ticks are filled with $0$ (or $-1$ for id fields, `NaN` for the PIT diagnostic).

### 2.0 What distribution each channel is surprise *for*

Every channel is $-\log P(\text{observed} \mid \text{conditioning})$ against one specific fitted
conditional distribution. $Z_t$ is the subtask at tick $t$; $\hat r$ is the joint model's trial-level
MAP recipe (`infer_recipe`, one draw for the whole trial); $\rho_j$ is the **cascade's** per-segment
recipe-cluster id (a separate stage-2 HMM state that *can* change across segments — not the same
object as $\hat r$, despite both being called "recipe"). $j$ indexes segments, $t$ indexes ticks.

| Channel | Distribution scored — cascade | Distribution scored — joint |
|---|---|---|
| $s_{\text{emit}}$ | $P(v_t, n_t \mid Z_t)$ | $P(v_t, n_t \mid Z_t)$ — **never $\hat r$** |
| $s_{\text{verb}}$ | $P(v_t \mid Z_t)$ | $P(v_t \mid Z_t)$ |
| $s_{\text{noun}}$ | $P(n_t \mid Z_t)$ | $P(n_t \mid Z_t)$ |
| $s_{\text{temporal}}$ | $P(D \ge d_{\text{elapsed}} \mid Z_j)$ | $P(D \ge d_{\text{elapsed}} \mid Z_j, \hat r)$ |
| $s_{\text{dur2}}$ | $P(D\ge d),\ P(D\le d) \mid Z_j$ | $P(D\ge d),\ P(D\le d) \mid Z_j, \hat r$ |
| $s_{\text{trans}}$ | $P(Z_j \mid Z_{j-1})$ | $P(Z_j \mid Z_{j-1}, \hat r)$ |
| $s_{\text{recipe}}$ | $P(\rho_j\mid\rho_{j-1})$ | not a probability — signed excess, see below |
| $s_{\text{pair}}$ | $-\text{PMI}$ under the tick's predictive mixture | same — the mixture, never a single state |

**Emissions are the one place recipe never enters, in either model.** $B^v, B^n$ are shared,
unindexed tables (`assemble_trace_joint`'s docstring: "emissions stay shared"); $P(v_t,n_t\mid Z_t)$
has no $R$ or $\hat r$ term to add. Duration and transition, by contrast, are genuinely
recipe-conditioned in the joint model — `dur_r[r_hat]`/`dur_p[r_hat]` and `log_trans[r_hat]` are
different numbers for different $\hat r$, so $Z_j$ alone does not pin down the distribution there.

**Hard vs. soft conditioning.** $Z_t$ in the emission row is never observed directly at tick $t$
scoring time, so the *actual* computed quantity marginalises over the causal belief
$\tilde\pi_t(k) = P(Z_t{=}k\mid o_{<t})$ rather than conditioning on one hard value:
$$
s_{\text{emit}}(t) = -\log P(v_t,n_t\mid o_{<t}) = -\log\sum_k \tilde\pi_t(k)\,P(v_t,n_t\mid Z_t{=}k).
$$
Duration and transition channels instead condition on the **hard** Viterbi decode $Z_j = z^*_j$ (the
segmentation is already fixed before these channels are scored — see §2.5), so no marginalisation
is needed there; the formula in the table above is evaluated directly at $Z_j=z^*_j$ (and $R=\hat r$
in the joint model).

**The joint model's $s_{\text{recipe}}$ is not $P(\cdot\mid\cdot)$ at all.** It measures how much
better $\hat r$'s own transition row explains the observed step than the recipe-averaged row does
for the *same* $(Z_{j-1}, Z_j)$ pair:
$$
s_{\text{recipe}} = \log P(Z_j\mid Z_{j-1}) - \log P(Z_j \mid Z_{j-1}, R{=}\hat r), \qquad
P(Z_j\mid Z_{j-1}) := \textstyle\sum_r P(R{=}r)\, P(Z_j\mid Z_{j-1}, R{=}r),
$$
a signed excess (§2.4), not a surprisal — it can be negative when $\hat r$ explains the transition
*better* than the average recipe does.

Every formula above bottoms out in one of three fitted objects, each defined precisely elsewhere:
the predictive occupancy $\tilde\pi_t(k)$ (forward recursion, [hsmm.md §4.2](hsmm.md)), the
categorical emission tables $B^v, B^n$ ([hsmm.md §1](hsmm.md)), the NB duration tail via the
regularised incomplete beta ([hsmm.md §2.1](hsmm.md)), or the Dirichlet-MAP transition matrix $A$
([hsmm.md §3](hsmm.md)). §2.1–2.4 substitute each of these in turn.

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

Concretely, with $(r,p) = (r_{z^*_t},\, p_{z^*_t})$ the fitted NB parameters of the believed state
and $I_p(a,b)$ the regularised incomplete beta (`jax.scipy.special.betainc`, [hsmm.md §2.1](hsmm.md)):

$$
s_{\text{temporal}}(t) \;=\; \begin{cases}
0 & d_{\text{elapsed}} = 1 \quad (P(D\ge1)=1,\text{ always true})\\[4pt]
-\log\bigl(1 - I_{p}(r,\, d_{\text{elapsed}}-1)\bigr) & d_{\text{elapsed}} \ge 2.
\end{cases}
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

Substituting the same incomplete-beta tails, with $(r,p) = (r_{z_j},\, p_{z_j})$ the closed
segment's fitted state:

$$
s_{\text{long}} = -\log\bigl(1-I_p(r,\,d-1)\bigr), \qquad
s_{\text{short}} = -\log I_p(r,\,d),
$$
$$
s_{\text{dur2}} \;=\; -\log\Bigl(\min\bigl\{1,\ 2\min\bigl(1-I_p(r,\,d{-}1),\ I_p(r,\,d)\bigr)\bigr\}\Bigr).
$$

**The left tail exists because the live channel structurally cannot catch abandonment.** A short
duration always has *high* survival probability, hence *low* $s_{\text{temporal}}$ — leaving a step
early is invisible to a monotone survival ramp, by construction. It needs its own retrospective
channel.

**The trial's own final segment is not scored** (`final_censored=True`, the default). That segment
is right-censored — observation stopped, the activity did not — so $P(D \le d)$ asks a question the
data cannot answer. Left in, it fires a spurious `left_early` flag at the final tick of any trial
whose last state has a long fitted mean, rendered as *"idle only took 8 ticks — you usually spend
about 186."* Measured on 419 healthy real trials, that one flag was **62% of the residual
trial-level false-positive rate** at tight $\alpha$ (0.100 → 0.038). The right tail would still be
valid for a censored segment, but `live_stall_surprise` already carries it, so nothing is lost.

> **The cost is not free, and it lands unevenly.** Suppressing that flag costs abandonment recall
> −4.2 points and omission −3.1 (26% of abandonment injections land at the trial end, and an
> omission of the second-to-last segment makes the final step the ground-truth one), against
> +3.4 precision and −2.1 healthy false alarms. F1 is essentially unchanged. The judgement is that
> the lost detections were credit for a flag that fires at every trial's end regardless of whether
> anything was injected — but it is a judgement, and `final_censored=False` restores the old
> behaviour.
>
> **Stale as of 2026-08-22.** The "26% land at the trial end" figure was `error_injection.
> inject_abandonment` picking the trailing idle segment, a bug fixed the same day (see
> [`synthetic.md`](synthetic.md) §2's interior-constraints note) — abandonment is now interior-only
> and can never land at the trial end, so that share is 0%, and the whole −4.2/+3.4/−2.1 trade this
> paragraph argues for needs a fresh measurement before being cited again. The mechanism argument
> for `final_censored=True` (a censored segment can't answer a two-sided question) is unaffected.
>
> The inflated means driving this have the same root cause: a state that is disproportionately the
> final segment of training trials has little uncensored duration data pinning it down. Several are
> fitted at 32–734 ticks. That fit problem is still open, and it is what makes §6's repetition test
> unusable.

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

$A$ here is the fitted, Dirichlet-MAP-normalised transition matrix ([hsmm.md §3](hsmm.md)), each
row summing to 1 over the $K-1$ off-diagonal entries since self-transitions are banned
($A_{kk}=0$).

Cascade recipe channel: the same thing one level up, evaluated at **every** segment boundary
(recipe self-transitions are legal and expected, so there is no boundary to skip):

$$
s_{\text{recipe}} \;=\; -\log A^{R}_{\rho_{j-1},\, \rho_j}.
$$

$A^R$ is the analogous transition matrix fit over recipe-cluster symbols by `recipe_hmm.py`
([recipe.md](recipe.md)), not the subtask HSMM's $A$.

**The joint model repurposes this channel entirely.** With per-recipe transition matrices, a more
useful question is *"is this transition ordinary in general but wrong for **this** recipe?"* So:

$$
\boxed{\;s_{\text{recipe}} \;=\; \log \bar A_{ji} \;-\; \log A^{(\hat r)}_{ji}\;}
\qquad \bar A = \sum_r \pi_r A^{(r)} .
$$

This is a **signed excess**, not a neg-log-probability. It can be negative, which has real
consequences downstream (§3.3).

### 2.5 Where "segment start" / "segment end" actually come from

Every boundary-indexed channel above ($s_{\text{trans}}$, $s_{\text{recipe}}$) and every
end-of-segment channel ($s_{\text{dur2}}$, PIT) reads its ticks off `seg_result["segments"]` /
`subtask_per_tick`, which is the **Viterbi (max-product) decode of the whole trial**
([recipe.md §1](recipe.md)):

$$
\bigl(\hat z_{1:J},\ \hat d_{1:J}\bigr) \;=\;
\arg\max_{J,\, z_{1:J},\, d_{1:J}}\ \log P\bigl(o_{0:T-1},\, z_{1:J},\, d_{1:J}\bigr).
$$

Concretely, a boundary is a tick where this single best explanation of the *entire* observed
sequence — jointly trading off emission likelihood, the duration prior, and the transition matrix
— switches state. It is **not**:

- **where the raw $(v_t, n_t)$ differs from $(v_{t-1}, n_{t-1})$.** Emission is a fitted per-state
  distribution, not a lookup key: a state can legitimately emit different (verb, noun) pairs on
  consecutive ticks (if $B^v_{k,\cdot}$/$B^n_{k,\cdot}$ put mass on more than one token) and still
  be one segment, while an anomalous single-tick observation does not by itself force a boundary —
  the duration and transition priors can make staying cheaper than switching.
- **where $\tilde\pi_t$'s argmax flips.** $\tilde\pi_t$ (§2.1) is the *causal*, tick-by-tick
  predictive belief and can flicker; $z^*$ is the *global* MAP path computed with the benefit of
  the whole trial. §3.5 documents the resulting known gap — `flag`/`flag_joint` index thresholds
  by $z^*$, which is only an exact stand-in for the live belief when $\tilde\pi_t$ is concentrated
  on it (`belief_concentration`, `belief_diagnostic`).

Because self-transitions are banned ($A_{kk}=0$, §2.4), consecutive Viterbi segments always differ
in state, so a maximal run of constant `subtask_per_tick` is exactly one segment — there is no
separate "did the state change" test to reconcile with this one.

---

### 2.6 The pair channel — `s_pair`, and why it needs a second corpus

**Opt-in.** `s_pair` is deliberately *not* in `surprise.CHANNELS`: every scorecard ORs `CHANNELS`
for its headline `raw` number, so adding an eighth there would move every result already recorded.
Use `CHANNELS_WITH_PAIR`, or `run_detect_eval.py --with-pair`.

The channel is the difference between the joint emission surprise and the two marginals:

$$
s_{\text{pair}}(t) \;=\; s_{\text{emit}}(t) - s_{\text{verb}}(t) - s_{\text{noun}}(t)
\;=\; -\operatorname{PMI}(v_t, n_t),
$$

larger meaning *less compatible*, matching every other channel's convention. It catches the case
none of the first three can state: **both words are individually ordinary, and their combination is
not** — `pour knife`, where `pour` is a common verb, `knife` is a common noun, and the pair is
nonsense.

**Its null is per-tick, and it has to be.** Under a *single* state the emission is a product by
construction ($P(v,n\mid Z{=}k) = P(v\mid k)P(n\mid k)$, [`hsmm.md`](hsmm.md) §1), so
$\operatorname{PMI}(v,n\mid Z{=}k)$ is identically zero and a per-state null would be degenerate.
All of $s_{\text{pair}}$'s signal comes from the predictive **mixture** $\sum_k \tilde\pi_t(k)
P(v\mid k)P(n\mid k)$, which is not a product even though every component is. So
`quantile.pair_quantile_threshold` builds the tick's own $(V,N)$ mixture joint and takes the exact
discrete tail over it — the same machinery as everywhere else in §3, just rebuilt per tick rather
than per state.

Two consequences follow from that, and both are costs:

- **The trace has to keep `pi_all`.** The $(T,K)$ predictive occupancy is normally discarded after
  the channels are computed; `assemble_trace(..., keep_pi_all=True)` retains it. Absent `pi_all`,
  `flag_joint` does not offer the channel at all rather than substituting an approximation.
- **It dominates the runtime of a sweep.** $O(T \cdot V \cdot N)$ with one sort per tick, against
  $O(K)$ table lookups for every other channel.

**On Breakfast it is close to redundant with $s_{\text{emit}}$**, because a 15 × 36 vocabulary of
scripted breakfast actions contains few pairs that are individually plausible but jointly absurd.
This channel is one of the reasons for the EPIC ingest ([`data.md`](data.md) §7): at 97 verbs and
305 nouns over unscripted kitchen visits, "each word fine, pair wrong" is a real and frequent
failure mode rather than a constructed one.

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

### 3.3 The five table builders

| Function | Support scored | Used by |
|---|---|---|
| `categorical_quantile_threshold` | one emission row, $V$ or $N$ entries | $s_{\text{verb}}$, $s_{\text{noun}}$ |
| `joint_quantile_threshold` | the $V \times N$ outer product per state | $s_{\text{emit}}$ |
| `transition_quantile_threshold` | one transition row | $s_{\text{trans}}$, cascade $s_{\text{recipe}}$ |
| `excess_quantile_threshold` | signed excess, null $i \sim A^{(\hat r)}(\cdot\mid j)$ | joint $s_{\text{recipe}}$ |
| `pair_quantile_threshold` | the $V \times N$ **mixture** joint, per tick | $s_{\text{pair}}$ (§2.6) |
| `sequence_thresholds` | an **empirical** sample of healthy-trial statistics | `sequence.py`'s swap / duration tests (§6) |

The first five score a *fitted discrete distribution*; `sequence_thresholds` scores an empirical
sample instead, weighting each observed value $1/N$ and passing it through the same
`_tail_threshold`, so the resulting cut is the empirical $(1-\alpha)$ quantile under the identical
strict-`>` convention. Every threshold in the package is therefore an $\alpha$-tail cut of *some*
null, never a hand-set nat value.

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

*Why not a flat nat floor instead.* The maximum achievable quantile threshold is 2.60/2.52 nats at
$K{=}20$ and 3.30/2.87 at $K{=}64$. Any floor set high enough to suppress dilution would therefore
sit above **every** state's threshold, replacing the per-state calibration with a constant and
making §3.1–§3.3 dead code. The correction has to be per-tick and derived, which is what the
inequality above provides.

**The bound is an equality in the common case, so flagging carries a numerical tolerance.** When
$z^*$ is near-deterministic and the observed token *is* its mode, $s_{\text{pure}} \approx 0$ and
the inequality closes: score and threshold become the same real number. They are nevertheless
reached by two independent floating-point routes — the score through a `logsumexp` over the full
mixture, the threshold through a direct $-\log\tilde\pi_t(z^*)$ — which disagree at the last bit
(~$10^{-17}$, one ULP of float64 at these magnitudes). A bare `>` resolves that noise as
"exceeded", and does so for *every* confidently-predicted tick, which is the modal tick of a
well-fitting model. The three dilution-corrected channels are therefore gated as

$$
\texttt{flag} \;=\; s(t) \;>\; \tau_t + \varepsilon,
\qquad \varepsilon = \texttt{EMISSION\_FLAG\_ATOL} = 10^{-6},
$$

which sits many orders of magnitude above the tie noise and many below the smallest surprise gap
this repo ever treats as meaningful (hundredths of a nat), so it cannot mask a real anomaly. The
other four channels compare against thresholds that are not derived from the score's own inputs,
have no equality case, and use a plain `>`.

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
flags[ch][t]  =  value[t] > threshold[t]                    # temporal, duration, transition
flags[ch][t]  =  value[t] > threshold[t] + 1e-6             # s_emit, s_verb, s_noun  (§3.4)
```

with `_base_flags` covering the six shared channels and each of `flag` / `flag_joint` adding its
own $s_{\text{recipe}}$ variant (cascade indexes by `from_recipe`; joint by `from_state` under
$\hat r$). The tolerance on the three emission channels is required by the equality case of the
dilution bound, not a sensitivity knob — see §3.4.

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

---

## 6. `sequence.py` — the retrospective segment detector

A second detector over the same trial, reading the **Viterbi segment sequence**
$z_1 \dots z_J$ (`segments_from_z`) rather than the tick stream. It shares no state with the seven
channels and is scored as its own arm by `run_sequence_eval.py`.

It was built to supply two things the tick pipeline cannot:

- **One verdict per event.** Each junction and each segment is tested exactly once, so a single
  structural error produces a single detection rather than a spread of tick flags.
- **It names the error type.** $s_{\text{trans}}$ fires identically for omission, transposition and
  repetition — the cascade genuinely cannot separate them ([`synthetic.md`](synthetic.md)). Three
  separate local edit tests can, because each asks a different question of the same transition row.

> ### Measured: the naming mechanism works, the detector does not carry its weight
>
> 419 real trials, $\alpha = 5\times10^{-3}$, trial-located ([`eval.md`](eval.md) §6):
>
> | arm | precision | recall | F1 | healthy FPR |
> |---|---|---|---|---|
> | tick-level | **0.728** | 0.453 | **0.558** | **0.141** |
> | sequence only | 0.412 | 0.107 | 0.170 | 0.086 |
> | union | 0.635 | **0.484** | 0.549 | 0.191 |
>
> **Transposition naming goes 0.000 → 0.380 standalone**, which is the capability this module
> exists for and which nothing else in the package can do. But standalone transposition *recall*
> is 0.186 against the tick arm's 0.377 — it names better and detects worse — and unioning the two
> arms lowers F1 (0.558 → 0.549) while raising healthy false alarms 0.141 → 0.191. In the union
> the naming win is also diluted to 0.149, because where the sequence test did not flag that exact
> step the tick arm's `CHANNEL_TO_TYPE` verdict wins it.
>
> Two specific defects, both upstream of this module rather than tuning problems:
>
> 1. **The repetition test is calibrated above its own signal.** The healthy $(1-\alpha)$ quantile
>    of the duration ratio is **3.97x expected**, while a Viterbi-merged duplicate produces ~2x.
>    Recall is 0.079. The heavy tail comes from the duration fits themselves — several states are
>    fitted with means of 32–734 ticks (see §2.3's censoring note) — so this test cannot work until
>    those fits do.
> 2. **The omission test strays more than it hits** (stray rate 0.289 vs recall 0.236).
>    `missing_step`'s 2-nat gate was tuned to choose *phrasing* once a channel had already fired,
>    not to decide whether anything fired at all, and it does not transfer to that role.
>
> The measured-good use is therefore narrower than "a third arm", and is what ships:
> `element_metrics.relabel_with_sequence` runs the swap test **only to correct the type** of an
> alarm the tick channels already raised. See below.

### How it is actually used: type relabelling, not detection

`element_metrics.relabel_with_sequence(tick_verdicts, seq_verdicts)` copies `is_anomaly` through
untouched and may only change `error_type`. Precision, recall and the healthy false-positive rate
are therefore **identical to the tick arm by construction** — verified, not asserted — and the only
metric that can move is `type_confusion`. Three narrowings make that trade positive:

| knob | default | why |
|---|---|---|
| `types` | `("transposition",)` | the only type the tick channels are *structurally* unable to name, rather than merely worse at |
| `from_types` | `("omission",)` | what `CHANNEL_TO_TYPE` resolves $s_{\text{trans}}$ to, and so the label a mis-read transposition wears — 0.862 of true transpositions |
| `scope` | `"trial"` | the swap test and the tick channels often flag *opposite halves* of the swapped pair, so step-exact matching misses most real agreements |

Restricting the **source** label is what removes the collateral damage: substitution (0.998) and
abandonment (0.921) come from channels a swap cannot be confused with, and an earlier unrestricted
version lost 0.008 and 0.025 on them for nothing.

Measured on 419 real trials, $\alpha = 5\times10^{-3}$:

| error type | tick arm | relabel (step) | **relabel (trial, shipping)** |
|---|---|---|---|
| transposition | **0.000** | 0.116 | **0.199** |
| omission | 0.904 | 0.880 | 0.872 |
| substitution / abandonment / repetition | — | unchanged | **unchanged** |

Net: a total blind spot becomes a **weak** one, for −0.032 on omission and nothing else. The
remaining ceiling is not the plumbing — trial scope already recovers most of what step scope
missed — but the swap test's own 0.186 detection recall: relabelling cannot fire on a trial where
the test stayed silent, and lowering its threshold to fix that reintroduces exactly the false
alarms this containment exists to avoid.

> `scope="trial"` is sound **because the harness injects exactly one anomaly per trial**, so a swap
> found anywhere and an alarm raised anywhere are almost certainly the same event. A live stream
> carrying several concurrent anomalies would need `scope="step"` or a windowed variant; the
> default is a property of the evaluation, not a general claim.

### The three tests

The base score of an observed sequence is $\sum_j \log A_{z_j z_{j+1}}$. Each test proposes one
local edit and measures what it would gain.

| Test | Edit | Statistic | Threshold |
|---|---|---|---|
| **transposition** | swap $z_j, z_{j+1}$ | local score after − before | healthy $(1-\alpha)$ quantile |
| **omission** | insert a bridging state $b$ between them | `narrate.missing_step`'s two-hop gain | `DEFAULT_MIN_BRIDGE_GAIN` $= 2$ nats |
| **repetition** | — | segment duration ÷ `Lexicon.expected_duration` | healthy $(1-\alpha)$ quantile |

"Local" for the swap test means the three transitions that actually change — into $z_j$, the
junction itself, and out of $z_{j+1}$ — with the flanking terms dropped at the sequence ends:

$$
\text{gain}_j \;=\;
\underbrace{\log A_{z_{j-1} z_{j+1}} + \log A_{z_{j+1} z_j} + \log A_{z_j z_{j+2}}}_{\text{swapped}}
\;-\;
\underbrace{\log A_{z_{j-1} z_j} + \log A_{z_j z_{j+1}} + \log A_{z_{j+1} z_{j+2}}}_{\text{observed}} .
$$

A positive gain means the recipe's own transition table prefers the swapped order to the one
observed. The omission test reuses `narrate.missing_step` directly — the same
$\arg\max_b \log A_{ab} + \log A_{bc}$ against the direct jump that `_order_queries` renders
*"did you skip B?"* from — so a sequence verdict and a narrated card can never disagree about
whether a step was skipped. The repetition test keys on duration because a duplicated segment
adjacent to its original merges into one over-long run in the decode, the same behaviour
[`synthetic.md`](synthetic.md) documents for the tick path.

**Transposition wins ties.** Where both the swap and the bridge clear their thresholds at one
junction, the junction is reported as a transposition and the omission test is skipped there: an
ordering violation explains an odd local transition, not the reverse. This is the priority argument
`element_metrics.CHANNEL_PRIORITY` applies to the tick channels, applied to the same evidence.

### Calibration and cost

The two magnitude thresholds are the $(1-\alpha)$ empirical quantile of each statistic's
distribution over **healthy** trials, via `quantile.sequence_thresholds` (§3.3) — the same
null-distribution discipline the per-tick channels use, with an empirical null in place of a fitted
one. The bridging test needs no such table: it inherits `missing_step`'s fixed nat gate.

At full scale this calibrates on ~2,600 junctions and ~3,000 segments, putting the $\alpha$ cut at
roughly the 13th-largest healthy gain — in the tail, but with enough support to be a genuine
quantile rather than a single-sample artifact. `run_sequence_eval.py` prints that count alongside
the threshold so the estimate's support is visible rather than implied.

The detector is **non-causal by construction**: the swap test at junction $j$ needs segment $j+1$
to have closed, so a verdict lags by one segment. That is the same latency $s_{\text{dur2}}$
already accepts, and it is the right trade for a retrospective *"did you mean to do that?"* prompt.

Every verdict remains an argmax over a fitted transition row or a ratio against a fitted NB mean,
so the auditability commitment of §1 holds here unchanged.
