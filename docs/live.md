# Live channels — a causal design for every surprise channel

**Status: implemented as an opt-in path, and measured (§8).** Code:
[`src/cook_ad/anomaly/live.py`](../src/cook_ad/anomaly/live.py); evaluation:
[`run_live_eval.py`](../run_live_eval.py); tests:
[`tests/test_live.py`](../tests/test_live.py). The shipped detector in [`anomaly.md`](anomaly.md)
is untouched, and every other result in this repo was measured on that hindsight detector. This
document says what each live channel computes, how it relates to the current one, and (§8) what
going live costs.

Notation follows the user's notes: $b$ is the belief table, $e_t$ is the ticks elapsed in the
current step, $D$ is a step's total duration, $h$ is the hazard, $A$ is the per-step transition
matrix, and $E^v, E^n$ are the emission tables. The repo's other docs write the emission tables as
$B^v, B^n$.

---

## 0. Why: what the shipped detector reads from the future

`compute_traces_joint` ([`eval.md`](eval.md) §1) computes two things from the **whole trial**
before scoring any tick:

- $\hat r$, the trial's recipe (`infer_recipe`, over the full $\log Z$)
- $z^*$, the Viterbi segmentation under $\hat r$ (`segment_all_conditioned`)

Everything downstream conditions on them:

| Channel | Score uses the future? | Threshold uses the future? |
|---|---|---|
| $s_{\text{emit}}, s_{\text{verb}}, s_{\text{noun}}$ | **yes**, joint only: $\tilde\pi_t$ is computed under $\hat r$ | **yes**: $\tau_{z^*_t}$ and the dilution term $-\log\tilde\pi_t(z^*_t)$ |
| $s_{\text{pair}}$ | yes, through $\hat r$ | yes, through $\hat r$ (otherwise per-tick and live) |
| $s_{\text{temporal}}$ | **yes**: $z^*_t$, and $e^*_t$ counted from $z^*$'s step start | no (global $-\log\alpha$) |
| $s_{\text{dur2}}$ | **yes**: $z^*$'s durations, **plus one tick**: it is placed on the step's *last* tick, before the next tick reveals that the step ended | no |
| $s_{\text{trans}}$ | **yes**: $z^*$'s boundaries and from/to states | yes, indexed by $z^*$'s from-state |
| $s_{\text{recipe}}$ | **yes**: $\hat r$ and $z^*$ | yes |

The LLM arm, by contrast, is strictly causal ([`llm.md`](llm.md) §1, "The second unit": the "step ended" note is
deliberately placed on the *next* tick). So every HSMM-vs-LLM number to date compares a hindsight
detector with a live one. How much hindsight is worth has never been measured.

---

## 1. The live state: one filter over (recipe, subtask, elapsed)

The recipe is one hidden variable per trial, so it can simply be added to the belief table:

$$
b_t(r,i,j) = P(R=r,\ z_t=i,\ e_t=j \mid o_{0:t}),
\qquad
b'_t(r,i,j) = P(R=r,\ z_t=i,\ e_t=j \mid o_{0:t-1}).
$$

This is **exact**. It is not an approximation of the joint model; it is the joint model's forward
pass, read out every tick. The cascade is the special case $K_R = 1$.

**Init.** $b'_0(r,i,1) = P(R=r)\,\pi^{\text{init}}_{r,i}$; all other cells are $0$.

**Predict (no observation).** Each recipe slice moves under its own dynamics:

$$
\begin{aligned}
b'_t(r,i,1) &= \sum_{k}\sum_j b_{t-1}(r,k,j)\; h^{(r)}_{kj}\; A^{(r)}_{ki} &&\text{a new step starts}\\
b'_t(r,i,j) &= b_{t-1}(r,i,j{-}1)\,\bigl(1-h^{(r)}_{i\,j-1}\bigr), \quad j\ge 2 &&\text{the step continues}
\end{aligned}
$$

A step that reaches column $M = D_{\max}$ ends there with certainty ($h_{\cdot M}=1$). That matches
`messages.forward_pass`, whose last pmf column holds $P(D\ge D_{\max})$
(`durations.duration_tables`): a longer step is bookkept as ending at $D_{\max}$, so no mass ever
sits past it.

**Update.** Multiply by the emission and normalise **over all of (r, i, j)**:

$$
b_t(r,i,j) = \frac{P(o_t\mid i)\; b'_t(r,i,j)}{P(o_t\mid o_{0:t-1})},
\qquad
P(o_t\mid o_{0:t-1}) = \sum_{r,i,j} P(o_t\mid i)\, b'_t(r,i,j).
$$

With `noun_tilt` ([`hsmm.md`](hsmm.md) §6.1), $P(o_t\mid i)$ becomes $P(o_t\mid i, r)$, and nothing
else changes.

**Readouts used below.**

| Readout | Definition | Meaning |
|---|---|---|
| $\tilde\pi_t(i)$ | $\sum_{r,j} b'_t(r,i,j)$ | subtask belief **before** $o_t$ |
| $w_t(r)$ | $\sum_{i,j} b_t(r,i,j)$ | recipe belief after $o_t$ ($w_{t-1}$ = before) |
| $P_{\text{mix},t}(x)$ | $\sum_{r,i,j} b'_t(r,i,j)\,P(x\mid i,r)$ | the model's forecast for tick $t$ |

**Built-in exactness test.** $\sum_t \log P(o_t\mid o_{0:t-1})$ must equal
$\log\sum_r P(R=r)\,Z_r$, the per-trial marginal that `infer_recipe` already returns
(`trial_ll`). If those disagree, the filter is wrong.

**Cost per tick:** $O(K_R\,K\,(K + D_{\max}))$. On EPIC ($K_R=16$, $K=128$, $D_{\max}=200$) that
is ~0.7M multiply-adds, which is trivial.

---

## 2. Four design rules

**R1. Observed things get predictive p-values; hidden things get posterior probabilities.**
The emission channels test $o_t$, which is observed. The right question is "how unlikely was
this, under the forecast?" Stall, transition, early-exit and recipe test **events nobody
observes**: was there a switch, is the person past their usual time. The right question there is
"how probable is it, given everything seen, that a bad event happened?"

**R2. Each live channel collapses to the current one.** If the belief is a point mass on the
hindsight answer $(\hat r, z^*, e^*)$, every live flag below reduces exactly to the current flag.
This is a design constraint and a unit test: the live channels test the *same* events as today,
under uncertainty, rather than inventing new ones.

**R3. Reuse the α-tail sets; don't recompute them.** Today, `quantile.py` turns each fitted row
into a threshold. Here, the same thresholds define a **set of bad outcomes** per row, such as
"transitions out of $k$ in the α-tail of $A^{(r)}_{k\cdot}$". The live channel sums belief over
that set. α keeps its meaning: the tail mass under the model's own row.

**R4. Posterior channels need a second knob, κ.** A posterior probability isn't a p-value, so
you flag when it exceeds a cutoff κ. κ = 0.5 means "more likely than not." Like α, κ is **not**
a false-alarm rate, and it has to be calibrated on healthy dev trials
(never in-sample; [`hsmm.md`](hsmm.md) §8.4 and [`eval.md`](eval.md) §7 record what in-sample selection cost here).

---

## 3. The one shared object: the switch posterior

Three channels (transition, early exit, recipe) are all questions about **the step that just
ended and the one that just began**. Before summing over anything, the switch term in the
predict step is:

$$
q_t(r,k,j,i) \;=\; \frac{b_{t-1}(r,k,j)\; h^{(r)}_{kj}\; A^{(r)}_{ki}\; P(o_t\mid i)}{P(o_t\mid o_{0:t-1})}
$$

$q_t(r,k,j,i)$ is the probability that, under recipe $r$, **step $k$ ended after exactly $j$
ticks, and step $i$ began at tick $t$**, given $o_{0:t}$. Summing it over $(r,k,j)$ gives
$\sum_r b_t(r,i,1)$, the "just started $i$" column, which is a consistency check.

Each channel keeps different indices and sums out the rest:

| Channel | Keeps | Bad set | Sums out |
|---|---|---|---|
| transition | $(r,k,i)$ | $i$ in the α-tail of $A^{(r)}_{k\cdot}$ | $j$ |
| early exit | $(r,k,j)$ | $j$ in the short α/2-tail of $D\mid k,r$ | $i$ |
| recipe | $(r,k,i)$ | $(k\to i)$ in $r$'s excess tail | $j$ |

**Never materialise $q_t$ in full** (on EPIC it has 52M entries per tick). Every channel factors:

- transition and recipe need $m_t(r,k) = \sum_j b_{t-1}(r,k,j)\,h^{(r)}_{kj}$, then a masked
  $(K\times K)$ product with $A^{(r)}$ and $P(o_t\mid\cdot)$;
- early exit needs $g_t(r,k) = \sum_i A^{(r)}_{ki}P(o_t\mid i)$, then a masked sum over $j$.

Both are $O(K_R K (K + D_{\max}))$.

---

## 4. The channels

### 4.1 Emission: $s_{\text{emit}}, s_{\text{verb}}, s_{\text{noun}}$ — live predictive p-value

**Question.** "Did the model, as a whole, see this observation coming?"

**Score.** Unchanged in form: $s_{\text{emit}}(t) = -\log P_{\text{mix},t}(v_t,n_t)$. The only
difference is that the belief no longer conditions on $\hat r$. For verb and noun, use the
forecast's marginals $P_{\text{mix},t}(v)$ and $P_{\text{mix},t}(n)$.

**Flag.** Replace the $z^*$-indexed threshold with the forecast's own tail:

$$
p_t = \sum_x P_{\text{mix},t}(x)\;\mathbb 1\bigl[P_{\text{mix},t}(x)\le P_{\text{mix},t}(x_t)\bigr],
\qquad \text{flag if } p_t \le \alpha,
$$

where $x$ ranges over $V$, $N$, or $V\times N$ for verb, noun and emit respectively. No sort is
needed; it is one masked sum. The emit version costs $O(K\,V\,N)$ per tick. That's 540 pairs
on Breakfast and ~30k on EPIC, which is cheap on a GPU.

**No dilution correction.** The correction ([`anomaly.md`](anomaly.md) §3.4) exists only because
today's score lives on the mixture's scale while its threshold lives on one state's scale. Here
both come from the same forecast, so there's nothing to correct, and therefore no $z^*$ needed.

**Collapse (R2).** If $\tilde\pi_t$ is a point mass on $z^*_t$, then $P_{\text{mix}} =
P(\cdot\mid z^*_t)$, the dilution term is $-\log 1 = 0$, and "$p_t\le\alpha$" holds exactly when
$o_t$'s tie group falls inside `_tail_threshold`'s tail set. That is the current flag, including
its strict-`>` tie convention.

**What this changes at a healthy step change.** Under the current scheme with a *live* $z^*$,
every real boundary is a false alarm: the belief still says "pour milk," and (take, bowl) is deep
in that state's tail. Under $p_t$, a next step holding 15% of the belief forecasts its own tokens
at ~13%, which is far above α, so there is no flag. See the worked example in §6.

**Known weaknesses, to measure:**

- **Deafness under a diffuse belief.** When the belief is spread over many states, the forecast
  is spread too, and little falls in the α-tail. A belief is most diffuse right *after* something
  odd happened, so a second anomaly right after a first one may go unflagged.
- **It absorbs order errors.** A skipped step forecasts the new step's tokens weakly, so $p_t$
  fires on what is really a transition anomaly. §5 handles the attribution.

**As measured: take step timing out of the emission test.** The plain $p_t$ above has a
failure that the design missed. The forecast mixes "the step continued" and "a new step began
now," weighted by the hazard. So a perfectly healthy step change that arrives earlier than the
duration model expected gets a small forecast probability, and gets flagged as a wrong token.
That is duration surprise counted a second time, in the emission channel. On 39 healthy Breakfast
test trials at α = 0.005, **all 28** live $s_{\text{emit}}$ flags sat on a step change, after a
step that had run a median 0.52 of its expected length. (The hindsight arm never sees this,
because its $-\log\tilde\pi_t(z^*_t)$ term excuses *every* boundary. That is also why its
emission channels are near-silent on healthy trials and catch little besides substitutions.)

The shipped live version therefore splits the forecast into its two hypotheses,

$$
\tilde\pi^{\text{stay}}_t(i) \propto \sum_{r}\sum_{j\ge 2} b'_t(r,i,j), \qquad
\tilde\pi^{\text{switch}}_t(i) \propto \sum_r b'_t(r,i,1),
$$

computes $p_t$ under each, and keeps the **larger**: a token is flagged only if it is in the α-tail
*whether or not* a step change happened at $t$ (`live.split_emission_pvalues`). Timing goes back
to the duration channels and order to the transition channel. A hypothesis with no predictive mass
drops out of the max, so at a point mass R2 still holds exactly (tested). On the same 39 trials,
healthy trials with an emission flag fell from 15 to 6. All 8 remaining flag ticks are step
changes into a step the transition row made unlikely, which is a transition anomaly showing up a
second time (§5). The plain mixture is kept as the `live_mix_L0` arm in §8.

**Unchanged and already live:** the verb/noun attribution margin ($s_{\text{noun}} -
s_{\text{verb}} > m$), and `conditional_expected` / `joint_expected` for *"I expected X"*. Both
already read $\tilde\pi_t$.

### 4.2 $s_{\text{pair}}$ — already live, once $\hat r$ is removed

`pair_quantile_threshold` already builds its threshold per tick from the forecast mixture, with
no $z^*$. Its only hindsight is $\hat r$ inside `pi_all`. Feed it §1's $\tilde\pi_t$ instead and
it is fully live with no other change. It is the existence proof for §4.1: the repo already pays
for, and trusts, a per-tick mixture tail.

### 4.3 Stall: replaces $s_{\text{temporal}}$

**Question.** "How probable is it that the person is past the usual upper limit for whatever step
they're in?"

**Bad set.** For each $(r,i)$, the elapsed times in the upper α-tail:
$U_{r,i} = \{\,j : P(D\ge j\mid i,r) < \alpha\,\}$.

$$
\text{stuck}_t = \sum_{r,i}\ \sum_{j\in U_{r,i}} b_t(r,i,j),
\qquad \text{flag on an upward crossing of } \kappa_{\text{stall}}.
$$

**Why $b_t$ and not $b'_t$.** The emission channels must not let $o_t$ into the belief, because
$o_t$ is the thing being tested. Here the thing being tested is the hidden elapsed time, and
$o_t$ is *evidence about it*. Leaving $o_t$ out would just discard information. (R1 is the rule
behind this.)

**How a real stall shows up.** Every tick, the hazard pushes mass out of the "pour milk" row into
the next steps. Every tick, (pour, milk) arrives, which only that row explains, so the update
pulls the mass back into high-$j$ cells of "pour milk." **Evidence for "stuck" means observations
keep supporting a row the duration model expected the person to leave.**

**Persistence.** stuck$_t$ stays high for the rest of the stall. So *narrate* once, on the
crossing, and re-arm only after it falls back below κ (hysteresis). This matches today's
narration, which speaks at the first crossing and not at the peak. *Flags*, as scored in §8, mark
every tick above κ, the same convention `s_temporal` is scored under.

**Collapse (R2).** A point mass at $(\hat r,z^*_t,e^*_t)$ gives stuck$_t = \mathbb 1[P(D\ge e^*_t)
< \alpha] = \mathbb 1[s_{\text{temporal}} > -\log\alpha]$, which is the current flag.

**Inherited limitation.** The model ends every step by $D_{\max}$. If $P(D\ge M\mid i,r) \ge \alpha$,
then $U_{r,i}$ is empty and state $i$ *cannot* stall. The current code clamps the same way. States fit
with means of 32–734 ticks ([`anomaly.md`](anomaly.md) §2.3) are exactly the ones at risk.

### 4.4 Transition: replaces $s_{\text{trans}}$

**Question.** "How probable is it that a step change happened at this tick, *and* that it was an
unusual one?"

**Bad set.** $T_{r,k} = \{\,i : -\log A^{(r)}_{ki} > \tau^{\text{trans}}_{r,k}\,\}$, taken
directly from `transition_quantile_threshold`, one table per recipe.

$$
\text{trans}_t = \sum_{r,k}\ \sum_{i\in T_{r,k}}\ \sum_j q_t(r,k,j,i),
\qquad\text{flag if } > \kappa_{\text{trans}}.
$$

It is a **joint** event: switch *and* tail. On a non-boundary tick, the switch probability is
small, so trans$_t$ is small no matter how odd any hypothetical transition would be. No boundary
mask is needed, unlike `flag_joint`.

**At L = 0 it is not as weak as it sounds.** If $o_t$ can only be explained by state $i$, the
update moves almost all the belief onto "just started $i$" in a single tick. A skipped step,
where the new step's tokens are explained only by $i$ and $A_{ki}$ is in the tail, gives
trans$_t \approx 1$ immediately. The weak case would be a **one-tick blip**: a stray token that
another state also explains, so the next tick takes the switch back -- the case lag was meant to
fix (§7). **On the fitted models it does not arise.** Every state's emission row is a near-delta
on its own (verb, noun) pair (97% of cells at the floor, `emission-rows-are-deltas`), so no other
state "accidentally" explains a stray token: even with the whole trial in view, the cheapest
account of a one-tick plate token is a one-tick plate step. Measured, lag changes nothing (§8).

**Narration.** "After $k$ you normally do $b$": take $(\hat k,\hat i) = \arg\max_{k,i}$ of the
$q$-mass, then run `missing_step` bridging over $A^{(\hat r_t)}$ unchanged, with $\hat r_t =
\arg\max w_t$.

**Collapse (R2).** A point mass on the hindsight switch gives $\mathbb 1[z^*_{(m)} \in
T_{\hat r, z^*_{(m-1)}}]$, which is the current flag, including the from-state indexing.

### 4.5 Early exit: replaces the short half of $s_{\text{dur2}}$

**Question.** "How probable is it that the step that just ended ended too early?"

**Bad set.** $S_{r,k} = \{\,j : P(D\le j\mid k,r) < \alpha/2\,\}$. The α/2 carries over the
two-sided doubling (§4.6).

$$
\text{early}_t = \sum_{r,k}\ \sum_{j\in S_{r,k}}\ \sum_i q_t(r,k,j,i),
\qquad\text{flag if } > \kappa_{\text{early}}.
$$

**Timing.** It fires on the **first tick of the next step**, the earliest moment the end is
knowable. The current $s_{\text{dur2}}$ is placed on the last tick of the ended step, which is one
tick earlier and leaks one tick of the future. [`llm.md`](llm.md) §1 explains why exactly that
leak inflates abandonment recall, which is the type this channel exists for.

**Censoring is automatic.** The trial's final step never ends inside the recording, so it never
produces a switch, so it can never fire. `final_censored=True` becomes unnecessary.

**Collapse (R2).** A point mass on the hindsight boundary gives
$\mathbb 1[P(D\le d_{(m)}) < \alpha/2]$. That is the current two-sided flag restricted to the
short side, because $P(D\le d)<\alpha/2$ forces $P(D\ge d) > 1-\alpha/2$, so the short tail is
the smaller one automatically.

### 4.6 The long half of $s_{\text{dur2}}$ — dropped

"Too long, judged after it ended" is the stall channel judged late. A live system has already
raised the stall flag while it was happening. Keeping both would double-alarm one event.

**Budget note.** Today, the long side gets α/2 inside $s_{\text{dur2}}$ and α inside
$s_{\text{temporal}}$. Live, only the stall channel remains, at α. So the long side's sensitivity
is that of today's $s_{\text{temporal}}$. Nothing is lost.

### 4.7 Recipe: replaces the joint $s_{\text{recipe}}$

**Question.** "Is this step change ordinary in general but wrong for the recipe the person seems
to be making?"

**Bad set.** $Q_{r,k} = \{\,i : \log\bar A_{ki} - \log A^{(r)}_{ki} > \tau^{R}_{r,k}\,\}$, from
`excess_quantile_threshold` per recipe (null: $i\sim A^{(r)}_{k\cdot}$, as today).

**The circularity, and its fix.** If $q_t$ is used as defined, the recipe belief has already
been updated by the very transition under test. An off-recipe transition then shifts belief
toward a recipe in which it is *normal*, and so explains itself away. This is Q1's mistake one
level up: testing something against a belief that has already seen it. So hold the recipe at its
pre-transition belief:

$$
\text{recipe}_t = \sum_r w_{t-1}(r)\ \sum_k\ \sum_{i\in Q_{r,k}}\ \sum_j q^{(r)}_t(k,j,i),
$$

where $q^{(r)}_t$ is $q_t$ renormalised **within** recipe $r$, and $w_{t-1}$ has not seen the
switch. Flag if it exceeds $\kappa_{\text{recipe}}$.

**Expected to be the weakest live channel.** Early in a trial, $w_{t-1}$ is diffuse, so no single
recipe's "wrong" set gets much weight. Hindsight is the *strongest* here, because $\hat r$ is
pinned by everything that happened *after* the transition. Also, the recipe layer's own fit is
poor: EM halves recipe ARI compared with its warm start. Measure it before trusting it, and
consider dropping it.

**Degenerate case.** At $K_R=1$, $\bar A = A^{(1)}$, so the excess is identically 0. The bad set
must be empty, which means passing +inf thresholds exactly as today (see the
`one-recipe-degenerate-channel` note).

**Cascade $s_{\text{recipe}}$: not designed.** It runs a second HMM over the *symbols of completed
steps*, using forward-backward (`_segment_recipe_path`), which is itself hindsight. A live version
would have to filter over *uncertain* symbols. It is deferred because the joint model is the
shipped path.

---

## 5. When several channels fire on one event

A skipped step $k\to i$ raises trans$_t$ (the switch is in the tail) *and* $p_t$ (the forecast
gave $i$'s tokens little mass), both on the same tick. The emission surprise is *caused* by the
transition, so name the transition. This is the same order `element_metrics.CHANNEL_PRIORITY`
already uses (transition before duration before stall before the emission channels).

Concretely: when trans$_t > \kappa$ on the same tick as an emission flag, report one alarm
labelled "transition." Only report "wrong object" when the emission flag fires on a tick with
little switch probability.

---

## 6. Worked example: a healthy step change

"Pour milk" (typical length ~5) → "take bowl," with the switch happening at elapsed 9. Before
$o_9$: $\tilde\pi = 0.85$ pour milk, $0.15$ take bowl. $P(\text{take,bowl}\mid\text{take bowl})
= 0.9$, and $\approx 0$ under pour milk.

| | Current, with whole-trial $z^*$ | Current scheme with a *live* $z^*$ | Live §4.1 |
|---|---|---|---|
| score | $-\log(0.15\cdot0.9)\approx2.0$ | same 2.0 | same 2.0 |
| threshold | $\tau_{\text{take bowl}} - \log 0.15 \approx \tau + 1.9$ | $\tau_{\text{pour milk}} - \log 0.85 \approx \tau + 0.16$, and pour milk's τ is small | $p_t\approx0.05 > \alpha$ |
| flag? | no, but only because hindsight knew the step | **yes, a false alarm at every boundary** | no |

The switch posterior: after $o_9$, almost all belief moves to "just started take bowl" (pour milk
can't explain the token). pour milk → take bowl is common, so it is not in $T$, and
trans$_9\approx0$. Elapsed 9 isn't short for pour milk, so early$_9\approx0$. Nothing fires.

---

## 7. Adding a lag L

Every posterior channel (§4.3–4.7) can condition on $o_{0:t+L}$ instead of $o_{0:t}$, and report
at tick $t+L$:

$$
q^{(L)}_t(r,k,j,i) \;\propto\; q_t(r,k,j,i)\cdot P\bigl(o_{t+1:t+L}\mid R=r,\ z_t=i,\ e_t=1\bigr),
$$

and likewise for stall with the state $(r,i,j)$. The second factor is a backward message over
$L$ ticks, at a cost of $O(L\cdot K_R K D_{\max})$ per tick. That is fixed-lag smoothing: the
same backward recursion as [`hsmm.md`](hsmm.md) §4.3, truncated at $t+L$.

| Channel | Expected gain from lag (as designed) | Measured (§8) |
|---|---|---|
| transition, early exit | **large**: the switch is decided from one tick at L=0, and a blip and a real switch look the same until $t+1$ | **none** -- with near-delta emissions $o_t$ alone already settles which step is running |
| recipe | large: $w$ sharpens with every step | none on the dish model; within noise on the clustered one |
| stall | small | none |
| emission $p_t$ | none directly: it is a *forecast* of $o_t$ | n/a |

L = 0, 2, 5, 10 and ∞ (the full smoother) agree to within 0.004 accuracy on Breakfast and are
identical on EPIC-Tent. Lag is the right tool for a model whose emission rows overlap; on these
fits it is free to set to 0.

Lag can still help the emission channels indirectly. An emission flag at $t$ can be **retracted**
at $t+L$ if the smoothed belief says a step started at $t$ and $o_t$ is typical for it. That is
an option, not part of the core design.

**Latency becomes a stated parameter:** "flags arrive within L ticks." L = 0 is fully live. L = ∞
is the smoother -- **not** what ships today: the shipped detector commits to Viterbi's single
best path (max-product) and thresholds emissions by it, whereas L = ∞ averages over every path
(sum-product). They agree only where the posterior piles onto one path, which is what R2 tests.

---

## 8. Measured

All headline numbers are **metric G** ([`eval.md`](eval.md) §0: alarm onsets find spans, merge 5,
deadline 10), with alarms per 10 min of error-free activity and step FPR beside them. Hidden-event
channels at κ = 0.5 unless an operating point says otherwise.

### 8.1 EPIC-Tent — real annotated mistakes (metric G)

Live arm at the hindsight arm's own α (0.005, `s_temporal` 1e-5), κ = 0.5, L = 0; nothing tuned.
EPIC-Tent is one recipe (`k_recipe: 1`), so the recipe channel is off in both arms.

**PREGO split** (15 test sessions; `epic_tent_report_G_llm_prego.json`):

| procedural | precision | recall | F1 | alarms / 10 min | step FPR |
|---|---|---|---|---|---|
| null: flag every step | 0.170 | 0.860 | 0.284 | 26.75 | 1.000 |
| null: flag every tick | 0.000 | 0.000 | 0.000 | 4.10 | 1.000 |
| HSMM hindsight | 0.458 | 0.355 | 0.400 | 2.90 | 0.142 |
| **HSMM live** | **0.528** | 0.355 | **0.425** | **2.33** | **0.125** |
| LLM predictive, no recipes | 0.149 | 0.841 | 0.254 | 30.22 | 0.973 |
| LLM predictive, + instruction order | 0.149 | 0.850 | 0.254 | 30.48 | 0.923 |

`procedural+temporal`: hindsight 0.545 / 0.345 / 0.423 (2.91 per 10 min); live 0.605 / 0.331 /
0.428 (2.35); LLM 0.201 / 0.863 / 0.326 (31.0); every-step null 0.225 / 0.871 / 0.358. The
default 20/9 split tells the same story (procedural: hindsight F1 0.366, live 0.378, LLM 0.216,
every-step null 0.233).

**Live finds exactly the spans hindsight finds** (identical per-type recall: order 0.411, omit
0.000, correction 0.000, repeat 0.333) **with ~20% fewer alarms on error-free activity.** It is
the only arm on the table that does not need the session to be over. Under G the flag-every-tick
null collapses to 0.000 (it scored F1 0.93 under D), and flag-every-step is the null to beat --
both HSMM arms do, at a tenth of its alarm rate; neither LLM arm does.

**PREGO-trimmed (metric E, the only protocol comparable with published PREGO):** hindsight 3/15
sessions caught (P 0.231, F1 0.214, false alarm on 8.1% of correct steps), live 2/15 (P 0.143,
F1 0.138, 9.7%), LLM no recipes 15/15 at 95.2% of correct steps flagged, flag-every-step null
15/15 at 100%. One session separates the two HSMM arms.

### 8.2 Breakfast — injected anomalies

Metric G on the injector's own spans, 378 train / 97 test trials per group, operating points
picked on **train** (by G-F1) and reported on **test** (`tools_live_summary.py --metric-g`; runs
`live_{dishknownG,dishG,lexhardG}_{train,test}.json`). Three models/settings:

* **dish, given** -- `joint_params_train_dish.npz` (recipe = labelled dish, K_R = 10, labels pinned
  in every fitting stage; `fit_dish_model.sh`), with the true dish handed to both arms;
* **dish, inferred** -- the same model, the dish read off the trial (hindsight: whole trial;
  live: the running posterior). Unlabelled, it names the dish on 94.1% of whole test trials;
* **cluster** -- the shipped `joint_params_train_lexhard.npz` (16 latent recipe clusters).

| setting | arm | α (κ) | precision | recall | F1 | alarms / 10 min | step FPR |
|---|---|---|---|---|---|---|---|
| dish, given | hindsight | 0.02 | 0.759 | 0.567 | **0.649** | **1.39** | **0.055** |
| dish, given | live L=0 | 0.035 (0.9) | 0.682 | 0.605 | 0.641 | 2.08 | 0.097 |
| dish, given | live L=2…∞ | 0.02 (0.3–0.9) | 0.741 | 0.574 | 0.647 | 1.49 | 0.074 |
| dish, inferred | hindsight | 0.02 | 0.755 | 0.560 | 0.643 | 1.39 | 0.054 |
| dish, inferred | live L=0 | 0.02 (0.7) | 0.704 | 0.584 | 0.638 | 1.83 | 0.086 |
| cluster | hindsight | 0.05 | 0.653 | 0.622 | 0.637 | 2.43 | 0.092 |
| cluster | live L=0 | 0.015 (0.05) | 0.679 | 0.617 | 0.646 | 2.23 | 0.105 |

Matched alarm budget (best train recall with train alarms ≤ 1 per 10 min), on test:

| setting | hindsight F1 / alarms / step FPR | live L=0 F1 / alarms / step FPR |
|---|---|---|
| dish, given | 0.649 / 1.39 / 0.055 | 0.648 / 1.54 / 0.077 |
| dish, inferred | 0.643 / 1.39 / 0.054 | 0.633 / 1.64 / 0.082 |
| cluster | 0.634 / 1.44 / 0.060 | 0.647 / 1.78 / 0.094 |

Per-type test recall at the F1 pick (dish, given): substitution 0.96 / 0.95, abandonment 0.59 /
**0.67**, omission 0.30 / **0.37**, transposition 0.65 / 0.66, repetition 0.26 / 0.31 (hindsight /
live L=0).

**Reading.** Going live costs **no F1** on any setting -- every live-vs-hindsight F1 gap is within
±0.01, against roughly ±0.03 of noise at 97 trials per type. What it costs is a little **alarm
budget**: at matched F1 the live arm raises ~0.1–0.4 more alarms per 10 min and flags 2–4 more
correct steps per hundred, in exchange for catching somewhat more abandonments and omissions. The
dish model is quieter than the cluster model at its F1 pick (1.4 vs 2.4 alarms per 10 min) at the
same F1, and **knowing the dish vs inferring it moves nothing** (≤ 0.006 F1) -- consistent with
94% unlabelled dish accuracy. Lag again buys nothing. One caveat that applies to every arm: train
alarm budgets do not hold on test (picked at ≤ 0.5 per 10 min on train, test lands at 1.2–1.5),
because healthy *train* trials are the ones the model was fitted to.

**One table with the nulls and the LLM** (`tools_breakfast_g_table.py`, `runs/breakfast_g_table.json`).
The LLM's cached answers exist only for `run_llm_eval.py`'s own pool (97 test trials, injected via
the older `joint_params.npz`), so that pool is the reference here: it was replayed from cache with
`--dump-streams` (no new requests) and every other arm was scored on the same streams, the HSMM
arms at the operating points picked on train above. Metric G:

| arm | precision | recall | F1 | alarms / 10 min | step FPR |
|---|---|---|---|---|---|
| null: flag every step | 0.224 | 0.899 | 0.359 | 26.87 | 1.000 |
| null: flag every tick | 0.000 | 0.000 | 0.000 | 4.81 | 1.000 |
| LLM predictive, no recipes | 0.202 | 0.840 | 0.326 | 27.26 | 0.800 |
| LLM predictive, + recipes | 0.254 | 0.835 | 0.390 | 19.58 | 0.639 |
| LLM predictive, + recipes, Auto-CoT | 0.258 | 0.845 | 0.395 | 19.78 | 0.618 |
| HSMM dish, given: hindsight | 0.769 | 0.560 | **0.648** | 1.39 | 0.056 |
| HSMM dish, given: live L=0 | 0.703 | 0.593 | 0.643 | 2.08 | 0.097 |
| HSMM dish, inferred: live L=0 | 0.727 | 0.577 | 0.644 | 1.83 | 0.087 |
| HSMM cluster: live L=0 | 0.684 | 0.598 | 0.638 | 2.23 | 0.107 |
| HSMM `joint_params.npz` at α=0.005 (the arm in the old LLM reports) | 0.804 | 0.393 | 0.528 | **0.50** | **0.022** |

The LLM arms sit at the flag-every-step null (no-recipes below it, with-recipes 0.03 above), flag
two thirds of correct steps and raise ~20–27 alarms per 10 min; every HSMM arm is at roughly a
tenth of that alarm rate with F1 ~0.64. The HSMM numbers reproduce the run_live_eval ones on a
different pool to within 0.01, so the pool is not what is being measured.

### 8.3 What the measurement corrected in the design

1. **The plain forecast p-value (§4.1) double-counts step timing.** All 28 healthy live emission
   flags on 39 test trials sat on step changes after early-ending steps; the timing split fixes
   it (15 → 6 trials with any emission flag).
2. **Lag buys nothing on these fits** (§7) -- near-delta emissions make $o_t$ decisive.
3. **The scoring, not the detector, sank live abandonment recall under metric A.** A's positive
   range ends one tick before a causal "ended too early" flag can fire, so every correct catch was
   charged as a stray: recall 0.00 at the honest tick vs 0.43 one tick earlier, at zero healthy
   false alarms (`causal-flags-need-one-tick-grace`). Metric G's 10-tick deadline removes the
   problem by construction.

### 8.4 Metric A, for continuity (Breakfast, 97 test trials per group, picked on test -- descriptive)

| | hindsight | live L=0 | live L=10 | live L=∞ |
|---|---|---|---|---|
| clustered model, A | 0.505 | 0.436 | 0.436 | 0.434 |
| clustered model, A +g1 | 0.547 | 0.560 | 0.562 | 0.558 |
| dish model, true dish given, A +g1 | 0.550 | 0.559 | 0.559 | 0.559 |

(accuracy; always-flag 0.457.) Plain A charges causal "ended early" catches as strays; read +g1.

Picked on **train** (378 trials per group), reported on test -- clustered model, A +g1:

| arm | α (κ) | train acc | test acc | test precision | test recall | test healthy-trial FPR |
|---|---|---|---|---|---|---|
| hindsight | 0.02 | 0.557 | 0.540 | 0.736 | 0.581 | 0.227 |
| live L=0 | 0.015 (0.05) | 0.533 | **0.560** | 0.695 | **0.664** | 0.330 |
| live L=10 | 0.01 (0.05) | 0.535 | 0.541 | 0.700 | 0.621 | 0.299 |
| live L=∞ | 0.01 (0.1) | 0.537 | 0.534 | 0.703 | 0.604 | 0.289 |

### 8.5 Checks that are tests, not measurements

1. **R2 as a unit test** -- a point-mass belief at the hindsight answer reproduces every current
   flag (`tests/test_live.py`, emission and all four hidden-event channels, non-vacuous: 12–20
   positive flags per channel exercised).
2. **The exactness test (§1)** -- $\sum_t\log P(o_t\mid o_{<t}) = $ `trial_ll` per trial, and the
   full-lag smoother reproduces `combine_sufficient_stats`' γ and expected transition count.
3. **κ** -- 0.5 by default; the Breakfast sweep picks it on train with α.

### 8.6 Still open

- Does the recipe channel (§4.7) earn its place live? Per-channel ablation on the dish model.
- Does $p_t$'s deafness after a first anomaly cost recall on multi-anomaly corpora? EPIC-Tent's
  per-type recall is identical between arms, which is weak evidence it does not.

## 9. Open questions

- Is a single κ per channel enough, or does the switch posterior need its own floor (flag only if
  $\sum q_t > c$) to stop tail-only noise from accumulating across many low-probability switches?
- Mid-p ($\tfrac12$ weight on the observed tie group) vs. the full-tie $p_t$ in §4.1. The full tie
  matches today's convention, but it is conservative on the near-deterministic states that
  dominate Breakfast.
- Should the stall bad set use α/2, so the stall and early-exit channels split one two-sided
  budget? Or should each get α, as today's $s_{\text{temporal}}$ effectively does?
