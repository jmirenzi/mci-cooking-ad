# Detectors: from beliefs to alarms

A detector reads a trial, one (verb, noun) pair per tick, and raises alarms. This doc covers the
shared contract, the α-tail **bad sets** every HSMM channel is tested against, the three HSMM
detectors (Live, Lag, Hindsight), and narration. The two LLM detectors implement the same contract
and are covered in [`llm.md`](llm.md). The model they read is in [`model.md`](model.md).

| Detector | Causal? | An alarm about tick t is known at | Reads |
|---|---|---|---|
| `LiveHSMMDetector` | yes | t | the filter b_t (§3) |
| `LagHSMMDetector(L)` | yes for finite L | t for emission channels, t + L for event channels | filter + L-tick backward message (§4) |
| `HindsightHSMMDetector` | no | T − 1 | r̂ and the Viterbi path of the whole trial (§5) |
| `LLMVerdictDetector`, `LLMPredictiveDetector` | yes, except protocol `batch` | see [`llm.md`](llm.md) | the rendered prefix |

---

## 1. The contract

`core.py` holds the types every detector shares:

- `Trial(trial_id, verbs, nouns, recipe=None, spans=(), group="healthy")`. `recipe` is a known
  dish (int), a (K_R,) log prior, or `None` (infer it). `spans` is the ground truth, as `Span(start,
  end, kind)`.
- `Alarm(tick, channel, score, known_at)`. `tick` is the tick the alarm is *about*, and is what
  metric G scores. `known_at` is the first tick at which the detector could have raised it.
- `Detection(alarms, trace)`. `trace` is detector-specific: `LiveTrace`, `HindsightTrace`, the LLM
  verdicts, or parse statistics.
- `OperatingPoint(alpha, kappa)`. `alpha` is one tail mass for every channel or a per-channel
  dict; `alpha_for(channel)` reads it. `kappa` is the posterior cutoff of the Live and Lag event
  channels.

`AnomalyDetector` has five methods. A causal detector implements `reset(recipe)` and
`step(verb, noun) -> [Alarm]`, plus `finish()` for alarms still pending at the end. The default
`detect_trial` is that loop. Detectors override `detect_trial` / `detect_many` with vectorised
fast paths, and `tests/test_detector_contract.py` checks that every fast path equals its step
loop. A non-causal detector (`causal = False`) raises `NotImplementedError` from `step()`.

**The seven HSMM channels.** Names are shared by Live and Hindsight, so tables compare like with
like:

| Channel | Live question (§3) | Hindsight question (§5) |
|---|---|---|
| `s_emit`, `s_verb`, `s_noun` | is o_t in the forecast's α-tail? | is o_t surprising under π̃ at z*_t? |
| `s_temporal` | P(past the usual upper limit) > κ? | −log P(D ≥ elapsed \| z*) > −log α? |
| `s_dur_two` | P(the segment that just ended was too short) > κ? | two-sided duration p-value of a closed segment < α |
| `s_transition` | P(a switch now, into the α-tail of A) > κ? | −log A[z*_prev, z*] above its α-threshold? |
| `s_recipe_transition` | as transition, against the recipe excess | recipe excess above its α-threshold? |

---

## 2. Bad sets

A **bad set** is a set of outcomes that carries at most α of the model's own probability. It is a
property of the fitted model; data chooses only α (and κ). The null hypothesis is always "this
came from the model as fitted". So α is a tail mass, **not** a false-alarm rate. False alarms are
measured on healthy data, not assumed from α. `class BadSets(hsmm, alphas)` builds every set for
one model over an α grid (`eval/harness.ALPHAS`).

### 2.1 Exact discrete tails

Every categorical row has a finite, known support, so its α-tail is computed exactly
(`thresholds.tail_threshold`). Given scores s = −log P and probabilities over one row:

1. drop zero-probability entries (the masked transition diagonal);
2. sort by descending score;
3. collapse ties. Under the strict `>` used when flagging, a tied group is in or out as a whole;
4. take the largest number of leading groups whose total mass is ≤ α.

This is a step function: α is achieved only up to the nearest group. The builders apply it to
different supports:

| Builder | Support | Channel |
|---|---|---|
| `categorical_quantile_threshold` | one emission row (V or N) | `s_verb`, `s_noun` |
| `joint_quantile_threshold` | the V × N product row of one state | `s_emit` |
| `transition_quantile_threshold` | one row of A^(r) | `s_transition` |
| `excess_quantile_threshold` | the excess log Ā − log A^(r), null i ~ A^(r)_{k·} | `s_recipe_transition` |

The joint emission tail is not recoverable from the two marginal tails, so `s_emit` builds the
product row. The excess tail is scored under A^(r), the distribution the observed transition is
drawn from, and **its threshold can be negative**: when a recipe's row is flatter than the
marginal, most transitions score below zero.

The duration channels are already tail probabilities, so they need no table. Their sets are
P(D ≥ d) < α (stall) and P(D ≤ d) < α/2 (early exit; the halving is the two-sided split).

### 2.2 Two readings of the same sets

- **Hindsight** reads thresholds per state: `BadSets.hindsight(r̂, α)` returns a
  `HindsightThresholds(emit, verb, noun, transition, recipe)`, memoised per (α, r̂).
- **Live and Lag** read *bins*. Every cell (stall: (r, i, j); early exit: (r, k, j); transition
  and recipe: (r, k, i)) stores the index of the first grid α at which it is bad. The sets are
  nested in α (`first_true` checks this), so one segment-sum over bins, then a cumulative sum,
  gives the bad mass at every grid α in one pass (`hsmm_live.binned_mass`). A sweep therefore
  computes each trace once and thresholds it many times.

`BadSets.alpha_index` raises `ValueError` if asked for an α that is not on the grid.

### 2.3 The one-recipe rule

At K_R = 1 the recipe excess is undefined: there is no other recipe for a transition to be
ordinary under. The arithmetic does not give zero, though. `marginal_log_trans` floors
probabilities, so on rare transitions the "excess" is a flooring artifact of up to ~6 nats. Two
things are both needed:

- the statistic is exactly zero (`HindsightHSMMDetector.recipe_transition_surprise`);
- the thresholds are +inf (`thresholds.recipe_excess_thresholds`).

A zero statistic alone is not enough: an artifact-derived threshold can dip below zero, and then
`0 > threshold` fires. The rule is gated on the number of recipes, not on a per-row "recipe row
equals marginal" test. That test would also match 112 of 1024 rows of the Breakfast K_R = 16 fit.
Those rows are whole states that are flat in every recipe, so the statistic ties its threshold
exactly and the strict `>` never fires. They are safe only while the comparison stays strict.

---

## 3. Live: the causal detector

`LiveHSMMDetector` runs the filter of [`model.md`](model.md) §3 and raises every alarm about tick t
at `step()` of tick t. Four rules shape its channels:

- **R1. Observed things get predictive p-values; hidden things get posterior probabilities.** The
  emission channels test o_t, which is observed: "how unlikely was this under the forecast?" The
  event channels test events nobody observes (was there a switch, is the person past their usual
  time): "how probable is it, given what was seen, that a bad event happened?"
- **R2. Each live channel collapses to its Hindsight counterpart.** With the belief a point mass
  on the hindsight answer (r̂, z*, e*), every live flag equals the Hindsight flag. This is a unit
  test (`tests/test_live.py`, every channel, non-vacuous).
- **R3. Reuse the bad sets** of §2, so α means the same thing in both detectors.
- **R4. Posterior channels need a second knob, κ.** A posterior is not a p-value, so an event
  channel flags when its posterior bad mass exceeds κ. κ = 0.5 is "more likely than not". Like α,
  κ is picked on train.

### 3.1 Emission channels: a timing-split forecast p-value

The forecast for tick t is the mixture P_mix(x) = Σ_i w(i) P(x | i), over x ∈ V, N or V × N. The
p-value of the observed x_t is the forecast mass on outcomes no more probable than it, ties
included:

$$
p_t = \sum_x P_{\text{mix}}(x)\;\mathbb 1\bigl[P_{\text{mix}}(x) \le P_{\text{mix}}(x_t)\bigr],
\qquad \text{flag if } p_t \le \alpha .
$$

**Split by timing.** The plain mixture over π̃ blends "the segment continued" with "a new segment
began now", weighted by the hazard. A healthy step change that comes earlier than the duration
model expected then gets a small forecast probability and is flagged as a wrong token. That is
duration surprise counted a second time. So the forecast is split into its two hypotheses,
w = stay_t and w = switch_t ([`model.md`](model.md) §3.3), each renormalised, and the **larger**
p-value is kept (`split_pvalues`). A token is flagged only if it is in the α-tail whether or not a
step change happened at t. Timing is left to the duration channels and order to the transition
channel. A hypothesis with less than `HYPOTHESIS_FLOOR` = 1e-12 of the forecast drops out of the
max, which keeps R2 exact at a point mass. `TIE_ATOL` = 1e-10 (in log probability) decides ties.

The split was a measured fix. On 39 healthy Breakfast test trials at α = 0.005, all 28 plain-
mixture emission flags sat on step changes after a step that ran a median 0.52 of its expected
length. With the split, healthy trials with any emission flag fell from 15 to 6, and the remaining
flags were step changes into a step the transition row made unlikely.

There is no dilution correction (compare §5.1). Score and threshold come from the same forecast.

**Known weakness.** A forecast spread over many states puts little in its α-tail. Belief is most
spread right after something odd, so a second anomaly just after a first may go unflagged.
EPIC-Tent's per-type recall is identical for Live and Hindsight, which is weak evidence that this
does not cost much.

### 3.2 The switch posterior

Three event channels ask about the segment that just ended and the one that just began. The
object behind them is

$$
q_t(r,k,j,i) = \frac{b_{t-1}(r,k,j)\,h_{rkj}\,A^{(r)}_{ki}\,P(o_t \mid i)\,P(o_{t+1:t+L} \mid r, z_t{=}i, e_t{=}1)}{P(o_t, o_{t+1:t+L} \mid o_{<t})},
$$

the probability that under recipe r, subtask k ended after exactly j + 1 ticks and i began at t.
The last factor in the numerator is the lag message e^β (L = 0: 1). Each channel keeps some indices and sums out the rest:

| Channel | Keeps | Bad set |
|---|---|---|
| transition (§3.4) | (r, k, i) | i in the α-tail of A^(r)_{k·} |
| early exit (§3.5) | (r, k, j) | j + 1 in the short α/2 tail of D \| k, r |
| recipe (§3.6) | (r, k, i) | k → i in r's excess tail |

q_t is never materialised (on EPIC it would hold 52M entries per tick). Every channel factors
through `log_end` and `log_m` = Σ_j log_end from `kernels.predict`, plus the arrival term
P(o_t | i) e^{β(r,i,0)} (`switch_terms`). Each costs O(K_R K (K + D)).

### 3.3 Stall (`s_temporal`)

"How probable is it that the person is past the usual upper limit for whatever step they are in?"

$$
\text{stall}_t = \sum_{r,i}\ \sum_{j:\,P(D \ge j+1 \mid i,r) < \alpha} \tilde b_t(r,i,j), \qquad \tilde b_t \propto b_t\, e^{\beta_t},
$$

flagged when > κ. It reads b_t, the belief **after** o_t. The emission channels must not let o_t
in, because o_t is what they test; here the tested thing is the hidden elapsed time, and o_t is
evidence about it. A real stall shows up as observations that keep supporting a row the duration
model expected the person to leave. Every tick above κ is flagged, as Hindsight's `s_temporal`
flags every tick above its threshold.

**Inherited limit.** Every segment ends by D. If P(D ≥ D | i, r) ≥ α the stall set of (r, i) is
empty and that state cannot stall. Hindsight has the same clamp.

### 3.4 Transition (`s_transition`)

"How probable is it that a switch happened at t *and* that it was an unusual one?"

$$
\text{trans}_t = \sum_{r,k}\ \sum_{i \in T_{r,k}}\ \sum_j q_t(r,k,j,i), \qquad
T_{r,k} = \{i : -\log A^{(r)}_{ki} > \tau^{\text{trans}}_{r,k}\}.
$$

It is a joint event, so on a tick with little switch probability it is small however odd a
hypothetical switch would be; no boundary mask is needed. At L = 0 it is not weak. If only state i
explains o_t, the update moves almost all belief onto "just started i" in one tick. The weak case
would be a one-tick blip that another state also explains. On the fitted models that case does
not arise, because emission rows are near-deltas ([`model.md`](model.md) §1).

### 3.5 Early exit (`s_dur_two`)

"How probable is it that the segment that just ended ended too early?"

$$
\text{early}_t = \sum_{r,k}\ \sum_{j:\,P(D \le j+1 \mid k,r) < \alpha/2}\ \sum_i q_t(r,k,j,i).
$$

It fires on the **first tick of the next segment**, the earliest moment the end is knowable.
Hindsight places `s_dur_two` on the segment's last tick, one tick earlier, which is one tick of
the future (§5.3). Censoring is automatic: a trial's final segment never ends inside the
recording, so it never produces a switch.

The long side of the two-sided test is dropped. "Too long, judged after it ended" is the stall
channel judged late, and a live system has already raised the stall alarm. The long side's
sensitivity is therefore the stall channel's, at α.

### 3.6 Recipe (`s_recipe_transition`)

"Is this step change ordinary in general, but wrong for the recipe the person seems to be making?"

Bad set: k → i with log Ā_ki − log A^(r)_ki above r's excess threshold. Using q_t as defined would
let the transition under test pull the recipe belief toward a recipe in which it is normal, so it
would explain itself away. The recipe is therefore held at its belief **before** the switch:

$$
\text{recipe}_t = \sum_r w_{t-1}(r)\ \sum_k\ \sum_{i \in Q_{r,k}}\ \sum_j q^{(r)}_t(k,j,i),
$$

with q^(r) renormalised within recipe r (`recipe_transition_prob`). This is the channel hindsight
helps most, since r̂ is pinned by everything after the transition. At K_R = 1 its bad set is empty
(§2.3).

### 3.6a Visit count (`s_count`, optional)

"Is the subtask beginning now one visit too many, or does it follow a subtask that never happened?"
Present only when the detector is given fitted visit counts (`hsmm/visit_counts.py:VisitCounts`,
§5.6), so the seven-channel detector is unchanged without them. A first-order model sees adjacent
pairs only: a step done twice with a legal step between, or skipped where its neighbours may follow
each other, leaves every transition ordinary.

The entry mass into i at t is the switch posterior summed over what was left,
m_t(r, i) = Σ_k Σ_j q^(r)_t(k, j, i) (`entry_one_tick`). At t = 0 it is the posterior itself.
Summing it over the past, within recipe r, gives the expected visits so far:

$$
\hat c_{t,r,k} = \sum_{t' < t} \frac{m_{t'}(r, k)}{P(r \mid o_{0:t'})}.
$$

A beginning of i at t is bad at α if its visit number v = ⌊ĉ_{t,r,i} + ½⌋ + 1 has
P̂(C_i ≥ v | r) ≤ α (an extra visit), or if some k whose usual successor is i has ĉ_{t,r,k} < ½ and
P̂(C_k = 0 | r) ≤ α (it follows a missing step). Then

$$
\text{count}_t = \min\Bigl(1,\ \sum_{r,i} m_t(r, i)\ \mathbb 1[\text{bad}_\alpha(r, i)]\Bigr),
$$

flagged at count_t > κ (`CountState.step`). Using expected visits rather than their distribution
is an approximation: the exact belief would also have to track which subtasks have been visited.

### 3.7 Worked example: a healthy step change

"Pour milk" (typical length ~5 ticks) is followed by "take bowl", with the switch at elapsed 9.
Before o_9, π̃ is 0.85 pour milk and 0.15 take bowl. P(take, bowl | take bowl) = 0.9, and ≈ 0
under pour milk.

| | Hindsight (whole-trial z*) | Hindsight's rule with a *live* z* | Live §3.1 |
|---|---|---|---|
| score | −log(0.15 · 0.9) ≈ 2.0 | the same 2.0 | the same token |
| threshold | τ_take bowl − log 0.15 ≈ τ + 1.9 | τ_pour milk − log 0.85 ≈ τ + 0.16 | under the switch hypothesis (take, bowl) is the mode: p ≈ 1 > α |
| flag? | no, because hindsight knew the step | **yes: a false alarm at every boundary** | no |

After o_9 almost all belief moves to "just started take bowl", since pour milk cannot explain the
token. Pour milk → take bowl is common, so trans_9 ≈ 0. Nine ticks is not short for pour milk, so
early_9 ≈ 0. Nothing fires.

### 3.8 When several channels fire on one event

A skipped step k → i can raise trans_t and the emission p-value on the same tick. The emission
surprise is caused by the transition. Each channel currently raises its own alarm; metric G merges
alarms within one run of constant (verb, noun), so this does not inflate the false-alarm count.
Naming one cause per event (transition before duration before stall before emission) matters only
for narration, which Live does not have yet.

---

## 4. Lag

`LagHSMMDetector(hsmm, bad, op, lag=L)` is Live with every **event** channel conditioned on L more
ticks: the switch posterior and the stall mass are multiplied by the backward message β_t over
o_{t+1..t+L} ([`model.md`](model.md) §5), and the alarm is raised at t + L. **Emission** alarms are
still raised at t (`known_at = t`): they test a forecast, which is final when o_t arrives.

- `step()` keeps the last L + 1 ticks of (b_{t−1}, b_t, e_t) and builds each window's β by
  backward steps from the newest tick.
- `finish()` flushes the last L ticks, with their windows cut at the end of the trial
  (`known_at = T − 1`).
- L = `math.inf` is the full smoother. It is a reference, not a causal detector, and refuses
  `step()`. It is the sum-product posterior over every path. That is not Hindsight, which commits
  to Viterbi's single best path.

**Measured: lag buys nothing on these fits.** On every Breakfast model, the best test-grid
metric-G F1 of L = 0, 5 and ∞ agree within 0.003. On EPIC-Tent the three arms are identical.
With near-delta emission rows, o_t alone already settles which step is running. Lag would matter
for a model whose emission rows overlap.

---

## 5. Hindsight

`HindsightHSMMDetector` reads the whole trial first, then scores each channel along one decode:

1. r̂ = the given dish, or argmax_r P(r) P(o | r) ([`model.md`](model.md) §6);
2. z*, the Viterbi segmentation under r̂ ([`model.md`](model.md) §7);
3. π̃_t under r̂, the one-recipe predictive occupancy.

`traces(trials)` batches all three over chunks of trials padded once (`hsmm/batch.py`), and
`assemble` builds the `HindsightTrace`. `flags(trace, op)` thresholds it, so a sweep reuses one
trace for every α. Every alarm has `known_at = T − 1`.

**What it reads from the future.** Everything conditions on r̂ and z*, which use the whole trial.
The emission channels' π̃ is computed under r̂. Their threshold is indexed by z*_t. The duration
and transition channels read z*'s boundaries. `s_dur_two` sits on a segment's last tick, before
the next tick reveals that the segment ended. Live (§3) was built to remove all of this. Under
metric G it costs no F1 (README, "Results").

### 5.1 Emission channels and the dilution correction

$$
s_{\text{emit}}(t) = -\log \sum_i \tilde\pi_t(i)\,P(v_t, n_t \mid i),
$$

and likewise s_verb, s_noun with the marginals. Scoring against π̃ (before o_t) rather than a
posterior that has seen o_t is what keeps the test honest. Scoring against the smoothed occupancy
would use the future.

The score lives on the mixture's scale; the threshold τ_{z*} is one state's α-quantile. Since
Σ_i π̃(i) P(o | i) ≥ π̃(z*) P(o | z*),

$$
s_{\text{emit}}(t) \le -\log\tilde\pi_t(z^*_t) + \bigl(-\log P(o_t \mid z^*_t)\bigr),
$$

so the mixture inflates surprise above z*'s own view by at most −log π̃_t(z*_t). Adding exactly
that to the threshold,

$$
\tau^{\text{emit}}_t = \tau_{z^*_t} - \log\tilde\pi_t(z^*_t),
$$

means an observation inside z*'s own α-tail can never be flagged by dilution alone.
`HindsightTrace.pi_at_zstar` carries π̃_t(z*_t).

**The bound is tight on the modal tick.** When z* is near-deterministic and o_t is its mode, score
and threshold are the same real number, reached by two floating-point routes that differ in the
last bit. A bare `>` would flag every confidently predicted tick. The emission channels therefore
flag at s > τ + `EMISSION_FLAG_ATOL` (1e-6). Before the tolerance, `s_emit` fired on ~99% of
healthy trials.

**Attribution.** Because emissions are a product, s_noun − s_verb > `ATTRIBUTION_MARGIN` (2 nats)
reads as "wrong object" (`"item"`), the reverse as "wrong action", otherwise `"none"`.

### 5.2 Stall (`s_temporal`)

$$
s_{\text{temporal}}(t) = -\log P\bigl(D \ge \text{elapsed}_t \mid z^*_t, \hat r\bigr),
$$

non-decreasing within a segment, flagged above −log α. It is a tail probability, so one global
threshold is per-state calibrated for free: it fires when elapsed time passes that state's α
upper quantile, whether the state usually takes 3 ticks or 90. The shipped per-channel default
uses α = 1e-5 for this channel (README, "Results").

### 5.3 Two-sided duration (`s_dur_two`)

When a segment closes at length d,

$$
s_{\text{dur2}} = -\log \min\bigl\{1,\ 2\min\bigl(P(D \ge d),\ P(D \le d)\bigr)\bigr\},
$$

placed on the segment's last tick, with attribution `"stuck"` if the upper tail is the smaller
one, else `"left_early"`. The short side exists because a survival ramp cannot see a step that
ended early: a short duration always has high survival. It is the channel that carries
abandonment.

**The trial's last segment is never scored.** It is censored, so P(D ≤ d) asks a question the data
cannot answer. When it was scored, it fired "idle only took 8 ticks — you usually spend about 186"
at the end of healthy trials.

### 5.4 Transition (`s_transition`)

s_trans = −log A^(r̂)[z_{m−1}, z_m] at segment m's first tick, flagged above the from-state's
α-threshold. The threshold is indexed by the **from** side, whose row defines the null.
`expected_next_state` records argmax_i A^(r̂)[z_{m−1}, i] for narration.

### 5.5 Recipe transition (`s_recipe_transition`)

$$
s_{\text{recipe}} = \log \bar A_{ki} - \log A^{(\hat r)}_{ki}, \qquad \bar A = \textstyle\sum_r \pi_r A^{(r)},
$$

a signed excess, not a surprisal: "ordinary in general, wrong for this recipe". It can be
negative, and so can its threshold, so the threshold is +inf on every tick that is not a segment
start. At K_R = 1 it is zero with +inf thresholds (§2.3).

### 5.6 Visit count (`s_count`, optional)

The visit-count statistics are fitted once, after training (`scripts/train/fit_visit_counts.py`).
The final model decodes its own training sessions, and for every recipe r and subtask k the
fraction of sessions with C_k ≥ c visits, smoothed as (n + 0.5)/(N_r + 1), is stored, along with
k's most frequent successor. Idle subtasks (most likely verb `stall`) are not counted. On the
decoded path z*:

$$
s_{\text{count}} = -\log \hat P(C_k \ge c \mid \hat r) \text{ at the start of the } c\text{-th visit of } k,
\qquad -\log \hat P(C_k = 0 \mid \hat r) \text{ for a } k \text{ never visited,}
$$

the second at the start of k's usual successor, else at the last segment. The threshold is −log α,
like the duration channels (the p-values are frequencies, not model tails). The smallest
reachable p is 0.5/(N_r + 1), so a recipe with few training sessions cannot fire. In the
validation grid the channel sits with the transition channels (`harness.TRANSITION_CHANNELS`).

---

## 6. Narration

`narrate(detection, detector, vocab)` turns a Hindsight detection into templated questions. Every
sentence is a direct read of a fitted quantity (an argmax over an emission or transition row, an NB
mean), so each question can be traced to the parameter behind it. A Live or Lag detection raises
`NotImplementedError`: narration needs a segmentation, and live narration is future work (README,
"Next").

It takes the detector rather than the HSMM so that a question's severity is measured against the
same threshold that gated its flag.

`Query(tick, segment_index, channel, kind, severity, ratio, text)`. At most one question per
segment and kind:

| Renderer | Channel | Kind | Example |
|---|---|---|---|
| `_item_query` | `s_noun`, attribution item | `item_substitution` | "Wait --, that's pour mustard -- based on what I'd seen up to then, I expected pour jelly." |
| `_action_query` | `s_verb`, attribution action | `wrong_action` | "...that looks like 'pour knife' -- I expected 'take knife'." |
| `_stall_query` | `s_temporal` | `stall` | "are you stuck on pour milk? You've been on it for 40 ticks; it usually takes about 12." |
| `_duration_query` | `s_dur_two` | `left_early` / `retro_stall` | "pour milk only took 2 ticks ... Did you skip part of it?" |
| `_order_query` | `s_transition` | `missing_step` / `out_of_order` | "did you skip take bowl?" / "after A you normally do B, not C." |

Rules worth knowing:

- **Severity** is s / threshold: low < 1.5 ≤ medium < 3 ≤ high, rendered as the hedge "I think",
  "I noticed", "Wait --". A non-positive threshold reads as high. A near-deterministic transition
  row can have a tiny positive threshold, and then the ratio is huge (~1e13 was seen); treat the
  ratio as an ordering, not a magnitude.
- **A stall question is asked at the first tick past the threshold**, not at the peak, because that
  is when a live system would speak.
- **"I expected X"** conditions on the token that was *not* flagged: x̂ = argmax_x Σ_i π̃(i)
  P(held | i) P(x | i) (`expected_given`). The naive argmax of z*'s own row can contradict the
  causal π̃ at a boundary ("that's cup, normally you use cup"), and marginalising only the flagged
  word can produce nonsense pairs ("pour kitchen"). When both words are flagged, the best pair is
  picked jointly (`expected_pair`).
- **Missing step.** An odd transition a → c is first explained as a skipped b* = argmax_b log A_ab
  + log A_bc, if that two-hop path beats the direct jump by ≥ `MIN_BRIDGE_GAIN` = 2 nats
  (`missing_step`).
- **The lexicon is r̂'s.** `Lexicon` is built from `select_recipe(params, r̂)`, so a quoted
  expected duration is the one the surprise was computed against. A subtask is named by its modal
  (verb, noun); the SIL pair renders as "idle".
- **`s_recipe_transition` is never narrated.** Latent recipe clusters have no names.

---

## 7. What the detectors can and cannot see

These are properties of the representation, measured on the current models. The full tables are in
README ("Results") and [`eval.md`](eval.md).

- **Live costs no F1.** Live matches Hindsight within ±0.01 metric-G F1 on every Breakfast model
  and on EPIC-Tent, with fewer or similar alarms per 10 minutes.
- **Substitution is the easy type** (recall 0.97–0.99 for every HSMM arm). Omission and repetition
  are the hard ones (0.29–0.44).
- **The decode hides some errors.** Viterbi re-explains ~70% of injected omissions as ordinary
  paths, and recipe re-inference flips r̂ on ~59% of transpositions.
- **On EPIC-Tent the emission channels never fire.** Most annotated errors leave the (verb, noun)
  stream unchanged (94% of `misuse` spans), and the state space is exact, so emissions are
  near-deterministic. The transition channel is the earliest channel on almost every caught order
  error. See [`data.md`](data.md) §3.4.

---

## 8. Code

| Concept | Where |
|---|---|
| Trial, Span, Alarm, Detection, OperatingPoint | `core.py` |
| the detector base class | `core.py:AnomalyDetector` |
| flags → alarms | `core.py:alarms_from_flags` |
| exact discrete tail | `detectors/thresholds.py:tail_threshold` |
| per-row thresholds | `thresholds.py:categorical_quantile_threshold`, `joint_quantile_threshold`, `transition_quantile_threshold`, `excess_quantile_threshold` |
| the one-recipe rule | `thresholds.py:recipe_excess_thresholds`; `hsmm_hindsight.py:HindsightHSMMDetector.recipe_transition_surprise` |
| bad sets, bins, α grid | `thresholds.py:BadSets` (`stall_bin`, `early_bin`, `trans_bin`, `recipe_bin`, `alpha_index`, `hindsight`) |
| Live detector | `detectors/hsmm_live.py:LiveHSMMDetector` (`step`, `trace`, `flags`, `detect_trial`) |
| timing-split p-values | `hsmm_live.py:forecast_pvalues`, `split_pvalues`, `LiveHSMMDetector.emission_pvalues` |
| switch posterior pieces | `hsmm_live.py:switch_terms` → `Switch` |
| event channels | `hsmm_live.py:stall_prob`, `transition_prob`, `early_exit_prob`, `recipe_transition_prob`, `events_one_tick`, `event_table` |
| per-tick scores of a trial | `hsmm_live.py:LiveTrace` |
| Lag | `detectors/hsmm_lag.py:LagHSMMDetector` (`beta`, `_window_beta`, `finish`) |
| Hindsight | `detectors/hsmm_hindsight.py:HindsightHSMMDetector` (`traces`, `assemble`, `thresholds`, `flags`, `detect_many`) |
| Hindsight channels | `HindsightHSMMDetector.emission_surprise`, `stall_surprise`, `duration_surprise`, `transition_surprise`, `recipe_transition_surprise` |
| Hindsight trace | `hsmm_hindsight.py:HindsightTrace` |
| visit counts (`s_count`, optional) | `hsmm/visit_counts.py:VisitCounts` (`fit`, `path_pvalues`, `p_at_least`, `p_zero`); `HindsightHSMMDetector(counts=)`, `count_surprise`; `LiveHSMMDetector(counts=)`, `hsmm_live.py:entry_one_tick`, `entry_table`, `CountState`; `scripts/train/fit_visit_counts.py` |
| narration | `detectors/narrate.py:narrate`, `Query`, `Lexicon`, `severity`, `expected_given`, `expected_pair`, `missing_step` |
| tests | `tests/test_detector_contract.py`, `test_live.py` (R2, exactness), `test_hindsight.py`, `test_thresholds.py`, `test_narrate.py`, `test_count_live.py` (s_count) |
