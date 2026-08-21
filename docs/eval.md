# `src/cook_ad/eval/` — batched inference and detection metrics

| File | Role |
|---|---|
| `batch.py` | many-trial analogues of `surprise.compute_trace{,_joint}` |
| `metrics.py` | recall / precision / latency / channel attribution at tick level |
| `element_metrics.py` | the **step**-level layer both the HSMM and the LLM are scored through |
| `counterfactual.py` | pairs each degraded trial against its own healthy counterfactual |
| `plotting.py` | two matplotlib figures per evaluation run |

`metrics.py` (ticks) is driven by `run_evaluation.py`; `element_metrics.py` (steps) by
`run_llm_eval.py` and documented in [`llm.md`](llm.md) §5, since the step unit exists to make the
LLM baseline comparable. The three layers measure the same detector at three granularities and are
kept separate rather than reconciled — a tick, a step and a trial are genuinely different questions.
`run_counterfactual.py` (§5) and `run_threshold_sweep.py` (§6) are the two analysis runners built
on top of them; `run_detect_eval.py` (§7) is the scorecard model selection runs through, and §8
lists the single-question diagnostics that localise a loss once the scorecard shows one.

---

## 1. `batch.py` — why batching is not just an optimisation

`surprise.compute_trace` handles one trial with `mask` all-`True`. Calling it in a loop over 500
trials **recompiles the JAX kernels for every distinct sequence length**, because $T$ is a static
shape. That dominates runtime completely.

`compute_traces` fixes this by padding the whole dataset to a single global
$T_{\max} = \max_i T_i$, so the three expensive ops compile **once**:

1. `_batched_pi` — `vmap`ped `predictive_occupancy`,
2. `segmentize.segment_all` — `vmap`ped Viterbi,
3. `_batched_recipe_gamma` — `vmap`ped recipe forward-backward.

The cheap per-trial numpy assembly then runs through `surprise.assemble_trace`, sliced back to each
trial's true length. Chunking (`chunk_size=16`) bounds peak memory; padding to the *global*
$T_{\max}$ rather than a per-chunk one is what keeps the compilation count at exactly one.

### The joint variant

`compute_traces_joint` differs in three ways that follow from the joint model's structure:

- **Recipe inference is a direct EM readout**, `joint_em.infer_recipe`, not a second
  forward-backward over segment symbols. It runs **once for the whole dataset up front** — it is
  cheap (forward pass only) and every trial's $\hat r_i$ is needed *before* that trial's
  recipe-conditioned segmentation can run.
- **Segmentation and occupancy are both recipe-conditioned**: `segment_all_conditioned` and
  `_batched_pi_conditioned` take per-trial tables gathered at $\hat r_i$ along the $K_R$ axis
  (`in_axes=(0,0,0,0,0,0,None)`), rather than one shared set broadcast across the batch.
- It returns `(traces, log_probs, r_hat, log_trans_marginal)` — all four are what
  `surprise.flag_joint` needs per trial.

Both functions return the tables they built internally for exactly this reason: the flagging stage
must score against the *same* tables the trace was built from, and handing them back removes the
opportunity to rebuild them inconsistently.

---

## 2. `metrics.py` — measuring a detector

### The evaluation design

Two populations:

- **healthy** trials with no injected anomaly — any flag is a false positive;
- **degraded** trials, one injected error each, carrying a ground-truth window $[t_0, t_1]$ from
  `synthetic/error_injection.py`.

A detection counts if any selected channel fires inside

$$
[\,t_0,\ t_1 + \texttt{latency\_tol}\,], \qquad \texttt{latency\_tol} = 5\ \text{ticks},
$$

and the latency is

$$
\text{lat} = \min\{\, t \in \text{flagged} : t_0 \le t \le t_1 + \texttt{latency\_tol} \,\} \;-\; t_0 .
$$

The tolerance exists because some channels are structurally retrospective — an abandonment is only
visible once the segment *closes*, so its flag lands after $t_1$ by construction.

### Every flag counts, everywhere — no persistence rule

A single flagged tick is a detection inside the window and a false positive outside it. There is
no run-length requirement anywhere in the pipeline, and adding one is the wrong instrument for
two reasons.

**It filters by error type, not by noise.** The structural channels' true signal is one tick wide
by construction — $s_{\text{trans}}$ fires at a segment's first tick and nowhere else — so any
run-length bar deletes their detections outright while sparing the duration channels, whose
signal is inherently extended. That is a selective mute on three of the seven channels, not a
uniform noise filter.

**It hides miscalibration rather than fixing it.** A channel that is not being gated by its
threshold produces the same symptom as genuine per-tick noise — many isolated flags — and a
persistence rule suppresses both indistinguishably. The two are separable at the threshold
instead: a calibrated channel's false-positive rate falls as $\alpha$ tightens, an ungated one is
flat in $\alpha$. `run_threshold_sweep.py` (§6) makes exactly that measurement per channel and is
the tool to reach for. `surprise.EMISSION_FLAG_ATOL` ([`anomaly.md`](anomaly.md) §3.4) exists
because three channels were ungated in precisely this sense.

For scale: treating ticks as independent at $\alpha = 0.05$ over a ~150-tick trial bounds the
any-flag trial rate at $1 - 0.95^{150} \approx 0.9994$. That bound is loose — ticks are not
independent, since $s_{\text{temporal}}$ is monotone within a segment and the transition channels
fire only at boundaries — and a correctly calibrated detector sits far below it. False alarms are
controlled by moving $\alpha$, which trades against recall along a measurable curve, rather than by
filtering the output, which trades against recall invisibly.

### The reported numbers

For each error type, with $\text{TP}$ = degraded trials detected in-window,
$\text{FP}_{\text{out}}$ = degraded trials flagged outside their window,
$\text{FP}_{\text{healthy}}$ = healthy trials flagged anywhere:

$$
\text{recall} = \frac{\text{TP}}{n}, \qquad
\text{precision} = \frac{\text{TP}}{\text{TP} + \text{FP}_{\text{out}} + \text{FP}_{\text{healthy}}},
$$

$$
\text{precision}_{\text{excl. healthy}} = \frac{\text{TP}}{\text{TP} + \text{FP}_{\text{out}}} .
$$

The second precision exists because $\text{FP}_{\text{healthy}}$ is the **same constant** pooled
into every error type's denominator — it partly *guarantees* the "constant floor across error
types" pattern by construction. Reporting both separates the type-specific false-alarm component
from the shared one. This is a deliberate anti-self-deception measure and worth preserving in any
rewrite.

### Channel attribution

$$
\mathrm{Attr}[\text{channel } c,\ \text{error type } e] \;=\;
\frac{\#\{\text{detected trials of type } e \text{ where } c \text{ fired in-window}\}}{\text{TP}_e}
$$

This matrix is the **isolation claim**: substitution should light up $s_{\text{noun}}$, abandonment
should light up the retrospective duration channel, omission/transposition should light up
$s_{\text{trans}}$. Rows are not mutually exclusive — several channels can fire on the same trial —
so columns need not sum to 1.

`detect(flags, channels)` takes a channel subset, which is the (currently unused) **ablation
knob**: rerun `evaluate` with one channel removed to measure its marginal contribution.

### `kl_sanity`

Explicitly **not** a detection metric. It pools noun-token histograms across the healthy and
degraded sets and reports their KL, answering only "did the perturbations move the distribution at
all". A near-zero value means the injection did nothing, i.e. the harness is broken. It reuses
`lifecycle.divergence.categorical_kl` rather than reimplementing.

---

## 3. `plotting.py`

Agg backend (headless), dpi 150, two figures per healthy source, tagged `synthetic` or `real`:

- **`detection_{tag}.png`** — grouped recall/precision bars per error type, with mean latency
  annotated above each group (`"n/a"` when nothing was detected, so `NaN` never renders as a
  number).
- **`attribution_{tag}.png`** — the channel × error-type matrix as a viridis heatmap with values
  printed in-cell (colour flipped at 0.6 for contrast) and the colourbar fixed to $[0,1]$ so
  figures from different runs are directly comparable.

`save_figures` writes both and returns their paths.

---

## 4. Two healthy sources, on purpose

`run_evaluation.py` evaluates against **both**:

| Source | Ground truth | Trade-off |
|---|---|---|
| **synthetic** — `generate_healthy`, ancestral samples from the fitted model | exact (segments are known because they were sampled) | measures the detector under a *perfectly specified* model; flatters it |
| **real** — held-out Breakfast trials via `trajectory_from_real` | Viterbi-decoded, hence fuzzier boundaries | includes real model misfit; harsher and more honest |

The synthetic source isolates detector behaviour from model misfit; the real source measures the
combination. Reporting only the first would be self-congratulatory; reporting only the second
would confound two failure modes.

For the joint model, `error_injection` is handed
`joint_params.collapse_to_marginal(joint_params)` rather than a per-trial conditioned model —
injecting an error doesn't need to know a trial's recipe, only what is typical for that subtask in
general.

Trials with fewer than `error_injection.MIN_SEGMENTS = 4` segments are filtered out before
evaluation: an out-of-order step is only out-of-order *relative to context*, so there must be
enough surrounding structure for the injection to constitute a genuine anomaly.

---

## 5. `counterfactual.py` — attributing flags instead of windowing them

A ground-truth window has a crisp start and end. An injected anomaly does not. One transposition
legitimately disturbs three boundaries — entering the wrong step, the junction, exiting into what
follows — plus a belief-state perturbation that decays over the segments after it. Any fixed window
covers some prefix of that and charges the rest as a false positive, even though the injection
caused it. Widening the window only moves the cutoff; it does not remove the need for one.

This layer removes the need entirely, by asking a question that has no window in it: **did the
injection change what the detector said?**

Each degraded trial is paired with the *same* trial with no injection, scored through the *same*
detector. `tick_map` ([`synthetic.md`](synthetic.md) §2.1) makes the two comparable by projecting
the healthy flags into the degraded trial's tick space:

$$
\texttt{projected}[c][i] \;=\; \texttt{healthy}[c]\bigl[\texttt{tick\_map}[i]\bigr],
\qquad
\texttt{attributable}[c][i] \;=\; \texttt{degraded}[c][i] \;\wedge\; \neg\,\texttt{projected}[c][i].
$$

A flag present in **both** is that detector's baseline noise on that trial and is discarded. A flag
present **only** in the degraded run is injection-attributable, wherever in the trial it lands.
Downstream disturbance is then credited rather than punished, by construction rather than by
tolerance.

Repetition's `tick_map` is many-to-one; the projection handles it by construction, since fancy
indexing simply reads the source flag once per copy.

### What is reported, and why it is three numbers

| Metric | Question | Window-dependent? |
|---|---|---|
| **detection rate** | did the injection change the output at all — $\ge 1$ attributable flag anywhere? | no |
| **localisation rate** | was the **earliest** attributable flag inside $[t_0, t_1 + \texttt{tol}]$? | yes |
| **false-alarm rate** | healthy trials only — identical to §2 | n/a |

Detection and localisation are separated because only the second can move with window width, and
collapsing them into one "recall" would let a window choice silently drive a number that reads as a
capability. False alarms are measured only on healthy trials, where there is no injection and
therefore nothing to attribute — which is the cleanest form of the question.

By construction $\text{detection} \ge \text{localisation}$, and detection is $\ge$ the in-window
recall of §2 on the same trials, since it credits strictly more flags.

### The matched null — what this is actually for

Beyond crediting downstream disturbance, the pairing supplies the null that a raw recall number
lacks: **the same detector, the same trial, the same range, with no injection present.** Ask
whether a *projected healthy* flag lands in the injection-touched range and you have the rate the
detector would score by being chatty in the right neighbourhood anyway. `run_counterfactual.py`
reports the three side by side per error type:

| | question |
|---|---|
| `observed` | did a **degraded** flag land in the range (what recall normally reports) |
| `chance` | did a **projected healthy** flag land there (the matched null) |
| `attributable` | was there a flag in the degraded run and *not* its counterfactual |

The null is matched per trial, so it absorbs trial length, step count, range width, and the
detector's own per-trial noise level — all things a uniform-random baseline gets wrong. Measured
on 419 real trials it sits at **0.017–0.055**, roughly 5x below what assuming uniform flag
placement predicts, because flags cluster at segment boundaries instead of spreading out. Any
"is this above chance?" argument built on a uniform assumption will therefore be far too
pessimistic; measure the matched null instead.

Because the two outcomes are paired, significance is **McNemar** on the discordant trials, not an
unpaired two-proportion test — which would both ignore the pairing and degenerate when the null
rate is zero.

### Cost

None beyond what is already computed. Every evaluation runs healthy trials through the detector
anyway (they are the false-positive pool), so pairing is post-hoc arithmetic over flags already in
hand — no extra inference, and for the LLM arm no extra requests.

---

## 6. `run_threshold_sweep.py` — picking $\alpha$

$\alpha$ is the detector's **only** sensitivity knob: there is no persistence rule, run-length
filter, or post-hoc smoothing anywhere downstream (§2), so this one number sets the whole
operating point. The shipped value is `surprise.DEFAULT_ALPHA` $= 5\times10^{-3}$, chosen from
this sweep — trial-located recall 0.453 at precision 0.728 with 14.1% of healthy trials raising
any alarm. Loosening to 0.05 buys 16 points of recall for roughly triple the false-alarm rate;
tightening to $10^{-3}$ returns 4 points of false alarms for 5 points of recall.

Reports accuracy, precision, recall and false-positive rate across an $\alpha$ grid spanning
$5\times10^{-1}$ to $10^{-10}$, at four granularities.

The sweep is cheap because of where $\alpha$ enters. Traces — predictive occupancy and Viterbi
segmentation, the entire JAX cost — do not depend on it, so they are computed **once** per group
(healthy plus each of the five error types) and the whole grid is swept by re-running
`surprise.flag_joint`, which is pure post-hoc thresholding over arrays already in memory.

### The four granularities, and which one to read

| Level | One test is | Positive means |
|---|---|---|
| `tick` | one tick | that tick is inside the injection-touched extent |
| `step` | one `textify.Step` | that step is touched |
| `trial` | one trial | the detector flagged **anywhere** |
| **`trial_loc`** | one trial | the detector flagged **in the right place** |

**`trial_loc` is the one to quote.** For a degraded trial the positive range is the whole
injection-touched extent — ground-truth window $\cup$ debris:

- a flag **inside** the range → TP; no in-range flag → FN;
- a flag **outside** the range → FP, counted *independently* of whether the trial was also hit;
- healthy trials have no range, so any flag → FP, none → TN.

Scoring the stray independently is the load-bearing choice. Collapsing it into the hit would let a
detector that finds the anomaly and *also* fires five times elsewhere score identically to one
that fires once, correctly — which is precisely the behaviour that matters for an assistant whose
failure mode is nagging. $\text{TP} + \text{FN}$ is exactly the degraded-trial count, so recall
stays clean; FP pools strays on degraded trials with flags on healthy ones.

`trial` is kept only as the degenerate comparison. It scores "flagged the wrong place" identically
to "flagged the right place", so it reads far more favourably than `trial_loc` and is not a
detection metric on its own.

### Two ground-truth conventions, on purpose

This sweep **counts injection debris as anomalous**; `element_metrics` ([`llm.md`](llm.md) §5)
**excludes** it. Both are correct for their own question. Debris is genuinely something the
injection moved, so flagging it is detecting the injection's effect — the question this sweep
asks. It is not the anomaly the injector was specifically testing for — the question
`element_metrics` asks. Unifying them would produce one convention that is wrong for one of the
two, so they are stated per consumer instead.

### Reading the curves

A correctly calibrated channel's false-positive rate falls toward zero as $\alpha$ tightens; a
channel whose FPR is flat across orders of magnitude of $\alpha$ is not being gated by its
threshold at all, which localises the problem to that channel rather than to the calibration.
Accuracy is base-rate dominated at every level here (the pool is 1 healthy : 5 degraded, and
positives are sparse within a trial), so read precision/recall/FPR and treat accuracy as a
sanity check against its own majority-class baseline.

---

## 7. `run_detect_eval.py` — the scorecard model selection runs through

Same `trial_loc` metric and ground-truth convention as §6, so numbers are directly comparable.
What it adds: the per-error-type and per-channel breakdown in one pass, a healthy pool built with
the batched `generate.trajectories_from_real_joint`, and `quantile.JointThresholdCache`. Those
take a 402-trial sweep from most of an hour to a couple of minutes, which is what makes detection
usable as a *selection* criterion rather than only a final report ([`hsmm.md`](hsmm.md) §7).

### Read accuracy, not F1

The pool is 1 healthy : 5 degraded and a stray is charged independently of a hit, so **flagging
every trial** scores precision $5/11$, recall 1, $F_1 = 0.625$, accuracy $5/11 = 0.455$; flagging
nothing scores accuracy $1/6$. An $F_1$ near 0.62 is what you get for free. Accuracy is the number
that separates a detector, and since

$$
\text{acc} \;=\; \frac{5R + (1 - h)}{6 + 5s}
$$

($R$ recall, $h$ healthy false-alarm rate, $s$ stray rate on degraded trials), **recall is worth
roughly five times $h$** — the opposite of the weighting a precision-focused sweep encourages.

### Two ways the comparison can flatter itself

**Who placed the injections.** `error_injection` perturbs whatever segmentation the *scoring
model itself* decoded, so models that differ in how they segment are graded on different degraded
streams. `--traj-params` points every model at one common source; report both directions.

**Which segments were hit.** The injectors draw a segment per trial, worth a few points on a
single run. Quote a mean over several `--seed` values, not the best one.

`report_final.py` enforces the rest: pick $\alpha$ on **train**, report at that same $\alpha$ on
**test**. Picking per split reports a number no deployment could reproduce.

### `trial_loc` caps false positives at one per trial — cross-check at step level

The stray is charged as a boolean: `fp += bool((~pos & mask).any())`. A trial that flags once
outside the range and one that flags five times cost the same single false positive, so total FP
is bounded by the trial count — not by anything a nagging assistant is bounded by. Charging the
stray independently of the hit (§6) separates "found it and also fired elsewhere" from "found it
cleanly"; it does not separate "once" from "five times".

`tools_alarm_load.py` measures the gap, counting contiguous runs of the unioned flag mask. On the
test split the ratio of alarms raised to false positives charged is ~1.1–1.4× for the HSMM arms
and **3.2× for the LLM arm**, so the cap flatters the chattier detector.

That can invert a comparison, so quote a `trial_loc` gain only alongside a step-level check
(`run_step_sweep.py`, the step-layer analogue of `run_threshold_sweep.py`). Neither granularity
is wrong — "did it bother the user about the right trial" versus "how often did it bother the
user" — but for an assistant whose failure mode is nagging, select on the step layer.

### Selecting regularisation out of sample

`smooth_params.py`'s `--strength` and `--backoff-tau` are regularisation, so choosing them on the
split the model was fit to chooses too little. On the train split the pooled backoff looks best
at $\tau = 2$; on a nested held-out fold (`make_dev_split.py`, 322 fit / 80 dev out of the 402
outer-train trials) $\tau = 2$ is the worst of the grid and $\tau = 30$ the best. The shift
strength is unaffected — $s = 0.7$ wins on both, $s \ge 1.5$ loses on both.

The symptom of getting it wrong is a train/test gap in flagged healthy steps, which is the
cheapest thing to watch:

| flagged healthy steps, at matched recall | train | test |
|---|---|---|
| cascade fit | 3.2% (85/2692) | 3.4% (22/649) |
| lexical+hard-EM, $\tau = 2$ (in-sample pick) | 0.8% (22/2692) | 4.3% (28/649) |
| lexical+hard-EM, $\tau = 30$ (dev pick) | — | 2.9% (19/649) |

A 4× gap is the transition table acting as a lookup of the training bigrams: a legal transition
that merely did not occur in those 402 trials costs ~32 nats out of sample, and backing off to
the pooled row is what covers it. With $\tau = 30$ the fit beats the cascade at both layers on
test — step precision 0.691 vs 0.626 at matched recall 0.511 with 19 vs 27 flagged healthy steps;
`trial_loc` accuracy 0.509 vs 0.453 at equal recall, healthy false-alarm rate 0.196 vs 0.351.

### The healthy false-alarm floor $\alpha$ cannot reach

Per channel, both fits show the pathology §6 names — a false-positive rate flat across orders of
magnitude of $\alpha$. `s_transition` and `s_recipe_transition` are unmoved from $5\times10^{-3}$
to $10^{-4}$ in both, so no threshold choice removes them.
`run_threshold_sweep_coordinate.py` is the instrument for spending the budget per channel rather
than through one shared $\alpha$.

---

## 8. Single-question diagnostics — `tools_*.py`

Each answers one question about a fitted model with no thresholds or sweeps in it, so a loss the
scorecard shows can be localised to a stage instead of guessed at. All are read-only.

| Script | Question |
|---|---|
| `tools_state_organisation.py` | do the fitted states correspond one-to-one with the observed $(v,n)$ inventory, and does the decode agree with the run structure? (purity, boundary F1, split pairs) |
| `tools_transition_ceiling.py` | of the junctions an injection creates, how many does the training data already contain — in this trial's own recipe, elsewhere, or nowhere? The "nowhere" share is what the transition channel *can* flag |
| `tools_launder.py` | of those flaggable junctions, how many survive the Viterbi decode, and how many then clear their threshold? Separates a decode loss from a calibration loss |
| `tools_oracle_recipe.py` | how much recall is lost to the MAP recipe being re-inferred from the degraded stream, by re-scoring with $\hat r$ pinned to the healthy decode's value |
| `tools_duration_power.py` | re-scores every healthy segment at 1 tick (abandonment) and $2\times$ (repetition) against its own fitted NB — what $s_{\text{dur two}}$ can deliver at the current duration spread. It predicts **that one channel**, not the error type: $s_{\text{temporal}}$ is not modelled here and carries most of repetition |
| `tools_alarm_load.py` | how many alarms are actually raised behind each false positive `trial_loc` charges, and what fraction of all alarms land in range |

`run_step_sweep.py` (the step-layer analogue of `run_threshold_sweep.py`) and `make_dev_split.py`
(a nested fit/dev split for selecting regularisation without touching the outer test split) are
the two runners the §7 protocol needs.

`tools_launder.py` and `tools_oracle_recipe.py` are the two that most often move a decision:
the first says whether a channel is failing to fire or never being shown the evidence, and the
second sizes the recipe-assignment loss, which no threshold change can recover.
