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

Same `trial_loc` metric and the same ground-truth convention as §6 (positive range = injection
window $\cup$ debris, strays charged independently of hits), so numbers from the two are directly
comparable. What it adds:

- the **per-error-type and per-channel** breakdown in one pass, so a change can be attributed to
  the channel it moved rather than only to the union;
- a healthy pool built with `synthetic.generate.trajectories_from_real_joint`, which pads the
  whole split to one global $T_{\max}$ instead of compiling the recipe-inference and Viterbi
  kernels once per distinct trial length;
- `quantile.JointThresholdCache`, since a sweep over $A$ alphas × $N$ trials × 6 source groups
  contains only $A$ distinct emission tables and $A \times K_R$ recipe-conditioned ones.

Together those take a full 402-trial sweep from most of an hour to a couple of minutes, which is
what makes detection usable as a selection criterion rather than a final report
([`hsmm.md`](hsmm.md) §7).

### Read accuracy, not F1

The pool is 1 healthy : 5 degraded and a stray flag is charged independently of a hit, so
**flagging every trial** scores

$$
\text{precision} = \tfrac{5}{11},\quad \text{recall} = 1,\quad F_1 = 0.625,\quad
\text{accuracy} = \tfrac{5}{11} = 0.455 ,
$$

and flagging nothing scores accuracy $1/6$. An $F_1$ near 0.62 is therefore not evidence of a
working detector — it is what you get for free. Accuracy is the number that separates one, and

$$
\text{acc} \;=\; \frac{5R + (1 - h)}{6 + 5s}
$$

with $R$ = recall, $h$ = healthy false-alarm rate, $s$ = stray rate on degraded trials. The
partials are $\partial\text{acc}/\partial R \approx 0.68$, $\partial\text{acc}/\partial s
\approx -0.34$, $\partial\text{acc}/\partial h \approx -0.14$: **recall is worth roughly five
times the healthy false-alarm rate**, which is the opposite of the weighting a precision-focused
threshold sweep encourages. `render_detect_compare_png.py` draws both reference lines.

### Two ways the comparison can flatter itself

**Who placed the injections.** `synthetic/error_injection.py` perturbs whatever segmentation the
**scoring model itself** decoded — so two models are graded on two different sets of degraded
streams, and models that differ in how they segment differ in exactly that. `--traj-params`
points every model at one common source. Report both directions; a gap that only exists in one is
not a gap.

**Which segments were hit.** The injectors draw a segment per trial, and that draw is worth a few
points on a single run. Quote a mean over several `--seed` values, not the best one.

`report_final.py` enforces the other half of the discipline: it picks $\alpha$ on **train** by
`trial_loc` accuracy and reports the model at that same $\alpha$ on **test**. Picking $\alpha$
per split would report a number no deployment could reproduce, since $\alpha$ has to be fixed
before the test trials are seen.

### `trial_loc` caps false positives at one per trial — cross-check at step level

The stray is charged as a boolean: `fp += bool((~pos & mask).any())`. A trial that flags once
outside the range and one that flags five times cost the same single false positive, and a
healthy trial that flags twenty times also costs one. Total FP is bounded by the trial count —
which is exactly the quantity a nagging assistant is *not* bounded by. Charging the stray
independently of the hit (§6) separates "found it and also fired elsewhere" from "found it
cleanly"; it does not separate "fired elsewhere once" from "fired elsewhere five times".

`tools_alarm_load.py` measures what that collapses, counting contiguous runs of the unioned flag
mask — the closest tick-level proxy for one Query card. On the test split the ratio of alarms
actually raised to false positives charged is roughly **1.1–1.4× for the HSMM and 3.2× for the
LLM arm**, so the cap flatters the chattier detector, and the LLM is much the chattier one.

That cap can invert a comparison. Two HSMM fits whose `trial_loc` accuracy differs by 5 points
in one direction can differ in the other at step level, where every flagged step is counted:

| test split | `trial_loc` acc | step precision | step recall | flagged steps on **healthy** trials |
|---|---|---|---|---|
| cascade fit, $\alpha = 5\!\times\!10^{-3}$ | 0.453 | 0.676 | 0.515 | 22 / 649 |
| lexical+hard-EM fit, $\alpha = 5\!\times\!10^{-3}$ | 0.517 | 0.580 | 0.588 | 35 / 649 |
| lexical+hard-EM fit, $\alpha = 5\!\times\!10^{-4}$ (recall matched) | — | 0.577 | 0.509 | 28 / 649 |

The healthy column settles what the two conventions could otherwise argue about: healthy trials
contain no injection and therefore no debris, so no difference in ground-truth convention
explains it. **At matched recall the second fit raises ~27% more false alarms on normal
behaviour**, and `trial_loc` cannot see it because those extra alarms land on trials already
charged their one false positive.

Neither granularity is wrong; they answer different questions ("did it bother the user about the
right trial" versus "how often did it bother the user"). The rule is to quote a `trial_loc`
improvement only alongside the step-level check, and to select an operating point on the
granularity that matches the failure mode you care about. For an assistant whose failure mode is
nagging, that is the step layer.

### Selecting the operating point on the step layer

Because the two layers can disagree, which one the operating point is chosen on is a real
decision, not a formality. `run_step_sweep.py` is `run_threshold_sweep.py`'s step-layer
counterpart — same trace-once-sweep-cheaply structure, scored through `evaluate_steps`.

Two things had to change to make the lexical+hard-EM fit win at this layer as well:

**Select regularisation out of sample.** `smooth_params.py`'s `--strength` and `--backoff-tau`
are regularisation; their whole job is generalisation, so choosing them on the split the model
was fit to systematically chooses too little. On the train split the pooled backoff looks best at
$\tau = 2$; on a nested held-out fold (`make_dev_split.py`, 322 fit / 80 dev out of the 402
outer-train trials) $\tau = 30$ is clearly better and $\tau = 2$ is the worst of the grid. The
shift strength is unaffected — $s = 0.7$ wins on both, and $s \ge 1.5$ loses on both.

**The symptom of getting it wrong is a train/dev gap in healthy false alarms**, which is the
cheapest thing to monitor:

| flagged healthy steps, at matched recall | train | test |
|---|---|---|
| cascade fit | 3.2% (85/2692) | 3.4% (22/649) |
| lexical+hard-EM, $\tau = 2$ (selected in-sample) | 0.8% (22/2692) | 4.3% (28/649) |
| lexical+hard-EM, $\tau = 30$ (selected on dev) | — | 2.9% (19/649) |

A 4× train/test gap is the transition table behaving as a lookup of the training bigrams: a legal
transition that merely did not occur in those 402 trials costs ~32 nats out of sample. Backing
off to the pooled row is exactly what covers it, and $\tau$ is how much.

With $\tau = 30$ the fit dominates the cascade at both layers on test — step precision 0.691
against 0.626 at matched recall 0.511, with 19 flagged healthy steps against 27; `trial_loc`
accuracy 0.509 against 0.453 at equal recall, with the healthy false-alarm rate roughly halved
(0.196 against 0.351).

### The healthy false-alarm floor $\alpha$ cannot reach

Splitting those healthy steps per channel shows the same pathology in both fits — the one §6
names, a false-positive rate flat across orders of magnitude of $\alpha$:

| flagged healthy steps | $5\!\times\!10^{-3}$ | $5\!\times\!10^{-4}$ | $10^{-4}$ |
|---|---|---|---|
| `s_transition` | 19 → 24 | 19 → 22 | 19 → 22 |
| `s_recipe_transition` | 19 → 19 | 19 → 16 | 19 → 16 |
| `s_temporal` | 4 → 12 | 3 → 8 | 2 → 7 |
| `s_dur_two` | 4 → 12 | 2 → 7 | 2 → 7 |

(cascade fit → lexical+hard-EM fit). The two transition channels are ungated in **both**: their
count is unmoved from $5\times10^{-3}$ to $10^{-4}$, so no threshold choice removes them. The
duration channels *are* gated, and their floor roughly tripled — the cost of
`refit_durations.py` tightening the duration fit, which buys abandonment and repetition recall
and pays for it here. `run_threshold_sweep_coordinate.py` is the instrument for spending that
budget per channel rather than through one shared $\alpha$.

---

## 8. Single-question diagnostics — `tools_*.py`

Each answers one question about a fitted model with no thresholds or sweeps in it, so a loss the
scorecard shows can be localised to a stage instead of guessed at. All are read-only.

| Script | Question |
|---|---|
| `tools_state_organisation.py` | do the fitted states correspond one-to-one with the observed $(v,n)$ inventory, and does the decode agree with the run structure? (purity, boundary F1, split pairs) |
| `tools_transition_sparsity.py` | usage-weighted transition row entropy, and what an unobserved transition actually costs in nats |
| `tools_transition_ceiling.py` | of the junctions an injection creates, how many does the training data already contain — in this trial's own recipe, elsewhere, or nowhere? The "nowhere" share is what the transition channel *can* flag |
| `tools_launder.py` | of those flaggable junctions, how many survive the Viterbi decode, and how many then clear their threshold? Separates a decode loss from a calibration loss |
| `tools_oracle_recipe.py` | how much recall is lost to the MAP recipe being re-inferred from the degraded stream, by re-scoring with $\hat r$ pinned to the healthy decode's value |
| `tools_duration_power.py` | re-scores every healthy segment at 1 tick (abandonment) and $2\times$ (repetition) against its own fitted NB — what $s_{\text{dur two}}$ can deliver at the current duration spread. It predicts **that one channel**, not the error type: $s_{\text{temporal}}$ is not modelled here and carries most of repetition |
| `tools_pace.py` | how much of the duration spread is between-trial (a participant's pace) rather than within-trial |
| `tools_alarm_load.py` | how many alarms are actually raised behind each false positive `trial_loc` charges, and what fraction of all alarms land in range |

`run_step_sweep.py` (the step-layer analogue of `run_threshold_sweep.py`) and `make_dev_split.py`
(a nested fit/dev split for selecting regularisation without touching the outer test split) are
the two runners the §7 protocol needs.

`tools_launder.py` and `tools_oracle_recipe.py` are the two that most often move a decision:
the first says whether a channel is failing to fire or never being shown the evidence, and the
second sizes the recipe-assignment loss, which no threshold change can recover.
