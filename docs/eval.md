# `src/cook_ad/eval/` — batched inference and detection metrics

| File | Role |
|---|---|
| `batch.py` | many-trial analogues of `surprise.compute_trace{,_joint}` |
| `metrics.py` | recall / precision / latency / channel attribution, plus the persistence rule |
| `element_metrics.py` | the **step**-level layer both the HSMM and the LLM are scored through |
| `counterfactual.py` | pairs each degraded trial against its own healthy counterfactual |
| `plotting.py` | two matplotlib figures per evaluation run |

`metrics.py` (ticks) is driven by `run_evaluation.py`; `element_metrics.py` (steps) by
`run_llm_eval.py` and documented in [`llm.md`](llm.md) §5, since the step unit exists to make the
LLM baseline comparable. The three layers measure the same detector at three granularities and are
kept separate rather than reconciled — a tick, a step and a trial are genuinely different questions.

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

### Cost

None beyond what is already computed. Every evaluation runs healthy trials through the detector
anyway (they are the false-positive pool), so pairing is post-hoc arithmetic over flags already in
hand — no extra inference, and for the LLM arm no extra requests.

---

## 6. `run_threshold_sweep.py` — picking $\alpha$

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
