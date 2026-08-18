# `src/cook_ad/llm/` + `eval/element_metrics.py` — the LLM baseline

A comparison detector: render each trial as text, hand it to a language model one step at a time,
and score its answers against the same injected errors the HSMM is scored on.

| File | Role |
|---|---|
| `llm/textify.py` | trajectory → steps → `'pour cereals for 19 seconds'`; tick↔step ground-truth mapping |
| `llm/prompts.py` | the two preprompts (with / without recipe descriptions) |
| `llm/client.py` | OpenAI-compatible chat client: cache, rate limit, budget guard |
| `llm/detect.py` | response grammar + parser, and the two protocol drivers |
| `eval/element_metrics.py` | the step-level metric layer **both** detectors are scored through |

Driven by `run_llm_eval.py`. Nothing here touches `eval/metrics.py`, `run_evaluation.py`, or any
tick-level figure — every number in [`eval.md`](eval.md) is exactly what it was.

> `anomaly/narrate.py` opens by stating that routing the system's user-facing questions through a
> language model would throw away their auditability. That still holds, and this module does not
> contradict it: the LLM here is a **baseline to measure against**, not a component of the
> detector. Nothing in `anomaly/` calls it.

---

## 1. The unit problem, and the step

The HSMM emits a value on all seven channels at **every tick**. The LLM reads a list of steps and
answers **once per step**. Neither can be scored in the other's unit, so both are converted to a
common one.

A **step** is one maximal run of constant $(v_t, n_t)$ — the run-length encoding of the
observation stream. Deliberately *not* the model's latent segmentation: $z^*$ comes from Viterbi,
which the LLM never sees, and scoring two detectors against two different segmentations would not
be a comparison.

Measured before committing to RLE (K=64, full corpus):

| Source | runs/trial (median) | run length (median) | 1-tick runs |
|---|---|---|---|
| real Breakfast trials | 6 | 11 s | — |
| synthetic ancestral samples | 7 | 11 s | 12% |

The fitted emissions are peaked enough that per-tick i.i.d. sampling still yields clean runs, so
both healthy sources render legibly and `synthetic/generate.py` needed no special-casing.

`tick_seconds` is 1.0 in both shipped configs, so a run length in ticks *is* a duration in
seconds. `assert_tick_seconds` makes that dependency explicit rather than letting "seconds"
silently become a lie if the binning ever changes.

### Names come from `Lexicon.verb`/`.noun`, not `.phrase`

`narrate.Lexicon.phrase` collapses the SIL sentinels to `"idle"` or to a bare noun. That is right
for a rendered query card and wrong here: it breaks the fixed `VERB NOUN` template the response
grammar and parser are built on, and an unparseable reply is scored as a parse failure rather than
as a verdict. SIL therefore renders literally as `stall kitchen`, and the preprompt explains the
token instead.

### Window → step, with one rule for all five injectors

`error_injection` gives ground truth in tick space. `gt_steps_for_window` maps
$[t_0, t_1]$ onto the steps it overlaps. Verified against each injector's index arithmetic:

| Injector | What the rendered text shows |
|---|---|
| substitution | the retagged tick splits a run into three; the middle becomes its own 1-second step |
| abandonment | the truncated segment becomes a 1-second step |
| omission | $t_0$ is the new boundary, i.e. the first tick of the following step |
| transposition | the window spans both swapped runs, visibly out of order |
| repetition | the duplicate is adjacent-identical to its original and **merges** into one double-length step |

Repetition's merge is not an approximation introduced here — it is the same behaviour
[`synthetic.md`](synthetic.md) already documents for the tick-level path, where Viterbi merges the
copy into an over-long segment.

### The ground-truth correction, from one rule

`step_covering_tick(source_steps, window[0])` recovers what *should* have happened, for every
injector, with a single lookup. It works because **each injector only rewrites ticks at or after
its window start**, leaving every earlier tick untouched:

| Injector | What the source step at $t_0$ is |
|---|---|
| substitution | the original run whose one tick was retagged |
| abandonment | the same step at its **full** original duration — the duration *is* the correction |
| omission | the deleted step |
| transposition | the step that should have come first of the swapped pair |
| repetition | the step that should have followed instead of the duplicate |

---

## 2. The two preprompts

Both carry the vocabulary verbatim (15 verbs, 36 nouns) so a proposed correction is guaranteed
in-vocabulary and therefore exactly scoreable — a free-text correction would need fuzzy matching,
which would put a tunable knob in the middle of a metric.

The five error-type definitions are lifted from the corresponding `error_injection.inject_*`
docstrings. The model is scored on type accuracy against the injector's ground-truth label, so it
has to be judged against the definition the injector actually implements; describing them any
other way would make `type_confusion` measure prompt drift rather than model capability.

**`with-recipes`** adds a description of each of the 10 recipes, derived from `labels.json`: the
modal collapsed subtask sequence plus each step's median duration, *with* a count of how often
that order actually occurs. That count matters — the model is told which recipes are rigidly
ordered and which are not:

```
cereals: stall_kitchen (~3s), pour_cereals (~17s), pour_milk (~12s), stall_kitchen (~3s)
    (most common order in 15/52 recorded trials; 7 distinct orders observed overall)
salat:   stall_kitchen (~4s), cut_fruit (~25s), take_bowl (~6s), put_fruit2bowl (~5s), ...
    (most common order in 2/52 recorded trials; 50 distinct orders observed overall)
```

Handing it a single canonical order with no such caveat would invite it to flag ordinary
participant variation as `transposition`.

> ### The `labels.json` asymmetry
>
> `labels.json` is validation-only everywhere else in this repo and is never fed to training
> ([`README.md`](README.md), cross-cutting conventions). The `with-recipes` preprompt is built
> from it, so **that arm sees ground-truth task structure the HSMM never had.**
>
> The gap between the two variants is the interesting measurement — how much of the LLM's
> performance is general knowledge about cooking versus knowledge of *these* recipes. The
> `with-recipes` number is **not** a like-for-like comparison against the HSMM, and
> `run_llm_eval.py` prints that caveat directly above the table rather than leaving it to a
> reader's memory.

---

## 3. Protocols

### `incremental` (default) — prefix-only

One request per step. Request *i* shows steps 0…*i* as a numbered list and asks for a verdict on
step *i*. The model sees a growing **prefix** and never the future, which is what makes its step
latency comparable to the HSMM's online channels (predictive occupancy, live stall).

```
system: <preprompt>
user:   1. stall kitchen for 2 seconds
        2. take bowl for 8 seconds
        3. pour cereals for 19 seconds      <- judge this one
```

Cost: `len(steps)` requests per trial, ~7 on this corpus.

**The model's own earlier answers are not fed back**, and that is deliberate. Causality — seeing
only the prefix — is the property the comparison depends on. Conversational self-feedback is a
*separate* property, and an earlier version of this module had it. Removing it bought two things:

1. **No error cascade.** A false alarm at step 2 previously sat in context for steps 3–7 and could
   bias them, so a miss at step 5 could not be distinguished from contamination by an earlier
   mistake. Every step is now an independent test, which is what `element_metrics` already assumed
   when it scored them as one test each.
2. **The sweep became parallelisable.** Every request is now a pure function of `(trial, step
   index)` instead of depending on a previous response. That is a precondition for issuing
   requests concurrently, and for submitting the whole sweep to an async batch endpoint — neither
   is possible when request *i+1* contains response *i*.

**Prompt layout is chosen for prefix caching.** One system turn plus one user turn that only ever
grows by appending a line, so request *i*'s prompt is a strict token prefix of request *i+1*'s. A
server with prefix caching (vLLM's APC, or hosted providers' implicit caching) then reuses almost
all of it, and the ~600-token system prompt is reused across the entire sweep. A single growing
user turn is used rather than repeated user turns because consecutive same-role messages are not
universally accepted.

> Changing the protocol changes the prompts, hence the cache keys. Responses collected under the
> older conversational protocol will not be reused, and results from the two are not directly
> comparable.

### `batch`

One request for the whole trial, one verdict line per step. ~7× cheaper, and **non-causal** — the
model sees every step before judging any of them. Its latency column is not comparable to the
incremental arm or to the HSMM, and the report labels it as such.

(Named before the Gemini/OpenAI *async batch endpoints* were under consideration. This is a
prompt-shape, not an async submission mode; if the latter is ever added it should be called
something else — `async` — rather than overloading this name.)

### The response grammar

```
No Anomaly
<TYPE> Anomaly. Correct move would have been VERB NOUN for NUMBER seconds
```

`parse_response` tries the strict regex, then a lenient rescue that scans for a type name anywhere
in the reply. A lenient hit still records `parse_ok=False`.

**Nothing unparseable is coerced to `No Anomaly`.** Most steps in most trials are normal, so
silently reading garbage as the majority class would inflate a model's apparent specificity for
free. Unparseable replies surface as `parse_failure_rate`; a model that cannot hold the format is
a finding about that model. Same rule in `batch`: a step the model returned no line for is a parse
failure for that step, not an implicit pass.

An out-of-vocabulary correction is dropped (the type is still kept), for the reason in §2.

---

## 4. Request budget — the binding constraint

This evaluation is **request-bound, not token-bound**. At ~7 steps/trial, a 6-condition sweep
(healthy + 5 error types) over 20 trials costs **~880 requests per prompt variant** under
`incremental`, or **120** under `batch`. The prompts are ~2.5–5 KB, so tokens are never the limit;
the per-day request cap always is.

| Provider | Model | Req/min | Req/day |
|---|---|---|---|
| **Gemini API** (default) | `gemini-3.5-flash-lite` | 15 | **500, per model** |
| OpenRouter | any `:free` variant | 20 | **50** (1000 after $10 lifetime credit) |

`DEFAULT_RPM` is pinned to the low end (15) because the token bucket only ever slows requests
down: being wrong low costs wall-clock time, being wrong high costs a 429 storm. Check your own at
[aistudio.google.com/rate-limit](https://aistudio.google.com/rate-limit) and pass `--rpm 30` to
halve sweep duration if your tier allows it.

**The daily cap is the one that bites.** Two back-to-back 10-trial incremental sweeps (~470
requests) exhausted it in practice. Gemini meters RPD *per model*, so `--model` is a genuine way
to keep working the same day — but the model id namespaces the response cache, so switching starts
from empty and **results collected under different models are not comparable**. Finish a variant
before switching. The runner prints cache state at startup so an accidental switch shows up as a
suddenly-empty cache rather than a surprise quota crash.

When a per-day quota does hit, the client raises `QuotaExhausted` immediately rather than burning
retries on something waiting cannot fix, and the runner writes a **partial report** of whatever
arms completed instead of discarding the run.

The default is therefore the Gemini API, not OpenRouter. OpenRouter's daily cap is per-account and
platform-wide — it does **not** depend on which free model you pick, so choosing a smaller model
there buys nothing. A 20-trial single-variant incremental sweep fits inside one Gemini free day
and does not fit inside an uncredited OpenRouter one.

Switch providers with a flag, not a code change — the client is a plain OpenAI-compatible
`/chat/completions` POST:

```bash
python run_llm_eval.py --base-url https://openrouter.ai/api/v1 \
    --api-key-env OPENROUTER_API_KEY --model <something>:free --rpm 20
```

Three further mitigations:

1. **`--dry-run`** prints the exact request count for your actual pool before anything is sent,
   and needs no API key.
2. **The cache.** Keyed on `sha256(base_url + model + messages + temperature)`. Reruns are free,
   and a sweep interrupted by a daily cap resumes where it stopped.
3. **`--protocol batch`** — ~7× fewer requests, at the cost of causal comparability.

`--max-requests` raises `BudgetExceeded` rather than silently truncating: a half-finished sweep
that reports as complete is worse than a crash, and the cache makes restarting cheap.

### The API key

`run_llm_eval.py` loads an untracked `.env` at startup (`--env-file` to point elsewhere). Copy the
tracked `.env.example` and fill it in:

```bash
cp .env.example .env      # then set GEMINI_API_KEY=...   (get one at aistudio.google.com/apikey)
```

`.env` is gitignored; `.env.example` is tracked so the required variables are documented in the
repo. Variables already present in the environment win over the file, so an explicit
`GEMINI_API_KEY=... python run_llm_eval.py` is never overridden by a stale file. A missing file is
not an error — exporting the variable directly is equally valid, and CI has no `.env`. The runner
checks for the key **before** building pools or running the HSMM arm, so a missing key fails in
one line instead of several minutes in.

---

## 5. `eval/element_metrics.py` — scoring both detectors identically

### The HSMM adapter, and why it keeps two masks

`step_verdicts_from_flags` turns per-tick channel flags into one `StepVerdict` per step, carrying
**both**:

- `is_anomaly` — any tick in the step is flagged;
- `persistent` — any tick in the step belongs to a run of $\ge$ `min_run` consecutive flagged ticks.

Keeping both is load-bearing. Collapsing an 11-tick step with a plain OR would re-run, per step,
exactly the multiple-testing arithmetic `metrics._persistent_mask` exists to fix: at
$\alpha = 0.05$ per tick, $1 - 0.95^{11} \approx 0.43$, so a step-level OR with no persistence
requirement would report the HSMM as flagging almost every step of every healthy trial. So
`persistent` drives false-positive determination and `is_anomaly` drives in-window detection —
mirroring the tick layer's asymmetry ([`eval.md`](eval.md) §2) for the same reason. For the LLM
the two are identical: it emits exactly one verdict per step, so there is no sub-step multiple
testing to correct for.

### Channel → predicted type

The LLM names a type in words. To compare, the HSMM has to name one too, from the channels that
fired. Measured in-window firing rates over 20 real trials:

| | `s_emit` | `s_verb` | `s_noun` | `s_temporal` | `s_dur_two` | `s_transition` | `s_recipe_trans` |
|---|---|---|---|---|---|---|---|
| substitution | 1.00 | 0.10 | **1.00** | 0.00 | 0.00 | 0.00 | 0.00 |
| abandonment | 0.40 | 0.05 | 0.25 | 0.15 | **0.60** | 0.15 | 0.05 |
| omission | 0.80 | 0.25 | 0.45 | 0.10 | 0.15 | **0.35** | 0.25 |
| transposition | 0.90 | 0.35 | 0.60 | 0.30 | 0.30 | **0.70** | 0.20 |
| repetition | 0.80 | 0.20 | 0.55 | **0.40** | **0.40** | 0.10 | 0.05 |

Two things follow, and both are decisions of principle that the table confirms rather than
choices tuned to make numbers look good:

**`s_emit` names no type at all.** It fires on 0.40–1.00 of *every* error type, which is by
construction — it is the joint marginal $-\log P(v,n\mid\cdot)$, so it carries no type information
its two component channels do not already carry. A step where only `s_emit` fires is a detection
with **no** predicted type, landing in `type_confusion`'s `"none"` column: the honest reading of
"something is off here but the model cannot say what."

**Priority is by specificity of evidence**, `s_transition` first: an ordering violation *explains*
an odd duration at that position, not the other way round. Then `s_dur_two`, which is signed and
so names a direction (`left_early` → abandonment, `stuck` → repetition) rather than merely
"unusual". Emission channels last, since a boundary the causal filter has not caught up to yet
fires them as a side effect of every structural error.

The residual confusions are the ones this repo already documents rather than new artefacts:
transposition → `omission` (both are `s_transition`; the cascade cannot separate them), and
repetition split between `repetition` and `substitution` (its ambiguity is stated outright in
[`synthetic.md`](synthetic.md)).

### What is reported

`evaluate_steps` returns the same `per_type` / `attribution` / `healthy` shape as
`metrics.evaluate`, so the printer and `eval/plotting.py` work on it unchanged. It keeps the
`precision` vs `precision_excl_healthy` split — `fp_healthy` is the same shared constant pooled
into every type's denominator at step level too, so the anti-self-deception argument in
[`eval.md`](eval.md) §2 carries over verbatim. `latency_tol = 5` ticks becomes `tol_steps = 1`,
since five ticks does not translate to a unit whose median width is eleven.

Three additions the tick layer structurally cannot provide:

| Field | What it measures |
|---|---|
| `type_confusion` | predicted × true anomaly type. The tick layer's attribution matrix only *proxies* this through which channel fired. |
| `correction_accuracy` | does the proposed correct move match the pre-injection step? Token match and duration match are reported **separately**, because abandonment leaves verb and noun untouched — a combined score would credit naming a step the detector never had to identify. |
| `parse_failure_rate` | LLM arms only. |

plus `step_level`, a pooled precision/recall over every step of every trial as one test each.

---

## 6. Running it

```bash
# cost the sweep first -- no key needed, no requests sent
python run_llm_eval.py --config configs/breakfast.yaml --dry-run

# the HSMM arm alone: no key, no budget, no network
python run_llm_eval.py --config configs/breakfast.yaml --skip-llm

# cheap smoke test (after `cp .env.example .env` and filling in GEMINI_API_KEY)
python run_llm_eval.py --config configs/breakfast.yaml --protocol batch --n 2 --max-real 2

# the real thing, both variants
python run_llm_eval.py --config configs/breakfast.yaml --variant both --n 20 --max-real 20

# score the 2-stage cascade as the HSMM arm instead of the joint model
python run_llm_eval.py --config configs/breakfast.yaml --cascade
```

### Which HSMM, and which checkpoint

The **joint** model is the default (`--cascade` switches back), on
`dataset/processed/breakfast/joint_params.npz`.

That choice changes **both** arms, not just the HSMM one: the model Viterbi-segments each healthy
trial, which decides where `error_injection` can place an error, and therefore what text the LLM
ends up reading. The LLM never sees the model, but it reads trials the model segmented.

The checkpoint was **not** chosen on training likelihood, which is unusable on this model — a
random recipe partition has been measured to beat the ground-truth partition by 175 nats, because
~65k per-recipe transition parameters sit against ~10⁴ observed segment transitions. Nor on recipe
ARI, which does not move detection (tripling it, 0.31 → 0.86, moved mean recall 0.823 → 0.821).
Nor on segmentation quality, which is indistinguishable across every available checkpoint
(boundary-F1 0.987–0.990 against the ground-truth action labels; injections land on a step
boundary 95% of the time; all six).

The one axis that separates them is **effective $K_R$** — how many of the 16 nominal recipe
components the fit actually uses — and more of them is measurably *worse*:

| checkpoint | init | eff $K_R$ | healthy FPR | step prec | step recall | mean prec |
|---|---|---|---|---|---|---|
| **`joint_params`** | cascade warm start | **5** | **0.050** | **0.951** | **0.802** | **0.947** |
| `joint_kmeans10` | k-means | 9 | 0.075 | 0.936 | 0.792 | 0.922 |
| `joint_noun10` | noun histogram | 11 | 0.075 | 0.938 | 0.762 | 0.915 |
| `joint_noun16` | noun histogram | 14 | 0.100 | 0.893 | 0.759 | 0.901 |

(40 real trials, 5 injections each, `min_run=10`, scored through this file's own step layer.)

Recall is flat — 0.880–0.910, no trend. Precision and healthy false-positive rate degrade
monotonically as effective $K_R$ rises. The mechanism is the same over-parameterisation that makes
training likelihood useless: splitting the corpus across more live components thins the data
behind each per-recipe transition and duration table, so the fitted rows are noisier, the quantile
thresholds derived from them looser, and spurious flags more frequent. **Extra recipe capacity
costs precision and buys no recall.**

Honest reading of the evidence: at $n = 40$ the individual gaps are small (healthy FPR 0.050 vs
0.100 is 2 trials against 4). The *monotone trend across four checkpoints in three separate
metrics* is the evidence, not any one pairwise difference. `joint_noun16` does hold the best
abandonment recall (0.90 vs 0.78) if that channel is ever the specific target.

`joint_clean.npz` and `joint_d09.npz` are not candidates: they are the same fit as `joint_params`
at iteration 200 and 320 rather than 100, with the same effective $K_R = 5$, and iterating past
~100 changes downstream metrics by nothing. Every joint checkpoint reports `converged: false` —
that is the `tol`-based stopping rule being structurally unable to fire against a period-3 limit
cycle in the objective, not an unfinished fit.

Healthy trajectories and all five injections are built **once per source** and handed to every
arm, so the LLM and the HSMM score byte-identical degraded trials. Regenerating them per arm would
silently compare two different datasets, which is the failure mode this whole layer exists to
avoid.
