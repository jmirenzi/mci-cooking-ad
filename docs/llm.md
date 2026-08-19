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
| `render_llm_compare_png.py` | comparison figures from the report JSON (layout only, no inference) |
| `render_llm_vs_hsmm_png.py` | one trial, all detectors, at their native resolutions |

Driven by `run_llm_eval.py`. `eval/metrics.py` and `run_evaluation.py` keep their own tick-level
path; `evaluate_steps` takes only optional extra arguments, so that path is unaffected by anything
here.

> `anomaly/narrate.py` opens by stating that routing the system's user-facing questions through a
> language model would throw away their auditability. That still holds, and this module does not
> contradict it: the LLM here is a **baseline to measure against**, not a component of the
> detector. Nothing in `anomaly/` calls it.

---

## 0. Headline result

**95 real trials** x (healthy + 5 injected errors), `gemma3:27b` served locally, conversational
protocol, joint HSMM at $\alpha = 5\times10^{-3}$. Scored **trial-located** (§5): one verdict per
trial, on whether the flag landed inside the injection-touched range.

| arm | precision | recall | F1 | stray/degraded | healthy FPR |
|---|---|---|---|---|---|
| **HSMM** | **0.618** | 0.423 | **0.503** | **0.219** | **0.211** |
| gemma3 (no recipes) | 0.439 | 0.537 | 0.483 | 0.554 | 0.663 |
| gemma3 (+ recipes) | 0.416 | **0.613** | 0.496 | 0.693 | 0.832 |

**On F1 the three are tied.** The HSMM's advantage is entirely in alarm volume: it reaches the
same F1 while flagging 21% of healthy trials against the LLM's 66–83%. For an assistant whose
failure mode is nagging a user with MCI, that is the axis that decides usability — but it is a
narrower claim than "the HSMM detects better."

**Recipe descriptions make the LLM flag more, not better**: recall +7.6 points, precision −2.3,
healthy false alarms +16.9. That arm also has the `labels.json` advantage the HSMM never had (§2),
and it runs on the conversational protocol, the LLM's stronger one (§3) — so both choices favour
the LLM and its false-alarm disadvantage is if anything understated.

### Detection is real on every error type

Per-type recall is low outside substitution, which raises the obvious question of whether those
detections are anything more than a chatty detector occasionally landing in the right place.
`run_counterfactual.py` ([`eval.md`](eval.md) §5) answers it by scoring each degraded trial
against **its own healthy counterfactual** — same trial, same range, no injection — which is a
matched null that absorbs trial length, range width, and the detector's own per-trial noise:

| error type | recall | matched null | lift | McNemar |
|---|---|---|---|---|
| substitution | 0.998 | 0.043 | **+0.955** | $\chi^2$ = 398 |
| abandonment | 0.384 | 0.017 | **+0.368** | $\chi^2$ = 148 |
| omission | 0.294 | 0.031 | **+0.263** | $\chi^2$ = 97 |
| transposition | 0.377 | 0.055 | **+0.322** | $\chi^2$ = 129 |
| repetition | 0.210 | 0.041 | **+0.169** | $\chi^2$ = 69 |

(419 real trials, $\alpha = 5\times10^{-3}$; every $\chi^2$ far past 3.84 at $p = 0.05$, with
discordance almost entirely one-directional.)

So the correct reading of the low per-type numbers is **low sensitivity, not absent detection**.
Note also how far the matched null sits below a uniform-random estimate: flags cluster at segment
boundaries rather than spreading across a trial, so a healthy run's flags usually miss the injected
range entirely. Any chance baseline derived by assuming uniform flag placement overstates it by
roughly 5x.

> **Every number on this page is in-sample.** `run_joint.py` fits on all of `sequences.json` and
> the evaluation scores a prefix of the same file; there is no held-out split. Treat these as
> upper bounds.

### The older step-level numbers

`evaluate_steps` also reports a pooled step-level precision/recall (§5). The last full-corpus
measurement of it — 447 real / 90 synthetic trials, incremental protocol — predates
`surprise.EMISSION_FLAG_ATOL`, the right-censoring fix, the removal of the persistence rule, and
the $\alpha$ change, so **its HSMM rows no longer describe the shipping detector** and are not
reproduced here. The LLM rows from that run remain valid: step-level precision 0.183 (real) and
0.150 (synthetic) against chance baselines of 0.125 and 0.114, i.e. **at roughly 1.2–1.5x chance**,
with `parse_failure_rate` 0.0025 over 19,253 requests — a capability result, not a formatting
artefact.

### Per-type: the HSMM dominates one type, the LLM leads on the rest

Trial-located recall, and the confusion diagonal (given a detection, was the type named right):

| | HSMM recall | LLM(+r) recall | HSMM diagonal | LLM(+r) diagonal |
|---|---|---|---|---|
| substitution | **0.989** | 0.979 | **1.000** | 0.935 |
| abandonment | 0.358 | **0.842** | 0.838 | **0.901** |
| omission | 0.316 | **0.337** | **0.879** | 0.058 |
| transposition | 0.253 | **0.589** | **0.000** | 0.088 |
| repetition | 0.200 | **0.316** | **0.789** | 0.281 |

The HSMM clearly dominates only **substitution** — its own emission channel's error type. It is
better at *naming* what it found on four of five types, but worse at finding it on four of five.

**Transposition is the structural blind spot**: the diagonal is exactly 0.000 because
$s_{\text{trans}}$ fires identically for omission, transposition and repetition and cannot
separate them. Note the counterfactual table above shows transposition detection *is* real
(lift +0.322) — the model notices, it just cannot say what it noticed.
`anomaly/sequence.py` ([`anomaly.md`](anomaly.md) §6) exists to close exactly that gap by testing
the swap hypothesis directly against the fitted transition row. It is built and unit-tested but
**not yet wired into a scored arm**, so no measured claim is made for it here.

> **The synthetic arm is not neutral ground for this comparison and should not be read as a second
> opinion.** Synthetic trials are ancestral samples from the HSMM, so they flatter it — which
> [`eval.md`](eval.md) already says — but they also actively *penalise* the LLM: healthy
> false-positive rate 0.767 there against 0.418 on real trials. The samples are valid under the
> model while containing (verb, noun) pairings and orderings no real cook would produce, so a
> detector reasoning from real-world priors is correct to call them anomalous. **Report the real
> arm.**

### Reading the false-positive rate

`healthy.false_positive_rate` is the fraction of healthy control trials with **any** flagged step.
Both arms are held to that same bar — there is no persistence filter or run-length requirement on
either side ([`eval.md`](eval.md) §2), so the two numbers are directly comparable and the figure
and the table read the same mask.

The arms do still differ in how many tests produce that flag: the HSMM makes ~150 per-tick tests
per trial that collapse into ~7 step verdicts, while the LLM makes ~7 step tests directly. That
asymmetry is real and is controlled where it belongs — at the threshold $\alpha$, which trades
false alarms against recall along a curve `run_threshold_sweep.py` measures ([`eval.md`](eval.md)
§6) — rather than by filtering one arm's output after the fact.

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
only the prefix — is the property the comparison depends on; conversational self-feedback is a
*separate* property, which the `conversational` protocol below supplies and this one withholds.
Withholding it buys two things:

1. **No error cascade.** Under self-feedback a false alarm at step 2 sits in context for steps 3–7
   and can bias them, so a miss at step 5 cannot be distinguished from contamination by an earlier
   mistake. Here every step is an independent test, which is what `element_metrics` assumes when it
   scores them as one test each.
2. **Parallelism.** Every request is a pure function of `(trial, step index)` rather than depending
   on a previous response — a precondition for issuing requests concurrently, and for submitting a
   sweep to an async batch endpoint. Neither is possible when request *i+1* contains response *i*.

**Prompt layout is chosen for prefix caching.** One system turn plus one user turn that only ever
grows by appending a line, so request *i*'s prompt is a strict token prefix of request *i+1*'s. A
server with prefix caching (vLLM's APC, or hosted providers' implicit caching) then reuses almost
all of it, and the ~600-token system prompt is reused across the entire sweep. A single growing
user turn is used rather than repeated user turns because consecutive same-role messages are not
universally accepted.

> Changing the protocol changes the prompts, hence the cache keys. Responses collected under the
> older conversational protocol will not be reused, and results from the two are not directly
> comparable.

### `conversational`

One request per step, with the model's own earlier replies fed back as assistant turns. Not the
default, but **kept runnable, because the two protocols are not equivalent in behaviour and the
non-default one is more accurate on this model.**

Same model, same 10 real trials, same seed, same prompt; only the protocol differs:

| protocol | tp | fp | fn | precision | recall |
|---|---|---|---|---|---|
| `incremental` (prefix-only) | 18 | 77 | 42 | 0.189 | 0.300 |
| `conversational` | 22 | **47** | 38 | **0.319** | **0.367** |

Self-feedback improves **both** precision and recall — the opposite of what the error-cascade
argument above predicts. The assistant turns act as a conservatism anchor: a model that has just
said "No Anomaly" five times is being shown five reasons to say it again, which suppresses the
prompt-length drift in section 0. This holds even though the conversational arm runs with the
prompt written for prefix-only ("a numbered list … judge only the last step"), which does not
describe what it actually sees.

So the prefix-only default buys parallelism and clean independence at a real cost in accuracy on
this model. That trade is worth restating whenever a number from either protocol is reported, and
the headline in section 0 was measured under `incremental`, i.e. under the weaker of the two.

Cannot be parallelised within a trial: request *i+1* literally contains response *i*. Measured
throughput was nonetheless comparable (~1.55 req/s), since ollama's prefix cache handles the
growing message list well.

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

### The HSMM adapter

`step_verdicts_from_flags` turns per-tick channel flags into one `StepVerdict` per step: a step is
`is_anomaly` if **any** tick inside it is flagged. That is the only mask — there is no persistence
variant, here or in the tick layer ([`eval.md`](eval.md) §2), so both arms are scored through one
rule and the two detectors' false-alarm numbers mean the same thing.

A step spans ~11 ticks, so the HSMM does carry more per-step exposure than the LLM's single
verdict. That is handled at the threshold rather than by filtering the output: $\alpha$ moves
false alarms and recall together along a curve `run_threshold_sweep.py` measures
([`eval.md`](eval.md) §6), whereas a run-length filter would mute the structural channels
(whose true signal is one tick wide) while sparing the duration channels.

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

plus two pooled views:

| Field | Unit | Question |
|---|---|---|
| `step_level` | one step | of all steps, which were correctly classified? |
| **`trial_located`** | one trial | did the detector flag **in the right place** on this trial? |

`trial_located` is the headline metric (§0) and mirrors `run_threshold_sweep.py`'s `trial_loc`
([`eval.md`](eval.md) §6) in step space rather than tick space. A degraded trial's positive range
is `gt_steps` $\cup$ `debris`; a flag inside it is a hit (TP) and no in-range flag is a miss (FN);
a flag outside it is a **stray**, charged as a false positive *independently* of whether the trial
was also hit, so finding the anomaly buys no absolution for firing elsewhere. Healthy trials have
no range: any flag is a false positive.

`stray_rate` (per degraded trial) and `healthy.false_positive_rate` (per healthy trial) are
reported separately rather than pooled into one FPR, because a degraded trial can contribute both
a TP and an FP and so is not a member of a well-defined negative class — `FP/(FP+TN)` would mix
the two populations and is deliberately not reported.

### What counts as one test — the `step_level` pool

Two rules define the pool, and both apply identically to every arm.

**One ground-truth window is one test.** A window can span several steps — a transposition spans
the swapped pair, averaging 2.01 ground-truth steps per trial. Flagging either half is a complete
detection of the one event that was injected, so the window contributes exactly one TP (any of its
steps flagged) or one FN (none flagged), never one per step. Scoring per step would book a false
negative for the unflagged sibling of a pair the detector had already found.

**Injection debris is excluded entirely.** A degraded step is *injection-touched* if it contains or
borders an edited tick, or if its own tick range spans a `tick_map` splice
(`textify.injection_touched_steps`, using the provenance fields in
[`synthetic.md`](synthetic.md) §2.1). A touched step that is **not** ground truth is neither the
injected anomaly nor a clean normal step: it is an artefact of the edit, and it is dropped from the
pool rather than counted either way.

The clearest case is substitution. Retagging one tick inside a run splits that run into three
steps, and the two surviving fragments have the durations they have *only* because of the
injection. A detector that flags one — correctly recovering the original duration — is neither
right nor wrong about anything the injector set out to test. Repetition produces the analogous case
when its duplicate merges with its original across a splice.

Both rules move the denominator, so `chance_precision` is computed from the same scoreable pool
rather than from `tp + fp + fn` — that quantity varies with how much a given detector flags and
would hand the two arms different baselines for identical data.

`evaluate_steps` takes `artifact_steps` per degraded trial; omitting it disables the debris rule,
which is what keeps `run_evaluation.py`'s tick-level path unchanged.

---

## 6. Running it locally (the default)

The client defaults to `http://localhost:11434/v1` with `gemma3:27b`. A local server removes every
constraint the hosted path imposes -- no key, no pacing, no daily quota -- and it *pins the model*,
which the hosted path cannot: `gemini-2.5-flash-lite` was retired mid-experiment here, invalidating
the results collected against it. For a baseline whose numbers end up in a writeup, that
reproducibility is the strongest argument for local, ahead of speed or cost.

```bash
# rootless install, no sudo
mkdir -p ~/.local/ollama && cd ~/.local/ollama
curl -sSL -o o.tar.zst https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64.tar.zst
tar --zstd -xf o.tar.zst && rm o.tar.zst
export PATH="$HOME/.local/ollama/bin:$PATH" LD_LIBRARY_PATH="$HOME/.local/ollama/lib:$LD_LIBRARY_PATH"
ollama serve &
ollama pull gemma3:27b
```

`--rpm` and `--concurrency` default *by destination*, because the right value is opposite in the
two cases: a localhost URL gets no pacing and 8 workers, anything else gets 15 rpm and 1 worker
(concurrency against a request-capped free tier just converts one 429 into eight).

### Sharing the GPU with JAX

JAX preallocates ~75% of the device on first use -- **37 GB of a 48 GB card**, measured -- which
starves a model server that has not loaded yet. `run_llm_eval.py` therefore sets
`XLA_PYTHON_CLIENT_PREALLOCATE=false` by default, after which JAX holds **572 MB** and JAX plus
`gemma3:27b` coexist at 29.5 GB.

Measured cost of disabling it, HSMM arm on 20 real trials: **68.83 s vs 68.97 s** -- nothing. This
workload is dominated by XLA compilation and a few large ops, not allocation churn, so there is no
trade-off to manage.

If you would rather keep the arms apart anyway, `--skip-llm` then `--skip-hsmm` **merge** into one
report at the same `--out`. The merge is guarded on the pool-defining arguments (seed, source,
trial counts, checkpoint): a run with a different `--seed` is refused rather than silently
producing a table that looks like a comparison and is not one.

### Measured throughput, and why concurrency disappoints here

Prefix-only requests are independent, so they are issued through a thread pool. n=10 real trials,
251 requests, RTX 6000 Ada:

| model | concurrency 1 | 8 | 16 |
|---|---|---|---|
| `gemma3:4b` | 104 s | **63 s** (1.66x) | 61 s (1.70x) |
| `gemma3:27b` | 850 s | 701 s (1.21x) | — |

Real, but far short of the ~7x that independent requests suggest, and it saturates by 8 workers.
The client is not the bottleneck; llama.cpp simply batches weakly. This is the ollama-vs-vLLM
difference: vLLM's continuous batching is what converts request independence into throughput.

### The configuration trap that dominates everything else

**Raising `OLLAMA_NUM_PARALLEL` without also bounding `OLLAMA_CONTEXT_LENGTH` will silently push
most of the model onto the CPU.** KV cache is allocated as `context_length x num_parallel`, and
these models default to enormous contexts -- 131072 for gemma3, 40960 for qwen3. At
`NUM_PARALLEL=8` that is an **80 GB** demand on a 48 GB card, so ollama loads only what fits:

```
llama_kv_cache: size = 81920.00 MiB (131072 cells, 10 layers, 8/8 seqs)
load_tensors: offloaded 21/63 layers to GPU      <- gemma3:27b, two thirds on CPU
```

Nothing errors. The run just becomes CPU inference. Measured on n=10, 251 requests:

| gemma3:27b | concurrency 1 | concurrency 8 |
|---|---|---|
| default context (21/63 layers on GPU) | 850 s | 701 s |
| `OLLAMA_CONTEXT_LENGTH=2048` (63/63 on GPU) | **212 s** | **167 s** |

**4x from one environment variable** -- far more than concurrency ever contributed. The longest
prompt this evaluation produces is 842 tokens (1476 with recipe descriptions), so 2048 is ample.

Gemma 3's sliding-window attention is a red herring here. ollama does churn ~250 MB KV checkpoints
in its log (`n_swa = 1024`) and that churn is real, but it is not the bottleneck — the layer-offload
lines above are. `gemma3:4b` has the same SWA architecture and scales fine, precisely because it
fits on the GPU whole. Read the offload count, not the SWA chatter.

### Model choice

`gemma3:27b` is the recommended local model. **`qwen3:32b` is a poor fit despite being a
reasonable model**, for a mechanical reason: its thinking mode cannot be disabled through the
OpenAI-compatible endpoint. `think: false` works only on ollama's native `/api/chat`
(4 completion tokens, 0.26 s), while the OpenAI path ignores it and also ignores
`chat_template_kwargs: {enable_thinking: false}`; `PARAMETER think false` in a Modelfile is
rejected as an unknown parameter. The result is 219-294 completion tokens per verdict against
gemma3's ~15, roughly 10x the generation cost for a one-line answer, plus a tendency to invent
out-of-vocabulary types ("Sequence Anomaly", "Time Anomaly"). Supporting it would mean teaching
the client ollama's native protocol, which is exactly the provider-specific coupling
`--base-url` exists to avoid.

So: **ollama is right for validating the pipeline and for n<=10; vLLM is what makes n=100+
pleasant.** Raise `OLLAMA_NUM_PARALLEL` to at least `--concurrency` if you use concurrency -- and
bound `OLLAMA_CONTEXT_LENGTH` when you do, or you will lose far more to CPU offload than
concurrency wins back.

## 7. Running it



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

# switch back to a hosted provider (needs GEMINI_API_KEY in .env)
python run_llm_eval.py --base-url https://generativelanguage.googleapis.com/v1beta/openai \
    --model gemini-3.1-flash-lite
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

(40 real trials, 5 injections each, scored through this file's own step layer.)

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

### Figures

`run_llm_eval.py` writes per-arm figures as it goes. `render_llm_compare_png.py` then reads only
the report JSON and draws the HSMM against the LLM:

```bash
python render_llm_compare_png.py --report dataset/processed/breakfast/llm_full_report.json
```

| figure | shows |
|---|---|
| `compare_detection_{source}` | recall and precision per error type, both detectors |
| `compare_steplevel` | pooled step-level precision / recall / F1, with the **chance-precision line** |
| `compare_type_confusion_{source}` | predicted x true type heatmaps, side by side, fixed [0,1] scale |
| `compare_type_accuracy` | the confusion diagonal alone -- given a detection, was the type right? |
| `compare_correction` | accuracy of the proposed "correct move", tokens and duration separately |
| `compare_latency` | mean detection latency in steps |
| `compare_healthy_fpr` | trial-level false-positive rate on healthy controls |

The chance line on `compare_steplevel` is not decoration. Step-level precision has to be read
against the base rate of anomalous steps or a detector that flags everything looks respectable:
`gemma3:4b` scored 0.166 precision at n=10, which is 1.16x chance, i.e. essentially nothing.
`step_level.chance_precision` is computed in `evaluate_steps` from the true step total rather than
derived in the renderer, because `tp+fp+fn` moves with how much a given detector flags and would
hand the two arms different baselines for identical data.

Being layout-only, the renderer can restyle a three-hour sweep's figures in seconds.

Healthy trajectories and all five injections are built **once per source** and handed to every
arm, so the LLM and the HSMM score byte-identical degraded trials. Regenerating them per arm would
silently compare two different datasets, which is the failure mode this whole layer exists to
avoid.
