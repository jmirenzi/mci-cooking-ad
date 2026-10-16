# Synthetic drift data from CaptainCook4D

The drift detector in [`next_steps.md`](next_steps.md) §2 needs streams where a person's normal
behaviour changes by a known amount at a known time. No public dataset has that, so this generator
builds it: multi-week, per-person streams of (verb, noun) actions in the EPIC-100 vocabulary, with
ground truth for onset, rate, size and the affected observable features.

It is a sanity and validity test, not a benchmark. Control matters more than realism, with one
exception: the output must be at the **logistics level** the HSMM is trained on (`take bowl`,
`pour water`), not the recipe-step level (`make roux`).

**Status (2026-10-15):** built end to end. All 31 conditions × 3 seeds × 8 persons are generated
under `dataset/processed/drift_synth/` (about 22k sessions, 470 MB, 70 s), and the sanity plots are
in (§8). Open for Jack: the duration scale a, β and the day-noise σ, and whether to lengthen waits
(sessions run 2.8 min against CC4D's 15.7). No detector has been run on the data.

---

## 1. Design (settled; changes go through Jack)

1. **Recipe structure comes from CaptainCook4D (CC4D) task graphs.** The LLM never invents it.
2. **LLM for structure, code for randomness.** An LLM runs offline, once per recipe step, to expand
   the step into vocabulary tokens and propose knob variants. Ordering, inclusion, durations, noise
   and drift are all sampled in seeded code.
3. **Knobs are global and hand-defined** in `configs/drift_synth/knobs.yaml`. They act on **tags**,
   and tags are assigned once per vocabulary item, not per recipe.
4. **Closed vocabulary**: every emitted token is in `dataset/processed/epic/vocab.json`.
5. **Fetch and put-back logistics are inserted by code** (a kitchen-state pass), not by the LLM.
6. **Ground truth is defined in observable space**: per recipe, token marginals, bigrams and
   per-token durations. It never refers to HSMM latent states, and `cook_ad.drift_synth` imports
   nothing from `hsmm`, `detectors` or `eval` (a test enforces this).

## 2. Pipeline

| Stage | What | Randomness |
|---|---|---|
| 1 graphs | load and validate task graphs; LLM labels each step prep/cook/serve/cleanup | none (LLM, cached) |
| 2 tags | LLM proposes tags per vocab item (`sour`, `fat`, `sweet`, `fine_motor`, `heat`, `cleaning`, `ingredient`, `tool`, `container`); Jack reviews | none |
| 3 expansion | LLM expands each step into core tokens (pass 1), then a calorie variant for steps whose tokens use a fat noun (pass 2), then one call per recipe for ≤ 2 sour sites; validated, retried 3×, cached | none |
| 4 kitchen state | `take` before first use, `put` after last use, `wash`/`dry` after cleaning-relevant uses (rates from `tidy`) | seeded |
| 5 durations | log-normal per verb, (verb, noun) backoff below 20 samples, from EPIC-100; gap distribution | seeded |
| 6 knobs | θ ∈ [0, 1] per knob drives variants, step order, logistics and duration scales | seeded |
| 7 nuisance | day-level θ noise (logit space), AR(1) speed, recognizer noise, recipe schedule and mix shift | seeded |
| 8 drift | θ(day) under none / sudden / incremental / gradual / recurring | seeded |
| 9 sample, export | one session = order → variants → logistics → repetition → durations → noise | seeded |
| 10 ground truth | Monte Carlo observable statistics at θ0 and θ1; JS and W1 per recipe | seeded |

**Knobs.**

| knob | family | acts on |
|---|---|---|
| `sour` | preference | P(apply sour `insert` variants) = θ |
| `calorie_conscious` | preference | P(apply fat `substitute`/`omit` variants) = θ; remaining fat-tagged durations × exp(−0.7θ) |
| `mise_en_place` | preference | step order: prep before cook |
| `tidy` | preference | put-back and wash/dry rates |
| `slow` | capability | every duration × exp(a(θ − 0.5)) |
| `fine_motor` | capability | duration and repetition probability of fine-motor-tagged verbs |

**Knob parameters** (`configs/drift_synth/knobs.yaml`): `tidy` puts back with P from 0.1 to
0.95 and washes tools and containers with P from 0 to 0.8, linear in θ. `slow` and `fine_motor` use
a = 1 (θ from 0 to 1 scales duration by 0.61 to 1.65). `fine_motor` repeats a token with
P = 0.02 + 0.3θ. `mise_en_place` uses β = 4. A capability knob's high θ is the impaired end.

**Step order.** A softmax over the steps available at each point is *not* uniform over
topological orders. Each candidate is therefore weighted by the number of orders that complete it
(`graphs.linear_extension_counter`, a DP over order ideals), times exp(β(2θ − 1)·is_prep). At
θ = 0.5 the bias vanishes and the order is exactly uniform over the graph's topological orders.

**Drift regimes** (`schedule.theta_at`): `none`; `sudden` (θ1 from onset); `incremental` (linear
over `duration_days`; every duration knob is exp(a(θ − 0.5)), so linear in θ is linear in log
duration); `gradual` (each day θ1 with P = the same ramp, else θ0); `recurring` (θ1 and θ0
alternate in `period_days` blocks from onset). Day-level noise is then added in logit space
(`noise.theta_for_day`), and every duration and gap is multiplied by an AR(1) day speed.

**Recognizer noise** (`noise.recognizer_noise`). One categorical draw per clean event: delete,
swap the verb, swap the noun, or keep. Then, independently, a spurious token takes the event's
tail. Confusions come from the session's own tokens, so they stay in the vocabulary. Every rate is
per clean event, as configured.

**Random streams** (`sampler.rng_for`). A person's traits come from (seed, person), each day's draws
from (seed, person, day), and each session's from (seed, person, day, session). None of them
includes the condition. Conditions with the same seed therefore share every day before onset
(common random numbers), and a drift condition differs from `null` only through θ. Every random
choice draws a fixed number of numbers regardless of θ, so streams stay aligned as θ moves.

**Stage 3 details** (`expand.py`):
- The schemas enumerate the vocabulary, so a reply cannot be out of vocabulary.
- Validation rejects:
  - `take X` of a logistic noun;
  - an add/pour/sprinkle pair whose noun is not an ingredient (the destination, "add pan");
  - an ingredient the recipe does not name (the dish itself and generic food are allowed);
  - repeated pairs;
  - variants on the wrong tag or with indices out of range.
- CC4D's "Take X" steps expand to nothing, because Stage 4 inserts the take.
- Asked per step, the model added lemon *and* vinegar to almost every step, hot chocolate
  included. Asked once per recipe, it picks one or two sites or none.

## 3. Sources

- **Recipe structure**: `dataset/captaincook4d/task_graphs/` (github.com/CaptainCook4D/annotations).
  START and END are ordinary steps; an edge `[a, b]` means a precedes b. Loading strips the
  sentinels after checking they are the only source and sink.
- **Durations, gaps and few-shot exemplars**: EPIC-100 raw narrations
  (`dataset/epic_kitchens/annotations`, loaded by `data/parse_epic.py:load_narrations`). CC4D's
  advertised fine-grained action annotations are not public. Its finest public unit is the step
  (median ~30–40 s), which is above logistics level, so CC4D step and recording durations are used
  only to calibrate and check session length.
- **Vocabulary**: EPIC-100's 97 verb and 305 noun classes plus SIL (`stall`, `idle`). Verbs and
  nouns are separate classes.

## 4. Output

Per person, under `dataset/processed/drift_synth/<condition>/seed<k>/<person>/`:

- `sessions/<day>_<idx>.json`: timestamped string events, θ for the day, and the clean events
  before recognizer noise;
- `sequences.json`, `labels.json`, `vocab.json`: the loader format of every other corpus
  ([`data.md`](data.md) §1). These are EPIC ids at 0.5 s ticks with gaps as (`stall`, `idle`),
  `trial_id = <person>_d<day>_s<idx>`, and `recipe_label` set to the CC4D recipe;
- `ground_truth.json`: condition, regime, onset, duration, moving knobs, θ0, θ1, per-day θ and
  mix-shift parameters.

## 5. Findings so far

The full report is `runs/drift_synth/reports/precheck.md`, written by `scripts/drift_synth/precheck.py`.

- All 24 CC4D graphs validate. Their topological-order counts run from 6 (spiced hot chocolate) to
  1.6 × 10¹⁰ (broccoli stir fry), and each counts in milliseconds.
- **`mise_en_place` has little room.** The graphs already force nearly every prep step before the
  first cooking step. 18 of 24 recipes, including every stovetop dish, have no (prep, cook) pair the
  knob could reorder. The six that do are all in the proposal.
- **EPIC verbs are coarse**: whisk, stir and beat are all `mix`; chop, slice and trim are all `cut`.
  There is no `wipe` (it is an instance of `wash`). `pepper` covers both bell and black pepper, and
  `sauce` covers soy sauce, salsa, sriracha, mayonnaise and marinara.
- Items with no EPIC noun: dental floss (pinwheels), frosting and cupcake liners (mug cake), the
  spiralizer (zoodles).
- **Phases** (`configs/drift_synth/phases/`, review copy `runs/drift_synth/reports/phases.md`):
  gemma3 needed two explicit rules. Adding to a pan already on the heat is cook. A recipe that never
  heats anything has no cook steps.
- **Tags** (`runs/drift_synth/reports/tags_summary.md`, with a list of likely errors): the proposal
  over-tags. Knife, spoon and hand come back as ingredients, whisk and dishwasher as heat, and 34
  verbs as fine-motor. Sour is only lemon, lime, vinegar and yoghurt.
- **EPIC durations** (`dataset/drift_synth/durations_epic.json`): take and put about 1.4 s, cut
  5.5 s, mix 7.6 s; 35% of gaps are zero. EPIC narrations do not cover long waits (microwaving,
  simmering), so sessions will be shorter than CC4D recordings (median 15 min) unless expansions
  emit `wait` tokens.
- **Expansions** (`configs/drift_synth/expansions/`): 257 steps, 0 skipped. 26 calorie variants,
  all `omit` (the model never proposed a lower-fat substitute). 21 sour inserts, some odd: lime in
  hot chocolate, lemon in pancake batter. The review sample is a seeded 10%.
- **First generation** (`null`, seed 0): 719 sessions over 8 persons and 70 days, generated in
  under a second. The median session is 2.9 min (p10–p90 2.2–4.5) against CC4D's ~15 min. EPIC's
  `wait` narrations are a few seconds long, so waiting is nearly free.
- **Approved set** (18 recipes, every knob covered by ≥ 6): scrambled eggs, herb omelet, tomato
  mozzarella salad, caprese bruschetta, cucumber raita, butter corn cup, sauteed mushrooms,
  broccoli stir fry, zoodles, pan-fried tofu, tomato chutney, microwave egg sandwich, microwave
  French toast, breakfast burritos, spiced hot chocolate, cheese pimiento, banana pancakes, coffee.

## 6. Conditions

`configs/drift_synth/conditions.yaml`, expanded by `scripts/drift_synth/generate.py:conditions`
into 31 conditions × 3 seeds × 8 persons × 70 days, with onset at day 28 and ramps of 21 days:

- `null` and `mix_shift_only`;
- every knob × {sudden, incremental, gradual}, with |Δθ| = 0.4;
- `pref_plus_capability` (sour and slow together);
- `sour` and `slow` sweeps at |Δθ| ∈ {0.05, 0.1, 0.2, 0.3, 0.45}, sudden.

θ0 ~ U(0.25, 0.75) per knob. θ1 moves up if that fits in [0, 1], otherwise down.

## 7. Tests

`tests/test_drift_synth_generator.py` runs on a toy world (`tests/_drift_fixtures.py`: two
recipes, hand-written expansions and tags, the real knobs) and covers the spec's nine tests:

1. vocab compliance;
2. sampled orders are topological;
3. byte-identical exports;
4. knob isolation: unlinked counts within MC noise (Bonferroni z), unlinked durations (KS), and `mise_en_place` moving bigrams only;
5. null equivalence at α = 0.01 after Bonferroni;
6. a monotone dose-response for `sour` (marginal JS) and `slow` (mean W1);
7. noise rates within 10% on 10k events;
8. uniform order at θ = 0.5 (χ²) on a graph where a naive softmax is not uniform;
9. `sudden` switches exactly at onset, and `gradual` tracks its ramp within 0.04 per week.

It also checks that pre-onset days are identical between `null` and a drift condition.

## 8. Sanity results

`scripts/drift_synth/sanity_plots.py` writes to `runs/drift_synth/reports/sanity/`. The numbers
below are means over the 18 recipes, 150 clean Monte Carlo sessions per recipe and θ, from
θ = 0.5 on every knob. "MC noise" is the same comparison against a rerun at θ = 0.5 with
another seed.

**Isolation** (`isolation.png`): the change at |Δθ| = 0.45 as a multiple of MC noise.

| knob | core-token marginals | take/put/wash marginals | bigrams | log durations |
|---|---|---|---|---|
| sour | 10.8× | 5.8× | 2.4× | 1.0× |
| calorie_conscious | 41.3× | 21.5× | 4.6× | 1.1× |
| mise_en_place | 1.1× | 1.0× | 3.9× | 1.0× |
| tidy | 1.0× | 11.6× | 6.0× | 1.0× |
| slow | 1.0× | 1.0× | 1.0× | 4.0× |
| fine_motor | 2.1× | 1.0× | 1.4× | 2.0× |

Each knob moves the feature group it targets and leaves the rest at noise. The off-target
entries are expected:

- `sour` and `calorie` move take/put/wash because an inserted or omitted ingredient brings or loses
  its own logistics. Any change in tokens also moves bigrams.
- `fine_motor` moves marginals through its repetitions.
- `calorie`'s fat-token speed-up is too small to see in durations (1.1×).
- `mise_en_place` is the weakest knob (3.9× on bigrams), as the pre-check predicted.

**Dose-response** (`dose_response.png`): monotone in |Δθ| for every knob on its own feature. At
|Δθ| ≤ 0.05 nothing is distinguishable from noise. `slow` reaches W1 = 0.44 against 0.11 noise,
while `sour` at 0.45 is 7× noise on marginal JS.

**Lengths** (`lengths.png`): the median session is 2.8 min, against 5.8 for EPIC sessions (2–15 min
band) and 15.7 for CC4D normal recordings. The median action is 2.2 s against EPIC's 1.6 s, with
more synthetic actions under 0.5 s (recognizer inserts and jitter). Sessions are short because EPIC
`wait` narrations are a few seconds long; waiting is not modelled.

**Granularity** (`examples.md`): synthetic sessions read at EPIC's level ("take onion, cut onion,
add onion, put onion"). Recognizer noise produces the odd pair ("turn-on garlic").

## 9. Does the anomaly detector work on this data?

`scripts/eval/eval_synth_ad.py`, results in `runs/drift_synth/eval_synth_ad{,_clean}.json`.

**Setup.**
- **The person:** seed 0, person 6 (θ0 slow = fine_motor = 0.41, 7 recipes), cooking for 320 days.
- **The model:** fitted on days 0–199 (265 sessions) with the four-stage dish preset (`configs/drift_synth/hsmm.yaml`: K = 128, recipe = the CC4D recipe).
- **Operating points:** best G-F1 on the five injected error types over days 200–259.
- **Test days 260–319:**
  - the injectors on the normal sessions;
  - the same sessions re-sampled with `slow` or `fine_motor` raised by 0.2, 0.4 or 0.55. Common random numbers make each anomalous trial differ from its normal twin only in θ.
- **Two versions of the person:** with the generator's recognizer noise, and without it (`--noise off`).

**Injected errors, test days, metric G** (78 trials per group; null F1 0.052–0.055):

| noise | arm | precision | recall | F1 | alarms / 10 min | run FPR | healthy-trial FPR | subst. | aband. | omission | transp. | repet. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| off | hindsight (α 1e-10) | 0.374 | 0.338 | 0.355 | 2.70 | 0.017 | 0.449 | 0.99 | 0.08 | 0.22 | 0.37 | 0.01 |
| off | live L=0 (α 1e-7, κ 0.9) | 0.370 | 0.344 | 0.357 | 2.80 | 0.018 | 0.449 | 0.99 | 0.08 | 0.22 | 0.38 | 0.03 |
| on | hindsight (α 1e-10) | 0.142 | 0.350 | 0.202 | 10.41 | 0.059 | 0.897 | 0.99 | 0.23 | 0.24 | 0.21 | 0.22 |
| on | live L=0 (α 1e-10, κ 0.9) | 0.144 | 0.350 | 0.204 | 10.22 | 0.058 | 0.910 | 0.99 | 0.23 | 0.22 | 0.22 | 0.23 |

**Capability pools vs their normal twins** (metric G alarm counting: one alarm per flagged run;
hindsight shown, live L=0 agrees within 0.1):

| noise | pool | min / trial | alarms / 10 min | trials with alarm | more alarms than twin | fewer | Δ alarms / trial | repetition recall (G) |
|---|---|---|---|---|---|---|---|---|
| off | healthy twins | 2.7 | 2.70 | 0.45 | | | | |
| off | slow +0.2 / +0.4 / +0.55 | 3.3 / 4.0 / 4.7 | 2.22 / 1.56 / 1.34 | 0.44 / 0.38 / 0.41 | 0.05 / 0.06 / 0.06 | 0.06 / 0.14 / 0.14 | +0.00 / −0.10 / −0.10 | |
| off | fine_motor +0.2 / +0.4 / +0.55 | 3.0 / 3.4 / 3.8 | 2.30 / 2.21 / 2.03 | 0.44 / 0.46 / 0.49 | 0.08 / 0.10 / 0.15 | 0.13 / 0.09 / 0.13 | −0.04 / +0.03 / +0.04 | 0.04 / 0.06 / 0.06 |
| on | healthy twins | 2.7 | 10.41 | 0.90 | | | | |
| on | slow +0.55 | 4.7 | 6.19 | 0.87 | 0.29 | 0.24 | +0.08 | |
| on | fine_motor +0.55 | 3.8 | 7.94 | 0.92 | 0.38 | 0.29 | +0.21 | 0.22 |

**What it says.**

- **Local errors are partly detected.** Substitution is caught (0.99). Transposition (0.37) and
  omission (0.22) are about Breakfast's level for omission and below it for transposition.
  Abandonment and repetition are nearly invisible without noise. With noise, recall on those two
  rises to ~0.22 only because the alarms are everywhere (healthy-trial FPR 0.90).
- **The healthy floor is structural.**
  - The operating point lands at the grid's strictest α in both runs, and the healthy alarms are mostly `s_transition` / `s_recipe_transition`.
  - Step orders are sampled uniformly over up to 10¹⁰ topological orders, so a held-out session often contains a transition never seen in 265 training sessions, and an unseen transition sits on the floor at any α.
  - Recognizer noise then adds `s_emit` flags on swapped tokens: 4× the alarm rate.
- **Capability decline is invisible to the per-trial detector at every level, with or without
  noise.** Even at 1.73× slower, a slowed trial raises the same number of alarms as its twin:
  - `s_temporal` fires more (raw flags 26 → 88), but on runs that are already alarmed.
  - A 1.7× slowdown is 0.55 in log duration, under one standard deviation of EPIC's per-verb spread (σ ≈ 0.65–1.0).
  - The `fine_motor` repetitions are normal for this person (at θ0 = 0.41, 14% of fine-motor tokens already repeat), so the model has learned them.

  This is a property of per-trial anomaly detection, not a bug: a whole-person shift is small in
  any one trial and visible only in aggregate. It is the case for the drift detector
  ([`next_steps.md`](next_steps.md) §2).

### 9.1 Lowering the healthy alarm floor

Three levers, one at a time and combined:

- **Training days:** 200 → 600. Validation and test are always the 60 + 60 days after training.
- **A routine (habit):** each person has a preferred topological order per recipe, and every
  available step is down-weighted by exp(−habit × its rank in that order): a sequential
  Mallows-style concentration. At habit = 0 the draws and the rng stream are unchanged, so every
  dataset in §6 is identical (`generator.yaml: person.habit`, `knobs.sample_topological_order`).
  Raising β would not do this: β only biases prep against cook by the person's `mise_en_place`
  θ, and at θ ≈ 0.5 the order stays uniform.
- **Recognizer noise scale:** × 0.5.

All runs use the same person, Hindsight at its validation-picked α; live L=0 agrees within 0.02 F1.

**Injected errors, test days, metric G:**

| setting | F1 | precision | recall | alarms / 10 min | run FPR | healthy-trial FPR | subst. | aband. | omission | transp. | repet. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline (200 days, habit 0, noise × 1) | 0.202 | 0.142 | 0.350 | 10.41 | 0.059 | 0.897 | 0.99 | 0.23 | 0.24 | 0.21 | 0.22 |
| 600 training days | 0.238 | 0.240 | 0.236 | 3.34 | 0.022 | 0.512 | 0.94 | 0.04 | 0.12 | 0.15 | 0.02 |
| habit 2 | 0.214 | 0.154 | 0.346 | 9.02 | 0.052 | 0.859 | 0.97 | 0.17 | 0.18 | 0.28 | 0.19 |
| noise × 0.5 | 0.263 | 0.202 | 0.378 | 7.17 | 0.043 | 0.782 | 0.99 | 0.19 | 0.19 | 0.41 | 0.08 |
| all three | 0.290 | 0.290 | 0.291 | 3.27 | 0.021 | 0.561 | 0.98 | 0.05 | 0.12 | 0.26 | 0.09 |
| noise × 0 (200 days, habit 0) | 0.355 | 0.374 | 0.338 | 2.70 | 0.017 | 0.449 | 0.99 | 0.08 | 0.22 | 0.37 | 0.01 |

**Where the healthy alarms come from** (share of raw flags on the healthy twins):

| setting | `s_emit` + `s_verb` + `s_noun` | `s_transition` | `s_recipe_transition` | `s_temporal` |
|---|---|---|---|---|
| baseline | 0.42 | 0.26 | 0.26 | 0.07 |
| 600 training days | 0.05 | 0.44 | 0.43 | 0.07 |
| habit 2 | 0.45 | 0.20 | 0.20 | 0.15 |
| noise × 0.5 | 0.56 | 0.22 | 0.22 | 0.01 |
| all three | 0.25 | 0.35 | 0.34 | 0.05 |
| noise × 0 | 0.02 | 0.39 | 0.39 | 0.19 |

**What it says.**

- **Noise is the biggest lever, then data.**
  - Removing noise halves healthy-trial FPR (0.90 → 0.45) and doubles F1.
  - Tripling the training data takes FPR to 0.51, mostly by learning the noise. Emission flags fall from 42% to 5% of the floor.
  - But that also costs recall on the structural errors: abandonment 0.23 → 0.04, repetition 0.22 → 0.02. The model has learned that a swapped or extra token is ordinary.
- **Habit 2 barely moves the floor** (0.90 → 0.86). Transition alarms on noisy data come
  mostly from noise-made transitions (an inserted token creates two new bigrams), not from the
  step order.
- **The floor that remains is transitions.** With noise off or enough data, `s_transition`
  is 70–80% of healthy alarms. No setting gets the healthy-trial FPR below ~0.45, and the picked
  α stays at the grid's strictest value.
- **`s_recipe_transition` duplicates `s_transition` on this data.** It fires on the same runs at
  every setting (dish given, 7 recipes), so it adds no information here.
- **Capability decline stays invisible in every setting.** Δ alarms vs twin is within ±0.3 at
  slow or fine_motor +0.55, so §9's verdict holds.

### 9.2 A global prior from a simulated population

`scripts/eval/eval_synth_prior.py`, `hsmm/train/global_prior.py`. Noise off throughout.

**The population:** 60 people (seed 1000, never test subjects), 30 days each, 2,336 sessions.
- Preference knobs θ0 ~ U(0.05, 0.95), capability knobs ~ U(0.4, 0.6), habit ~ U(0, 1).
- Their tick streams are run-length counted per recipe into first-run and run-to-run transition counts, keyed by (verb, noun). States are pairs, so these counts are the prior with no population EM, and they project onto any person's state inventory.
- One prior serves every person model.

**The fine-tune:** the person's four-stage dish fit (§9), then post-fit, like `smoothing.backoff_counts`:
- Each transition and init row gets λ × the population's row for that recipe and pair. λ is in pseudocounts per row: λ = 10 is the population worth about 10 of the person's own observations of what follows that action.
- Cells are first lifted to count 1, so under the MAP mode a cell the person never produced is worth exactly λ·P_global.
- λ = 0 is the person-only model bit for bit. A transition nobody produced stays on the floor.
- "Global only" drops the person's own init and transition counts.

**The test persons:** seed 0, persons 0–3, habit 2, capability U(0.4, 0.6), 1.3 sessions a day.
Each is trained on its first 10, 30, 100 or 200 days, validated on days 200–259 (injected errors,
α picked by G-F1) and tested on days 260–319. Every arm and N uses the same pools, placed by the
person's N = 200 model.

**Mean over the 4 persons, Hindsight, test days, metric G:**

| train days (≈ sessions) | arm | F1 | precision | recall | alarms / 10 min | healthy-trial FPR |
|---|---|---|---|---|---|---|
| 10 (13) | person only | 0.152 | 0.088 | 0.609 | 29.46 | 0.97 |
| 10 | + prior λ = 1 | **0.200** | 0.143 | 0.422 | 15.04 | 0.81 |
| 10 | global only | 0.197 | 0.140 | 0.363 | 10.98 | 0.82 |
| 30 (40) | person only | 0.241 | 0.177 | 0.400 | 8.90 | 0.73 |
| 30 | + prior λ = 1 | **0.301** | 0.334 | 0.357 | 6.95 | 0.50 |
| 30 | global only | 0.277 | 0.296 | 0.301 | 5.61 | 0.55 |
| 100 (130) | person only | 0.360 | 0.418 | 0.318 | 2.00 | 0.30 |
| 100 | + prior λ = 1 | 0.362 | 0.550 | 0.305 | 2.23 | 0.31 |
| 100 | global only | 0.322 | 0.460 | 0.253 | 1.24 | 0.24 |
| 200 (260) | person only | **0.398** | 0.587 | 0.302 | 0.97 | 0.21 |
| 200 | + prior λ = 1 | 0.377 | 0.643 | 0.295 | 1.62 | 0.25 |
| 200 | global only | 0.338 | 0.562 | 0.250 | 0.89 | 0.18 |

λ = 3, 10, 30 and 100 are within 0.01 F1 of λ = 1 at 10–30 days, and no better than it later
(`runs/drift_synth/eval_synth_prior_p0*.json`).

![Cold-start curves](figures/drift_synth_prior_curves.png)

(`./py scripts/eval/eval_synth_prior.py --stage plot`)

**F1 change from the prior (λ = 1), per person:**

| train days | p00 | p01 | p02 | p03 |
|---|---|---|---|---|
| 10 | +0.09 | +0.07 | +0.08 | −0.05 |
| 30 | +0.10 | +0.12 | +0.13 | −0.10 |
| 100 | −0.17 | +0.05 | +0.09 | +0.05 |
| 200 | −0.18 | +0.03 | +0.04 | +0.03 |

**What it says.**

- **The prior is a clear win at cold start.** At 10–30 days mean F1 rises by 0.05–0.06 and the
  healthy-trial FPR falls from 0.73 to 0.50. With 100+ days it is a wash on average.
- **λ hardly matters,** because at the strict α the detector asks only whether a transition is
  possible: any prior mass turns "never seen" into "allowed". At α = 1e-10 on p00 (200 days)
  the prior cuts the healthy-trial FPR from 0.16 to 0.03. It also removes every
  transition-channel catch of omission (29 → 0) and transposition (60 → 0): the person-only
  model was detecting them as "a transition this person never made".
- **Every loss is an operating-point failure.** In all four losing cells (p00 at 100 and 200 days,
  p03 at 10 and 30) the prior arm's validation pick jumped from α ≤ 1e-7 to α = 0.015–0.05,
  where alarms explode. Where the pick stayed strict, the prior helped in 12 of 12 cells.
  - **A split α** (transition channels picked separately from the rest, `harness.hindsight_split_grid`) recovers part of p00's loss: F1 0.22 → 0.29–0.30 at 100–200 days, still below person-only 0.40.
  - **Next:** a prior that fades with the person's own row count, paired with the split α.
- **Why persons differ:** p01 and p02 cook high-freedom recipes (stir fry 1.6×10¹⁰ orders, scrambled eggs 3.6×10⁶); p00 and p03 cook constrained ones. At 30 days, test transitions the person had never produced are 13% for p01/p02 and 6–7% for p00/p03. At 200 days they are 0.6–1.4% for all four, and 0.03–0.15% are unseen by the population too. This explains part of the pattern, but not p03's early loss.

### 9.3 Calibration against CaptainCook4D

`scripts/drift_synth/calibrate_cc4d.py` writes `runs/drift_synth/reports/cc4d_calibration.{md,json}`.
The real data are CC4D recordings with no Order Error or Missing Step tag: 5–13 per recipe.

- **Order variety matches uniform rollouts.** At equal sample size, the share of distinct orders is
  real 0.57–1.00 against generator 0.61–1.00. Real orders are about 25% closer to their medoid than
  uniform ones (normalized Kendall).
- **Habit:** real within-person distance 0.086 (91 pairs), between-person 0.104 (477 pairs). Habit 2
  gives 0.022 / 0.099, and the best grid fit is habit ≈ 0–0.25. Set aside for now: 2–3 lab repeats
  per person make the within-person figure mostly noise.
- **Real recordings break the task graphs.** For sautéed mushrooms 0 of 7 normal recordings follow its
  edges, tomato chutney 1 of 5, pan-fried tofu 5 of 9. People interleave and overlap steps (by
  annotated start time), which the graphs, and so the generator, cannot produce.
- **Durations are 4–5× short on every kind of step, not only waits.** The median recording has
  716 s of step time against 169 s for a synthetic session; steps with a `wait` token are 4.0×
  short, the rest 4.6×. The largest gaps are "cook for N minutes" steps the expansion gave no
  `wait` (pan-fried tofu, "cook 5 to 6 minutes": 327 s real, 8 s generated).
  - **Step-time calibration** (built, off by default): `calibrate_cc4d.py --write-scales` writes
    `configs/drift_synth/step_time_scale.json`, where each step's scale is the CC4D median step
    time divided by the uncalibrated generator's median.
    - Scales run from 2.8× (banana pancakes) to 9.1× (sautéed mushrooms), with a 4.15× global median.
    - The sampler multiplies every event's duration and following gap by its step's scale; logistics take their owning step's (`sampler.step_owners`). Single events are capped at 600 s.
    - Calibrated sessions: median 807 s against CC4D's 716 s of step time.
    - Enable with `generator.yaml: durations.step_scale` and `max_event_s`.
  - **The calibrated rerun** (`scripts/drift_synth/overnight_cal.sh`, `eval_synth_prior.py --variant cal`)
    uses 1 s ticks, d_max 300 (`configs/drift_synth/hsmm_cal.yaml`), training sizes 10 / 30 / 200
    days and 20 soft-EM iterations, to fit in a night. After iteration 20 the objective moved less
    than 0.2% in 13 of 16 uncalibrated fits.

## 10. Code

| Concept | Where |
|---|---|
| data types (Step, RecipeGraph, Variant, StepExpansion, Event, Session) | `drift_synth/types.py` |
| task graph loading, validation, topological-order counting | `drift_synth/graphs.py:load_task_graphs`, `validate_graph`, `linear_extension_counter`, `count_linear_extensions` |
| step phases (LLM, cached) | `drift_synth/graphs.py:assign_phase`, `load_phased`; `configs/drift_synth/phases/` |
| CC4D word → EPIC class table (pre-check estimate only) | `drift_synth/lexicon.py:NOUNS`, `VERBS`, `nouns_in` |
| LLM calls with a JSON schema | `drift_synth/llm.py:client_for`, `ask_json`; `llm/client.py:ChatClient(response_format=)` |
| step expansions (two passes, per-recipe sour sites), validation, EPIC exemplars | `drift_synth/expand.py:expand_step`, `sour_sites`, `validate_expansion`, `ExemplarIndex`, `build_expansions`; `configs/drift_synth/expansions/` |
| tags | `drift_synth/tagging.py:propose_tags`, `write_tags`, `load_tags` (refuses unreviewed); `configs/drift_synth/tags.yaml` |
| assets bundle, expansion I/O | `drift_synth/assets.py:Assets`, `load_assets`, `save_expansions`, `load_expansions` |
| person routine (habit) | `drift_synth/sampler.py:make_person` (`habit_orders`), `knobs.sample_topological_order(preferred, habit)`; `generator.yaml: person.habit` |
| knobs: variants, step order, duration scales, repetition, tidy rates | `drift_synth/knobs.py:apply_variants`, `sample_topological_order`, `slow_multiplier`, `token_multiplier`, `repetition_prob`, `tidy_rates`; `configs/drift_synth/knobs.yaml` |
| take / put / wash | `drift_synth/kitchen_state.py:insert_logistics` |
| durations and gaps | `drift_synth/durations.py:fit_duration_priors`, `fit_from_intervals`, `sample_duration`, `sample_gap` |
| day noise, speed, recognizer noise | `drift_synth/noise.py:theta_for_day`, `day_speed_factor`, `recognizer_noise` |
| recipe schedule, mix shift, regimes | `drift_synth/schedule.py:recipes_for_day`, `mix_weights`, `theta_at` |
| one session, one person | `drift_synth/sampler.py:sample_session`, `make_person`, `generate_person` |
| export | `drift_synth/export.py:export`, `to_ticks` (reuses `parse_epic.to_frame_segments`, `expand_verb_noun_to_ticks`) |
| observable ground truth | `drift_synth/ground_truth.py:observable_stats`, `drift_magnitude`, `count_z`, `affected_features` |
| pre-check report | `scripts/drift_synth/precheck.py` |
| offline assets (stages 1–3) | `scripts/drift_synth/build_assets.py --stage phases\|tags\|expansions` |
| generation (stages 4–9) | `scripts/drift_synth/generate.py`; `configs/drift_synth/generator.yaml`, `conditions.yaml` |
| sanity plots | `scripts/drift_synth/sanity_plots.py` |
| global prior + fine-tune (§9.2) | `hsmm/train/global_prior.py:count_global`, `apply_global_prior` (fixed or fading: `person_counts`, `n0`), `state_pairs`; `eval_synth_prior.py --stage eval-split`; `scripts/eval/eval_synth_prior.py`; `harness.hindsight_split_grid`, `split_op` |
| CC4D calibration (§9.3) | `scripts/drift_synth/calibrate_cc4d.py` (`--write-scales`); `sampler.step_owners`; `assets.load_step_scale`; `configs/drift_synth/step_time_scale.json`; `scripts/drift_synth/overnight_cal.sh` |
| anomaly detector on synthetic pools (§9) | `scripts/eval/eval_synth_ad.py` (`--train-days`, `--habit`, `--noise-scale`); `configs/drift_synth/hsmm.yaml`; repetition spans from `sampler.sample_session` (`meta.repeats`); `export.step_labels` |
| tests | `tests/test_drift_synth_graphs.py`, `tests/test_drift_synth_generator.py`, `tests/test_drift_synth_expand.py`, `tests/_drift_fixtures.py` |
