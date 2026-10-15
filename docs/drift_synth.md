# Synthetic drift data from CaptainCook4D

The drift detector in [`next_steps.md`](next_steps.md) §2 needs streams where a person's normal
behaviour changes by a known amount at a known time. No public dataset has that, so this generator
builds it: multi-week, per-person streams of (verb, noun) actions in the EPIC-100 vocabulary, with
ground truth for onset, rate, size and the affected observable features.

It is a sanity and validity test, not a benchmark. Control matters more than realism, with one
exception: the output must be at the **logistics level** the HSMM is trained on (`take bowl`,
`pour water`), not the recipe-step level (`make roux`).

**Status (2026-10-15):** 18 recipes approved. Stage 1 phases are built and committed.
Stage 2 tags are proposed and **waiting for Jack's review** (`configs/drift_synth/tags.yaml`,
`reviewed: false`). Stage 3 (expansions) is not built. Stages 4–10 are built and tested on a toy
world, but cannot run on the real recipes until Stages 2–3 are done.

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
| 3 expansion | LLM expands each step into core tokens and knob variants; validated, retried 3×, cached | none |
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

## 8. Code

| Concept | Where |
|---|---|
| data types (Step, RecipeGraph, Variant, StepExpansion, Event, Session) | `drift_synth/types.py` |
| task graph loading, validation, topological-order counting | `drift_synth/graphs.py:load_task_graphs`, `validate_graph`, `linear_extension_counter`, `count_linear_extensions` |
| step phases (LLM, cached) | `drift_synth/graphs.py:assign_phase`, `load_phased`; `configs/drift_synth/phases/` |
| CC4D word → EPIC class table (pre-check estimate only) | `drift_synth/lexicon.py:NOUNS`, `VERBS`, `nouns_in` |
| LLM calls with a JSON schema | `drift_synth/llm.py:client_for`, `ask_json`; `llm/client.py:ChatClient(response_format=)` |
| tags | `drift_synth/tagging.py:propose_tags`, `write_tags`, `load_tags` (refuses unreviewed); `configs/drift_synth/tags.yaml` |
| assets bundle, expansion I/O | `drift_synth/assets.py:Assets`, `load_assets`, `save_expansions`, `load_expansions` |
| knobs: variants, step order, duration scales, repetition, tidy rates | `drift_synth/knobs.py:apply_variants`, `sample_topological_order`, `slow_multiplier`, `token_multiplier`, `repetition_prob`, `tidy_rates`; `configs/drift_synth/knobs.yaml` |
| take / put / wash | `drift_synth/kitchen_state.py:insert_logistics` |
| durations and gaps | `drift_synth/durations.py:fit_duration_priors`, `fit_from_intervals`, `sample_duration`, `sample_gap` |
| day noise, speed, recognizer noise | `drift_synth/noise.py:theta_for_day`, `day_speed_factor`, `recognizer_noise` |
| recipe schedule, mix shift, regimes | `drift_synth/schedule.py:recipes_for_day`, `mix_weights`, `theta_at` |
| one session, one person | `drift_synth/sampler.py:sample_session`, `make_person`, `generate_person` |
| export | `drift_synth/export.py:export`, `to_ticks` (reuses `parse_epic.to_frame_segments`, `expand_verb_noun_to_ticks`) |
| observable ground truth | `drift_synth/ground_truth.py:observable_stats`, `drift_magnitude`, `count_z`, `affected_features` |
| pre-check report | `scripts/drift_synth/precheck.py` |
| offline assets (stages 1–3) | `scripts/drift_synth/build_assets.py --stage phases\|tags` |
| generation (stages 4–9) | `scripts/drift_synth/generate.py`; `configs/drift_synth/generator.yaml`, `conditions.yaml` |
| tests | `tests/test_drift_synth_graphs.py`, `tests/test_drift_synth_generator.py`, `tests/_drift_fixtures.py` |
