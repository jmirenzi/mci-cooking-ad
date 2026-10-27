# Synthetic drift data from CaptainCook4D

The drift detector in [`next_steps.md`](next_steps.md) §2 needs streams where a person's normal
behaviour changes by a known amount at a known time. No public dataset has that, so this generator
builds it: per-person streams of cooking sessions, (verb, noun) actions in the EPIC-100
vocabulary, with ground truth for onset, rate, size and the affected observable features. The
same generator, with no drift, is the synthetic test bed for the anomaly detector (§7).

It is a sanity and validity test, not a benchmark. Control matters more than realism, with one
exception: the output must be at the **logistics level** the HSMM is trained on (`take bowl`,
`pour water`), not the recipe-step level (`make roux`). Time is the session index: one recipe a
session, no days (a day layer, e.g. mealtimes, can go on top later). How it got here is in
[`logbook.md`](logbook.md) (2026-10-15 onward).

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
| 2 tags | LLM proposes tags per vocab item (`sour`, `fat`, `sweet`, `salty` (salt only, by hand), `fine_motor`, `heat`, `cleaning`, `ingredient`, `tool`, `container`); Jack reviews | none |
| 3 expansion | LLM expands each step into core tokens (pass 1), then a lower-fat variant for steps whose tokens use a fat noun (pass 2), then one call per recipe for ≤ 2 sour sites; the flavour pass adds 1–2 sweet / salty / fatty insert sites per recipe (LLM) and an omit for each core token with the knob's nouns (code); validated, retried 3×, cached | none |
| 4 kitchen state | `take` before first use, `put` after last use, `wash` after the last use of a tool or container (rates from `tidy`); a tool or container's last use is the last token of the phase of its last named use (`add garlic` happens in the pan); never for food made in the session (`kitchen_state.PREPARED`), and no put-back straight after the recipe's own `put` | seeded |
| 5 durations | step totals: a log-normal per CC4D step (median of its normal recordings, one pooled log-sd 0.38; `durations.StepTimes`); within a step each action's share is its EPIC log-normal draw (per verb, (verb, noun) backoff below 20 samples) over the step's expected EPIC total as written; take/put/wash: EPIC per verb; no gaps; every draw's log-sd × 0.5 with the mean kept (`durations.narrowed`, `generator.yaml: durations.sigma_scale`) | seeded |
| 6 knobs | θ ∈ [0, 1] per knob drives variants, step order, logistics and duration scales | seeded |
| 7 nuisance | per-session θ noise (logit space), AR(1) speed over sessions, one recipe per session from the mix weights, mix shift; recognizer noise (off) | seeded |
| 8 drift | θ(n) under none / sudden / incremental / gradual / recurring, n = session index | seeded |
| 9 sample, export | one session = order → variants (additions spaced off their twins, `knobs.space_inserts`) → logistics → back-to-back identical actions merged into one → repetition → durations → noise | seeded |
| 10 ground truth | Monte Carlo observable statistics at θ0 and θ1; JS and W1 per recipe | seeded |

**Knobs.**

| knob | family | acts on |
|---|---|---|
| `sour` | preference (flavour) | adds or drops lemon, lime, vinegar |
| `sweet` | preference (flavour) | adds or drops sugar, syrup |
| `salty` | preference (flavour) | adds or drops salt |
| `fatty` | preference (flavour) | adds butter, oil, cheese, cream; or the LLM's lower-fat substitute/omit; fat-tagged durations × exp(0.7(θ − 0.5)) |
| `mise_en_place` | preference | step order: prep before cook |
| `tidy` | preference | put-back and wash rates |
| `slow` | capability | every duration × exp(a(θ − 0.5)) |
| `fine_motor` | capability | duration and repetition probability of fine-motor-tagged verbs |

**Knob parameters** (`configs/drift_synth/knobs.yaml`): `tidy` puts back with P from 0.1 to
0.95 and washes tools and containers with P from 0 to 0.8, linear in θ. `slow` and `fine_motor` use
a = 1 (θ from 0 to 1 scales duration by 0.61 to 1.65). `fine_motor` repeats a token with
P = 0.6·max(0, θ − 0.5), after a pause of at least 2 s, and session noise never carries its θ across
0.5, so a healthy person (θ0 < 0.5) never repeats. `mise_en_place` uses β = 4. A capability
knob's high θ is the impaired end.

**Flavour knobs are two-sided around θ = 0.5** (`knobs.variant_prob`). Above it, the dish gets the
flavour added with P = 2(θ − 0.5)·affinity[recipe][knob], once per session, at one of its 1–2
insert sites picked uniformly (`knobs.pick_inserts`). Below it, each `substitute` or `omit` applies
with P = 2(0.5 − θ). At 0.5 the recipe is cooked as written. Affinity
(`configs/drift_synth/affinity.yaml`: rare 0.1, sometimes 0.4, common 1.0) is the chance that a
flavour-lover (θ = 1) changes this dish in a session, so lime in hot chocolate stays possible but
rare. Every addition is `add <ingredient>`, and so is the recipe's own salt. An addition that would land next to
the same action (more salt straight after the recipe's salt) moves to the end of the nearest step
where it does not, so it reads as a second addition, never as one action twice.

**No action appears twice in a row** in a clean stream except a `fine_motor` repetition. Besides
spacing additions: a put-back right after the recipe's own `put` of the same thing is dropped, and
any remaining identical pair is emitted as one event (`meta.merged`; in practice the burritos'
two sauces, which are both EPIC `sauce`, in about 57% of sessions). Every duration draw is
narrowed (log-sd × 0.5, same mean): EPIC's and CC4D's spreads mix many people, and at full width
one person's `take banana` varied 13× between sessions. Removal ignores
affinity: an ingredient the recipe has can always be left out.

**Step order.** A softmax over the steps available at each point is *not* uniform over
topological orders. Each candidate is therefore weighted by the number of orders that complete it
(`graphs.linear_extension_counter`, a DP over order ideals), times exp(β(2θ − 1)·is_prep). At
θ = 0.5 the bias vanishes and the order is exactly uniform over the graph's topological orders.

**Drift regimes** (`schedule.theta_at`), in sessions: `none`; `sudden` (θ1 from `onset`);
`incremental` (linear over `duration` sessions; every duration knob is exp(a(θ − 0.5)), so linear
in θ is linear in log duration); `gradual` (each session θ1 with P = the same ramp, else θ0);
`recurring` (θ1 and θ0 alternate in `period`-session blocks from onset). Session noise is then
added in logit space (`noise.theta_for_session`), and every duration is multiplied by a speed that
is an AR(1) over sessions (`noise.speed_factor`).

**Durations** (`sampler.sample_session`, `durations.py`). One source per quantity. A step's total
is drawn from CC4D (a log-normal per step, its normal recordings' median and a log-sd pooled over
all 88 steps, 0.38; `calibrate_cc4d.py --write-step-times` writes
`configs/drift_synth/step_times.json`). Each of the step's actions lasts its EPIC draw times
(the step's CC4D draw) / (the step's written actions' expected EPIC time), so an added action
(salt for a salty cook) adds its own share and a removed one takes its share away. Take, put and
wash keep EPIC time and are added on top: in CC4D the time between steps (median 15–190 s a
recipe) is far more than they need (10–25 s). Then speed, `slow` and the per-token `fine_motor`
and `fatty` factors multiply; one event is capped at 600 s. There are no gaps between actions,
except a fine-motor repeat's ≥ 2 s pause (which idle scrubbing then removes, merging the repeat
into one longer run: open).

**Recognizer noise** (`noise.recognizer_noise`; off: every rate is 0 in `generator.yaml`, the
code path is kept). One categorical draw per clean event: delete,
swap the verb, swap the noun, or keep. Then, independently, a spurious token takes the event's
tail. Confusions come from the session's own tokens, so they stay in the vocabulary. Every rate is
per clean event, as configured.

**Random streams** (`sampler.rng_for`). A person's traits come from (seed, person); session n's
state (θ noise, speed, recipe) from (seed, person, n) and its content from another stream on the
same keys. None of them includes the condition. Conditions with the same seed therefore share every session before onset
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
- **Step totals**: CC4D step annotations (`annotation_json/complete_step_annotations.json`),
  normal recordings only (3–11 a step). CC4D's advertised fine-grained action annotations are not
  public; its finest public unit is the step (median ~30–40 s).
- **Take/put/wash durations, the split within a step, and few-shot exemplars**: EPIC-100 raw
  narrations (`dataset/epic_kitchens/annotations`, loaded by `data/parse_epic.py:load_narrations`).
  Of the 171 distinct recipe tokens, 32 have their own (verb, noun) row, 135 a verb row and 4 the
  global row; take/put/wash have 8–17k samples each.
- **Vocabulary**: EPIC-100's 97 verb and 305 noun classes plus SIL (`stall`, `idle`). Verbs and
  nouns are separate classes.

## 4. Output

Per person, under `dataset/processed/drift_synth/<condition>/seed<k>/<person>/`:

- `sessions/<trial_id>.json`: timestamped string events, the session's θ, and the clean events
  before recognizer noise;
- `sequences.json`, `labels.json`, `vocab.json`: the loader format of every other corpus
  ([`data.md`](data.md) §1). These are EPIC ids at `export.tick_seconds` (1 s) with any gap as
  (`stall`, `idle`), `trial_id = <person>_s<nnnn>` (labels carry `session` = n), and
  `recipe_label` set to the CC4D recipe;
- `ground_truth.json`: condition, regime (in sessions), moving knobs, θ0, θ1, per-session θ, speed
  and recipe, and mix-shift parameters.

## 5. Conditions

`configs/drift_synth/conditions.yaml`, expanded by `scripts/drift_synth/generate.py:conditions`
into 37 conditions × 3 seeds × 8 persons × 90 sessions, with onset at session 36 and ramps of 27
sessions:

- `null` and `mix_shift_only`;
- every knob × {sudden, incremental, gradual}, with |Δθ| = 0.4 (`fine_motor` + 0.6, so every healthy θ0 < 0.4 ends above its 0.5 repetition hinge);
- `pref_plus_capability` (sour and slow together);
- `sour` and `slow` sweeps at |Δθ| ∈ {0.05, 0.1, 0.2, 0.3, 0.45}, sudden.

θ0 ~ U(0.25, 0.75) per knob. θ1 moves up if that fits in [0, 1], otherwise down.

## 6. Tests

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
9. `sudden` switches exactly at onset, and `gradual` tracks its ramp within 0.04 per 7-session block.

It also checks that pre-onset sessions are identical between `null` and a drift condition, that
step totals come from CC4D and logistics from EPIC with no gaps, and that the session index is
the only clock. `tests/test_drift_synth_inject.py` covers the step-level injectors.

## 7. The anomaly detector on this data

`scripts/eval/eval_synth_prior.py`, end to end in `scripts/drift_synth/prior_experiment.sh`
(about 4 h). Results go to `runs/drift_synth/prior/`; the numbers are in [`logbook.md`](logbook.md).

**People.**
- **Population** (the prior): 60 people (seed 1000, never test subjects), 40 sessions each,
  preference knobs θ0 ~ U(0.05, 0.95), capability U(0.4, 0.6), `fine_motor` U(0, 0.4), habit
  U(0, 1).
- **Test persons:** seed 0, persons 0–3, habit 2, the same capability ranges, 400 sessions each.
  They train on their first 10, 30 or 250 sessions; sessions 250–324 validate (α and κ are
  picked there by best F1) and 325–399 test.

**Streams.** Recognizer noise off, 1 s ticks, and every stream idle-scrubbed before the model
(`data/scrub.py`): "A, pause, A" merges into one run, and `scrub_kept.json` keeps each kept
tick's original tick. Without scrubbing, two actions in three were separated by an idle run, so
the model's transitions went A → idle → B and a skipped, moved or repeated step left no unusual
transition.

**Models.** Each test person's model is the training pipeline ([`training.md`](training.md),
stages 1–4b, `configs/drift_synth/hsmm.yaml`: K = 128, d_max 300, the recipe = the CC4D recipe).
Then the **global prior** is added post-fit (`hsmm/train/global_prior.py`):
- The population's tick streams are run-length counted per recipe into first-run and run-to-run
  transition counts keyed by (verb, noun). States are pairs (lexical init), so these counts are
  the prior with no population EM and project onto any person's states.
- Each transition and init row gets λ × the population's row for that recipe and pair, after the
  person's cells are lifted to count 1, so under the MAP mode a cell the person never produced is
  worth exactly λ·P_global. λ = 0 is the person-only model bit for bit; a transition nobody
  produced stays on the floor.
- The arms are person only (λ = 0) and λ = 1.
- A prior makes a never-seen transition rare rather than impossible, so Hindsight picks the
  transition channels' α apart from the rest (`harness.hindsight_split_grid`). Live picks α × κ.

**Pools: step-level injected errors** (`drift_synth/inject.py`). Each held-out session gets one
real error edited into its clean events, using the generator's step labels; every other event
keeps its duration. The stream is re-exported and scrubbed (`map_spans` moves the spans with the
kept ticks). Each error and its span:

| error | edit | span |
|---|---|---|
| omission | all recipe actions of one step dropped (its fetches stay) | from the step's old place to the next step's first action |
| transposition | a step, with its fetches and put-backs, moved to before a step it depends on in the task graph | the moved step's first action, and the jumped step's first action |
| repetition | a step's actions done again after the next step | the repeat's first action |
| abandonment | an action of ≥ 8 s cut to 5–20% (≥ 1.5 s) and the rest of its step skipped | the cut action |
| substitution | an action's verb or noun replaced by one the person never uses in that recipe | that action |


**Are they real errors?** The healthy-support oracle (`eval/synth_oracle.py`) reads ~3,600 more
healthy sessions of the same person (sessions no pool uses) and asks of an injected stream:
a run bigram never seen; a per-recipe run count outside its healthy range; a run length outside
the healthy 0.2–99.8% range; and, apart, an order of first occurrences no healthy session has.
`gen_pool` prints the share each type passes. Errors injected at the tick level into a model's
segmentation (the Breakfast injectors) were mostly not errors here (gaps, optional put-backs), which
is why these injectors edit steps.

**Open.**
- Scrubbing merges a `fine_motor` repeat ("A, pause, A") into one longer run, so a repetition
  drift would be visible only as duration.
- Dirichlet(1) recipe weights are uneven: a person can cook one recipe 60% of the time and
  another twice in 200 sessions, so per-recipe test counts vary a lot.
- Cold start: with 10 training sessions most healthy sessions still alarm.

## 8. Code

| Concept | Where |
|---|---|
| data types (Step, RecipeGraph, Variant, StepExpansion, Event, Session) | `drift_synth/types.py` |
| task graph loading, validation, topological-order counting | `drift_synth/graphs.py:load_task_graphs`, `validate_graph`, `linear_extension_counter`, `count_linear_extensions` |
| step phases (LLM, cached) | `drift_synth/graphs.py:assign_phase`, `load_phased`; `configs/drift_synth/phases/` |
| LLM calls with a JSON schema | `drift_synth/llm.py:client_for`, `ask_json`; `llm/client.py:ChatClient(response_format=)` |
| step expansions (two passes, per-recipe sour sites), validation, EPIC exemplars | `drift_synth/expand.py:expand_step`, `sour_sites`, `validate_expansion`, `ExemplarIndex`, `build_expansions`; `configs/drift_synth/expansions/` |
| tags | `drift_synth/tagging.py:propose_tags`, `write_tags`, `load_tags` (refuses unreviewed); `configs/drift_synth/tags.yaml` |
| flavour pass and affinity | `drift_synth/expand.py:add_flavour_variants`, `flavour_sites`, `code_omits`, `propose_affinity`; `build_assets.py --stage flavour\|affinity`; `assets.load_affinity` (refuses unreviewed); `configs/drift_synth/affinity.yaml`; `runs/drift_synth/reports/flavour_review.md` |
| assets bundle, expansion I/O | `drift_synth/assets.py:Assets`, `load_assets`, `save_expansions`, `load_expansions` |
| person routine (habit) | `drift_synth/sampler.py:make_person` (`habit_orders`), `knobs.sample_topological_order(preferred, habit)`; `generator.yaml: person.habit` |
| knobs: variants, step order, duration scales, repetition, tidy rates | `drift_synth/knobs.py:variant_prob`, `pick_inserts`, `apply_variants`, `variant_layout`, `space_inserts`, `sample_topological_order`, `slow_multiplier`, `token_multiplier`, `repetition_prob`, `keep_side_of_hinge`, `tidy_rates`; `configs/drift_synth/knobs.yaml` |
| take / put / wash | `drift_synth/kitchen_state.py:insert_logistics`, `PREPARED` |
| durations (EPIC actions, CC4D step totals) | `drift_synth/durations.py:fit_duration_priors`, `fit_from_intervals`, `narrowed`, `sample_duration`, `mean_duration`, `StepTimes`, `fit_step_times`, `sample_step_total`; `sampler.step_units`; `configs/drift_synth/step_times.json`; `assets.load_step_times` |
| session noise, speed, recognizer noise | `drift_synth/noise.py:theta_for_session`, `speed_factor`, `recognizer_noise` |
| recipe schedule, mix shift, regimes (in sessions) | `drift_synth/schedule.py:recipe_for_session`, `mix_weights`, `theta_at` |
| one session, one person | `drift_synth/sampler.py:sample_session`, `make_person`, `generate_person` |
| export | `drift_synth/export.py:export`, `to_ticks` (reuses `parse_epic.to_frame_segments`, `expand_verb_noun_to_ticks`) |
| observable ground truth | `drift_synth/ground_truth.py:observable_stats`, `drift_magnitude`, `count_z`, `affected_features` |
| offline assets (stages 1–3) | `scripts/drift_synth/build_assets.py --stage phases\|tags\|expansions\|flavour\|affinity` |
| generation (stages 4–9) | `scripts/drift_synth/generate.py`; `configs/drift_synth/generator.yaml`, `conditions.yaml` |
| sanity plots | `scripts/drift_synth/sanity_plots.py` |
| example rollouts (hand-set people × every recipe, for reading) | `scripts/drift_synth/rollout_examples.py` → `runs/drift_synth/reports/rollout_examples.md` |
| tests | `tests/test_drift_synth_graphs.py`, `test_drift_synth_generator.py`, `test_drift_synth_expand.py`, `test_drift_synth_inject.py`, `_drift_fixtures.py`; `test_scrub.py`, `test_global_prior.py` |
| CC word → EPIC class table (exemplar lookup) | `drift_synth/lexicon.py:NOUNS`, `VERBS`, `nouns_in` |
| CC4D step totals and the calibration check | `scripts/drift_synth/calibrate_cc4d.py --write-step-times configs/drift_synth/step_times.json`; `sampler.step_owners` |
| global prior + fine-tune (§7) | `hsmm/train/global_prior.py:count_global`, `apply_global_prior`, `state_pairs`; `harness.hindsight_split_grid` |
| idle scrubbing (§7) | `data/scrub.py:scrub_idle`, `map_spans`, `scrub_sequence`; `eval_synth_prior.py:_scrub_export` |
| step-level injectors, healthy-support oracle (§7) | `drift_synth/inject.py`; `eval/synth_oracle.py:HealthySupport`, `person_support`; `eval_synth_prior.py:gen_pool`; `harness.trials_from_groups` |
| the experiment (§7) | `scripts/eval/eval_synth_prior.py` (`--stage population\|persons\|train\|eval\|eval-live\|summary`, `--core-count`); `scripts/drift_synth/prior_experiment.sh`; `configs/drift_synth/hsmm.yaml` |
