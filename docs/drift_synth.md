# Synthetic drift data from CaptainCook4D

The drift detector in [`next_steps.md`](next_steps.md) §2 needs streams where a person's normal
behaviour changes by a known amount at a known time. No public dataset has that, so this generator
builds it: multi-week, per-person streams of (verb, noun) actions in the EPIC-100 vocabulary, with
ground truth for onset, rate, size and the affected observable features.

It is a sanity and validity test, not a benchmark. Control matters more than realism, with one
exception: the output must be at the **logistics level** the HSMM is trained on (`take bowl`,
`pour water`), not the recipe-step level (`make roux`).

**Status (2026-10-15):** pre-checks done (§5), waiting on recipe approval. Stages 1–10 are not
built yet.

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

**Step order.** A softmax over the steps available at each point is *not* uniform over
topological orders. Each candidate is therefore weighted by the number of orders that complete it
(`graphs.linear_extension_counter`, a DP over order ideals), times exp(β(2θ − 1)·is_prep). At
θ = 0.5 the bias vanishes and the order is exactly uniform over the graph's topological orders.

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

## 5. Pre-check findings

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
- **Proposed set** (18 recipes, every knob covered by ≥ 6): scrambled eggs, herb omelet, tomato
  mozzarella salad, caprese bruschetta, cucumber raita, butter corn cup, sauteed mushrooms,
  broccoli stir fry, zoodles, pan-fried tofu, tomato chutney, microwave egg sandwich, microwave
  French toast, breakfast burritos, spiced hot chocolate, cheese pimiento, banana pancakes, coffee.

## 6. Code

| Concept | Where |
|---|---|
| data types (Step, RecipeGraph, Variant, StepExpansion, Event, Session) | `drift_synth/types.py` |
| task graph loading, validation, topological-order counting | `drift_synth/graphs.py:load_task_graphs`, `validate_graph`, `linear_extension_counter`, `count_linear_extensions` |
| CC4D word → EPIC class table (pre-check estimate only) | `drift_synth/lexicon.py:NOUNS`, `VERBS`, `nouns_in` |
| pre-check report | `scripts/drift_synth/precheck.py` |
| tests | `tests/test_drift_synth_graphs.py` |
