# Data: corpora, ticks, splits, pools, injection

Everything a detector reads is an integer (verb, noun) pair per tick. This doc covers how the three
corpora become those streams, how they are split, and the fixed **pools** every detector is scored
on: Breakfast with injected errors, EPIC-Tent with annotated ones.

---

## 1. The corpora

| | Breakfast | EPIC-KITCHENS-100 | EPIC-Tent |
|---|---|---|---|
| what | 10 scripted breakfast recipes | unscripted kitchen visits | one person assembling a tent, 29 sessions |
| trials | 503 (participant, recipe) pairs | 357 of 633 sessions (2–15 min) | 29 |
| verbs / nouns (with SIL) | 15 / 36 | 98 / 306 | 8 / 12 |
| tick | 1.0 s | 0.5 s | 1.0 s |
| K / K_R | 64 / 16 (or 10 dishes) | 64 / 16 | 16 / 1 |
| D_max | 200 | 200 (= 100 s) | 200 |
| anomalies | injected (§6) | none | **annotated** by two observers (§3.4) |
| driver | `data/parse_breakfast.py` | `data/parse_epic.py` | `data/parse_epic_tent.py` |
| config | `configs/breakfast.yaml` | `configs/epic.yaml` | `configs/epic_tent.yaml` |

Unqualified numbers in these docs are Breakfast. EPIC-KITCHENS is ingested and has fitted models
from the vocabulary-embedding work ([`logbook.md`](logbook.md)), but no current evaluation runs on
it.

Every driver writes the same files to `dataset/processed/<corpus>/`, so every script works on any
corpus:

```
sequences.json   [{"trial_id", "verb_ids": [T], "noun_ids": [T], "terminal_idle_ticks"}]
labels.json      [{"trial_id", "recipe_label", "subtask_labels": [T], ...}]    VALIDATION ONLY
vocab.json       {"verbs": {name: id}, "nouns": {...}, "recipes": {...}}
split*.json      {"seed", "test_frac", "train_trial_ids", "test_trial_ids"}
```

Both lists are ordered by sorted `trial_id`, so they zip without a join.

**`labels.json` is never fed to a likelihood.** The model is fitted unsupervised on the integer
streams; labels exist to score clusters, to build the `with-recipes` LLM prompt, and to pin dishes
in the `dish` training preset. Keeping them in a separate file is the structural guarantee.

---

## 2. From annotations to ticks

**Breakfast deduplication.** The same trial is recorded by up to five cameras, so 1,712 files
describe 503 trials. `find_trial_files` groups by (participant, recipe) and picks one view by a
fixed camera priority, logging any trial whose views disagree. `_check_against_config` asserts
the trial, vocabulary and recipe counts and raises on a mismatch.

**SIL and the verb/noun split.** A label `pour_milk` splits on the first underscore. SIL ("nothing
labelled is happening") maps to a configured sentinel pair, `stall kitchen` on Breakfast, so the
product emission is defined on every tick and idling is an ordinary state. SIL covers 7.3% of
Breakfast frames. Vocabulary ids are the
sorted token sets, so they are a deterministic function of the corpus.

**Tick expansion** (`tick_expansion.expand_verb_noun_to_ticks`). A tick's value is the plurality
(verb, noun) **pair** over its frames. The pair is voted on jointly, not verb and noun
separately, so every emitted observation is one the annotator actually wrote. `labels.json` is
binned the same way, so it aligns with `sequences.json` index for index.

**The trailing idle is trimmed** (`parse_breakfast.trim_terminal_idle`, both drivers; the count is
kept as `terminal_idle_ticks`). A trial's last segment is treated as censored, and censoring
imputation pushes a state that is *always* last toward ever longer durations with nothing pulling
it back: the terminal-idle state had converged to a mean of 734 ticks against a median observed
run of 5. Trimming removes 4.5% of Breakfast ticks and empties no trial.

**EPIC-KITCHENS differences** (`parse_epic.py`, annotations only):

- *A trial is a visit*, which may hold several goals; the 2–15 minute band keeps plausibly
  single-purpose ones.
- *Narrations overlap* (28% of consecutive pairs). `resolve_overlaps` is first-come-wins. Letting
  the later one interrupt would make `A B A`, a self-transition the model cannot represent.
- *fps varies per video*, so timestamps are the source of truth, re-quantised on a 20 fps grid.
- *There are no recipe labels*. `derive_dish_labels` takes each session's top TF-IDF food noun as a
  proxy, with `unknown` for the 33 sessions that mention none. It is partly circular, since it is
  derived from the same observations the recipe clustering reads.
- *0.5 s ticks*: at 1 s, 25% of narrations would vanish (1.9% at 0.5 s).
- EPIC's own class ids are kept; `build_vocab` raises on a name collision (`kitchen` is a real EPIC
  noun, so the SIL noun there is `idle`).

---

## 3. EPIC-Tent

29 sessions of one task. It is the only corpus whose anomalies nobody placed. Annotations only:
`scripts/data/fetch_epic_tent.py` downloads ~16 MB and asserts the published row counts (the
166 GB video release is not needed).

### 3.1 One task, so one recipe

`recipe_label` is the constant `"tent"` and K_R = 1. The recipe layer is a no-op by construction:
joint EM converges, and the recipe channel is off by the one-recipe rule
([`detectors.md`](detectors.md) §2.3).

### 3.2 Actions are the states

The 12 action classes factor injectively into 12 (verb, noun) pairs, so the lexical warm start
hands over the state space and the fitted model reaches per-tick subtask ARI 1.000. K = 16 leaves
room to split a state with a bimodal duration.

### 3.3 Overlaps cost the interrupter

10.5% of consecutive segments overlap, because instruction reading interrupts other actions.
First-come-wins drops 127 of 1,252 segments, 79 of them instruction reads. Gaps are 22% of
wall-clock time, so SIL is a substantial state here.

### 3.4 `anomalies.json`, and what is observable

Nine annotated error types, mapped onto the repo's names. `motor` and `failure` (physical mishaps)
map to nothing.

| EPIC-Tent | repo type | n | spans that leave the (verb, noun) stream unchanged |
|---|---|---|---|
| `misuse` | substitution | 164 | 94% |
| `order` | transposition | 136 | 90% |
| `search` + `slow` | stall | 40 | 80% |
| `omit` | omission | 16 | 44% (onset tick only) |
| `correction` | correction | 8 | 75% |
| `repeat` | repetition | 3 | |

Most errors **do not touch the observation stream**. `misuse` is annotated inside a correctly
labelled action (the wrong guyline during `place guyline`), so the stream is identical with or
without it. This is a fact about the representation, and why `misuse` is in no scored group
(§3.8). `order` is the informative contrast: the pair is unchanged, but its *position* is wrong,
which a transition model can see.

Per record:

- the position comes from the error's own frames, never from a join to an action segment;
- `omit` collapses to its onset tick ("omission is the absence of an action");
- there is no `tick_map`: nothing was spliced, and a real error has no healthy twin.

### 3.5 Healthy fragments and the excised training corpus

**No session is error-free.** The healthy population is therefore each session's maximal
error-free intervals of ≥ 10 ticks (`clean_fragments`: 170 fragments, 73% of ticks). A fragment's
flags are sliced out of the whole session's decode, never re-decoded: the belief part-way through
an assembly is not the start-of-trial prior.

`--exclude-anomalies` writes those fragments as `sequences_clean.json` / `labels_clean.json`, one
sequence per fragment. The HSMM is fitted **only** on them: a model fitted on whole sessions learns
the errors as normal and alarms less on both counts (measured under the retired metric D, all six
mapped types: F1 0.316 → 0.371 and order recall 0.300 → 0.425 after excision). Fragments are cut out, not masked, because the training code reads a
prefix mask as the sequence length. `split_clean*.json` inherit session membership, so a fragment
never lands opposite its session.

### 3.6 The prescribed order

`order` errors are annotated against the printed instruction sheet, not against what people did.
`CANONICAL_ORDER` records it (median onset per action, agreeing with the paper's Table 1;
`read instruction` interleaves freely and is excluded). A first-order transition model cannot
represent this reference: the transition into an out-of-order segment has median corpus count 23
against a 22 baseline. The LLM's EPIC-Tent `with-recipes` prompt states the order instead.

### 3.7 PREGO's split

PREGO's split is recovered exactly from their supplementary rule: the 7 unrated sessions plus
those with median confidence < 0.6 are test. That gives 14 train / 15 test, and the 15 test ids
(1–7, 10, 16, 17, 18, 20, 23, 25, 28) equal the keys of PREGO's own prediction file. The nearest
sessions to the threshold are 0.583 and 0.627. `scripts/data/epic_tent_splits.py` writes
`split_prego.json`. The default split is a seeded 20 / 9 (`split.json`).

### 3.8 Error groups

Results are reported per group, each scored with every other type **removed**, i.e. relabelled
normal, which is what PREGO's scorer does:

| group | labels |
|---|---|
| `procedural` | order, omit, correction, repeat (PREGO's subset: 163 instances) |
| `procedural+temporal` | + search, slow |

An out-of-group span leaves the ground truth and its ticks rejoin the healthy population, so the
healthy fragments are recomputed per group (`eval/pools.group_truth`).

### 3.9 Errors inside a deleted action

When overlap resolution swallows an action, an error committed during it has nothing left in the
stream (p20 ties the top for 28 s inside a 119 s `insert support`). `host_swallowed` moves those
15 records to `unobservable_records`, so no arm is charged for a miss preprocessing made. They
stay excised from training. Omissions and errors between actions are not affected.

---

## 4. Splits

`scripts/data/split_dataset.py` writes `split.json` (inputs and outputs, so it is reproducible from
the file). `data/split.py` reads it; an unknown part raises rather than returning nothing.

- Breakfast: seeded 80/20 by trial (402 train / 101 test). The same participant can be on both
  sides: this measures generalisation to new trials, not new people.
- `scripts/data/make_dev_split.py` carves a nested fit/dev split out of train for selecting
  regularisation ([`training.md`](training.md) §3), so the outer test split is never touched.
- EPIC-Tent: `split.json`, `split_prego.json` and their `split_clean*.json` versions (§3.5, §3.7).

---

## 5. Canonical pools

Every detector is scored on fixed files, written once by `scripts/data/build_pools.py`.

**Breakfast** (`dataset/processed/breakfast/pools/{train,test}.pkl`). For each trial of the split:

1. decode it with the **placement model** `joint_params.npz` (`trajectories_from_real_joint`);
2. drop it if it has fewer than `MIN_SEGMENTS` = 4 segments;
3. inject all five error types into it in turn, from one rng (seed 0), before touching the next
   trial (`build_pool`).

Each entry is `{"trial_id", "healthy", "injected": {type: trajectory}}`. The train pool has 378
usable trials, the test pool 97. Because each trial is finished before the next is started, a pool
built on a prefix of the split is a prefix of the full pool (`tests/test_pools.py`).

One placement model for every scored model matters: injections land in the placement model's
segmentation, and if each scored model placed its own, models would be graded on different
streams. **The test pool is bitwise the pool the cached LLM answers were collected on**, so every
row of the headline table (HSMM, LLM and the nulls) is scored on the same trials. The pool's `meta`
records this.

**EPIC-Tent** (`dataset/processed/epic_tent/pools/{split,split_prego}_test.pkl`): the test
sessions and their annotation records, frozen. Nothing is sampled.

`eval/pools.injected_per_trial` builds the same kind of pool on the fly, for EPIC-Tent's injected
control arm and the trial figure.

---

## 6. Injected errors

`synthetic/error_injection.py`. Each injector edits one healthy decoded trajectory and returns the
new streams, a `window`, the anomalous `anomaly_ticks`, the scored `spans`, and provenance
(`tick_map`, `edited_ticks`).

| Type | Edit | Span scored by metric G |
|---|---|---|
| substitution | one interior segment's verb **or** noun (chosen at random) → a token with no training evidence for that state | the segment |
| abandonment | truncate an interior segment to 5–20% of its length | the onset tick |
| omission | delete an interior segment | the segment now following the gap |
| transposition | swap two adjacent interior segments | **two** spans, one per moved segment |
| repetition | duplicate an interior segment in place | the inserted copy |

- **Interior only.** Every injector skips the first and last segment (Breakfast trials open and
  close on idle), and abandonment needs d ≥ 2. Before this rule, 3.2% of abandonments were no-ops
  and 21% landed on the trailing idle.
- **Substitution's token** is drawn uniformly from the tokens whose raw training count for that
  state is within 5% of the row's floor (`SUBSTITUTION_UNSEEN_BAND`), rather than the single token
  the scored model ranks worst. `select="near"` instead uses the nearest embedding neighbour.
- **Abandonment's span is the onset only.** The trial continues after the cut, so a span to the
  end would make recall nearly free. The metric's 10-tick deadline admits the retrospective
  channels that see it.
- **Downstream disturbance is not labelled.** After `A B D C E` the `E` onset is also unexpected,
  but labelling it would make span width grow with how far the disturbance travels.
- **Repetition is ambiguous.** The copy either creates a re-entry the transition channel sees, or
  the decode merges it into one over-long segment the stall channel sees.

`synthetic/generate.py` also samples healthy trials from a fitted model (`sample_trajectory_joint`,
`generate_healthy_joint`); the tests use these, the evaluation does not.

---

## 7. Code

| Concept | Where |
|---|---|
| Breakfast parsing, dedupe, trailing idle | `data/parse_breakfast.py:find_trial_files`, `build_dataset`, `trim_terminal_idle` |
| frames → ticks | `data/tick_expansion.py:expand_verb_noun_to_ticks`; `data/labels.py:build_trial_labels` |
| EPIC-KITCHENS | `data/parse_epic.py:resolve_overlaps`, `to_frame_segments`, `derive_dish_labels`, `build_vocab` |
| EPIC-Tent | `data/parse_epic_tent.py:build_dataset`, `load_error_labels`, `build_truth`, `clean_fragments`, `excise`, `host_swallowed` |
| fetching EPIC-Tent | `scripts/data/fetch_epic_tent.py` |
| splits | `scripts/data/split_dataset.py`, `make_dev_split.py`, `epic_tent_splits.py`; `data/split.py:load_split`, `filter_sequences` |
| canonical pools | `scripts/data/build_pools.py`; `eval/pools.py:build_pool`, `save_pool`, `load_pool`, `to_streams` |
| EPIC-Tent pools and groups | `eval/pools.py:freeze_epic_tent_pool`, `load_epic_tent_pool`, `group_truth`, `slice_flags`, `trim_at_first_mistake` |
| decoding real trials | `synthetic/generate.py:trajectories_from_real_joint` |
| injection | `synthetic/error_injection.py:inject`, `inject_substitution`, `inject_abandonment`, `inject_omission`, `inject_transposition`, `inject_repetition` |
| tests | `tests/test_parse_breakfast.py`, `test_parse_epic.py`, `test_parse_epic_tent.py`, `test_split.py`, `test_pools.py` |
