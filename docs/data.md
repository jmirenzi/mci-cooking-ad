# `src/cook_ad/data/` — from raw annotations to integer tick sequences

**Job:** turn a corpus's frame- or timestamp-level text annotations into the three JSON
artifacts every downstream module consumes, and do it deterministically enough that the counts
can be asserted against the config.

Two corpora are ingested, through two drivers that emit the **same three files in the same
shapes**. That contract is the point: every runner, config path, eval script and figure works on
either corpus without a branch. §1–§5 describe the Breakfast driver; §6 applies to both; §7 is
EPIC and states only what differs; §8–§9 cover the configs and the train/test split.

| File | Role |
|---|---|
| `config.py` | 6-line YAML loader. Nothing else. |
| `tick_expansion.py` | frame-interval segments → per-tick values (majority vote) |
| `labels.py` | ground-truth per-tick action labels — **validation only** |
| `parse_breakfast.py` | Breakfast driver: parse → dedupe → vocab → tick-expand → write JSON |
| `parse_epic.py` | EPIC-KITCHENS-100 driver: same output contract, four different problems (§5) |
| `split.py` | reads a `split.json` and filters sequences/labels to one partition (§7) |

---

## 1. What the raw data looks like

Breakfast `segmentation_coarse` ships one `.txt` per *(participant, camera, recipe)*:

```
dataset/breakfast_actions/segmentation_coarse/cereals/P03_cam01_P03_cereals.txt
```

whose contents are frame intervals with a coarse action label:

```
1-30 SIL
31-150 take_bowl
151-428 pour_cereals
...
```

Two facts drive the whole module:

1. **Frames are 1-indexed, inclusive, and contiguous.** Segment $j$'s `end` is segment $j+1$'s
   `start` $- 1$; there are no gaps to fill.
2. **The same physical trial is recorded by up to five cameras**, so the 1712 files on disk
   describe only **503** distinct trials. Naively globbing `*.txt` triple-counts the corpus.

---

## 2. Deduplication: one file per trial

`find_trial_files` parses each filename against

```
^(P\d+)_([a-zA-Z0-9]+)_P\d+_([a-zA-Z0-9]+)\.txt$      →  (participant, camera, recipe)
```

groups by `(participant, recipe)`, and picks a canonical view by fixed priority:

```python
CAMERA_PRIORITY = ["cam01", "cam02", "stereo01", "webcam01", "webcam02"]
```

Views are annotation-identical for all but a handful of trials, so this is nearly a free choice —
but "nearly" is not "always", so `_warn_on_disagreements` compares the raw text of every view of
every multi-view trial and logs a warning naming the file actually used. The disagreement is
surfaced, not silently resolved.

`_check_against_config` then hard-asserts four counts against the YAML
(`n_unique_trials`, `vocab.verbs`, `vocab.nouns`, `n_recipes`) and raises rather than proceeding.
This is the tripwire that catches "I pointed it at the wrong dataset root" before it becomes
"why is my model bad".

---

## 3. The `SIL` token and the verb/noun factorisation

Every coarse label except `SIL` is `verb_noun` and splits on the *first* underscore
(`pour_milk` → `("pour", "milk")`; `take_cup_from_shelf` → `("take", "cup_from_shelf")`).

`SIL` — "nothing labelled is happening" — is not a verb/noun pair, so it is mapped to a
configured sentinel pair:

```yaml
ambient_gaps:
  sil_verb: "stall"
  sil_noun: "kitchen"
```

That is what makes the product model $P(v,n\mid Z) = P(v\mid Z) P(n \mid Z)$ well-defined on
*every* tick: idleness becomes an ordinary in-vocabulary token pair rather than a missing
observation requiring special handling in the likelihood. The cost is that a fitted state can
legitimately be an "idle" state, and `narrate.Lexicon.phrase` has to translate
`("stall", "kitchen")` back to the word "idle" so the user never sees the sentinel leak.

The config's `inject_synthetic_gaps: false` records an audit decision: `SIL` already covers 7.3%
of frames, dense enough that no artificial gaps need be inserted to teach the model about idling.

Vocabularies are built by sorting the observed token sets, so ids are a deterministic function of
the corpus — reproducible across machines without persisting a counter.

---

## 4. Tick expansion: the frames → ticks reduction

`tick_expansion.expand_to_ticks` is the only place time resolution changes. With

$$
f = \texttt{fps} = 15 \ \text{frames/s}, \qquad \tau = \texttt{tick\_seconds} = 1.0\ \text{s},
\qquad \Phi = f\tau = 15 \ \text{frames/tick},
$$

a trial of $F$ total frames becomes

$$
T = \left\lceil \frac{F}{\Phi} \right\rceil \quad \text{ticks},
$$

and tick $i \in \{0,\dots,T-1\}$ owns the (1-indexed, inclusive) frame range

$$
\mathcal{F}_i = \bigl[\,\lfloor i\Phi \rfloor + 1,\ \min(\lfloor (i{+}1)\Phi \rfloor,\, F)\,\bigr].
$$

The tick's value is the **plurality label over its own frames**:

$$
x_i \;=\; \arg\max_{x} \sum_{\phi \in \mathcal{F}_i} \mathbf{1}\!\left[ \text{label}(\phi) = x \right].
$$

Implementation detail that matters: the function first materialises a dense
`frame_values[1..F]` array by writing each segment's label over its whole frame range, *then*
bins. Doing it that way — rather than computing interval overlaps analytically — is what makes
the straddling case ("this tick spans the end of `take_bowl` and the start of `pour_cereals`")
fall out for free with the correct tie-break semantics.

**Why the same function handles verbs, nouns, and labels.** `expand_verb_noun_to_ticks` binds
the verb and noun into a single tuple *before* binning:

$$
x_i = \arg\max_{(v,n)} \#\{\phi \in \mathcal F_i : (\text{verb},\text{noun})(\phi) = (v,n)\},
$$

then unzips. That is deliberately **not** the same as majority-voting verb and noun
independently — independent votes could produce a $(v,n)$ pair that occurred in *no* frame of
that tick. Binning the pair guarantees every emitted observation is one the annotator actually
wrote down. `labels.build_trial_labels` calls the scalar `expand_to_ticks` on the raw label
strings with the *identical* binning, so `labels.json[i]` and `sequences.json[i]` are aligned
index-for-index by construction.

### 4.1 Consequence for the duration model

Because $\tau = 1$ s, tick counts *are* seconds, and `d_max_ticks` reads directly as a wall-clock
bound: $D_{\max} = 200$ on full Breakfast (≈ p99 of segment lengths), $D_{\max} = 50$ on the mini
subset (whose observed max is 65 and p99 is 44). Anything longer right-censors into the last
duration bin — see [`hsmm.md`](hsmm.md).

---

## 5. The three outputs

```
dataset/processed/{breakfast,breakfast_mini,epic}/
├── sequences.json   [{"trial_id", "verb_ids": [T ints], "noun_ids": [T ints],
│                      "terminal_idle_ticks": int}, ...]          # last key: §6
├── labels.json      [{"trial_id", "recipe_label": str, "subtask_labels": [T strs],
│                      "participant_label": str}, ...]            # last key: EPIC only, §7.4
└── vocab.json       {"verbs": {name: id}, "nouns": {...}, "recipes": {...}}
```

Both drivers write exactly this, which is what lets every downstream runner take a `--sequences` /
`--labels` / `--vocab` path and not care which corpus produced it. The two extra keys are additive
and optional; nothing that reads only the original three is affected.

Both lists are ordered by sorted `trial_id`, so they zip without a join — though
`run_recipe.py:_load_and_join` still joins on `trial_id` defensively rather than relying on it.

**`labels.json` is quarantined by design.** Its module docstring says so in capitals: nothing in
it is ever fed to EM. The entire model is fit unsupervised on integer `(verb_ids, noun_ids)`
streams; the labels exist only so `recipe_hmm.adjusted_rand` / `matched_accuracy` can score
recovered clusters after the fact. Keeping them in a physically separate file is the cheap
structural guarantee that a stray `labels` variable can never leak into a likelihood.

---

## 6. The trailing idle is trimmed, not modelled

`parse_breakfast.trim_terminal_idle` strips each trial's final run of SIL ticks, and **both**
drivers call it (`--keep-terminal-idle` on either one turns it off). The trimmed count is recorded
per trial as `sequences.json[i]["terminal_idle_ticks"]`, so nothing is lost silently.

The first reason is uninteresting: the recipe is over during that run, so there is nothing to gain
by scoring those ticks. The second is the reason the function exists.

**A trailing idle is not right-censored, but the model has no way to know that.** The E-step
treats every trial's final segment as still in progress and weights it by the survival function
$P(D \ge d)$ rather than the pmf; `durations.impute_censored_histogram` then redistributes that
mass across $d \ge d_{\text{observed}}$ *proportional to the current fit*. For a state that is
**always** the final segment, that is a fixed point with nothing pulling it back — the fit says
"long", the imputation places mass further out in the tail, the refit says "longer". Measured on
the shipped checkpoint: the terminal-idle state converged to a mean duration of **734 ticks**
against observed runs with a median of 5, and it appeared nowhere except as a trial's last segment
(19/19 occurrences).

Trimming removes the *concentration*, not the censoring. A trimmed trial's final segment is a real
action, and which state plays that role varies from trial to trial, so no single state accumulates
a runaway estimate. The E-step's censoring handling is deliberately left alone — see
[`hsmm.md`](hsmm.md) §2.3.

Measured cost on the full Breakfast corpus: 98% of trials end in a SIL run, but the runs are short
(median 5 ticks, p90 12), so this removes 4.5% of all ticks and no trial becomes empty.

---

## 7. The second corpus — EPIC-KITCHENS-100

`parse_epic.py` emits the same `sequences.json` / `labels.json` / `vocab.json` contract from
EPIC-KITCHENS-100. **Annotations only** — no video is downloaded or read. Clone
[`epic-kitchens/epic-kitchens-100-annotations`](https://github.com/epic-kitchens/epic-kitchens-100-annotations)
(~89 MB) and point `--root` at it:

```bash
./py -m cook_ad.data.parse_epic --root dataset/epic_kitchens/annotations \
    --config configs/epic.yaml --out-dir dataset/processed/epic
```

Only `EPIC_100_train.csv` and `EPIC_100_validation.csv` are read. `EPIC_100_test_timestamps.csv`
is the held-out challenge set and carries no verb/noun labels, so there is nothing to build a
sequence from.

**Why a second corpus at all.** Breakfast has 15 verbs and 36 nouns over 10 recipes; EPIC has 97
verbs and 305 noun classes over unscripted kitchen visits. Several claims the detector rests on
are untestable at Breakfast's scale — most directly, that a substitution can be *graded* by
semantic distance rather than merely flagged. That test needs a vocabulary large enough for "near"
and "far" to differ, which is what the similarity kernel ([`hsmm.md`](hsmm.md) §8) is scored on.
The `s_pair` channel ([`anomaly.md`](anomaly.md) §2.6) has the same motivation: it is redundant
with $s_{\text{emit}}$ on Breakfast and is the point of moving corpora.

Four things differ from Breakfast, and each is a decision rather than a detail.

### 7.1 A trial is a *visit*, so it has to be filtered

EPIC's recording protocol was to capture every kitchen visit, so one video is one visit. But a
visit is not a **goal**: a long one may be cook-then-wash-up-then-unpack, which breaks the joint
model's one-recipe-per-trial assumption outright. The config's duration band keeps visits short
enough to be plausibly single-purpose:

```yaml
data:
  min_session_minutes: 2
  max_session_minutes: 15   # keeps 357 of 633
```

This is a modelling constraint leaking into the ingest, and it is worth naming as such: the
sessions thrown away are not bad data, they are data the *recipe latent* cannot represent.

### 7.2 Narrations overlap — 28% of consecutive pairs

The HSMM emits one state per tick, so the ties have to be broken. `resolve_overlaps` is
**first-come-wins**: a narration's start is pushed past whatever already occupies those seconds,
and one swallowed entirely by its predecessor is dropped.

The alternative — letting the later narration interrupt the earlier — is not merely worse, it is
**unrepresentable**. Interrupting turns one segment into `A B A`, and `params._row_normalize`'s
`mask_diag` bans self-transitions structurally ($A_{kk} = 0$, see [`hsmm.md`](hsmm.md) §3), so the
re-entry is impossible rather than merely unlikely. The re-derived segmentation would have zero
likelihood under the model that consumes it.

`to_frame_segments` then converts the disjoint narrations to **contiguous** 1-indexed frame
segments with SIL filling every gap. Contiguity is required, not cosmetic: `expand_to_ticks`
majority-votes over a dense frame array, so an uncovered frame would vote `None`.

### 7.3 fps varies per video (29.97 to 90)

EPIC's frame columns cannot be pooled across videos, so **timestamps are the source of truth** and
are re-quantised onto one synthetic grid, `FPS_REF = 20`. That is an order of magnitude finer than
the 0.5 s tick, so binning is unaffected — and it lets `tick_expansion.expand_verb_noun_to_ticks`
be reused verbatim (§4) rather than reimplemented against a second time convention.

### 7.4 There are no recipe labels

EPIC annotates actions, not dishes, so the recipe layer is **fully unsupervised** and
`labels.json` has to carry something derived to score it against. `derive_dish_labels` takes the
food noun with the highest TF-IDF per session — TF so a long session does not outscore a short
one, IDF so a noun everybody touches (water, oil) loses to one that actually distinguishes this
session.

Only nouns in `FOOD_CATEGORIES` are eligible. The rest — appliances, crockery, cutlery, cleaning
equipment — are **91% of all observations** and barely vary with the dish, so a label read off all
nouns would mostly encode "this is a kitchen".

Two honest caveats, both stated in the code:

- **Abstain, don't fabricate.** 33 of 357 sessions mention no food noun at all and get
  `NO_DISH = "unknown"`. Sessions that are not cooking still mention *something*, so a label
  forced onto them would be noise scored as signal.
- **It is partly circular.** The dish label is derived from the observations, so scoring a
  clustering built from the same features against it measures less than a human annotation would.
  It is a working proxy, not ground truth. [`recipe.md`](recipe.md) §5 has the consequences.

`participant_label` is kept alongside as a **control**: a clustering that recovers participants
rather than dishes is a real failure mode on this corpus, and this is what makes it visible. Both
labels are validation-only, exactly as on Breakfast.

### 7.5 The vocabulary keeps EPIC's own class ids

`build_vocab` uses EPIC's published verb/noun class ids verbatim and appends the two SIL tokens at
the end, so a real class never changes id and altering the duration filter cannot silently
renumber the vocabulary. Classes absent from the filtered subset simply get an all-zero emission
column.

It **raises** on a name collision, which is not hypothetical: Breakfast's SIL noun is `kitchen`,
and `kitchen` IS a real EPIC noun class. Reusing it would silently merge the idle state with a real
object. Hence `configs/epic.yaml`:

```yaml
ambient_gaps:
  sil_verb: "stall"       # not an EPIC verb class
  sil_noun: "idle"        # NOT "kitchen" -- that IS one; build_vocab raises
```

### 7.6 EPIC-scale fits need two things Breakfast did not

Both are pure optimisations — no result depends on either — but without them an EPIC fit does not
finish, so they belong here rather than in a footnote.

- **`cook_ad.xla_env.disable_gpu_autotuning()`.** The E-step is a stack of scans nested under two
  `vmap`s, producing fusions with shapes XLA's GPU autotuner has no heuristic for, so it
  benchmarks kernel variants for minutes. Measured at $K = 128$, $K_R = 16$, $D = 200$:
  compilation **did not finish in 4 minutes** with autotuning on, and took **3.0 s** with it off.
  Runtime is unaffected — this workload is elementwise and reduction bound, not matmul bound. Call
  it from a runner's top, before any computation: XLA reads the flag when the backend comes up, so
  setting it later is silently a no-op. An explicitly-set `--xla_gpu_autotune_level` is respected.
- **Length-bucketed chunking** in `em.py` ([`hsmm.md`](hsmm.md) §5), which trims padded ticks per
  chunk instead of padding every chunk to the corpus-wide $T_{\max}$. Verified bit-identical.

---

## 7b. The third corpus — EPIC-Tent

`parse_epic_tent.py`, 29 sessions of one person assembling a camping tent outdoors. Annotations
only, ~16 MB from the authors' annotation repo; the 166 GB data.bris release carries the video and
the gaze traces and is not needed, because the detector consumes a symbolic `(verb, noun)` tick
stream. `tools_fetch_epic_tent.py` downloads what is needed and asserts the published row counts.

This corpus is here for one reason the other two cannot supply: **two observers annotated real
errors, with begin and end frames.** Everything else in this repo detects anomalies that
`synthetic/error_injection.py` manufactured. So EPIC-Tent writes a fourth artifact,
`anomalies.json`, and a fifth and sixth (`sequences_clean.json` / `labels_clean.json`) for the
contamination ablation in §7b.5.

### 7b.1 A trial is a participant, and there is only one task

29 sessions, one each, all building the same tent. There is no recipe taxonomy to recover and no
dish to derive, so `recipe_label` is the constant `"tent"` and `k_recipe: 1`. The recipe layer is
a no-op by construction rather than an unsupervised guess — which turns out to matter twice:

- Joint EM **converges** here, in 9 iterations. On Breakfast and EPIC it lands in a period-3 limit
  cycle and never meets `tol`. The limit cycle is a recipe-layer phenomenon.
- `s_recipe_transition` is degenerate and needed a fix in two places. At one recipe the
  recipe-attributable excess is undefined — there is no other recipe for a transition to be
  ordinary under — but the arithmetic does not produce zero: `joint_params.marginal_log_trans`
  floors probabilities, so on rare transitions (p ≈ 1e-14) the "excess" is a flooring artifact of
  up to ~6 nats, and float noise elsewhere. Before any fix the channel was the earliest on 17 of 25
  caught spans. `assemble_trace_joint` now returns an exact-zero statistic at `K_R == 1`, and
  `quantile.recipe_excess_thresholds` returns `+inf` thresholds — the second half was found a week
  later, when the error-excised fit (whose artifact-derived thresholds dip below zero) still fired 6
  ticks on `0 > threshold` that the contaminated fit (thresholds all ≥ 0) had hidden. The gate is on
  the number of recipes, not a per-row "recipe row equals marginal" test: that test also matches 112
  of 1024 rows of the Breakfast `k_recipe=16` fit and would have moved its numbers.

### 7b.2 The 12 action classes factor injectively into `(verb, noun)`

| action | verb | noun | | action | verb | noun |
|---|---|---|---|---|---|---|
| assemble support | assemble | support | | pickup/open supportbag | open | supportbag |
| insert stake | insert | stake | | pickup/open tentbag | open | tentbag |
| insert support | insert | support | | pickup/place ventcover | place | ventcover |
| insert support tab | insert | support_tab | | place guyline | place | guyline |
| instruction | read | instruction | | spread tent | spread | tent |
| pickup/open stakebag | open | stakebag | | tie top | tie | top |

12 classes → 12 distinct pairs, so **8 verbs and 12 nouns** including the `stall`/`idle` sentinel.
That is unusual and it is why `k_subtask: 16`: on Breakfast the labels are already `verb_noun`, on
EPIC there are 3806 pairs the subtask layer has to group, and here the pairs *are* the states.
`recipe/lexical_init`'s anchored emissions hand the state space over rather than discovering it,
and the fitted model reaches a **per-tick subtask ARI of 1.000**. 16 rather than 13 leaves the
weak-limit prior room to split a state whose duration is bimodal — the paper's finer 38-way
taxonomy distinguishes tent corners (`insert stake c1`..`c4`), and the public release collapses
them.

`tick_seconds: 1.0`, not EPIC's 0.5. `llm/textify.py` renders durations as seconds and asserts
1-second ticks, so the LLM arm cannot run on a 0.5 s corpus at all; and at 1 s no segment is
shorter than a tick, against the 1.9% that motivated EPIC's finer grid. At that binning
`d_max_ticks: 200` censors 2 of 1912 runs.

### 7b.3 Overlaps cost the interrupter, and that is the instruction reads

10.5% of consecutive action segments overlap, because the paper's own design note is that
instruction reading *interrupts* other subtasks. `resolve_overlaps` is reused unchanged —
first-come-wins, for the reason §7.2 gives — which drops 127 of 1252 segments, **79 of them
instruction reads**, a quarter of all 304. The interrupter is systematically the one that loses.
Worth remembering when reading a transition-channel result: the context around an error is
sometimes the host action rather than the interrupting read.

Gaps are 22% of wall-clock time (median 2.8 s, p90 9.9 s) and 17% of ticks after binning, so SIL
is a substantial state here rather than a rounding artifact.

### 7b.4 `anomalies.json` — ground truth that nobody placed

Nine annotated error types, mapped onto the repo's vocabulary. `motor` (219 instances) and
`failure` (42) map to nothing: they are physical mishaps with no signature in a `(verb, noun)`
stream, so scoring against them would only depress recall for something invisible.

| EPIC-Tent | repo type | n |
|---|---|---|
| `misuse` | `substitution` | 164 |
| `order` | `transposition` | 136 |
| `search` + `slow` | `stall` | 40 |
| `omit` | `omission` | 16 |
| `correction` | `correction` | 8 |
| `repeat` | `repetition` | 3 |
| | **total** | **367** |

**One of those six is not observable in a `(verb, noun)` stream, and it is the biggest.**
`misuse` is annotated *within a correctly-labelled action* — the wrong guyline during a step the
annotator still called `place guyline` — so the tick stream is identical whether or not it
happened. Measured: **94% of `misuse` spans leave the observed pair unchanged** (order 90%, stall
80%, correction 75%, omit 44%). Mapping it to `substitution` is still the right call — it is the
closest thing this corpus has, and excluding it would hide the limitation rather than measure it
— but its recall is a fact about the representation, not about the detector. See
[`eval.md`](eval.md) §7c, where the emission channels are shown firing exactly zero times.

`PROCEDURAL_TYPES` names PREGO's four-type subset (`order`, `omit`, `correction`, `repeat`), which
comes to exactly their published **163** instances across 28 of 29 participants. Keeping it as a
named subset is what lets their numbers stay comparable without throwing away `misuse`, the
closest match this corpus has to the repo's `substitution`.

Three properties of the records, each a decision:

- **Position comes from the error's own frames**, never from a join to an action segment. 35 of
  367 records lose their host segment to the overlap resolution above, and 3 `search` rows carry
  `action_label = -1` because the participant was hunting for a part *between* subtasks. The
  `action_label` on a record is context.
- **`omit` collapses to its onset tick.** The annotation README is explicit: "End time codes for
  omission should not be used as omission is the absence of an action." The annotated end is where
  the next thing happened.
- **No `tick_map` and no `edited_ticks`.** Nothing was spliced, so `textify.injection_touched_steps`
  correctly finds no debris, and the observation stream is not copied into the record — it is in
  `sequences.json`, and it is the same stream, because a real anomaly has no un-anomalous twin.

### 7b.5 `clean_fragments` and the excised corpus

**No trial is error-free.** Under the mapped taxonomy all 29 carry something; p08 is free of
*procedural* error but has `misuse`. Two consequences:

`anomalies.json` also records each trial's `clean_fragments` — its maximal error-free tick
intervals of length ≥ 10 (below that, a fragment's first ticks are still inside the previous
anomaly's `span_metrics.DEADLINE`). **170 fragments, 17 008 ticks, 73% of the corpus.** That is the
healthy population: a false-positive rate here is a fragment-level quantity, not a trial-level one.
Flags for a fragment must be *sliced out of the full trial's decode* — re-decoding a fragment alone
would start it from the init distribution, and the belief state part-way through an assembly is not
the start-of-trial prior.

`--exclude-anomalies` writes those fragments as `sequences_clean.json` / `labels_clean.json`, one
sequence per fragment (`p18_f0`, `p18_f1`, …). Fitting on it is the second half of the
contamination ablation: every training session contains annotated errors, so a model fitted on all
of them partly models the errors it should flag. **Excision rather than masking**, because
`hsmm/em.pad_batch` returns a *prefix* length mask and both `_chunk_ticks` and `_length_order` read
`sum(mask)` as the sequence length — a mid-sequence `False` would leave those disagreeing with the
mask and silently corrupt right-censoring. Cutting costs only the transitions that straddle a cut
and needs no EM change. `tools_epic_tent_split.py` derives `split_clean.json` by *inheriting*
session membership, so a fragment cannot land opposite the session it was cut from.

### 7b.6 `CANONICAL_ORDER`, and why the LLM arm needs it

EPIC-Tent's `order` errors are annotated against the **printed instruction sheet**, not against
what participants actually did. `CANONICAL_ORDER` records that sequence, derived two ways that
agree: the median normalised onset of each action over the 29 sessions, and the workflow column of
the paper's Table 1. `read/instruction` is deliberately absent — it interleaves freely (median
position 0.36 over 300 segments).

This is not documentation. Measured on the corpus: **the transition into an out-of-order segment
has median count 23, against a 22 baseline over all 1223 transitions**; only 5% of order-error
incoming edges have count ≤ 2 against 3% at baseline. A "backward jump in canonical rank" heuristic
scores precision 0.14 / recall 0.50 / F1 0.22 at a 0.14 base rate. A first-order transition model —
which is what the HSMM has — cannot represent this reference, because its bigrams are estimated
from error-laden sessions. `llm/prompts.assembly_block` states the prescribed order to the
predictive arm instead. See [`eval.md`](eval.md) §7c.

### 7b.7 PREGO's split, and a data-quality note on the uncertainty ratings

The per-frame self-rated uncertainty exists only for participants 8–29 — the rating procedure was
added partway through collection. PREGO's split is nonetheless **exactly recoverable**: their
CVPR'24 supplementary (section C) states the rule — "the videos that form the test set have a
median confidence score under 0.6 ... the remaining 7 do not [have scores] and are assigned to the
test set." The release ships *uncertainty*, so confidence is its complement. That rule gives
**14 train / 15 test**, and the 15 test ids (1–7, 10, 16, 17, 18, 20, 23, 25, 28) are exactly the
keys of PREGO's own `step_anticipation/data/predictions/output_miniROAD_Epic-tent-O.json`. The
nearest participants to the threshold sit at 0.583 and 0.627, so it is not a knife-edge.
`tools_epic_tent_split.py` writes it as `split_prego.json`. (An earlier version put the unrated
seven in *train* and ranked by mean; it matched the 14/15 counts with the wrong membership.)

Most uncertainty files carry exactly 2 rows more than the release README's `total_frame_count`, a
harmless systematic off-by-two; six disagree by more than 1%, and two look like mis-assigned
copies (participant 14's file has participant 9's row count, and 29's has 8's). PREGO read the same
files, and neither suspect is near the threshold.

### 7b.8 Error groups, and what "removing" the other types means

Results are reported per error group, each scored with every other type **removed**:

| group | EPIC-Tent labels | repo types |
|---|---|---|
| `procedural` | order, omit, correction, repeat — PREGO's subset | transposition, omission, correction, repetition |
| `procedural+temporal` | the above + search, slow | + stall |

`misuse` is in neither, deliberately: 94% of its spans leave the `(verb, noun)` stream unchanged
(§7b.4), so it measures the representation rather than the detector. `motor` and `failure` were
never mapped.

"Removed" follows PREGO, and it is worth being exact about what they did, because the paper does
not say. Their input is a collapsed step sequence, which has no slot for a misuse; their scorer
(`step_anticipation/src/models/llm_hf.py`) does

```python
correct = matches[:-1]      # every step before the last is scored as correct
mistake = matches[-1]       # the last step of each trimmed test sequence is THE mistake
```

and each test video is trimmed at the end of its first procedural mistake. So the non-procedural
types were never excised from anything — they were **relabelled as correct**. Here the equivalent
is: an out-of-group span drops out of the ground truth, and its ticks rejoin the healthy
population, so a flag on one is a false positive. `run_epic_tent_eval.group_truth` implements
it, and recomputes the healthy fragments per group accordingly — `anomalies.json`'s stored
`clean_fragments` exclude all six mapped types and are used only to build the training corpus.

The one part of PREGO's protocol **not** adopted is the trimming: this run scores whole sessions
with every in-group mistake a positive, where PREGO scores one positive per video. Same subset,
same removal rule, different denominator.

---

## 8. Reading the three configs

| | `breakfast.yaml` | `breakfast_mini.yaml` | `epic.yaml` |
|---|---|---|---|
| recipes | 10 | 3 (cereals, coffee, tea) | none annotated — derived dish (§7.4) |
| trials | 503 | 152 | 357 of 633 sessions |
| verbs / nouns | 15 / 36 | 15 / 36 | 98 / 306 |
| tick | 1.0 s | 1.0 s | 0.5 s |
| $K$ | 64 | 20 | 64 |
| $K_R$ | 16 | 6 | 16 |
| $D_{\max}$ | 200 (200 s) | 50 (50 s) | 200 (100 s) |
| `global_damping` | 0.7 | 0.0 | 0.7 |
| warm start | cascade | cascade | lexical only (§8.2) |

### 8.1 The two Breakfast configs

The vocabularies are *identical* (15 verbs, 36 nouns) in both — the mini subset uses only ~6
verbs and ~10 nouns in practice, but keeps the global id space so a model fit on mini can be
compared against one fit on full without remapping. `build_mini_dataset.py` filters the already-
processed full artifacts rather than re-parsing, which is why it can't change the vocabulary.

The `global_damping` difference is not cosmetic; see the duration-shrinkage section of
[`hsmm.md`](hsmm.md) for why full scale needs it and mini does not.

### 8.2 Every EPIC value that differs is a decision

`configs/epic.yaml` carries the rationale inline; the four that most affect results:

**`tick_seconds: 0.5`, not Breakfast's 1.0.** At a 1 s tick, **25%** of EPIC narrations are
shorter than one tick and vanish entirely, against **1.9%** at 0.5 s. Going finer still buys
little and scales message passing linearly in $T$. Note the knock-on: $D_{\max} = 200$ is 100 s
here, not 200 s, and it is sized against the segments that actually exist in the tick stream
(p99 = 48 ticks, so it censors ~0%).

**`k_subtask: 64`, and specifically *not* one state per $(v,n)$ pair.** Breakfast's lexical warm
start assigns one subtask per observed pair ([`recipe.md`](recipe.md) §4) because there are only
48 of them. EPIC has **3806** distinct pairs, so the subtask layer must *discover* groupings
rather than be handed them. 64 is chosen over the 128 that pair coverage alone would argue for
(57% vs 68%) because $K$ also sets the per-recipe transition matrix size, and that is what gets
starved: at $K = 128$ only **2.1%** of $A^{(r)}$'s cells survive the Dirichlet-MAP mode, against
**31.3%** for Breakfast at $K = 64$. That measurement is also what motivates `noun_tilt`
([`hsmm.md`](hsmm.md) §6.1) — with the recipe-conditioned transition likelihood mostly floor,
$\rho$ is close to noise.

**`joint_em.warm_start: false`.** No cascade artifacts exist for EPIC, so `run_joint.py` has
nothing to warm-start from; use `run_joint_lexical.py`. `chunk_size: 2` is likewise set
explicitly rather than derived — `run_joint_lexical.py` would otherwise compute
$\max(1, \texttt{em.chunk\_size} / K_R) = 1$, and compile cost at these shapes is not monotone in
chunk size, so it has to be measured rather than reasoned about.

**`recipe_features: nouns` and `idf_recipes: true`.** Both are non-default, and **both must move
together** — see `recipe/lexical_init.cluster_recipes` and [`recipe.md`](recipe.md) §4. The
default $(v,n)$-pair histogram is far too sparse at 3806 pairs, and without IDF the equipment
nouns that dominate every session drown the dish. Measured dish ARI on EPIC: **−0.009 → 0.538**
with both on. Breakfast keeps the defaults.

---

## 9. Train/test splits — `split_dataset.py` and `split.py`

Splitting is a separate step from parsing, and the split lives in its own file, for the same
reason `labels.json` does: it is a fact about the *experiment*, not about the corpus.

```bash
python split_dataset.py --sequences dataset/processed/epic/sequences.json \
    --labels dataset/processed/epic/labels.json \
    --out dataset/processed/epic/split.json --test-frac 0.2 --seed 0
```

`split.json` records `{seed, test_frac, train_trial_ids, test_trial_ids}` — the inputs as well as
the outputs, so a split is reproducible from the file alone. `data/split.py` is the read side:
`load_split`, then `filter_sequences` / `filter_labels` to one partition. It raises on an unknown
part rather than returning an empty list, which is what stops a typo'd `--split-part` from
quietly scoring zero trials. Every runner that takes `--split-file` / `--split-part` goes through
it.

Two things to know before quoting a number off a split:

- **It splits over `trial_ids` directly, not grouped by participant.** The same person appears in
  both partitions, so this measures generalisation to new *trials*, not to new *users*. On a small
  corpus a recipe can also land entirely on one side; `_print_summary` prints the per-recipe
  train/test counts and warns explicitly when that happens rather than leaving it to be discovered
  downstream.
- **`--labels` affects only the printed summary.** With it, the summary groups by `recipe_label`;
  without it, it falls back to the trailing token of the trial id (`P03_juice` → `juice`), which
  is a Breakfast filename convention and is wrong on any other corpus. Pass it on EPIC.

### 9.1 A related read-only tool

`explore_participant_variability.py` reads the processed `sequences.json` / `labels.json` directly
— no HSMM checkpoint, no fitting — and asks whether the corpus supports personalisation at all:
whether participants cluster into a few distinct step orderings within a recipe (against a
shuffled null), whether a step's duration distribution across participants is bimodal rather than
unimodal, and whether a participant's normalised speed or ingredient use holds across the
different recipes they performed. It writes PNGs and a markdown report next to the dataset. Pure
data description, so it is the right thing to run *before* designing anything in
[`lifecycle.md`](lifecycle.md) rather than after.

---

The **nested** dev split used for selecting regularisation is a different tool —
`make_dev_split.py`, which carves a fit/dev fold out of the training partition without touching
the outer test split. [`eval.md`](eval.md) §7 has the protocol and the measurement showing what
in-sample selection of those hyperparameters costs.
