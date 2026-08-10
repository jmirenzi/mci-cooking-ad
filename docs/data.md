# `src/cook_ad/data/` — from raw annotations to integer tick sequences

**Job:** turn the Breakfast Actions corpus's frame-level text annotations into the three JSON
artifacts every downstream module consumes, and do it deterministically enough that the counts
can be asserted against the config.

| File | Role |
|---|---|
| `config.py` | 6-line YAML loader. Nothing else. |
| `tick_expansion.py` | frame-interval segments → per-tick values (majority vote) |
| `labels.py` | ground-truth per-tick action labels — **validation only** |
| `parse_breakfast.py` | the driver: parse → dedupe → vocab → tick-expand → write JSON |

---

## What the raw data looks like

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

## Deduplication: one file per trial

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

## The `SIL` token and the verb/noun factorisation

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

## Tick expansion: the frames → ticks reduction

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

### Consequence for the duration model

Because $\tau = 1$ s, tick counts *are* seconds, and `d_max_ticks` reads directly as a wall-clock
bound: $D_{\max} = 200$ on full Breakfast (≈ p99 of segment lengths), $D_{\max} = 50$ on the mini
subset (whose observed max is 65 and p99 is 44). Anything longer right-censors into the last
duration bin — see [`hsmm.md`](hsmm.md).

---

## The three outputs

```
dataset/processed/breakfast/
├── sequences.json   [{"trial_id", "verb_ids": [T ints], "noun_ids": [T ints]}, ...]
├── labels.json      [{"trial_id", "recipe_label": str, "subtask_labels": [T strs]}, ...]
└── vocab.json       {"verbs": {name: id}, "nouns": {...}, "recipes": {...}}
```

Both lists are ordered by sorted `trial_id`, so they zip without a join — though
`run_recipe.py:_load_and_join` still joins on `trial_id` defensively rather than relying on it.

**`labels.json` is quarantined by design.** Its module docstring says so in capitals: nothing in
it is ever fed to EM. The entire model is fit unsupervised on integer `(verb_ids, noun_ids)`
streams; the labels exist only so `recipe_hmm.adjusted_rand` / `matched_accuracy` can score
recovered clusters after the fact. Keeping them in a physically separate file is the cheap
structural guarantee that a stray `labels` variable can never leak into a likelihood.

---

## Reading the two configs

| | `breakfast.yaml` | `breakfast_mini.yaml` |
|---|---|---|
| recipes | 10 | 3 (cereals, coffee, tea) |
| trials | 503 | 152 |
| $K$ | 64 | 20 |
| $K_R$ | 16 | 6 |
| $D_{\max}$ | 200 | 50 |
| `global_damping` | 0.7 | 0.0 |

The vocabularies are *identical* (15 verbs, 36 nouns) in both — the mini subset uses only ~6
verbs and ~10 nouns in practice, but keeps the global id space so a model fit on mini can be
compared against one fit on full without remapping. `build_mini_dataset.py` filters the already-
processed full artifacts rather than re-parsing, which is why it can't change the vocabulary.

The `global_damping` difference is not cosmetic; see the duration-shrinkage section of
[`hsmm.md`](hsmm.md) for why full scale needs it and mini does not.
