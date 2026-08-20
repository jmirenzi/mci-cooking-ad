# Where the detector's accuracy goes, and what moves it

This doc is the measurement trail behind the training-side changes in
`recipe/lexical_init.py`, `run_hard_em.py`, `smooth_params.py` and `refit_durations.py`. Read
[`eval.md`](eval.md) §6 first for the metric; everything below is `trial_loc`, the
deployment-shaped question ("did it flag, and in the right place?").

All numbers are the 402-trial train / 101-trial test split from `split_dataset.py`
(`dataset/processed/breakfast/split.json`), scored by `run_detect_eval.py`.

---

## 0. Read accuracy, not F1

With a 1-healthy : 5-degraded pool and strays charged independently of hits
([`eval.md`](eval.md) §6), the trivial **always-flag** detector scores

$$
\text{acc} = \tfrac{5}{11} = 0.455, \qquad \text{precision} = \tfrac{5}{11}, \qquad F_1 = 0.625 .
$$

An F1 of 0.62 is therefore *not evidence of a detector*. Every model in this repo's history sits
near it. `trial_loc` accuracy is the number that separates them, and

$$
\text{acc} \;=\; \frac{5R + (1 - h)}{6 + 5s}
$$

with $R$ = recall, $h$ = healthy false-alarm rate, $s$ = stray rate on degraded trials. The
partials are $\partial\text{acc}/\partial R \approx 0.68$, $\partial\text{acc}/\partial s
\approx -0.34$, $\partial\text{acc}/\partial h \approx -0.14$: **recall is worth about five times
what the healthy false-alarm rate is**, which is the opposite of the intuition the
precision-focused threshold sweeps encourage.

---

## 1. What was NOT wrong

Measured on the cascade-warm-started `joint_params_train.npz`:

| | |
|---|---|
| occupancy-weighted state emission purity | 0.986 |
| boundary F1 vs the (verb,noun) run structure | 0.975 |
| usage-weighted transition row entropy | 0.79 nats (effective out-degree 2.3) |
| surprise of a transition never observed | ~30 nats |

The emissions, the segmentation and the sparsity of $A^{(r)}$ were all fine. The loss is entirely
in what happens when the detector is handed a *degraded* trial.

---

## 2. The three leaks

**Viterbi laundering.** `tools_launder.py` tracks each injection-created junction that the
training data never contains, through the decode, to the flag:

| | omission | transposition |
|---|---|---|
| trials with a never-seen created junction | 0.442 | 0.712 |
| ...junction still present in the decode | **0.135** | 0.413 |
| ...and the transition channel fires there | 0.114 | 0.161 |

After the §4 changes the same measurement reads:

| | omission | transposition |
|---|---|---|
| trials with a never-seen created junction | 0.222 | 0.534 |
| ...junction still present in the decode | 0.204 (**92%** of them) | 0.505 (**95%**) |
| ...and the transition channel fires there | 0.198 (**97%** of survivors) | 0.505 (**100%**) |

The channel is now essentially perfectly efficient on what reaches it; what limits it is how much
reaches it, which is §2's data ceiling and the recipe flip. (The *fraction* of trials with a
never-seen junction drops because the baseline's duplicate states inflate the bigram inventory
with spurious novelty — pairs of duplicate states that stand for one real transition.)

**70% of omission anomalies are re-explained away by the decoder before any channel sees them.**
The cause is duplicate states: 10 of the 48 distinct (verb,noun) pairs are split across two to
four states (`stall/kitchen` across four), and a duplicate state is a legal alternative route
that costs less than the ~30-nat transition the injection created.

**MAP recipe flips.** The detector re-infers $\hat r$ from the degraded stream. On a transposition
it moves to a different cluster ~40-60% of the time — and the cluster it moves to is, by
selection, one whose $A^{(r)}$ finds the new ordering ordinary. `tools_oracle_recipe.py` pins
$\hat r$ to the healthy decode's value and measures the prize:

| | inferred $\hat r$ | pinned $\hat r$ | flip rate |
|---|---|---|---|
| omission | 0.370 | **0.511** | 0.172 |
| transposition | 0.566 | **0.728** | 0.402 |

**A data-limited ceiling.** `tools_transition_ceiling.py` splits injection-created junctions three ways: ~33%
already occur in the trial's own recipe cluster (real Breakfast order variation — not anomalies),
~18% occur elsewhere in the corpus, ~49% never. Perfect recipe conditioning caps omission recall
near 0.67, not 1.0. Repetition is capped separately by duration spread: banned self-transitions
force Viterbi to merge the duplicated run, so a 2x duration is the only signal, and at the fitted
CV of 0.66 that clears the $\alpha=0.05$ tail only 28% of the time (`tools_duration_power.py`). A per-trial
pace factor explains 26% of duration variance — z-score 0.99 → 1.15 — so it is not the fix
(`tools_pace.py`).

---

## 3. Likelihood does not rank these fits

Three direct observations, all on the train split:

- the lexical warm start **starts** at objective −12948, higher than the cascade-warm-started
  joint EM ever **converges** to (−14245);
- soft EM from that warm start then *lowers* the objective for a stretch while degrading per-tick
  subtask ARI from 0.999 to 0.940;
- the highest-likelihood model of all (−11863, soft EM followed by hard EM) has subtask ARI 0.935
  and is beaten on detection by models a thousand nats below it.

Select on `run_detect_eval.py --split-part train`, not on the objective. That is what every
number in §5 does.

---

## 4. The four changes

**`recipe/lexical_init.py` — one subtask state per observed (verb,noun) pair.** The observation
stream is piecewise-constant coarse action labels, so the distinct pairs *are* the action
inventory; recovering them needs `sequences.json` alone, never `labels.json`. Recipes are seeded
by spherical k-means on bag-of-pairs histograms, emissions held by a per-state anchor prior.
At iteration 0 this reaches per-tick subtask ARI **0.9991** and recipe ARI **0.817**, and it
removes the duplicate states — novel-junction survival through the decode goes from 30% to
**97%** for omission and 58% to **97%** for transposition.

**`run_hard_em.py` — Viterbi EM instead of soft EM.** Every surprise channel scores against
`z_star`, so the model may as well be fit to `z_star`. Hard EM converges in ~5 iterations at
~15 s each (soft EM: 60 iterations at ~45 s), holds subtask ARI at 0.999, and reaches a *higher*
marginal likelihood than soft EM does.

**`smooth_params.py` — undo the MAP mode's erasure of singleton transitions.**
`params._row_normalize` takes numerator $\max(c-1, \text{floor})$. At ~150 transitions per recipe
over a 64x63 grid, a large share of the model's legal transitions are singletons, and subtracting
a whole count files them next to transitions that never happened. Adding $s \in (0,1)$ back lifts
singletons above the floor while leaving never-observed cells at it — the gradation the
structural channels need. $s \approx 0.7$ measures best; $s = 1$ (the full Dirichlet posterior
mean) is worse, because it also lifts never-seen transitions from ~32 nats to ~9.

**`refit_durations.py` — refit durations from the model's own decode.** A hard-assignment
duration M-step at low `kappa` tightens the fitted CV from 0.63 to 0.42 and lifts both duration
channels without moving the healthy false-alarm rate.

---

## 5. Results

Operating point $\alpha$ chosen on **train** by `trial_loc` accuracy, then applied unchanged to
**test** (`report_final.py`). Both at $\alpha = 2\times10^{-2}$.

| | train acc | train P | train R | **test acc** | **test P** | **test R** |
|---|---|---|---|---|---|---|
| always-flag reference | 0.455 | 0.455 | 1.000 | 0.455 | 0.455 | 1.000 |
| cascade warm start (before) | 0.497 | 0.676 | 0.550 | 0.453 | 0.589 | 0.559 |
| lexical + soft EM + hard EM + shift + duration refit | **0.556** | **0.741** | **0.596** | **0.517** | **0.648** | **0.623** |

Per-error-type test recall moves from `subs 0.97 / aban 0.49 / omis 0.37 / tran 0.57 / repe 0.39`
to `subs 1.00 / aban 0.62 / omis 0.39 / tran 0.71 / repe 0.39`.

Reproduce:

```bash
./py run_joint_lexical.py --split-part train --out runs/joint_lex_a50.npz --anchor 50 --max-iters 60
./py run_hard_em.py --split-part train --out runs/joint_softhard.npz \
    --init-from runs/joint_lex_a50.npz --keep-init-emissions --iters 5
./py smooth_params.py --in runs/joint_softhard.npz --out runs/joint_sh_s07t2.npz --strength 0.7 --backoff-tau 2
./py refit_durations.py --in runs/joint_sh_s07t2.npz --out runs/joint_sh_k0.npz --kappa 0.001
./py run_detect_eval.py --joint-params runs/joint_sh_k0.npz --split-part test --tag sh_k0
```

---

## 6. What is still on the table

- **Recipe stability** is the largest single remaining lever: §2's oracle is worth roughly +0.04
  accuracy, and nothing here closes it. Reducing $K_R$ to 12 makes the *healthy* clustering nearly
  perfect (recipe ARI 0.937, matched accuracy 0.908) but does not improve detection — consistent
  with the earlier finding that recipe ARI and detection are decoupled.
- Transition backoff to the recipe's own state-occupancy unigram (`smooth_params.py
  --unigram-tau`) was built to decouple "wrong recipe" from "wrong order". Measured across
  tau = 1 … 100 it never beats the plain shift, and at the taus large enough to actually lift a
  wrong-order transition off the floor it destroys the signal it was meant to preserve (omission
  recall 0.37 → 0.04 at tau = 100). The tension looks structural: the cells that give the recipe
  posterior its escape hatch are exactly the cells `s_recipe_transition` needs to be sharp on.
  A recipe assignment built causally from the trial prefix — which is what a live deployment
  would have anyway — sidesteps it without touching the transition rows, and is the obvious next
  thing to try.
- The duration channels are at their own ceiling, not below it: `tools_duration_power.py` on the
  final model puts the 1-tick (abandonment) ceiling at 0.699 and the 2x (repetition) ceiling at
  0.235 for alpha = 2e-2, against observed 0.67 and 0.37.
- Repetition sits at 0.39 against a duration-limited ceiling near 0.5.
