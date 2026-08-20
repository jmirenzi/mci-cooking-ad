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

**Quote the seed-averaged numbers.** `error_injection` draws which segment each injector
perturbs, and that draw moves a single run by a few points. Over four injection seeds at
$\alpha = 2\times10^{-2}$:

| | train acc | train P | train R | **test acc** | **test P** | **test R** |
|---|---|---|---|---|---|---|
| cascade warm start (before) | 0.502 ± 0.004 | 0.675 ± 0.004 | 0.559 ± 0.007 | 0.468 ± 0.009 | 0.601 ± 0.009 | 0.578 ± 0.011 |
| final | **0.552 ± 0.008** | **0.735 ± 0.005** | **0.594 ± 0.012** | **0.508 ± 0.008** | **0.636 ± 0.010** | **0.619 ± 0.006** |

The per-seed test gap runs +3.0 to +6.4 accuracy points and is positive on every seed; the
single-seed table below is seed 0, the most favourable of the four. Read the averages.

| | train acc | train P | train R | **test acc** | **test P** | **test R** |
|---|---|---|---|---|---|---|
| always-flag reference | 0.455 | 0.455 | 1.000 | 0.455 | 0.455 | 1.000 |
| cascade warm start (before) | 0.497 | 0.676 | 0.550 | 0.453 | 0.589 | 0.559 |
| lexical + soft EM + hard EM + shift + duration refit | **0.556** | **0.741** | **0.596** | **0.517** | **0.648** | **0.623** |

Per-error-type test recall moves from `subs 0.97 / aban 0.49 / omis 0.37 / tran 0.57 / repe 0.39`
to `subs 1.00 / aban 0.62 / omis 0.39 / tran 0.71 / repe 0.39`.

### The comparison is not an artifact of who placed the injections

`synthetic/error_injection.py` injects into whatever segmentation the **scoring model itself**
decoded, so two models are normally graded on two different (though closely related) sets of
degraded streams — and the models here differ precisely in their segmentation. That is a real
confound, so `run_detect_eval.py --traj-params` exists to point every model at one common
source. Train split, alpha = 2e-2:

| scoring model | injections from the cascade fit's decode | injections from the final model's decode |
|---|---|---|
| cascade warm start (before) | 0.496 | 0.501 |
| final | **0.540** | **0.556** |

The gap survives in both columns, and the baseline barely moves between them — so the
improvement is in the detector, not in a friendlier injection set.

Reproduce:

```bash
./py run_joint_lexical.py --split-part train --out runs/joint_lex.npz \
    --anchor 50 --max-iters 60 --init-prior-scale 0.0
./py run_hard_em.py --split-part train --out runs/joint_sh.npz \
    --init-from runs/joint_lex.npz --keep-init-emissions --iters 5
./py smooth_params.py   --in runs/joint_sh.npz   --out runs/joint_sh_s.npz --strength 0.7 --backoff-tau 2
./py refit_durations.py --in runs/joint_sh_s.npz --out runs/joint_final.npz --kappa 0.001
./py run_detect_eval.py --joint-params runs/joint_final.npz --split-part test --tag final
```

Run end to end from a clean tree this reproduces train 0.556 / test 0.517 exactly (seed 0).

### `--init-prior-scale 0.0` is load-bearing, and uncomfortable

The Dirichlet prior belongs on the iteration-0 transition counts — without it,
`params._row_normalize`'s `max(c-1, floor)` sends a bigram seen once to the floor, which is the
same defect §4's third change exists to undo. Adding it (`--init-prior-scale 1.0`, the default)
does everything you would expect: soft EM **converges** rather than running out its budget, the
objective ends higher (−12594 against −12931), and per-tick subtask ARI stays at 0.9991 instead
of degrading to 0.94.

Detection prefers the other basin, by a lot: **train `trial_loc` accuracy 0.556 against 0.518**,
far outside the ±0.008 injection-seed noise. The measurable difference in the finished models is
transition sparsity — median row entropy 0.470 against 0.518 — which is exactly the quantity
`s_transition` reads. Running the first E-step against an almost-hard transition structure
appears to keep the responsibilities concentrated and the final $A^{(r)}$ sharper; that is an
observation, not a proven mechanism.

This is the same disagreement as §3, and it is the reason the flag exists and defaults to the
coherent value rather than the winning one. Do not "fix" it without re-measuring.

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
- **`s_dur_two` is already delivering what the fitted durations allow.**
  `tools_duration_power.py` re-scores every healthy segment at 1 tick (what abandonment
  truncates to) and at 2x (what repetition collapses to, since banned self-transitions force
  Viterbi to merge the duplicated run), against that (recipe, state) cell's own fitted NB. At
  alpha = 2e-2 on the final model that predicts 0.699 and 0.235; `s_dur_two` observed 0.661 and
  0.259. Read it as a prediction for **that one channel**, not a bound on the error type:
  abandonment's union recall is 0.669 and `s_dur_two` supplies 0.661 of it, so there the two
  coincide — but repetition's union is 0.365, of which the live survival channel `s_temporal`
  (which this diagnostic does not model) carries 0.341. The number moves only when the duration
  fit's spread moves, which is why the `refit_durations.py` CV 0.63 -> 0.42 shows up here.
- Repetition is the weakest type at 0.37, and both channels that catch it are duration-driven,
  so it moves only with the duration fit's spread.

---

## 7. Axes already swept, so you don't sweep them again

Every one of these was measured on train-split `trial_loc` accuracy against the §5 pipeline;
none of them beat it.

| axis | tried | best |
|---|---|---|
| warm start | cascade, lexical | lexical |
| iteration-0 prior scale | 0.0, 1.0 | **0.0** (see §5) |
| $K$ (subtask) | 48, 64 | 64 — 48 costs 0.008 |
| $K_R$ (recipe) | 8, 10, 12, 16, 24, 32 | 16 |
| soft-EM iterations before hard EM | 10, 20, 30, 40, 54–60 | 54–60; 10–40 all sit at 0.518 |
| hard EM | off, 5, 20 iterations | 5 |
| hard-EM `alpha_trans` | 0.1, 0.5, 2.0 | flat at 0.554–0.556 |
| hard-EM `kappa` | 1, 5, 20 | flat at 0.556 |
| emission anchor | 50, frozen | 50 — freezing costs 2.4 points of recall |
| transition shift $s$ | 0.3, 0.5, 0.7, 0.9, 1.0, 2, 4 | 0.7; 0.3–0.9 within 0.007 |
| pooled backoff $\tau$ | 0, 1, 2, 3, 4, 5, 10, 30 | 2; 0–5 within 0.006 |
| unigram backoff $\tau$ | 1, 3, 10, 20, 50, 100 | none — see §6 |
| duration refit `kappa` | 0.001, 1, 20, 100 | 0.001 |

The plateau across the last four rows is worth reading on its own: the result is not perched on
a tuned operating point.
