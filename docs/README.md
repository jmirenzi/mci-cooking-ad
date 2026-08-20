# `cook_ad` — repo layout and system overview

Anomaly detection during cooking for a user with MCI (mild cognitive impairment). The system
watches a stream of coarse `(verb, noun)` action labels, maintains a probabilistic belief about
*which subtask the user is in* and *which recipe they are making*, and raises a small number of
**auditable, templated confirmation questions** when the observed behaviour departs from what the
model expects.

The whole thing is a classical latent-variable pipeline — an explicit-duration HSMM fit with
EM — deliberately *not* a neural model. Every question the system asks ("did you skip
toasting?") traces back to a single argmax over a fitted parameter row. That auditability is
the design constraint the rest of the architecture serves.

---

## Reading order

| Doc | Covers | Read it when you want to know… |
|---|---|---|
| this file | repo layout, end-to-end pipeline, notation | where anything lives, what runs in what order |
| [`data.md`](data.md) | `src/cook_ad/data/` | how raw Breakfast annotations become integer tick sequences |
| [`hsmm.md`](hsmm.md) | `src/cook_ad/hsmm/` | the core model: durations, message passing, EM, the joint recipe mixture |
| [`recipe.md`](recipe.md) | `src/cook_ad/recipe/` | Viterbi segmentation, the cascade's stage-2 recipe HMM, the joint warm start |
| [`anomaly.md`](anomaly.md) | `src/cook_ad/anomaly/` | the seven surprise channels, quantile calibration, narration |
| [`lifecycle.md`](lifecycle.md) | `src/cook_ad/lifecycle/` | frozen/live dual model, bounded preference updates, drift reports |
| [`eval.md`](eval.md) | `src/cook_ad/eval/` | batched trace computation, precision/recall/latency metrics, figures |
| [`synthetic.md`](synthetic.md) | `src/cook_ad/synthetic/` | ancestral sampling and the five canonical injected errors |
| [`llm.md`](llm.md) | `src/cook_ad/llm/`, `eval/element_metrics.py` | the LLM comparison baseline and the step-level metrics that make it comparable |

---

## The problem, stated as math

An observation stream for one trial is
$$
o_{0:T-1}, \qquad o_t = (v_t, n_t) \in \{1..V\}\times\{1..N\},
$$
one observation per **tick** (1 second; $V = 15$ verbs, $N = 36$ nouns on the full Breakfast
corpus, each including a `SIL` "nothing happening" token).

Behind that stream sit two latent variables:

- a **subtask** $Z \in \{1,\dots,K\}$ that persists for a *segment* of ticks
  ($K = 64$ nominal on full Breakfast — a *weak-limit* over-parameterisation, not a claim that
  there are 64 subtasks),
- a **recipe** $R \in \{1,\dots,K_R\}$ ($K_R = 16$ nominal, covering 10 real recipes plus
  headroom).

The generative story is an **explicit-duration HSMM** (a.k.a. hidden semi-Markov model /
segment model). Segments $j = 1,2,\dots$ carry a state $z_j$ and a duration $d_j$:

$$
\begin{aligned}
R &\sim \operatorname{Cat}(\boldsymbol\pi) \\
z_1 &\sim \operatorname{Cat}(\boldsymbol\pi^{\text{init}, R}) \\
d_j \mid z_j &\sim \operatorname{NB}_{\ge 1}\!\left(r^{R}_{z_j},\, p^{R}_{z_j}\right) \\
v_t \mid z_j &\sim \operatorname{Cat}(B^v_{z_j,\cdot}), \quad
n_t \mid z_j \sim \operatorname{Cat}(B^n_{z_j,\cdot}) \quad \text{i.i.d. for all } t \text{ in segment } j \\
z_{j+1} \mid z_j &\sim \operatorname{Cat}(A^{R}_{z_j,\cdot}), \qquad A^{R}_{kk} = 0 .
\end{aligned}
$$

Three modelling commitments are load-bearing everywhere downstream:

1. **Explicit durations.** A plain HMM forces $D \sim \text{Geometric}$, whose mode is always
   $d = 1$ — useless for "you've been stirring for 90 seconds, are you stuck?". The negative
   binomial gives a two-parameter, over-dispersible duration with a usable *hazard*.
2. **Banned self-transitions**, $A_{kk} = 0$. This makes the segmentation identifiable: a run of
   constant $z$ is *exactly one* segment, never an ambiguous concatenation. Run-length encoding a
   Viterbi path is therefore lossless (`anomaly/narrate.py:segments_from_z`).
3. **Product emissions**, $P(v,n \mid Z) = P(v\mid Z)\,P(n \mid Z)$. This is what lets the
   detector *attribute* an anomaly to the object ("you grabbed mustard") versus the action
   ("you're pouring, not spreading") — see [`anomaly.md`](anomaly.md).

### Two models, not one

The repo contains **two** fits of the above, and most modules have a plain and a `_joint`
variant:

| | **Cascade** (2-stage) | **Joint** (mixture) |
|---|---|---|
| Fit by | `hsmm/em.py` then `recipe/recipe_hmm.py` | `hsmm/joint_em.py` |
| Subtask dynamics | one shared $A, \pi^{\text{init}}, (r,p)$ | per-recipe $A^{(r)}, \pi^{\text{init},(r)}, (r,p)^{(r)}$ |
| Emissions | $B^v, B^n$ | $B^v, B^n$ — **shared across recipes** |
| Recipe inferred by | a second, flat HMM over *segment symbols* | $\hat r_i = \arg\max_r \log\pi_r + \log Z_{ir}$ |
| Artifact | `hsmm_params.npz` + `recipe_params.npz` | `joint_params.npz` |

The cascade is fit first and then used as the **warm start** for the joint model
(`recipe/warm_start.py`), because a symmetric random init plus a symmetric likelihood can never
break the recipe-permutation symmetry on its own.

---

## Repository layout

```
mci-cooking-ad/
├── configs/
│   ├── breakfast.yaml          full corpus: 503 trials, K=64, K_R=16, D_max=200
│   └── breakfast_mini.yaml     cereals+coffee+tea: 152 trials, K=20, K_R=6, D_max=50
├── dataset/
│   ├── breakfast_actions/      raw Breakfast `segmentation_coarse` .txt annotations
│   └── processed/breakfast{,_mini}/
│       ├── sequences.json      per-trial verb_ids / noun_ids  (TRAINING input)
│       ├── labels.json         per-tick ground-truth action labels (VALIDATION ONLY)
│       ├── vocab.json          verb/noun/recipe → id maps
│       ├── hsmm_params.npz     cascade stage 1
│       ├── recipe_params.npz   cascade stage 2
│       ├── joint_params.npz    joint model (+ .meta.json checkpoint sidecar)
│       ├── flow/               exported JSON for rendering
│       └── figures/            rendered PNGs
├── src/cook_ad/
│   ├── data/       parsing, tick binning, label extraction
│   ├── hsmm/       the model: durations, emissions, messages, EM, joint EM
│   ├── recipe/     Viterbi segmentation, recipe HMM, cascade→joint warm start
│   ├── anomaly/    surprise channels, quantile thresholds, narration, sequence detector
│   ├── lifecycle/  frozen/live dual model, online updates, drift detection
│   ├── eval/       batched traces, tick/step/counterfactual metrics, plots
│   ├── synthetic/  ancestral sampling + the five injected error types
│   └── llm/        trial-as-text rendering + the LLM comparison baseline
├── tests/          one pytest module per src module of substance
└── *.py            top-level runner / export / render scripts (see below)
```

### `src/` at a glance

```
                     data/  ──── sequences.json ─────┐
                                                      │
                                                      ▼
    hsmm/  ── em.run_em ──────────────────► hsmm_params.npz ──┐
      │                                            │           │
      │                          recipe/segmentize │           │
      │                                            ▼           │
      │        recipe/  ── recipe_hmm.run_em ─► recipe_params.npz
      │                                            │           │
      │        recipe/warm_start.cascade_to_joint ◄┴───────────┘
      │                     │
      └── joint_em.run_joint_em ──────────► joint_params.npz
                                 │
                                 ▼
        anomaly/  ── compute_trace → flag → narrate ──► Query cards
             ▲                │
             │                ▼
        lifecycle/       eval/  ── metrics on synthetic/ injected errors
                              ▲
                         synthetic/
```

Dependency direction is strictly downward: `hsmm` depends on nothing else in `cook_ad`;
`recipe`, `anomaly`, `synthetic`, `lifecycle`, `eval` all depend on `hsmm`; `eval` and `anomaly`
depend on `recipe`; nothing depends on `eval`.

---

## End-to-end pipeline

Each step writes an artifact the next step reads, so you can stop and restart anywhere.

**1 — Build the dataset** (`data/parse_breakfast.py`, run as a module main)

```bash
python -m cook_ad.data.parse_breakfast --config configs/breakfast.yaml --out-dir dataset/processed/breakfast
```

Deduplicates camera views → one sequence per `(participant, recipe)` trial, bins 15 fps frames
into 1 s ticks by majority vote, writes `sequences.json` / `labels.json` / `vocab.json`. Optionally
carve a 3-recipe subset:

```bash
python build_mini_dataset.py --recipes cereals coffee tea
```

**2 — Fit the subtask HSMM** (cascade stage 1)

```bash
python run_experiment.py --config configs/breakfast.yaml
```

10 random restarts of MAP-EM; keeps the best final log-likelihood. → `hsmm_params.npz`.

**3 — Fit the recipe HMM** (cascade stage 2)

```bash
python run_recipe.py --config configs/breakfast.yaml
```

Viterbi-segments every trial with the stage-1 model, then fits a flat discrete HMM over the
resulting *segment-symbol* sequences. Scores clusters against `labels.json` with ARI and
Hungarian-matched accuracy. → `recipe_params.npz`.

**4 — Fit the joint model**

```bash
python run_joint.py --config configs/breakfast.yaml   # add --resume to continue a checkpoint
```

Warm-starts from the two cascade artifacts, then runs a single deterministic joint EM with
resumable checkpoints every 5 iterations. → `joint_params.npz` + `.meta.json`.

**4b — Or fit the joint model without the cascade**

The same artifact, from a warm start built out of the observation stream instead of steps 2–3,
then a Viterbi-EM polish and two post-fit re-parameterisations. This is the route that fits the
better detector; steps 2–3 are still the reference implementation of the cascade.

```bash
python run_joint_lexical.py --split-part train --out runs/joint_lex.npz \
    --anchor 50 --max-iters 60 --init-prior-scale 0.0
python run_hard_em.py --split-part train --out runs/joint_sh.npz \
    --init-from runs/joint_lex.npz --keep-init-emissions --iters 5
python smooth_params.py    --in runs/joint_sh.npz  --out runs/joint_sh_s.npz --strength 0.7 --backoff-tau 2
python refit_durations.py  --in runs/joint_sh_s.npz --out runs/joint_final.npz --kappa 0.001
```

| Step | What it does | Documented in |
|---|---|---|
| `run_joint_lexical.py` | one subtask state per observed (verb,noun) pair; recipes seeded by bag-of-pairs k-means; emissions held by a per-state anchor prior | [`recipe.md`](recipe.md) §4 |
| `run_hard_em.py` | Viterbi EM — fits the model to the MAP segmentation the detector actually reads | [`hsmm.md`](hsmm.md) §7 |
| `smooth_params.py` | restores the singleton transitions the Dirichlet-MAP mode erases | [`hsmm.md`](hsmm.md) §3 |
| `refit_durations.py` | hard-assignment duration M-step, sweepable in `kappa` without a full EM run | [`hsmm.md`](hsmm.md) §2.5 |

`--init-prior-scale 0.0` is deliberate and is *not* the default — see [`recipe.md`](recipe.md) §4
before changing it.

**5 — Analyse / evaluate / demo**

```bash
python run_anomaly.py    --inject-noun --plot   # single-trial channel trace
python run_evaluation.py                        # full 5-error precision/recall/latency sweep
python run_lifecycle.py                         # frozen/live dual-model demo
python run_rollout_demo.py --scenario all       # per-user calibrated rollout
python run_llm_eval.py --dry-run                # LLM baseline: cost the sweep before sending
python run_llm_eval.py --skip-llm               # ...or score just the HSMM at step level
python render_llm_compare_png.py                # HSMM vs LLM figures from the report JSON
python run_threshold_sweep.py                   # accuracy vs alpha, per granularity
python run_counterfactual.py                    # is detection attributable to the injection?
python run_sequence_eval.py                     # segment-sequence detector vs the tick channels
python run_detect_eval.py --split-part train    # one trial_loc scorecard: alpha curve x error type x channel
```

`run_detect_eval.py` reports the same `trial_loc` metric `run_threshold_sweep.py` defines, on the
same ground-truth convention, plus the per-error-type and per-channel breakdown — and fast enough
to be used as a *selection* criterion between fits rather than only as a final report
([`eval.md`](eval.md) §7). **Read accuracy, not F1**: with a 1-healthy : 5-degraded pool the
trivial always-flag detector already scores F1 0.625, and `report_final.py` is what fixes
$\alpha$ on train before quoting test. **And never quote a `trial_loc` gain on its own** — it
charges at most one false positive per trial, so a detector can improve on it while raising more
alarms; [`eval.md`](eval.md) §7 has the cross-check and the measured case where that happens. The `tools_*.py` scripts ([`eval.md`](eval.md) §8) answer
one question each about a fitted model, for localising a loss the scorecard surfaces.

`run_counterfactual.py` scores each degraded trial against its own healthy counterfactual, which
supplies the matched null a raw recall number lacks ([`eval.md`](eval.md) §5) — reach for it before
concluding anything about whether a detection rate is meaningful.

`run_threshold_sweep.py` is the calibration diagnostic: it computes traces once and re-flags them
across an $\alpha$ grid, reporting precision/recall/FPR at tick, step and trial granularity.
Reach for it before adding any filtering rule — a channel whose false-positive rate is flat in
$\alpha$ is not being gated by its threshold, and no amount of downstream filtering fixes that
(see [`eval.md`](eval.md) §6).

`run_llm_eval.py` is the odd one out: it scores an **LLM reading each trial as text** against the
HSMM on a shared unit (one run-length-encoded step), on byte-identical injected errors. It is a
comparison baseline, not part of the detector -- see [`llm.md`](llm.md), and note the request-budget
arithmetic there before running it against a metered API.

**6 — Export and render**

```bash
python export_flow.py        && python render_flow_html.py --out flow.html
python export_flow.py        && python render_flow_png.py
python export_flow_joint.py  && python render_flow_compare_png.py   # cascade vs joint
python export_anomaly.py     && python render_anomaly_png.py
```

`export_*.py` runs inference and dumps JSON; `render_*.py` does layout only and performs no
inference. That split exists so figures can be re-styled without re-running EM.

`main.py` is an unused scaffolding stub from `uv init`.

---

## Notation used throughout the docs

| Symbol | Meaning | Code |
|---|---|---|
| $T$ | ticks in a trial | `T`, `t_max`, `mask.sum()` |
| $K$ | nominal subtask count (weak limit) | `k_subtask` |
| $K_R$ | nominal recipe count (weak limit) | `k_recipe` |
| $V, N$ | verb / noun vocabulary sizes | `vocab_verbs`, `vocab_nouns` |
| $D_{\max}$ | duration truncation | `d_max` |
| $\boldsymbol\pi^{\text{init}}$ | initial-state distribution, $(K,)$ | `init_counts` → `log_init` |
| $A$ | subtask transition matrix, $(K,K)$, zero diagonal | `trans_counts` → `log_trans` |
| $B^v, B^n$ | emission matrices, $(K,V)$, $(K,N)$ | `verb_counts`, `noun_counts` |
| $(r_k, p_k)$ | NB duration parameters per state | `dur_r`, `dur_p` |
| $\boldsymbol\pi$ | recipe mixture weights, $(K_R,)$ | `pi_counts` → `log_pi` |
| $\gamma_t(k)$ | smoothed occupancy $P(Z_t{=}k \mid o_{0:T-1})$ | `gamma` |
| $\tilde\pi_t(k)$ | **predictive** occupancy $P(Z_t{=}k \mid o_{0:t-1})$ | `pi_all` |
| $\rho_{ir}$ | recipe responsibility for trial $i$ | `rho` |
| $z^*_t$ | Viterbi (MAP) subtask at tick $t$ | `z_star`, `subtask_per_tick` |
| $\alpha$ | tail-probability level for flagging (0.05) | `DEFAULT_ALPHA` |

**Weak limit** appears constantly: rather than doing nonparametric inference over an unknown
number of states, the model fixes $K$ generously larger than the truth and uses a sparse
Dirichlet prior ($\alpha/K < 1$) to push unneeded states toward zero mass. The number that
matters is the *effective* $K$ read off the decode
(`recipe/recipe_hmm.py:effective_k`), never the nominal one.

---

## Cross-cutting conventions worth knowing before you read any module

**Everything is in log space.** `logsumexp` throughout; probabilities are only exponentiated at
the last possible moment. Two floors recur: `FLOOR = 1e-12` (before any `log`) and
`EPS = 1e-8` (before any division).

**Padding is masked, never trusted.** Batches pad to a common $T_{\max}$ with the *valid dummy
id 0* — never `-1`, because JAX gathers wrap out-of-range indices silently instead of erroring.
A `mask` array gates every use.

**Right-censoring is structural, not an edge case.** The last segment of every trial is still in
progress when observation stops. The forward recursion weights it by the **survival** function
$P(D \ge d)$ rather than the pmf $P(D = d)$; the M-step imputes its unobserved remainder. Getting
this wrong is the single easiest way to silently bias every duration in the model — see
[`hsmm.md`](hsmm.md).

**Chunking is a memory necessity, not a style choice.** At $K = 64$, $T_{\max} \approx 650$, a
full 503-sequence `vmap` of the $(T,K,K)$ transition posterior allocates > 10 GB and OOMs. The
E-step loops over chunks of 8 trials (and $\max(1, 8 / K_R) = 1$ for the joint model).

**`labels.json` is never fed to training.** It exists solely to score recovered latents. It lives
in a separate file from `sequences.json` for exactly that reason.
