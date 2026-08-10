# `src/cook_ad/recipe/` — segmentation, the recipe layer, and the joint warm start

| File | Role |
|---|---|
| `segmentize.py` | max-product (Viterbi) decoding of the HSMM → segments |
| `recipe_hmm.py` | the cascade's stage-2 discrete HMM over segment symbols, + clustering metrics |
| `warm_start.py` | cascade artifacts → the joint model's iteration-0 parameters |

This package sits between "a fitted HSMM" and "a per-trial interpretation of what happened".

---

## 1. `segmentize.py` — Viterbi for an explicit-duration HSMM

### The objective

Where the forward pass computes $\log \sum$, Viterbi computes $\log \max$. It returns the single
most probable segmentation:

$$
\bigl(\hat z_{1:J},\ \hat d_{1:J}\bigr) \;=\;
\arg\max_{J,\, z_{1:J},\, d_{1:J}}\ \log P\bigl(o_{0:T-1},\, z_{1:J},\, d_{1:J}\bigr),
$$

subject to $\sum_j d_j = T$. Note this maximises over the *number* of segments too — the model
chooses its own segmentation granularity.

### The recursion

`viterbi_decode` is `messages.forward_pass` with `logsumexp` → `max`, on the same
$(D_{\max}, K)$ window carry and the same cumsum trick, plus backpointers:

$$
\begin{aligned}
A^{*}(t,k) &= \max_{d}\ \Bigl[\,F(t{-}d{+}1,k) + \log \widetilde P(D{=}d\mid k) + \log L(t{-}d{+}1,t,k)\Bigr],
& \texttt{dur\_bp}(t,k) &= \arg\max_d\ (\cdot)\\
A^{\text{surv}}(t,k) &= \max_{d}\ \Bigl[\,F(t{-}d{+}1,k) + \log P(D{\ge}d\mid k) + \log L(t{-}d{+}1,t,k)\Bigr],
& \texttt{dur\_bp\_surv}(t,k) &= \arg\max_d\ (\cdot)\\
F(t{+}1,k') &= \max_{k}\ \bigl[A^{*}(t,k) + \log A_{kk'}\bigr],
& \texttt{prev\_bp}(t,k') &= \arg\max_k\ (\cdot)
\end{aligned}
$$

(all in log space; backpointers are stored as lookback $r = d-1$).

**Two duration-weighted scores, not one.** $A^*$ uses the pmf and drives every ordinary segment.
$A^{\text{surv}}$ uses the survival function and is consulted at exactly one place: a trial's
final tick, where the segment is right-censored. So the terminal state is

$$
\hat k^{*} \;=\; \arg\max_k\ A^{\text{surv}}\bigl(T_{\text{true}}{-}1,\ k\bigr),
$$

and the traceback pops that first segment using `dur_bp_surv` and every earlier one using
`dur_bp`. Same censoring logic as the sum-product backward pass, in max-product form.

### Traceback

`traceback` is plain numpy (it is inherently sequential and cheap):

```
e ← T_true − 1 ;  k ← k*  ;  first ← True
while e ≥ 0:
    r ← (dur_bp_surv if first else dur_bp)[e, k]      # lookback
    d ← r + 1 ;  u ← e − r
    emit segment (k, d)
    if u == 0: break                                   # reached the initial boundary
    k ← prev_bp[u − 1, k] ;  e ← u − 1
reverse
```

The `prev_bp[u-1]` index is the off-by-one that `_boundary_from_astar` handles in `messages.py`:
backpointers are recorded for *boundary time* $t+1$ but stored at index $t$.

Output per trial: `{"segments": [(state, duration), ...], "subtask_per_tick": (T,) int64}`.

### Three drivers

| Function | Tables |
|---|---|
| `segment_all(hsmm_params, ...)` | normalises, then one shared table set for the whole batch |
| `segment_all_from_log_probs(log_probs, ...)` | same, given already-normalised tables |
| `segment_all_conditioned(joint_log_probs, r_hat, ...)` | **per-trial** tables gathered at $\hat r_i$ |

The conditioned variant is the joint model's path: each trial is decoded under its own MAP
recipe's init/transition/duration tables, while emissions stay shared. `_assemble_segment_results`
is factored out so both paths use bit-identical traceback logic.

### Why banning self-transitions pays off here

Because $A_{kk} = 0$, consecutive Viterbi segments always differ in state. A run of constant
`subtask_per_tick` is therefore **exactly one segment** — run-length encoding is lossless, which is
what `narrate.segments_from_z` and `warm_start._init_trans_hard_counts` rely on.

The one wrinkle: $D_{\max}$ truncation can split a genuinely longer action into two adjacent
segments of the same state. `export_flow.py` detects this seam explicitly (a segment of length
$\ge D_{\max}$ immediately followed by a same-state segment) and marks it `clipped`, so a
rendering never reports "you did X twice".

---

## 2. `recipe_hmm.py` — the cascade's stage 2

### The model

Stage 1 gives every trial a sequence of **segment symbols**
$s_1, s_2, \dots, s_S$ with $s_j \in \{1..K\}$ (just the state ids, durations discarded). Stage 2
is a plain discrete HMM over that sequence, where each "tick" is a whole subtask segment:

$$
\begin{aligned}
R_1 &\sim \operatorname{Cat}(\boldsymbol\pi^{\text{init},R}), \qquad
R_{j+1} \mid R_j \sim \operatorname{Cat}\bigl(A^{R}_{R_j,\cdot}\bigr),\\
s_j \mid R_j &\sim \operatorname{Cat}\bigl(E_{R_j,\cdot}\bigr), \qquad E \in \mathbb{R}^{K_R \times K}.
\end{aligned}
$$

```python
RecipeParams(init_counts (K_R,), trans_counts (K_R,K_R), emit_counts (K_R,K))
```

**Two deliberate contrasts with the subtask HSMM:**

1. **No duration model.** Segment lengths were already consumed by stage 1; here each observation
   *is* a segment.
2. **The diagonal is used.** $A^R_{rr} > 0$ — recipe self-transitions are not merely allowed, they
   are the expected case (a trial mostly stays in one recipe). This is the opposite of the subtask
   layer's banned diagonal, and `init_weak_limit_recipe_params` accordingly does *not* zero it.

The same $\alpha$ split applies: $\alpha = 0.5$ sparsity on init/trans (weak limit over $K_R$),
$\alpha = K$ (per-category 1) on the emission, which is an ordinary closed-vocabulary categorical.

### Forward-backward

`_forward_backward` is textbook:

$$
\begin{aligned}
\alpha_0(r) &= \log\pi^{\text{init},R}_r + \log E_{r, s_0}, &
\alpha_j(r) &= \log E_{r,s_j} + \operatorname*{logsumexp}_{r'}\bigl[\alpha_{j-1}(r') + \log A^R_{r'r}\bigr],\\
\beta_{S-1}(r) &= 0, &
\beta_j(r) &= \operatorname*{logsumexp}_{r'}\bigl[\log A^R_{rr'} + \log E_{r',s_{j+1}} + \beta_{j+1}(r')\bigr],
\end{aligned}
$$

$$
\log Z = \operatorname*{logsumexp}_r \alpha_{S_{\text{true}}-1}(r), \qquad
\gamma_j(r) = \alpha_j(r) + \beta_j(r) - \log Z,
$$

$$
\xi_{rr'} = \sum_{j} \exp\!\bigl[\alpha_j(r) + \log A^R_{rr'} + \log E_{r', s_{j+1}} + \beta_{j+1}(r') - \log Z\bigr].
$$

Padding is handled exactly as in `hsmm/messages.py`: $\alpha$ **freezes** past the true end
(`jnp.where(mask[t], alpha_new, alpha_prev)`), and $\beta$ is force-set to $0$ at the *true* final
index $S_{\text{true}}-1$ rather than at the padded array's last index. That force-set is the
firewall — garbage computed beyond the true sequence can never flow back into real positions.
$\log Z$ is likewise read at $\alpha[S_{\text{true}}-1]$.

EM is the same restart-loop shape as `hsmm.em.run_em`.

### Decode: per-trial recipe by majority vote

The model permits recipe transitions *within* a trial, but the quantity being scored against
Breakfast's `recipe_label` is a single id per trial. `decode_recipe` therefore takes the
per-segment posterior argmax and majority-votes:

$$
\hat R_i \;=\; \operatorname{mode}_j\ \Bigl(\arg\max_r \gamma^{(i)}_j(r)\Bigr).
$$

### Scoring the clusters

Three metrics, all hand-rolled to avoid a scikit-learn dependency.

**`effective_k(recipe_ids, min_frac=0.02)`** — the number of clusters holding a non-negligible
share of trials. This is the number to report from a weak-limit model, *not* the nominal $K_R$.
A plain distinct-label count over-reports: a handful of stray-assigned trials in an otherwise
unused nominal state is noise, so clusters below `min_frac` of the dataset are excluded.

**`adjusted_rand`** — chance-corrected agreement between two labelings. From the contingency
table $n_{ab}$ with margins $a_\cdot, b_\cdot$ and $\binom{n}{2}$ notation:

$$
\mathrm{ARI} = \frac{\sum_{ab}\binom{n_{ab}}{2} \;-\; \dfrac{\sum_a \binom{a_\cdot}{2}\sum_b\binom{b_\cdot}{2}}{\binom{n}{2}}}
{\tfrac12\Bigl[\sum_a\binom{a_\cdot}{2} + \sum_b\binom{b_\cdot}{2}\Bigr] \;-\; \dfrac{\sum_a \binom{a_\cdot}{2}\sum_b\binom{b_\cdot}{2}}{\binom{n}{2}}} .
$$

ARI $= 0$ for chance agreement, $1$ for identical partitions, and can go negative. Degenerate
labelings (zero denominator) return $1.0$, matching sklearn's convention.

**`matched_accuracy`** — clustering has no canonical label order, so accuracy is only meaningful
after an optimal one-to-one alignment. Solve

$$
\max_{\sigma \ \text{injective}} \ \sum_{a} n_{a,\sigma(a)}
$$

with the Hungarian algorithm (`scipy.optimize.linear_sum_assignment` on $-n$), then report
$\bigl(\sum_a n_{a,\sigma(a)}\bigr)/n$. The contingency table and value arrays are returned too, so
the recovered mapping can be inspected rather than taken on faith.

### The cascade's structural weakness

Stage 2 sees only the *symbol sequence*. It never sees durations, never sees the raw
observations, and — critically — **cannot feed back**: a recipe hypothesis cannot change how the
subtask layer segments. Stage 1's transition matrix is a single recipe-agnostic $A$, so
"after pouring cereal you normally pour milk" and "after pouring coffee you normally pour water"
are averaged into one matrix. That averaging is precisely what the joint model removes.

---

## 3. `warm_start.py` — cascade → joint

### Why a warm start at all

The joint objective $\sum_i \operatorname{logsumexp}_r(\log\pi_r + \log Z_{ir})$ is invariant to
permuting recipe labels. From a symmetric random init, EM has no gradient to break that symmetry
and stalls with $K_R$ near-identical recipes. `cascade_to_joint` builds an iteration-0
`JointHSMMParams` in which the recipes are **already genuinely differentiated**, so EM starts
inside one basin rather than on the saddle between all $K_R!$ of them.

### The construction

1. Viterbi-segment every trial with the cascade HSMM (`segmentize.segment_all`).
2. Decode each trial's recipe id $\hat R_i$ with the cascade recipe HMM.
3. Copy the **shared** emission counts verbatim (they are already recipe-agnostic — no work to do).
4. Mixture weights are the raw assignment histogram: $c^\pi_r = \#\{i : \hat R_i = r\}$.
5. For each recipe $r$, build **hard** histograms from that recipe's trials only:

$$
c^{\text{init},(r)}_k = \sum_{i:\hat R_i = r}\!\mathbf{1}[z^{(i)}_1 = k],
\qquad
c^{\text{trans},(r)}_{jk} = \sum_{i:\hat R_i = r}\ \sum_{m}\!\mathbf{1}[z^{(i)}_m = j,\, z^{(i)}_{m+1} = k],
$$

$$
\text{hist}^{(r)}_k(d) = \sum_{i:\hat R_i = r}\ \sum_{m} \mathbf{1}\bigl[z^{(i)}_m = k,\ \min(d^{(i)}_m, D_{\max}) = d\bigr].
$$

6. Fit durations by the same shrinkage step as the real M-step, using the **cascade's own** fitted
   $(r_k, p_k)$ as the global shape:

$$
\hat n^{(r)}_k(d) = \text{hist}^{(r)}_k(d) + \kappa\, P_{\text{cascade}}(D = d \mid k),
$$

then method-of-moments → Newton → closed-form $p$ (see [`hsmm.md`](hsmm.md) §2).

### Three deliberate approximations, each flagged in the code

**Sparse-recipe fallback.** A recipe with fewer than `MIN_TRIALS_PER_RECIPE = 2` assigned trials
cannot support a stable histogram (one trial's segment sequence is not a distribution). Those
recipes inherit the cascade's recipe-agnostic init/trans plus Gaussian noise
(`FALLBACK_NOISE_SCALE = 0.05`) for symmetry breaking. Their duration histogram is left all-zero,
so the shrinkage step falls back entirely to the global shape. This is applied **per recipe**, not
as an all-or-nothing switch for the whole warm start.

**No censoring imputation.** Viterbi segments carry no censoring information, so the final segment
of each trial is treated as exactly observed. That is a one-time optimism (durations start
slightly short) which the first real joint M-step corrects.

**Hard counts, not soft.** The warm start uses $\arg\max$ assignments rather than responsibilities
$\rho_{ir}$. It only needs to be a good *starting point*, not an unbiased estimate — EM immediately
softens everything on iteration 1.

### Where the zero diagonal comes from

`_init_trans_hard_counts` needs no explicit diagonal zeroing: consecutive Viterbi segments always
differ in state, because the underlying HSMM's transition matrix already had $A_{kk} = -\infty$ at
segmentation time. The final `* (1 - I)` in `cascade_to_joint` is belt-and-braces, and matters only
for the fallback path where `fallback_trans` is copied wholesale.
