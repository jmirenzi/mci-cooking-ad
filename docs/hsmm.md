# `src/cook_ad/hsmm/` — the model

This is the mathematical core. Everything else in the repo consumes what this package produces.

| File | Role |
|---|---|
| `params.py` | `HSMMParams` container, weak-limit init, Dirichlet-MAP normalisation |
| `emissions.py` | 15 lines: per-tick, per-state emission log-likelihood |
| `durations.py` | negative-binomial pmf/survival/cdf/hazard, censoring imputation, the NB M-step |
| `messages.py` | forward / backward / predictive-occupancy recursions and E-step statistics |
| `em.py` | the batched EM driver for the single (recipe-agnostic) HSMM |
| `joint_params.py` | `JointHSMMParams`, per-recipe normalisation, marginal / conditional collapses |
| `joint_em.py` | joint EM over the recipe mixture, plus recipe inference |

---

## 1. The generative model, precisely

A trial is a sequence of **segments** $j = 1,\dots,J$. Segment $j$ has state $z_j$ and duration
$d_j \ge 1$; the segments tile the ticks $0..T-1$.

$$
\begin{aligned}
z_1 &\sim \operatorname{Cat}(\boldsymbol\pi^{\text{init}}), &&\\
d_j \mid z_j = k &\sim \operatorname{NB}_{\ge 1}(r_k, p_k), &&\\
o_t \mid z_j = k &\sim B^v_{k,\cdot} \otimes B^n_{k,\cdot} \quad \text{i.i.d.}, && t \in \text{segment } j,\\
z_{j+1} \mid z_j = k &\sim A_{k,\cdot}, && A_{kk} = 0 .
\end{aligned}
$$

The joint likelihood of one trial with a given segmentation is

$$
P(o_{0:T-1}, z_{1:J}, d_{1:J})
= \pi^{\text{init}}_{z_1}\prod_{j=1}^{J} P(d_j \mid z_j)\, L(u_j,\, u_j{+}d_j{-}1,\, z_j)
\prod_{j=1}^{J-1} A_{z_j z_{j+1}},
$$

where $u_j = \sum_{i<j} d_i$ is segment $j$'s start tick and

$$
L(a,b,k) \;=\; \prod_{t=a}^{b} P(v_t \mid k)\,P(n_t \mid k)
$$

is the **segment likelihood**. Inference marginalises over $\{z_j, d_j, J\}$.

### `emissions.py`

$$
\log \ell_t(k) \;=\; \log B^v_{k, v_t} + \log B^n_{k, n_t}
$$

is a single fancy-index and add, `(T,K)`, with padded ticks zeroed by mask (contributing
$\log 1 = 0$). The conditional independence of $v$ and $n$ given $Z$ is *the* assumption that
later lets `anomaly/` attribute a surprise to the object or the action separately.

---

## 2. `durations.py` — the negative binomial

### Support convention

Durations live on $d = 1,2,3,\dots$ (a segment occupies at least one tick), implemented as a
standard NB on $d' = d - 1 \ge 0$:

$$
\boxed{\;P(D = d) \;=\; \frac{\Gamma(d - 1 + r)}{\Gamma(r)\,\Gamma(d)}\; p^{\,r} (1-p)^{\,d-1}\;}
\qquad d \ge 1 .
$$

Moments (**note the shift** — forgetting the $+1$ is a real bug source, and
`narrate.Lexicon.expected_duration` carries a comment about it):

$$
\mathbb{E}[D] = 1 + \frac{r(1-p)}{p}, \qquad
\operatorname{Var}[D] = \frac{r(1-p)}{p^{2}} .
$$

The variance-to-mean ratio of $d'$ is $1/p > 1$: NB is **over-dispersed** relative to Poisson,
and $r \to \infty$ with $rp$ fixed degenerates to Poisson. Real cooking-step durations are
over-dispersed, which is exactly why NB was chosen; it is also why the estimator has to guard the
$r\to\infty$ direction (§2.4).

### 2.1 Tail functions via the regularised incomplete beta

All three tails reduce to $I_p(a,b)$ (`jax.scipy.special.betainc`):

$$
P(D \le d) = I_p(r,\, d), \qquad
P(D \ge d) = \begin{cases} 1 & d \le 1\\ 1 - I_p(r,\, d-1) & d \ge 2\end{cases}
$$

The $d \le 1$ branch is handled by an explicit `jnp.where`, not by letting `betainc` see a second
shape parameter of $0$ (which is out of its domain). Both were verified against
`scipy.stats.nbinom` to ~1e-6.

The **hazard**

$$
h(d) \;=\; \frac{P(D=d)}{P(D \ge d)} \;=\; P(D = d \mid D \ge d)
$$

is guarded: once the fitted tail has genuinely underflowed, $\log P(D{=}d) - \log P(D{\ge}d)$
would be $-\infty - (-\infty) = \text{NaN}$, so the code returns $-\infty$ instead.

### 2.2 Truncation at $D_{\max}$

`duration_tables(dur_r, dur_p, d_max)` builds two `(K, D_max)` tables. The **last column of the
pmf table is overwritten with the survival value**:

$$
\widetilde{P}(D = d) = \begin{cases}
P(D=d) & d < D_{\max}\\[2pt]
P(D \ge D_{\max}) & d = D_{\max}
\end{cases}
$$

so the infinite NB tail is absorbed rather than truncated away. A segment genuinely longer than
$D_{\max}$ is bookkept as ending at $D_{\max}$, and $\sum_d \widetilde P(D=d) = 1$ exactly. Every
histogram elsewhere in the repo (`warm_start._hard_duration_histogram`,
`duration_drift.duration_histogram`) clamps into the last bin to match this convention.

### 2.3 Right-censoring and ECM imputation

The final segment of every trial is right-censored: we know only $D \ge d_c$. The E-step
therefore returns **two** histograms per state — `xi_dur[k,d]` (exactly-observed segment
durations) and `cens[k,d_c]` (censored ones).

`impute_censored_histogram` redistributes each censored count over the compatible durations under
the *current* fit:

$$
\hat n_k(d) \;=\; \xi^{\text{dur}}_k(d) \;+\; \sum_{d_c \le d}
\operatorname{cens}_k(d_c)\;\underbrace{\frac{P_{\text{old}}(D = d \mid k)}{P_{\text{old}}(D \ge d_c \mid k)}}_{= \; P_{\text{old}}(D = d \mid D \ge d_c,\, k)} .
$$

This is a textbook **ECM** step: it is re-run every M-step, so as $(r,p)$ improve so does the
imputation. It is mass-conserving — for a fixed $d_c$ the weights sum to 1 over $d \ge d_c$ by
construction, since numerator and denominator come from the same table. Nothing is created or
destroyed, only redistributed.

Skipping this biases every duration **downward**, because the longest segment in each trial is
precisely the one that gets truncated.

### 2.4 The NB M-step

From an imputed histogram define the sufficient statistics

$$
N_k \;=\; \sum_d \hat n_k(d), \qquad S_k \;=\; \sum_d \hat n_k(d)\,(d - 1) .
$$

**$p$ given $r$ has a closed form.** Setting $\partial \ell/\partial p = 0$:

$$
\boxed{\; \hat p_k(r) \;=\; \frac{r N_k}{r N_k + S_k} \;}
$$

**$r$ does not.** Profiling $p$ out, the score is

$$
g(r) \;=\; \frac{\partial \ell}{\partial r}
\;=\; \sum_d \hat n(d)\Bigl[\psi(d - 1 + r) - \psi(r)\Bigr] \;+\; N \log \hat p(r),
$$

$$
g'(r) \;=\; \sum_d \hat n(d)\Bigl[\psi_1(d - 1 + r) - \psi_1(r)\Bigr]
\;+\; N\left(\frac{1}{r} - \frac{N}{rN + S}\right),
$$

with $\psi$ = digamma, $\psi_1$ = trigamma. `newton_update_r` runs a **fixed 5 iterations** of
Newton, $r \leftarrow r - g(r)/g'(r)$, inside a `jax.lax.scan` (fixed count keeps shapes static
and the whole thing jittable).

**Why Newton is seeded from method-of-moments, not from `r_old`.** This is the sharpest
numerical point in the module, and the docstring records a directly measured failure. $g(r)$ is
*not* globally well-behaved: it decreases past the true root, reaches a minimum, then flattens
back toward $0$ as $r \to \infty$ **without ever re-crossing** (the NB $\to$ Poisson degeneracy).
Newton started on the far side of that minimum has no path back and diverges up the flattening
tail — starting at $r = 10$ against data truly generated by $r = 4$ diverged to $r \approx 38{,}000$
within 30 iterations.

The fix is to seed from the method of moments. With $\mu' = S/N$ and
$\sigma'^2 = \bigl(\sum_d \hat n(d)(d-1)^2\bigr)/N - \mu'^2$ the moment estimates of $d'$:

$$
r_{\text{mom}} \;=\; \frac{\mu'^{\,2}}{\sigma'^{\,2} - \mu'}
\qquad\text{(valid only when } \sigma'^2 > \mu' \text{, i.e. genuinely over-dispersed).}
$$

Under-dispersed histograms — which NB simply cannot represent, and which are common in sparse
cells — fall back to a fixed $r = 5$. `r_old` survives only as the *starved-state* fallback:
if $N_k \approx 0$ the update leaves that state's $r$ completely unmoved rather than letting a
numerically-live-but-meaningless step drift it.

### 2.5 Shrinkage for the joint model — `fit_durations_shrunk`

The joint model needs $(r,p)$ per **(recipe, state)** cell: $K_R \times K$ = 1024 cells at full
scale, of which ~931 were measured near-empty. Method-of-moments seeding on a nearly-empty
histogram is exactly where §2.4's failure mode lives. So each cell's histogram is inflated with a
pooled global shape:

$$
\hat n^{\text{shrunk}}_{r,k}(d) \;=\; \hat n_{r,k}(d) \;+\; \kappa\, P_{\text{global}}(D = d \mid k),
\qquad \kappa = 5 .
$$

$\kappa$ is a **pseudocount budget in units of expected segments**: a starved cell is lifted into
the well-behaved regime, while a well-populated cell ($N_{r,k} \gg \kappa$) is essentially
unperturbed. This is a hierarchical prior in all but name — each recipe's copy of state $k$ is
shrunk toward the recipe-marginal shape of state $k$.

**Order is load-bearing** and the docstring says so:

```
impute per cell with that cell's own old (r,p)
    → pool over recipes for one global per-state fit
        → shrink each cell toward the global pmf
            → re-fit per cell
```

Shrinking before imputing would charge censored mass against the wrong $(r,p)$.

**Global damping.** The pooled global fit is itself refit from scratch each M-step, and for a
near-empty state its pooled histogram is thin and noisy. Measured on a real full-scale
checkpoint: $r_{\text{global}}$ swung $471 \to 38$ in one M-step for a state where every recipe's
own cell had ~0 occupancy. Because $\kappa P_{\text{global}}$ is injected into **all $K_R$**
copies of that state, one noisy estimate perturbs 16 cells simultaneously — shared instability,
not per-cell noise that averages out. The fix is an EMA across M-step calls:

$$
r^{(m)}_{\text{global}} \;=\; \lambda\, r^{(m-1)}_{\text{global}} \;+\; (1-\lambda)\, r^{\text{fresh}},
\qquad \lambda = \texttt{global\_damping} = 0.7 \ \text{(full)},\ 0.0 \ \text{(mini)} .
$$

$\lambda = 0$ reproduces the undamped behaviour exactly. The EMA state lives in
`run_joint_em`'s local variables, **not** in `JointHSMMParams` — a deliberate choice so existing
checkpoints stay loadable, at the cost of damping re-warming over a few iterations after a
resume.

---

## 3. `params.py` — parameterisation and Dirichlet-MAP

### The container

```python
HSMMParams(init_counts (K,), trans_counts (K,K), verb_counts (K,V),
           noun_counts (K,N), dur_r (K,), dur_p (K,))
```

Note that the categorical families are stored as **Dirichlet concentration parameters**
(prior + accumulated data), not as normalised probabilities. Durations are stored as point
estimates, because NB is not conjugate to anything convenient — a distinction that propagates all
the way to `lifecycle/`, where verb/noun/trans get bounded online count bumps and durations
structurally cannot.

### Weak-limit initialisation

Per restart, each row is drawn from a symmetric Dirichlet and rescaled to the target
concentration:

$$
\mathbf{q} \sim \operatorname{Dir}(\alpha/W,\dots,\alpha/W), \qquad \text{counts} = \alpha \mathbf{q} .
$$

A random draw per restart is the **only** source of symmetry breaking: a symmetric prior plus a
symmetric likelihood has a symmetric posterior, so EM would otherwise sit at a saddle. This is why
restarts matter here more than in a typical fit.

Two different $\alpha$ regimes, on purpose:

- **init / trans**: $\alpha = 0.5$, so $\alpha/K \ll 1$ — a *sparsity* prior. This is the weak
  limit: unneeded states get pushed to the floor rather than absorbing noise.
- **verb / noun**: $\alpha = W$ (i.e. per-category $\alpha = 1$) — an ordinary flat prior over a
  closed vocabulary. Per-category $\alpha = 1$ is log-concave and never triggers the clipped-MAP
  hazard below.

Durations initialise from a uniform mean in $[3, 40]$ ticks and $r \in [1,20]$, converted via
$p = r/(r + \mu')$.

The transition diagonal is drawn and then zeroed (`trans_counts * (1 - I)`). The discarded mass
is ~$1/K$ of a prior row — negligible, and cheaper than deriving a $(K{-}1)$-wide Dirichlet.

### Normalisation is a *clipped* MAP

The mode of $\operatorname{Dir}(c_1,\dots,c_W)$ is $\hat\theta_w = (c_w - 1)/\sum_{w'}(c_{w'} - 1)$,
valid only when every $c_w \ge 1$. With $\alpha/K < 1$ a near-empty state can drive
$c_w - 1$ **negative**, which is not a probability and would feed $\log$ a non-positive number.
`_row_normalize` therefore computes

$$
\hat\theta_w \;=\; \frac{\max(c_w - 1,\, \varepsilon)}{\max\!\bigl(\sum_{w'} \max(c_{w'} - 1,\, \varepsilon),\, \varepsilon\bigr)},
\qquad \varepsilon = 10^{-12},
$$

**flooring before every `log`, never after.** When all $c_w \ge 1$ this is exactly the textbook
MAP; when some are not, it degrades gracefully instead of producing NaN.

`mask_diag=True` (used for the transition rows) zeroes the diagonal *before* the row sum, not
just after taking the log — otherwise a structurally banned self-transition would quietly steal
normalisation mass from the $K-1$ real entries.

`to_log_probs` bundles the four normalised log-tables plus the two duration tables into
`HSMMLogProbs`. Everything downstream consumes `HSMMLogProbs`, never raw counts.

### What the mode does to rare transitions

Subtracting a whole count is a rounding error when a cell holds hundreds and an erasure when it
holds one. With $\alpha_{\text{trans}}/K = 0.0078$, a bigram observed **exactly once** has
$c - 1 = 0.0078$ and floors; one observed twice has $c - 1 = 1.008$. Benign for the emission rows
(prior $\alpha_{\text{emit}} = \text{width}$, i.e. 1 per category, so the mode is the plain data
frequency), not benign for the joint model's transition rows: at $K_R = 16$ there are ~150
observed transitions per recipe over a $64 \times 63$ grid, so a large share of the model's
*legal* transitions are singletons.

The result is an $s_{\text{transition}}$ miscalibrated in both directions at once. Rare but legal
transitions score like impossible ones, and since the per-state $\alpha$-quantile threshold is
computed against that same distorted row, the threshold rises to cover them — which is then what
stops genuinely impossible transitions clearing it.

`smooth_params.py` undoes this post-fit, with no inference change: storing $c + s$ makes
`_row_normalize`'s numerator $c - 1 + s$.

| $s$ | a singleton | a never-observed cell |
|---|---|---|
| $0$ | at the floor | at the floor |
| $0 < s < 1$ | at $s$, off the floor | at the floor, ~32 nats |
| $1$ | the posterior mean | ~9 nats |

$s \approx 0.7$ measures best; $s = 1$, the principled predictive distribution, is worse because
it also lifts never-observed transitions to ~9 nats and the structural channels lose their top
end. `--backoff-tau` additionally mixes each recipe's rows toward the pooled-over-recipes row —
the transition analogue of $\kappa$ in §2.5, and the only pooling those rows otherwise get. Both
are regularisation: select them on held-out data ([`eval.md`](eval.md) §7).

---

## 4. `messages.py` — inference

All three recursions share one trick and one data structure.

**The cumsum trick.** Segment likelihoods over arbitrary windows are needed constantly. With
$\mathrm{Cum}[i,k] = \sum_{t<i}\log \ell_t(k)$ (prepended with a zero row),

$$
\log L(a, b, k) \;=\; \mathrm{Cum}[b{+}1, k] - \mathrm{Cum}[a, k],
$$

so any window is an $O(1)$ subtraction. `_padded_cumsum` edge-replicates by $D_{\max}$ on both
sides so a `dynamic_slice` of any window stays in bounds; validity is then enforced by **explicit
masks**, never by relying on the padding value.

**The window carry.** Each recursion is a single `jax.lax.scan` over ticks with a
$(D_{\max}, K)$ carry. In the forward pass `window[r, k] = F(t - r, k)`, the log-probability that
a *new segment* starts at boundary tick $t-r$; lookback $r$ corresponds to duration $d = r+1$.
Advancing a tick pushes a new row on the front and drops the last.

### 4.1 Forward pass (Yu 2010, explicit-duration)

Define

$$
F(u,k) = P\bigl(\text{a segment in state } k \text{ starts at } u,\; o_{0:u-1}\bigr).
$$

Then per tick $t$:

$$
\begin{aligned}
A^{*}(t,k) &= \sum_{d=1}^{D_{\max}} F(t{-}d{+}1,\,k)\; \widetilde P(D{=}d \mid k)\; L(t{-}d{+}1,\, t,\, k)
&&\text{segment \textit{ends exactly} at } t,\\[4pt]
A^{\text{occ}}(t,k) &= \sum_{d=1}^{D_{\max}} F(t{-}d{+}1,\,k)\; P(D \ge d \mid k)\; L(t{-}d{+}1,\, t,\, k)
&&\text{segment \textit{occupies} } t,\ \text{may continue},\\[4pt]
F(t{+}1,\,k') &= \sum_{k} A^{*}(t,k)\, A_{k k'} .
\end{aligned}
$$

The normaliser and the incremental log-likelihood are

$$
c_t \;=\; \sum_k A^{\text{occ}}(t,k) \;=\; P(o_{0:t}),
\qquad
\log P(o_t \mid o_{0:t-1}) \;=\; \log c_t - \log c_{t-1},
$$

which is the per-tick `log_norm`, and $\log Z = \sum_t \log\text{norm}_t = \log c_{T-1}$.

**This is where right-censoring is handled, and it costs nothing.** Because the normaliser uses
the *survival*-weighted $A^{\text{occ}}$ rather than the pmf-weighted $A^*$, a sequence that stops
mid-segment is scored correctly at any true length with **no per-sequence special-casing**. The
mask simply freezes the carry past $T_{\text{true}}$.

Base case: $\texttt{window}[0,\cdot] = \log \boldsymbol\pi^{\text{init}}$, all other rows
$-\infty$.

### 4.2 Predictive occupancy — `predictive_occupancy`

This is the object the anomaly detector actually needs:

$$
\boxed{\;\tilde\pi_t(k) \;=\; P\bigl(Z_t = k \mid o_{0:t-1}\bigr)\;}
$$

— the belief over the current subtask **before** seeing tick $t$'s observation. Scoring $o_t$
against a posterior that already conditions on $o_t$ would be circular; scoring it against the
smoothed $\gamma_t$ would additionally use the future.

It is computed by the *same* recursion as forward, with one extra term per step that reads the
cumsum window through $t-1$ (excluding $o_t$) and pairs it with the survival weight (the segment
may still be in progress):

$$
\tilde\pi_t(k) \;\propto\; \sum_{d=1}^{D_{\max}} F(t{-}d{+}1,\,k)\; P(D \ge d \mid k)\;
L(t{-}d{+}1,\; t{-}1,\; k),
$$

row-normalised over $k$. Crucially the *carry advance* still uses the real $A^*$ computed **with**
$o_t$, so later ticks benefit from $o_t$ once it is legitimately observed. Only the readout is
causal.

### 4.3 Backward pass

Mirrored scan run in reverse (`jax.lax.scan(..., reverse=True)`) with carry
`bwin[r,k] = ` $B^*(t{+}1{+}r,\,k)$:

$$
\begin{aligned}
G(u, k) &= \sum_{d=1}^{D_{\max}} w_d(k)\; L(u,\, u{+}d{-}1,\, k)\; B^{*}(u{+}d{-}1,\, k),\\
B^{*}(t, k) &= \sum_{k'} A_{k k'}\, G(t{+}1,\, k'),
\qquad B^{*}(T_{\text{true}}{-}1,\, k) = 1 .
\end{aligned}
$$

The base case is at $T_{\text{true}} - 1 = \bigl(\sum \texttt{mask}\bigr) - 1$, **not** at a fixed
array index — the mask is a prefix of `True`s and the padded tail must never contaminate real
positions.

**The censoring weight is the subtle part**, and the docstring flags it as something a first
draft got wrong:

$$
w_d(k) = \begin{cases}
P(D \ge d \mid k) & \text{if } u + d - 1 = T_{\text{true}} - 1 \quad (\text{this segment is the censored one})\\
\widetilde P(D = d \mid k) & \text{otherwise.}
\end{cases}
$$

Using the pmf unconditionally double-counts the boundary segment as both possibly-exact and
implicitly censored.

### 4.4 E-step sufficient statistics — `combine_sufficient_stats`

Runs forward + backward, recovers $F$ from $A^*$ via `_boundary_from_astar`, then combines.

**Transitions:**
$$
\xi^{\text{trans}}_{ji} \;=\; \frac{1}{Z}\sum_{t=0}^{T_{\text{true}}-2} A^{*}(t,\,j)\; A_{ji}\; G(t{+}1,\, i).
$$

**Segment posterior.** For a segment starting at $u$ in state $k$ with duration $d$:

$$
q(u, k, d) \;=\; \frac{1}{Z}\, F(u,k)\; w_d(k)\; L(u,\, u{+}d{-}1,\, k)\; B^{*}(u{+}d{-}1,\, k),
$$

with $w_d$ as in §4.3 (and $B^* = 1$ on the final segment, so it drops out). Then

$$
\xi^{\text{dur}}_k(d) = \!\!\sum_{u:\,u+d-1 < T_{\text{true}}-1}\!\! q(u,k,d),
\qquad
\operatorname{cens}_k(d) = \!\!\sum_{u:\,u+d-1 = T_{\text{true}}-1}\!\! q(u,k,d),
$$

$$
\gamma_t(k) \;=\; \sum_{d=1}^{D_{\max}} \ \sum_{u \,=\, t-d+1}^{t} q(u,k,d),
$$

the inner range-sum again done by cumsum in $O(1)$ per $(t,d)$.

**Why this is scanned over $d$, not vectorised.** For fixed $d$, every quantity needed reduces to
a shift-and-add over already-materialised `(T,K)` arrays. Scanning over $d$ keeps peak memory at
$O(TK)$ per step instead of $O(T D_{\max} K)$ overall. Likewise `xi_trans` is scanned over $t$:
the dense $(T,K,K)$ form allocates > 10 GB at $K=64,\ T=650$ batched over 503 sequences — a
measured OOM, not a hypothetical.

---

## 5. `em.py` — the single-model EM driver

### E-step

`e_step` normalises, optionally tempers, and loops over chunks of `chunk_size` sequences,
`vmap`-ing emissions + `combine_sufficient_stats` within each chunk and summing statistics. The
accumulated statistics are:

$$
\begin{aligned}
c^{\text{init}}_k &= \textstyle\sum_i \gamma^{(i)}_0(k), &
c^{\text{trans}}_{jk} &= \textstyle\sum_i \xi^{\text{trans},(i)}_{jk},\\
c^{v}_{kw} &= \textstyle\sum_i \sum_t \gamma^{(i)}_t(k)\,\mathbf{1}[v^{(i)}_t = w], &
c^{n}_{kw} &= \textstyle\sum_i \sum_t \gamma^{(i)}_t(k)\,\mathbf{1}[n^{(i)}_t = w].
\end{aligned}
$$

The emission accumulations are one-hot einsums, `"ntk,ntv->kv"`.

**Deterministic annealing** (off by default) divides the emission and duration log-tables by a
temperature $\Upsilon$:

$$
\log \tilde P \;=\; \frac{1}{\Upsilon}\log P, \qquad
\Upsilon_m = \max\bigl(1,\ 2.0 \cdot 0.9^{\,m}\bigr).
$$

$\Upsilon > 1$ flattens the likelihood surface, letting early iterations escape shallow local
optima before the true objective is restored at $\Upsilon = 1$. Note the transition and initial
tables are *not* tempered.

### M-step

Fresh Dirichlet posteriors each iteration — this is standard **batch** MAP-EM, not an
accumulation across iterations:

$$
c^{\text{new}}_{kw} \;=\; \frac{\alpha}{W} \;+\; \hat c_{kw},
$$

with the transition result re-masked by $(1 - I)$. Durations go through
`impute_censored_histogram` → `newton_update_r` → `update_p_given_r` (§2).

### Restart loop

`run_em` is a plain Python loop: fresh random init, iterate to $|\Delta \log Z| < \texttt{tol}$ or
`max_iters`, keep the best by **final total log-likelihood**. There is no held-out metric at this
stage of the pipeline, so model selection is in-sample by construction.

---

## 6. The joint model — `joint_params.py` / `joint_em.py`

### What changes

$$
P\bigl(o^{(i)}\bigr) \;=\; \sum_{r=1}^{K_R} \pi_r \; P\bigl(o^{(i)} \mid \theta_r\bigr),
$$

one **discrete recipe latent per trial**. Parameters:

```python
JointHSMMParams(init_counts (K_R,K), trans_counts (K_R,K,K),
                verb_counts (K,V),   noun_counts (K,N),      # ← SHARED
                dur_r (K_R,K), dur_p (K_R,K), pi_counts (K_R,),
                kernel_v=None, kernel_n=None,                 # ← SHARED, optional (see kernel.py)
                noun_tilt=None)                                # ← per-recipe, optional (§6.1)
```

Emissions are **shared across recipes**; dynamics (init, transitions, durations) are per-recipe.
That is the whole modelling claim: *"pour milk" looks the same in cereals and in coffee; what
differs is when you do it and how long it takes.* Sharing emissions also keeps the E-step cheap —
`loglik` is computed once per trial and broadcast across all $K_R$ forward-backward passes.

`kernel_v` / `kernel_n` are the fixed semantic-neighbourhood kernel (see `kernel.py`'s module
docstring) — shared, like the emission counts they modulate, since they describe vocabulary
structure, not recipe identity. `noun_tilt` (below) is the one field that breaks the "content is
recipe-agnostic" rule on purpose.

`to_log_probs_joint` mirrors `to_log_probs` with `vmap` wherever the helper assumes a single
recipe's rank (`mask_diag` builds `jnp.eye(counts.shape[0])`, so it must see a `(K,K)` slice, not
`(K_R,K,K)`; the duration tables assume 1-D `dur_r`). `pi` is fed through as one `(1,K_R)` row.

### E-step

Per trial $i$ and recipe $r$, run the ordinary HSMM machinery to get $\log Z_{ir}$, then

$$
\log \rho_{ir} \;=\; \log \pi_r + \log Z_{ir} - \operatorname*{logsumexp}_{r'}\bigl(\log\pi_{r'} + \log Z_{ir'}\bigr),
$$

$$
\mathcal{L} \;=\; \sum_i \operatorname*{logsumexp}_{r}\bigl(\log \pi_r + \log Z_{ir}\bigr)
$$

is the objective. Note `rho` is computed from the normalised log form — never
`exp(raw log_z)`, which underflows immediately.

Statistics are $\rho$-weighted:

$$
c^{\pi}_r = \sum_i \rho_{ir}, \qquad
c^{\text{trans},(r)}_{jk} = \sum_i \rho_{ir}\, \xi^{\text{trans},(i,r)}_{jk},
$$

and for the **shared** emissions the recipe axis is collapsed *first*,

$$
\bar\gamma^{(i)}_t(k) \;=\; \sum_r \rho_{ir}\, \gamma^{(i,r)}_t(k),
$$

so the emission accumulation is shape-identical to the single-recipe version.

Memory scales an extra $K_R$-fold (the `(chunk, K_R, T, K)` gamma), hence the config's derived
`chunk_size = max(1, 8 // K_R) = 1` at full scale.

### M-step and monotonicity

Dirichlet MAP per recipe for init/trans/$\pi$, shared MAP for emissions, and
`fit_durations_shrunk` (§2.5) for durations. Because the shrinkage duration fit is **not an exact
M-step**, $\mathcal L$ can dip slightly; `run_joint_em` therefore *warns* rather than raises when
the objective decreases by more than `tol`. (The saved full-scale history shows exactly this:
a rise to about $-16431$, then sub-nat oscillation.)

### Recipe-modulated emissions — `noun_tilt`

**The problem this solves.** Emissions are shared, so a recipe is distinguished only by its
init/trans/duration tables — content reaches the recipe latent only indirectly, via which states
a recipe happens to favour. That indirect path is starved at any real $K$: `params._row_normalize`'s
MAP mode floors any transition cell observed once or never, and most per-recipe transition cells
land there (measured on EPIC at $K{=}128$: 2.1% of cells survive; Breakfast at $K{=}64$: 31.3%).
With the recipe-conditioned transition likelihood mostly floor, $\rho$ is close to noise.

**The fix.** `noun_tilt` $\in \mathbb R^{K_R \times N}$ gives the recipe latent one *direct* channel
into content: a rank-1 per-recipe reweighting of the shared noun table,

$$
P_r(n\mid k) \;=\; \operatorname*{softmax}_n\bigl(\log P(n\mid k) + a_r[n]\bigr),
\qquad
\log Z_r[k] \;=\; \operatorname*{logsumexp}_n\bigl(\log P(n\mid k) + a_r[n]\bigr),
$$

so the tick loglik under recipe $r$ is

$$
\log L_r(t,k) \;=\; \log L_{\text{shared}}(t,k) \;+\; a_r[n_t] \;-\; \log Z_r[k].
$$

`joint_params.tilt_terms(log_emit_n, noun_tilt, k_r)` computes $(a_r, \log Z_r)$ (or an all-zero
pair when `noun_tilt is None`, making the term an exact no-op — `test_noun_tilt_zero_matches_none_in_e_step`).
`joint_params.log_emit_n_recipe(log_probs, r)` returns the resulting $(K,N)$ table for one recipe.

**The cost stays $O(\text{chunk}\times K_R \times T)$, not $K_R$-fold on `sequence_loglik`.**
$\log L_r(t,k)$ splits into a $(t)$-only term ($\log L_{\text{shared}}$, still computed exactly
once per trial) plus a $(t,k)$-only broadcast add ($a_r[n_t] - \log Z_r[k]$) formed inside the
recipe `vmap`, at the `(chunk,K_R,T,K)` shape `gamma` already occupies. `_e_step_chunk` and
`_recipe_logz_chunk` (decode) apply this identically — they must, or `infer_recipe`'s $\hat r$
stops matching the $\rho$ training actually converged to.

**Why it works.** Because every HSMM path covers every tick exactly once, $\sum_t a_r[n_t]$
contributes additively to $\log Z_{ir}$ — the rho-weighted noun bag lands directly in the recipe
responsibility, bypassing the starved transition channel entirely.

**M-step: a GIS coordinate step, not an exact solve.** `m_step`'s `tilt_steps` (default 0, a
no-op) runs that many generalized-iterative-scaling updates *after* the shared `noun_counts`
update, matching the model's predicted per-recipe noun marginal to the $\rho$-weighted noun bag
(`stats["tilt_noun_counts"]`) and per-recipe state occupancy (`stats["occ"]`):

$$
a_r \;\mathrel{+}=\; \eta \Bigl(\log \hat c_r \;-\; \log q_r\Bigr), \qquad
q_r[n] = \operatorname*{logsumexp}_k\bigl(\log\widehat{\text{occ}}_r[k] + \log P_r(n\mid k)\bigr),
$$

then re-centred (the shift is unidentifiable — $\log Z_r$ absorbs any constant) and clipped to
$\pm$`tilt_max`. Because this holds the just-updated shared `noun_counts` fixed rather than
re-solving jointly, the M-step is only *approximately* exact — the same status as the shrinkage
duration fit, and for the same reason `run_joint_em` warns rather than raises on a small decrease.

**Not free everywhere.** `select_recipe` / `collapse_to_marginal` return only the shared emission
counts (there is no per-recipe axis to hold a tilt), so both `warnings.warn` when `noun_tilt is
not None` — propagating the tilt into `surprise`/`quantile`/`narrate` is not yet done.

**Seeding.** `lexical_init.lexical_to_joint(..., noun_tilt_init=True)` seeds $a_r$ from the SAME
cluster assignment `cluster_recipes` already computed — $a_r[n] = \log(\text{cluster-}r\text{ noun
freq}) - \log(\text{global noun freq})$, centred and clipped to `noun_tilt_clip` — needing no
extra pass over the data.

### The anchoring dial — `lam`

A per-trial external recipe prior, added to the E-step's responsibility term with a strength dial:

$$
\log \rho_{ir} \;=\; \log \pi_r + \log Z_{ir} + \lambda \cdot \log(\text{prior}_{ir}).
$$

One parameter spans three behaviours: $\lambda{=}0$ is unchanged; small $\lambda$ biases the
assignment toward the prior while the likelihood can still override it; large $\lambda$ freezes
the assignment to the prior. "Freeze the recipe assignment for $N$ iterations, then release it" is
therefore a *schedule* over $\lambda$ (`make_lam_schedule`, forms `const`/`geom`/`linear`/`freeze`),
not separate machinery — freezing only changes the EM *path*, not the fixed point at $\lambda{=}0$.

**Validity.** At a fixed $\lambda$, $\exp(\lambda \log\text{prior}_{ir})$ is a $\theta$-independent
per-$(i,r)$ constant folded into the generative model, so EM's usual monotonicity argument still
applies to an exact M-step — `test_objective_non_decreasing_at_constant_lam`. A *moving* $\lambda$
changes the objective between iterations by construction, so `run_joint_em` suppresses its
decrease-warning on any iteration where $\lambda$ actually moved (a "decrease" there is comparing
two different objectives, not a broken M-step) and does not treat that iteration as convergence.

**The reordering trap.** `e_step` sorts trials by length before chunking (`_length_order`) so
chunks are length-homogeneous, and slices `verb_ids`/`noun_ids`/`mask` with that permutation —
`recipe_log_prior` must use the *same* permutation or it silently applies to the wrong trial
(`test_lam_prior_respects_the_length_reorder`). `infer_recipe`, by contrast, does **not** reorder
(decode's per-chunk cost is cheap enough that the reorder isn't worth the bookkeeping), so there
`recipe_log_prior` is sliced plain `[start:end]` — using `_length_order` there would be the bug.

**Where the prior comes from is a separate question.** `run_joint_lexical.py --recipe-prior
warmstart` builds a one-hot-ish prior from the lexical warm start's own cluster assignment — needs
no new source, so it is enough to exercise the dial, but is not a semi-supervised signal from
session content. See `docs/recipe.md` §4 for that (deferred, lower-priority) idea and the
circularity it would need to address.

### Per-state emission priors — `emit_prior_v` / `emit_prior_n`

The shared emission M-step's $\alpha_{\text{emit}}/\text{width}$ term is a *flat* prior: right
when the states are anonymous, wrong once the initialisation has given state $k$ a meaning
(`recipe/lexical_init.py` anchors state $k$ on one observed $(v,n)$ pair), since a flat prior lets
EM relicense that state onto whatever the responsibilities hand it. `m_step` therefore accepts
$(K,V)$ / $(K,N)$ prior **matrices** in place of the scalar. Default `None` leaves the flat-prior
behaviour unchanged.

### Restarts, checkpoints, convergence

`run_joint_em` is a **single deterministic run** from `init_params` — no restart loop, because the
warm start (see [`recipe.md`](recipe.md)) is deterministic. It is resumable via
`start_iteration` / `init_history` / `init_prev_obj`, and calls an injected
`on_checkpoint(iteration, params, history)` callback every `checkpoint_every` iterations *after*
the M-step, so the saved params are always exactly the state the next E-step would consume. The
module does **no file I/O itself** — persistence lives in `run_joint.py`, which writes atomically
(temp file + `os.replace`) so a killed process never leaves a half-written checkpoint.

`converged` is `True` only if the tolerance criterion actually fired, never merely because
`max_iters` was reached. A caller resuming should keep calling until `converged`, not until
`history` stops growing.

### Recipe inference — `infer_recipe`

$$
\hat r_i \;=\; \arg\max_r \bigl(\log \pi_r + \log Z_{ir}\bigr)
$$

plus the full posterior $\rho_{i\cdot}$ and the marginal $\log P(o^{(i)})$. This uses
`messages.forward_pass` **directly** rather than the heavier `combine_sufficient_stats` — it needs
only $\log Z$, not $\xi$ or $\gamma$ — so it skips the backward pass entirely. The $\log Z$ it
produces is bit-identical to training's (both are `sum(forward_pass(...)[0])`), so decode can
never drift numerically from the E-step.

### Collapses (`joint_params.py`)

Three ways to get a single-recipe view out of the joint model, each with a distinct purpose:

| Function | Formula | Used for |
|---|---|---|
| `marginal_log_trans` | $\bar A_{jk} = \sum_r \pi_r A^{(r)}_{jk}$ | the $s_{\text{recipe transition}}$ baseline: "how surprising is this transition *in general*" |
| `collapse_to_marginal` | $\pi$-weight every count family into a plain `HSMMParams` | callers with no per-trial recipe concept — e.g. `synthetic.error_injection` |
| `select_recipe(r̂)` | slice recipe $\hat r$'s dynamics + shared emissions | the *exact* per-trial model that anomaly scoring and narration use |

The last distinction matters for honesty of output: a `narrate.Lexicon` built from
`select_recipe` reports expected durations consistent with the surprise actually computed for
that trial, whereas one built from the marginal would quote a cross-recipe average.

---

## 7. Viterbi EM — `run_hard_em.py`

An alternative optimiser for the same joint model, replacing §6's soft E-step with the MAP
segmentation:

$$
\max_{\theta}\ \max_{z}\ \log P_\theta(o, z)
\qquad\text{in place of}\qquad
\max_{\theta}\ \log \sum_{z} P_\theta(o, z)
$$

The wrong estimator if you want calibrated posterior uncertainty, the right one here because
**every surprise channel scores against `z_star`** — so the counts the model is fit from and the
decode the detector reads become the same object.

One iteration: `infer_recipe` → `segment_all_conditioned` → hard init/transition counts and
per-$(r,k)$ duration histograms → the same Dirichlet-MAP renormalisation and `fit_durations_shrunk`
the soft M-step uses. Right-censoring is handled as in §2.3. Emissions stay at the lexical anchor
([`recipe.md`](recipe.md) §4). It converges in ~5 iterations at ~15 s each against soft EM's 60 at
~45 s, holds per-tick subtask ARI at 0.999, and reaches a higher marginal likelihood than the soft
run it started from. `--init-from` takes $K$ and $K_R$ from the checkpoint, not the config.

### The objective is not the selection criterion

Three measurements on the train split, all disagreeing with the likelihood ranking:

- the lexical warm start *begins* at $-12948$, above what a cascade-warm-started joint EM
  *converges* to ($-14245$);
- soft EM from it lowers the objective for a stretch while per-tick subtask ARI degrades from
  0.999 to 0.940;
- the highest-likelihood fit available ($-11863$) has subtask ARI 0.935 and is beaten on detection
  by fits a thousand nats below it.

Model selection between fitting routes therefore runs through `run_detect_eval.py` and
`run_step_sweep.py` ([`eval.md`](eval.md) §7), never through `history[-1]`.
