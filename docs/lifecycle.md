# `src/cook_ad/lifecycle/` — adapting over weeks without going blind

| File | Role |
|---|---|
| `online_update.py` | `PreferenceEvent`, the bounded count bump |
| `state_manager.py` | the frozen/live `DualModel`, confirmation routing, consolidation |
| `divergence.py` | localized KL drift between live and frozen |
| `duration_drift.py` | statistical duration-shift report (with a documented integration gap) |

---

## 1. The problem this package solves

A detector that adapts to everything it sees stops detecting. If the user substitutes almond milk
for cow's milk, confirms it once, and the model absorbs it fully, then the next fifty
substitutions are silent — including the ones that are *not* preferences.

But a detector that never adapts nags forever.

The resolution is a **dual model** plus a **bounded** update, so that accommodation is real but
strictly capped, and permanent change requires a human in the loop.

---

## 2. The dual model — `state_manager.py`

```python
DualModel(frozen: HSMMParams,   # slow reference; moves ONLY at consolidation
          live:   HSMMParams)   # short-horizon copy; absorbs confirmed preferences
```

`init_dual_model` sets `live = frozen` with no defensive copy: JAX arrays are immutable and every
update produces a fresh array via `.at[...].set(...)`, so sharing the reference is safe.

### Confirmation routing

A `Query` from `narrate.py` carries a `PreferenceEvent`. When the user answers, the outcome routes
through `handle_confirmation`:

| Outcome | Meaning | Effect |
|---|---|---|
| `"preference"` | user affirmed their surprising action | bounded bump to **live** only |
| `"breakdown"` | user recognised a mistake | **update nothing**; flag the incident |

The `"breakdown"` branch is the important one. A recognised error is *evidence the model was
right*; learning from it would train the detector to expect the very failures it exists to catch.
The confirmation oracle here is **supplied, not learned** — this package assumes the answer, it
does not infer it.

`frozen` never moves in this function. Only `consolidate` moves it.

---

## 3. The bounded bump — `online_update.py`

```python
PreferenceEvent(channel ∈ {"verb","noun","trans"}, state, token)
```

For `"trans"`, `state` is the **FROM** state and `token` is the **TO** state — matching how
`quantile.py` indexes transition thresholds.

The update is a single capped increment to one Dirichlet count cell:

$$
\boxed{\;
c^{\text{live}}_{k,w} \;\leftarrow\; \min\Bigl(c^{\text{live}}_{k,w} + \delta,\;\; c^{\text{frozen}}_{k,w} + B\Bigr)\;}
\qquad \delta = 1.0,\quad B = 5.0 .
$$

### Why the ceiling is relative to *frozen*, not absolute

This is the mechanism that keeps the detector alive. No matter how many times a substitution is
accepted within a window, the live cell can rise **at most $B$ pseudocounts above the reference**.
Its normalised probability therefore stays well below the state's dominant token:

$$
\hat\theta^{\text{live}}_{kw}
= \frac{c^{\text{frozen}}_{kw} + B - 1}{\sum_{w'} \bigl(c^{\text{live}}_{kw'} - 1\bigr)}
$$

so with a well-populated state ($\sum_{w'} c_{kw'} \gg B$) the token remains rare, and
$-\log \hat\theta^{\text{live}}_{kw}$ stays above the $\alpha$-quantile threshold. **One
acceptance does not blind the detector.** Only a caretaker consolidating the change into the
frozen baseline does.

`bounded_bump` is pure and functional — it returns a new array and mutates nothing.

### Why durations are absent from `COUNT_FIELD`

```python
COUNT_FIELD = {"verb": "verb_counts", "noun": "noun_counts", "trans": "trans_counts"}
```

The three listed families are **conjugate Dirichlet counts**, so "add $\delta$ pseudo-observations"
is a well-defined Bayesian operation. `dur_r` / `dur_p` are **NB point estimates**, not counts —
there is no cell to bump. Giving durations an online update would require carrying a duration
*histogram* as the parameterisation, which is flagged as future work in both this module and
`duration_drift.py`.

---

## 4. Consolidation is a reset, not an increment

```python
consolidate(dual, approved="all" | [PreferenceEvent, ...])
```

Weekly review. The approved live drift **defines the new frozen baseline**, and live is then
re-initialised *from that new frozen*:

$$
\text{frozen}' \leftarrow \text{apply}(\text{approved}), \qquad \text{live}' \leftarrow \text{frozen}' .
$$

This is the precision-growth guard. If consolidation merely *added* accepted drift onto frozen,
counts would compound week over week: the posterior would become ever more concentrated, the model
ever more confident, and eventually nothing could ever be surprising again. Because frozen is
**re-based** and live is short-horizon, precision has nowhere to compound.

With an explicit approval list, `_apply_approved` copies only the named `(state, token)` cells from
live into frozen. Everything unapproved is discarded when live resets — silently reverted, by
construction, with no separate rollback path to get wrong.

---

## 5. Localizing drift — `divergence.py`

Consolidation review needs to know *where* the model drifted, not just that it did. The KL
divergence between two categorical rows,

$$
\mathrm{KL}(p \,\|\, q) \;=\; \sum_w p_w \bigl(\log p_w - \log q_w\bigr),
$$

is computed from log-probability arrays with cells where $p_w = 0$ masked **before** the multiply —
otherwise a floored or banned entry with $\log q_w = -\infty$ gives $0 \cdot (-\infty) = \mathrm{NaN}$.

`model_divergence(live, frozen)` breaks $\mathrm{KL}(\text{live} \,\|\, \text{frozen})$ down by
family **and by state**:

| Key | Shape | Quantity |
|---|---|---|
| `init` | scalar | $\mathrm{KL}$ of the initial-state distribution |
| `trans` | $(K,)$ | per-FROM-state $\mathrm{KL}$ of $P(Z' \mid Z{=}k)$ |
| `verb` | $(K,)$ | per-state $\mathrm{KL}$ of $P(v \mid Z{=}k)$ |
| `noun` | $(K,)$ | per-state $\mathrm{KL}$ of $P(n \mid Z{=}k)$ |
| `per_state` | $(K,)$ | `trans + verb + noun` — the drift heat map over subtasks |
| `total` | scalar | `init + sum(per_state)` |

`per_state` is the actionable artifact: sort it descending and you have the review agenda ("find
the large differences, then query the user about them").

The direction is $\mathrm{KL}(\text{live} \| \text{frozen})$, which weights by the *new* behaviour —
it asks "how surprising is the new model's typical behaviour under the old model", the right
question for a drift alert.

**Durations are omitted, and the docstring says why:** they are never updated online, so live and
frozen durations are identical and their KL is identically zero. `duration_drift.py` closes with an
explicit note that if durations ever gain a live/frozen split, *this claim becomes wrong and must
change in the same commit*.

---

## 6. `duration_drift.py` — the statistical report

Durations can't be handled by the bump/consolidate machinery, so they get a standalone report
comparing a recent window of trials against a reference window.

### Pipeline per state

1. **Build histograms.** `duration_histogram` bins segments the same way
   `durations.duration_tables` indexes columns (column 0 = $d{=}1$, last column absorbs
   $d \ge D_{\max}$).

   `_usable_segments` **drops each trial's final segment** — the only structurally right-censored
   one. Including it without imputation biases the mean downward; dropping it is the honest cheap
   fix for a lightweight report (a full ECM imputation would be the expensive correct one).

2. **Fit NB per window.** `fit_nb` reuses `durations.newton_update_r` + `update_p_given_r`
   verbatim — *the exact M-step estimator*, not a second duration-fitting code path that could
   drift from it.

3. **Compare.**

$$
\mu = 1 + \frac{r(1-p)}{p}, \qquad \Delta\mu = \mu_{\text{recent}} - \mu_{\text{frozen}},
$$
$$
\mathrm{KL} = \mathrm{KL}\bigl(\text{NB}_{\text{recent}} \,\|\, \text{NB}_{\text{frozen}}\bigr)
\quad \text{over the } D_{\max}\text{-bin pmf, via } \texttt{divergence.categorical\_kl}.
$$

4. **Test.** A one-sided Mann–Whitney U test on the **raw** duration lists, with the alternative
   chosen by the sign of $\Delta\mu$. Non-parametric on purpose: duration distributions are skewed
   and small-$n$, so a $t$-test's assumptions do not hold.

### The reporting gate

$$
\texttt{reportable} \;=\;
\underbrace{\bigl(n_{\text{recent}} \ge 5 \ \wedge\ n_{\text{frozen}} \ge 5\bigr)}_{\text{enough data}}
\ \wedge\ \underbrace{|\Delta\mu| \ge 2\ \text{ticks}}_{\text{practically significant}}
\ \wedge\ \underbrace{p < 0.05}_{\text{statistically significant}} .
$$

All three conjuncts are load-bearing, and the docstring is blunt about why KL alone is not a
decision rule: *"at ~5 sessions/week and a handful of instances per state, a KL of 0.3 says nothing
about whether a shift is real versus sampling noise at that n."* KL measures distance between
fitted distributions; it says nothing about the sampling uncertainty of the fits themselves.

`narrate_drift` renders reportable rows through the **same** `narrate.Lexicon` used for live
queries, so a subtask has one name everywhere:

> "You've been slower at pour milk this week — about 34 ticks now vs your usual 21 (p=0.012)."

### The documented integration gap

The module ends with an explicit statement of what it does **not** do: `duration_drift` produces a
report and rendered strings but does not feed back into `state_manager` the way
`PreferenceEvent`/`handle_confirmation` does for verb/noun/trans, because `HSMMParams.dur_r/dur_p`
are point estimates and `bounded_bump`'s pattern doesn't apply. Closing it — giving duration a
live/frozen split inside `DualModel` itself — is scoped as separate future work.

`run_lifecycle.py` demonstrates the whole loop end to end: pick the most committed subtask, find
its least likely noun, replay $n$ confirmed-preference events, and watch $s_{\text{noun}}$ decay
toward but never through the flag threshold.
