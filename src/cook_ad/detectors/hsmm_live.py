"""class LiveHSMMDetector: causal alarms from the filter (docs/detectors.md §3).

Emission channels test o_t against the forecast made before seeing it (a p-value, flagged at
p <= alpha). Event channels are posterior probabilities that a bad hidden event happened
(flagged at P > kappa); the bad sets come from BadSets, binned so one pass gives every alpha.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

from cook_ad.core import Alarm, AnomalyDetector, Detection, OperatingPoint, Trial
from cook_ad.hsmm import kernels

CHANNELS = ("s_emit", "s_verb", "s_noun", "s_temporal", "s_dur_two", "s_transition",
            "s_recipe_transition")
COUNT = "s_count"         # optional event channel, only with fitted visit counts (§5.7)
EMISSION = CHANNELS[:3]
EVENTS = CHANNELS[3:]
TIE_ATOL = 1e-10          # outcomes this close in log forecast probability count as tied
HYPOTHESIS_FLOOR = 1e-12  # a stay/switch hypothesis with less mass cannot excuse a token


class Bins(NamedTuple):
    """Each cell's first bad alpha index (G = never), sliced to the trial's recipes."""
    stall: jnp.ndarray    # (K_R', K, D)
    early: jnp.ndarray    # (K_R', K, D)
    trans: jnp.ndarray    # (K_R', K, K)
    recipe: jnp.ndarray   # (K_R', K, K)


class LiveTrace(NamedTuple):
    """Per-tick channel scores of one trial (true length)."""
    p: dict               # emission channel -> (T,) p-value
    events: dict          # event channel -> (T, G) posterior bad mass at every grid alpha
    alphas: np.ndarray    # (G,) the grid
    log_norm: np.ndarray  # (T,) log P(o_t | o_<t)
    pi_tilde: np.ndarray  # (T, K) log P(z_t = i | o_<t)


# ---------------------------------------------------------------------------------------------
# Emission p-values (docs/detectors.md §3.1)


def _pvalue(log_mix, obs):
    """Forecast mass on outcomes no more probable than the observed one, ties included."""
    log_obs = jnp.take_along_axis(log_mix, obs[:, None], axis=1)
    return jnp.sum(jnp.where(log_mix <= log_obs + TIE_ATOL, jnp.exp(log_mix), 0.0), axis=1)


@jax.jit
def forecast_pvalues(log_w, log_emit_v, log_emit_n, verbs, nouns):
    """(p_emit, p_verb, p_noun), each (T,), of o_t under the mixture sum_i w_t(i) P(o | i)."""
    lv = logsumexp(log_w[:, :, None] + log_emit_v[None], axis=1)
    ln = logsumexp(log_w[:, :, None] + log_emit_n[None], axis=1)
    joint = logsumexp(log_w[:, :, None, None] + log_emit_v[None, :, :, None]
                      + log_emit_n[None, :, None, :], axis=1)
    p_emit = _pvalue(joint.reshape(joint.shape[0], -1), verbs * log_emit_n.shape[1] + nouns)
    return p_emit, _pvalue(lv, verbs), _pvalue(ln, nouns)


def split_pvalues(log_stay, log_switch, log_emit_v, log_emit_n, verbs, nouns):
    """The larger p-value of o_t under 'the subtask continued' and 'a subtask began at t'."""
    def normalise(x):
        tot = logsumexp(x, axis=1, keepdims=True)
        return x - tot, jnp.exp(tot[:, 0]) > HYPOTHESIS_FLOOR

    stay, has_stay = normalise(log_stay)
    switch, has_switch = normalise(log_switch)
    ps = forecast_pvalues(stay, log_emit_v, log_emit_n, verbs, nouns)
    pw = forecast_pvalues(switch, log_emit_v, log_emit_n, verbs, nouns)
    return tuple(jnp.maximum(jnp.where(has_stay, a, 0.0), jnp.where(has_switch, b, 0.0))
                 for a, b in zip(ps, pw))


# ---------------------------------------------------------------------------------------------
# Event posteriors at one tick (docs/detectors.md §3.2-3.6). beta = 0 is L = 0.


class Switch(NamedTuple):
    """The pieces of q_t(r, k, j, i): subtask k ended at t-1 after j+1 ticks, i began at t."""
    b_pred: jnp.ndarray   # (K_R, K, D)
    log_end: jnp.ndarray  # (K_R, K, D)
    log_m: jnp.ndarray    # (K_R, K)
    arrive: jnp.ndarray   # (K_R, K) log P(o_t, o_{t+1..t+L} | r, i begins at t)
    log_z: jnp.ndarray    # () log P(o_t, o_{t+1..t+L} | o_<t)


def switch_terms(log_b_prev, beta, e, is_first, log_b0, log_A, log_h, log_stay):
    b_pred, log_end, log_m = kernels.predict(log_b_prev, log_A, log_h, log_stay)
    b_pred = jnp.where(is_first, kernels.initial_belief(log_b0, log_h.shape[-1]), b_pred)
    log_z = logsumexp(b_pred + e[None, :, None] + beta)
    return Switch(b_pred, log_end, log_m, e[None, :] + beta[..., 0], log_z)


def binned_mass(log_cells, bins, n_alphas):
    """out[g] = total mass of the cells that are bad at alphas[g]."""
    per_bin = jax.ops.segment_sum(jnp.exp(log_cells).ravel(), bins.ravel(),
                                  num_segments=n_alphas + 1)
    return jnp.cumsum(per_bin)[:n_alphas]


def stall_prob(log_b, beta, bins, n_alphas):
    """P(elapsed time is past the upper alpha-tail | o_0..o_{t+L}), §3.3."""
    post = log_b + beta
    return binned_mass(post - logsumexp(post), bins, n_alphas)


def early_exit_prob(sw, log_A, bins, n_alphas):
    """P(a subtask ended at t-1 in its short alpha/2 tail | ...), §3.5."""
    into = logsumexp(log_A + sw.arrive[:, None, :], axis=-1)
    return binned_mass(sw.log_end + into[..., None] - sw.log_z, bins, n_alphas)


def transition_prob(sw, log_A, bins, n_alphas):
    """P(the k -> i switch at t is in the alpha-tail of A[r, k, :] | ...), §3.4."""
    cells = sw.log_m[:, :, None] + log_A + sw.arrive[:, None, :] - sw.log_z
    return binned_mass(cells, bins, n_alphas)


def recipe_transition_prob(sw, log_A, beta, e, bins, n_alphas):
    """As transition_prob against r's excess tail, holding the recipe belief at t-1, §3.6."""
    w_prev = logsumexp(sw.b_pred, axis=(1, 2))
    z_r = logsumexp(sw.b_pred + e[None, :, None] + beta, axis=(1, 2))
    lead = jnp.where(jnp.isfinite(w_prev), w_prev - z_r, -jnp.inf)
    cells = lead[:, None, None] + sw.log_m[:, :, None] + log_A + sw.arrive[:, None, :]
    return binned_mass(cells, bins, n_alphas)


class Dynamics(NamedTuple):
    """The arrays of an hsmm.model.View, without its recipe slice (so it can enter jit)."""
    log_b0: jnp.ndarray
    log_A: jnp.ndarray
    log_h: jnp.ndarray
    log_stay: jnp.ndarray


def events_one_tick(log_b_prev, log_b, beta, e, is_first, dyn, bins, n_alphas):
    """(stall, early, trans, recipe), each (G,), at one tick; the switch events are 0 at t = 0."""
    sw = switch_terms(log_b_prev, beta, e, is_first, *dyn)
    stall = stall_prob(log_b, beta, bins.stall, n_alphas)
    zero = jnp.zeros((n_alphas,), dtype=stall.dtype)
    early = jnp.where(is_first, zero, early_exit_prob(sw, dyn.log_A, bins.early, n_alphas))
    trans = jnp.where(is_first, zero, transition_prob(sw, dyn.log_A, bins.trans, n_alphas))
    recipe = jnp.where(is_first, zero, recipe_transition_prob(sw, dyn.log_A, beta, e,
                                                              bins.recipe, n_alphas))
    return stall, early, trans, recipe


_events_one_tick = jax.jit(events_one_tick, static_argnames=("n_alphas",))


def entry_one_tick(log_b_prev, log_b, beta, e, is_first, dyn):
    """((K_R', K) P(a segment of i begins at t, recipe r | o), (K_R',) P(r | o)), at one tick:
    q_t(r, k, j, i) summed over the subtask left and its elapsed time (§5.7); at t = 0 every
    subtask begins, so it is the posterior itself."""
    sw = switch_terms(log_b_prev, beta, e, is_first, *dyn)
    cells = sw.log_m[:, :, None] + dyn.log_A + sw.arrive[:, None, :] - sw.log_z
    later = logsumexp(cells, axis=1)
    post = log_b + beta
    post = post - logsumexp(post)
    first = logsumexp(post, axis=-1)
    return jnp.exp(jnp.where(is_first, first, later)), jnp.exp(logsumexp(post, axis=(1, 2)))


@jax.jit
def entry_table(dyn, log_b_all, beta, loglik, mask):
    """(T, K_R', K) entry posteriors and (T, K_R') recipe posteriors of a padded trial."""
    log_b_prev = jnp.concatenate([log_b_all[:1], log_b_all[:-1]], axis=0)
    is_first = jnp.arange(loglik.shape[0]) == 0

    def body(xs):
        return entry_one_tick(*xs, dyn)

    ent, pr = jax.lax.map(body, (log_b_prev, log_b_all, beta, loglik, is_first))
    return jnp.where(mask[:, None, None], ent, 0.0), pr


_entry_one_tick = jax.jit(entry_one_tick)


class CountState:
    """The live count channel's running state: expected visits so far, per recipe and subtask,
    conditional on the recipe (sum over past ticks of P(i begins at t | r, o))."""

    def __init__(self, counts, recipes, k, k_recipe=None):
        if isinstance(recipes, slice):            # the view holds every recipe (hsmm.model.View)
            recipes = range(k_recipe)[recipes]
        self.counts, self.recipes = counts, [int(r) for r in np.atleast_1d(recipes)]
        self.c = np.zeros((len(self.recipes), k))

    def step(self, entry, p_r, alphas):
        """(G,) bad-event mass at this tick, then add the tick's entries to the counts. A
        beginning of i that would be its first visit (c_i < 1/2) is bad at alpha if
        counts.p_first <= alpha, with "not yet happened" read as c_k < 1/2."""
        entry, p_r = np.asarray(entry, dtype=float), np.asarray(p_r, dtype=float)
        cond = entry / np.maximum(p_r[:, None], 1e-300)
        alphas = np.asarray(alphas, dtype=float)
        bad = np.zeros(len(alphas))
        for a, r in enumerate(self.recipes):
            if not self.counts.known(r):
                continue
            c = self.c[a]
            for i in np.flatnonzero((entry[a] > 1e-12) & (c < 0.5)):
                i = int(i)
                if i in self.counts.ignore:
                    continue
                p = self.counts.p_first(r, i, lambda k: c[k] < 0.5)
                bad += entry[a, i] * (p <= alphas)
        self.c += cond
        return np.minimum(bad, 1.0)


@jax.jit
def event_table(dyn, bins, alphas, log_b_all, beta, loglik, mask):
    """(T, G) event posteriors for a whole padded trial; padded ticks are 0."""
    n_alphas = alphas.shape[0]
    log_b_prev = jnp.concatenate([log_b_all[:1], log_b_all[:-1]], axis=0)
    is_first = jnp.arange(loglik.shape[0]) == 0

    def body(xs):
        return events_one_tick(*xs, dyn, bins, n_alphas)

    out = jax.lax.map(body, (log_b_prev, log_b_all, beta, loglik, is_first))
    return tuple(jnp.where(mask[:, None], x, 0.0) for x in out)


# ---------------------------------------------------------------------------------------------
# The detector


class LiveHSMMDetector(AnomalyDetector):
    """Causal: every alarm about tick t is raised at step() of tick t."""
    name = "hsmm-live"
    causal = True
    L = 0

    def __init__(self, hsmm, bad_sets, op: OperatingPoint, counts=None):
        """`counts`: hsmm/visit_counts.VisitCounts; adds the s_count event channel (§5.7):
        the posterior mass that a subtask beginning now for the first time skips one that
        should have come before it, under the expected visit counts so far. Without it: the
        seven-channel detector."""
        self.hsmm, self.bad, self.op, self.counts = hsmm, bad_sets, op, counts
        self.channels = CHANNELS + (COUNT,) if counts is not None else CHANNELS
        self._alpha_idx = {c: bad_sets.alpha_index(op.alpha_for(c)) for c in EVENTS}
        if counts is not None:
            a = op.alpha_for(COUNT, 0.0)
            self._alpha_idx[COUNT] = bad_sets.alpha_index(a) if a > 0 else None
        self.reset()

    # ---- per-trial state

    def reset(self, recipe=None):
        self.hsmm.reset(recipe)
        r = self.hsmm.current.recipes
        b = self.bad
        self.bins = Bins(b.stall_bin[r], b.early_bin[r], b.trans_bin[r], b.recipe_bin[r])
        self.t, self._history = 0, {}   # tick -> (b_{t-1}, b_t, e_t), the last L+1 ticks only
        self._count = (CountState(self.counts, r, self.hsmm.tables.log_emit_v.shape[0],
                                  self.hsmm.k_recipe) if self.counts is not None else None)

    def step(self, verb, noun):
        forecast = self.hsmm.predict()
        p = self.emission_pvalues(forecast.stay[None], forecast.switch[None], [verb], [noun])
        b_prev = self.hsmm.belief
        self.hsmm.update(verb, noun)
        e = self.hsmm.tables.log_emit_v[:, verb] + self.hsmm.tables.log_emit_n[:, noun]
        self._history[self.t] = (b_prev, self.hsmm.belief, e)
        self._history.pop(self.t - self.L - 1, None)
        alarms = self._emission_alarms(self.t, {c: float(x[0]) for c, x in p.items()})
        alarms += self._event_steps()
        if self._count is not None:
            ent, pr = _entry_one_tick(b_prev if b_prev is not None else self.hsmm.belief,
                                      self.hsmm.belief, jnp.zeros_like(self.hsmm.belief), e,
                                      self.t == 0, self._dynamics())
            mass = self._count.step(ent, pr, self.bad.alphas)
            g = self._alpha_idx[COUNT]
            if g is not None and mass[g] > self.op.kappa:
                alarms.append(Alarm(self.t, COUNT, float(mass[g]), self.t))
        self.t += 1
        return alarms

    def _event_steps(self):
        """Event alarms that became known at this tick (Lag defers them)."""
        return self._event_alarms_at(self.t, beta=jnp.zeros_like(self.hsmm.belief),
                                     known_at=self.t)

    def _event_alarms_at(self, s, beta, known_at):
        b_prev, b, e = self._history[s]
        stall, early, trans, recipe = _events_one_tick(
            b if b_prev is None else b_prev, b, beta, e, s == 0, self._dynamics(),
            self.bins, n_alphas=len(self.bad.alphas))
        probs = dict(zip(EVENTS, (stall, early, trans, recipe)))
        return [Alarm(s, c, float(probs[c][self._alpha_idx[c]]), known_at) for c in EVENTS
                if float(probs[c][self._alpha_idx[c]]) > self.op.kappa]

    def _dynamics(self):
        v = self.hsmm.current
        return Dynamics(v.log_b0, v.log_A, v.log_h, v.log_stay)

    def _emission_alarms(self, t, p):
        return [Alarm(t, c, p[c], t) for c in EMISSION if p[c] <= self.op.alpha_for(c)]

    # ---- channels

    def emission_pvalues(self, log_stay, log_switch, verbs, nouns):
        """{s_emit, s_verb, s_noun: (T,) p-value} under the timing-split forecast, §3.1."""
        t = self.hsmm.tables
        p = split_pvalues(jnp.asarray(log_stay), jnp.asarray(log_switch), t.log_emit_v,
                          t.log_emit_n, jnp.asarray(verbs), jnp.asarray(nouns))
        return dict(zip(EMISSION, p))

    # ---- whole trial

    def beta(self, fr):
        """Backward messages for the event channels: none at L = 0."""
        return kernels.fixed_lag_beta(fr.view.log_A, fr.view.log_h, fr.view.log_stay, fr.loglik,
                                      fr.n, 0)

    def trace(self, trial: Trial) -> LiveTrace:
        """Every channel score at every tick, vectorised over the trial."""
        self.reset(trial.recipe)
        fr = self.hsmm.filter(trial.verbs, trial.nouns, recipe=trial.recipe)
        verb, noun, _ = self.hsmm.pad(trial.verbs, trial.nouns)
        p = self.emission_pvalues(fr.log_stay, fr.log_switch, verb, noun)
        ev = event_table(self._dynamics(), self.bins,
                         jnp.asarray(self.bad.alphas), fr.log_b, self.beta(fr), fr.loglik, fr.mask)
        n = fr.n
        events = {c: np.asarray(x)[:n] for c, x in zip(EVENTS, ev)}
        if self.counts is not None:
            ent, pr = entry_table(self._dynamics(), fr.log_b, jnp.zeros_like(fr.log_b), fr.loglik,
                                  fr.mask)
            ent, pr = np.asarray(ent)[:n], np.asarray(pr)[:n]
            cs = CountState(self.counts, self.hsmm.current.recipes, ent.shape[-1], self.hsmm.k_recipe)
            events[COUNT] = np.stack([cs.step(ent[t], pr[t], self.bad.alphas) for t in range(n)])
        return LiveTrace({c: np.asarray(x)[:n] for c, x in p.items()}, events,
                         self.bad.alphas, np.asarray(fr.log_norm)[:n], fr.occupancy)

    def flags(self, trace: LiveTrace, op: OperatingPoint | None = None):
        """{channel: (T,) bool} at the operating point."""
        op = op or self.op
        out = {c: trace.p[c] <= op.alpha_for(c) for c in EMISSION}
        for c in EVENTS:
            g = self.bad.alpha_index(op.alpha_for(c))
            out[c] = trace.events[c][:, g] > op.kappa
        if self.counts is not None:
            a = op.alpha_for(COUNT, 0.0)
            out[COUNT] = (trace.events[COUNT][:, self.bad.alpha_index(a)] > op.kappa if a > 0
                          else np.zeros(len(trace.log_norm), dtype=bool))
        return {c: out[c] for c in self.channels}

    def known_at(self, channel, t, n):
        return t if channel in EMISSION else min(t + self.L, n - 1)

    def detect_trial(self, trial: Trial) -> Detection:
        tr = self.trace(trial)
        flags = self.flags(tr)
        n = len(trial.verbs)
        alarms = []
        for t in range(n):
            for c in self.channels:
                if flags[c][t]:
                    score = tr.p[c][t] if c in EMISSION else tr.events[c][t, self._alpha_idx[c]]
                    alarms.append(Alarm(t, c, float(score), self.known_at(c, t, n)))
        return Detection(alarms, trace=tr)
