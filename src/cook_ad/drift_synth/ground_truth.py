"""Stage 10: ground truth in observable space. Per recipe, at a fixed theta, Monte Carlo over
clean sessions (no recognizer noise, no day noise, speed 1): token counts per session, token
marginals, bigrams, and log durations per token. Drift is measured between two such summaries.
Nothing here refers to a model's latent states."""
from dataclasses import dataclass

import numpy as np
from scipy.stats import wasserstein_distance

from cook_ad.drift_synth.noise import NoiseCfg
from cook_ad.drift_synth.sampler import rng_for, sample_session

_GT = 99  # stream tag, distinct from the generator's


def tok(verb, noun):
    return f"{verb} {noun}"


@dataclass
class ObsStats:
    recipe_id: str
    n: int
    counts: dict      # token -> per-session counts (np.ndarray, length n)
    marginal: dict    # token -> share of all tokens
    bigram: dict      # "a > b" -> share of all bigrams
    logdur: dict      # token -> np.ndarray of log seconds

    def dur_quantiles(self, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
        return {t: np.quantile(x, qs) for t, x in self.logdur.items()}


def observable_stats(recipe_id, theta, n_mc, seed, assets):
    graph, exps = assets.graphs[recipe_id], assets.expansions[recipe_id]
    off = NoiseCfg().off()
    per_session, bigrams, logdur = [], {}, {}
    for i in range(n_mc):
        s = sample_session(graph, exps, theta, 1.0, assets, off, rng_for(seed, _GT, i))
        seq = [tok(e.verb, e.noun) for e in s.events]
        c = {}
        for t, e in zip(seq, s.events):
            c[t] = c.get(t, 0) + 1
            logdur.setdefault(t, []).append(np.log(e.t_end - e.t_start))
        for a, b in zip(seq, seq[1:]):
            k = f"{a} > {b}"
            bigrams[k] = bigrams.get(k, 0) + 1
        per_session.append(c)
    tokens = sorted({t for c in per_session for t in c})
    counts = {t: np.array([c.get(t, 0) for c in per_session], dtype=float) for t in tokens}
    total = sum(x.sum() for x in counts.values())
    nb = sum(bigrams.values())
    return ObsStats(recipe_id, n_mc, counts,
                    {t: counts[t].sum() / total for t in tokens},
                    {k: v / nb for k, v in sorted(bigrams.items())},
                    {t: np.asarray(x) for t, x in sorted(logdur.items())})


def js_divergence(p, q):
    """Jensen-Shannon divergence in bits between two {key: prob} dicts."""
    keys = sorted(set(p) | set(q))
    a = np.array([p.get(k, 0.0) for k in keys])
    b = np.array([q.get(k, 0.0) for k in keys])
    m = 0.5 * (a + b)

    def kl(x):
        nz = x > 0
        return float(np.sum(x[nz] * np.log2(x[nz] / m[nz])))
    return 0.5 * kl(a) + 0.5 * kl(b)


def duration_w1(s0, s1):
    """{token: W1 between log-duration samples}, for tokens seen in both."""
    return {t: float(wasserstein_distance(s0.logdur[t], s1.logdur[t]))
            for t in sorted(set(s0.logdur) & set(s1.logdur))}


def drift_magnitude(s0, s1):
    """Per recipe: JS on marginals and on bigrams; W1 on log duration (per token, plus the
    marginal-weighted mean and the max)."""
    w1 = duration_w1(s0, s1)
    weight = {t: 0.5 * (s0.marginal.get(t, 0) + s1.marginal.get(t, 0)) for t in w1}
    total = sum(weight.values()) or 1.0
    return {
        "marginal_js": js_divergence(s0.marginal, s1.marginal),
        "bigram_js": js_divergence(s0.bigram, s1.bigram),
        "duration_w1_mean": sum(w1[t] * weight[t] for t in w1) / total,
        "duration_w1_max": max(w1.values(), default=0.0),
        "duration_w1": w1,
    }


def count_z(s0, s1):
    """{token: z} for the per-session count difference (Welch, MC noise as the yardstick)."""
    out = {}
    for t in sorted(set(s0.counts) | set(s1.counts)):
        a = s0.counts.get(t, np.zeros(s0.n))
        b = s1.counts.get(t, np.zeros(s1.n))
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        diff = b.mean() - a.mean()
        out[t] = 0.0 if diff == 0 else (diff / se if se > 0 else np.inf * np.sign(diff))
    return out


def affected_features(stats0, stats1, threshold):
    """Feature names whose change exceeds `threshold`: {"count_z", "bigram_js", "duration_w1"}.
    count:<token> by |z|, bigrams as one feature, duration:<token> by W1 on log seconds."""
    out = [f"count:{t}" for t, z in count_z(stats0, stats1).items() if abs(z) > threshold["count_z"]]
    if js_divergence(stats0.bigram, stats1.bigram) > threshold["bigram_js"]:
        out.append("bigram")
    out += [f"duration:{t}" for t, w in duration_w1(stats0, stats1).items()
            if w > threshold["duration_w1"]]
    return sorted(out)
