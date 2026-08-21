"""Does this dataset actually support personalization? Three questions, one script:

1. Order multimodality: within a recipe, do participants cluster into a few distinct step
   orderings (e.g. coffee black vs. coffee + cream + sugar), beating a shuffled null -- or is
   the variation just noise around one typical order?
2. Duration multimodality: for a given step, is the duration distribution across participants
   bimodal (two ways of doing that step) rather than unimodal?
3. Personal traits: does a participant's normalized speed, or their use of a given ingredient,
   stay consistent across the different recipes they performed?

Reads the already-processed dataset/processed/breakfast/{sequences,labels}.json (see docs/data.md)
-- no HSMM checkpoint needed, this is pure data description. Writes PNGs + a markdown report to
dataset/processed/breakfast/reports/participant_variability/.
"""
import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.cluster import hierarchy  # noqa: E402
from scipy.spatial.distance import squareform  # noqa: E402

AMBIENT_NOUN = "kitchen"  # SIL sentinel noun (configs/breakfast.yaml ambient_gaps); not a step
RNG_SEED = 0


# ---------------------------------------------------------------------------
# Loading + segment collapsing
# ---------------------------------------------------------------------------

def load_trials(sequences_path, labels_path):
    sequences = json.load(open(sequences_path))
    labels = json.load(open(labels_path))
    labels_by_id = {e["trial_id"]: e for e in labels}
    trials = []
    for seq in sequences:
        lab = labels_by_id[seq["trial_id"]]
        participant, recipe = seq["trial_id"].rsplit("_", 1)
        assert recipe == lab["recipe_label"]
        trials.append({
            "trial_id": seq["trial_id"], "participant": participant, "recipe": recipe,
            "subtask_labels": lab["subtask_labels"][: len(seq["verb_ids"])],
        })
    return trials


def collapse_segments(subtask_labels):
    """Consecutive-run collapse: tick-level labels -> [(verb, noun, duration_ticks), ...].

    verb_noun splits on the first underscore, matching parse_breakfast.label_to_verb_noun.
    """
    segments = []
    for label in subtask_labels:
        if label == "SIL":
            verb, noun = "SIL", AMBIENT_NOUN
        else:
            verb, noun = label.split("_", 1)
        if segments and segments[-1][0] == verb and segments[-1][1] == noun:
            segments[-1][2] += 1
        else:
            segments.append([verb, noun, 1])
    return [tuple(s) for s in segments]


def real_steps(segments):
    """Drop SIL/ambient segments -- idle time, not a cooking step."""
    return [s for s in segments if s[0] != "SIL"]


# ---------------------------------------------------------------------------
# 1. Order multimodality per recipe
# ---------------------------------------------------------------------------

def edit_distance(seq_a, seq_b):
    """Levenshtein distance over step tokens (verb, noun)."""
    n, m = len(seq_a), len(seq_b)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def distance_matrix(token_seqs):
    n = len(token_seqs)
    d = np.zeros((n, n))
    for i, j in combinations(range(n), 2):
        dist = edit_distance(token_seqs[i], token_seqs[j])
        d[i, j] = d[j, i] = dist
    return d


def mean_within_cluster_distance(d, cluster_ids):
    """Mean pairwise distance among trials sharing a cluster, averaged over clusters."""
    totals = []
    for c in set(cluster_ids):
        idx = [i for i, ci in enumerate(cluster_ids) if ci == c]
        if len(idx) < 2:
            continue
        totals.append(np.mean([d[i, j] for i, j in combinations(idx, 2)]))
    return float(np.mean(totals)) if totals else float("nan")


def cluster_recipe(recipe, trials, out_dir, n_clusters=2, n_null=500, rng=None):
    """Cluster trials of one recipe by step-order edit distance into k clusters, then ask: is
    THIS split meaningfully tighter than a random split of the same trials into groups of the
    same sizes, using the same real distance matrix?

    Under genuine unimodal variance (participants doing the same recipe with idiosyncratic but
    non-clustered noise), no bipartition of the trials is much tighter than any other, so the
    best k=2 split should look about like a random one. Under real multimodality (two distinct
    step orders), the split that separates the two modes is much tighter than a random split
    that mixes them -- that's the signal this null isolates, as opposed to just asking "is there
    any order structure at all" (trivially yes, since it's a recipe).
    """
    token_seqs = [tuple((v, n) for v, n, _ in real_steps(t["segments"])) for t in trials]
    if len(token_seqs) < 6:
        return None

    d = distance_matrix(token_seqs)
    condensed = squareform(d, checks=False)
    link = hierarchy.linkage(condensed, method="average")
    cluster_ids = hierarchy.fcluster(link, t=n_clusters, criterion="maxclust")
    observed = mean_within_cluster_distance(d, cluster_ids)

    n = len(token_seqs)
    sizes = [np.sum(cluster_ids == c) for c in set(cluster_ids)]
    null_scores = []
    for _ in range(n_null):
        perm = rng.permutation(n)
        random_ids = np.empty(n, dtype=int)
        pos = 0
        for c, size in enumerate(sizes):
            random_ids[perm[pos:pos + size]] = c
            pos += size
        null_scores.append(mean_within_cluster_distance(d, random_ids))
    null_scores = np.array(null_scores)
    # fraction of random splits that are looser (higher within-cluster distance) than the
    # clustering actually found -- high means the found split is a real, non-random structure
    percentile = float((null_scores > observed).mean())

    fig, ax = plt.subplots(figsize=(8, 4))
    hierarchy.dendrogram(link, ax=ax, no_labels=True, color_threshold=0.7 * max(link[:, 2]))
    ax.set_title(f"{recipe}: step-order dendrogram (n={len(trials)})")
    ax.set_ylabel("edit distance")
    fig.tight_layout()
    fig.savefig(out_dir / f"order_dendrogram_{recipe}.png", dpi=130)
    plt.close(fig)

    cluster_summaries = []
    for c in sorted(set(cluster_ids)):
        idx = [i for i, ci in enumerate(cluster_ids) if ci == c]
        counts = defaultdict(int)
        for i in idx:
            counts[token_seqs[i]] += 1
        modal_seq, modal_n = max(counts.items(), key=lambda kv: kv[1])
        cluster_summaries.append({
            "cluster": int(c), "n": len(idx),
            "modal_sequence": " -> ".join(f"{v} {n}" for v, n in modal_seq),
            "modal_count": modal_n,
        })

    min_cluster_n = min(c["n"] for c in cluster_summaries)
    return {
        "recipe": recipe, "n_trials": len(trials), "observed_within_dist": observed,
        "null_mean": float(null_scores.mean()), "null_std": float(null_scores.std()),
        "percentile": percentile, "min_cluster_n": min_cluster_n, "clusters": cluster_summaries,
    }


# ---------------------------------------------------------------------------
# 2. Step duration bimodality
# ---------------------------------------------------------------------------

def fit_1d_gmm2(x, n_iter=200, n_restarts=5, rng=None):
    """Two-component 1D Gaussian mixture via EM, numpy only (no sklearn dependency).
    Returns (log_likelihood, params) for the best of n_restarts random inits.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    best = None
    for _ in range(n_restarts):
        mu = rng.choice(x, size=2, replace=False)
        sigma = np.full(2, x.std() + 1e-6)
        pi = np.array([0.5, 0.5])
        ll_prev = -np.inf
        for _ in range(n_iter):
            # E-step
            resp = np.stack([
                pi[k] / (sigma[k] * np.sqrt(2 * np.pi)) *
                np.exp(-0.5 * ((x - mu[k]) / sigma[k]) ** 2)
                for k in range(2)
            ])
            resp_sum = resp.sum(axis=0)
            resp_sum = np.where(resp_sum < 1e-300, 1e-300, resp_sum)
            resp = resp / resp_sum
            ll = np.sum(np.log(resp_sum))
            # M-step
            nk = resp.sum(axis=1)
            nk = np.where(nk < 1e-6, 1e-6, nk)
            mu = (resp * x).sum(axis=1) / nk
            sigma = np.sqrt((resp * (x - mu[:, None]) ** 2).sum(axis=1) / nk) + 1e-6
            pi = nk / n
            if abs(ll - ll_prev) < 1e-8:
                break
            ll_prev = ll
        if best is None or ll > best[0]:
            best = (ll, {"mu": mu.tolist(), "sigma": sigma.tolist(), "pi": pi.tolist()})
    return best


def gaussian_ll(x):
    x = np.asarray(x, dtype=float)
    mu, sigma = x.mean(), x.std() + 1e-6
    return float(np.sum(-0.5 * np.log(2 * np.pi * sigma ** 2) - 0.5 * ((x - mu) / sigma) ** 2))


def duration_bimodality(recipe, trials, out_dir, rng, min_n=15, bic_margin=6.0):
    """For each (verb, noun) step that appears once per trial in most trials of this recipe,
    compare a 1-component vs 2-component Gaussian fit on its duration via BIC. Flags steps where
    two components win convincingly -- candidate 'two ways to do this step'.
    """
    by_step = defaultdict(list)
    for t in trials:
        for v, n, dur in real_steps(t["segments"]):
            by_step[(v, n)].append(dur)

    flagged = []
    for (v, n), durs in by_step.items():
        if len(durs) < min_n:
            continue
        durs = np.array(durs, dtype=float)
        ll1 = gaussian_ll(durs)
        bic1 = -2 * ll1 + 2 * np.log(len(durs))
        ll2, params2 = fit_1d_gmm2(durs, rng=rng)
        bic2 = -2 * ll2 + 5 * np.log(len(durs))  # 2 means, 2 sigmas, 1 free mixing weight
        if bic1 - bic2 > bic_margin:
            flagged.append({
                "recipe": recipe, "verb": v, "noun": n, "n": len(durs),
                "bic1": float(bic1), "bic2": float(bic2), "delta_bic": float(bic1 - bic2),
                "means": sorted(params2["mu"]),
            })

    for f in sorted(flagged, key=lambda r: -r["delta_bic"])[:6]:
        durs = np.array(by_step[(f["verb"], f["noun"])], dtype=float)
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.hist(durs, bins=min(20, max(5, len(durs) // 3)), color="#4a6d8c", edgecolor="white")
        for m in f["means"]:
            ax.axvline(m, color="#c44e52", ls="--", lw=1.3)
        ax.set_title(f"{recipe}: {f['verb']} {f['noun']} (n={f['n']}, ΔBIC={f['delta_bic']:.1f})")
        ax.set_xlabel("duration (ticks)")
        fig.tight_layout()
        safe = f"{f['verb']}_{f['noun']}".replace("/", "-")
        fig.savefig(out_dir / f"duration_{recipe}_{safe}.png", dpi=130)
        plt.close(fig)

    return flagged


# ---------------------------------------------------------------------------
# 3. Personal traits across recipes
# ---------------------------------------------------------------------------

def participant_features(trials):
    """One row per (participant, recipe): normalized duration, idle fraction, noun usage set."""
    by_recipe_durs = defaultdict(list)
    rows = []
    for t in trials:
        segs = t["segments"]
        total = sum(d for _, _, d in segs)
        idle = sum(d for v, _, d in segs if v == "SIL")
        nouns = {n for v, n, _ in real_steps(segs)}
        rows.append({"participant": t["participant"], "recipe": t["recipe"],
                     "total_ticks": total, "idle_frac": idle / total if total else 0.0,
                     "nouns": nouns})
        by_recipe_durs[t["recipe"]].append(total)
    recipe_median = {r: float(np.median(v)) for r, v in by_recipe_durs.items()}
    for row in rows:
        row["norm_duration"] = row["total_ticks"] / recipe_median[row["recipe"]]
    return rows


def speed_consistency(rows, out_dir, min_recipes=3):
    """Split-half correlation of a participant's normalized speed: recipes sorted alphabetically,
    even-indexed vs. odd-indexed half. Positive correlation = speed is a trait of the participant,
    not recipe-specific noise.
    """
    by_participant = defaultdict(list)
    for r in rows:
        by_participant[r["participant"]].append((r["recipe"], r["norm_duration"]))

    evens, odds, participants = [], [], []
    for p, recs in by_participant.items():
        if len(recs) < min_recipes:
            continue
        recs = sorted(recs)
        evens.append(np.mean([v for i, (_, v) in enumerate(recs) if i % 2 == 0]))
        odds.append(np.mean([v for i, (_, v) in enumerate(recs) if i % 2 == 1]))
        participants.append(p)

    if len(evens) < 4:
        return {"n_participants": len(evens), "r": float("nan")}

    evens, odds = np.array(evens), np.array(odds)
    r = float(np.corrcoef(evens, odds)[0, 1])

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(evens, odds, color="#4a6d8c")
    lo, hi = min(evens.min(), odds.min()), max(evens.max(), odds.max())
    ax.plot([lo, hi], [lo, hi], color="#999", ls="--", lw=1)
    ax.set_xlabel("normalized speed, even-indexed recipes")
    ax.set_ylabel("normalized speed, odd-indexed recipes")
    ax.set_title(f"per-participant speed consistency (r={r:.2f}, n={len(evens)})")
    fig.tight_layout()
    fig.savefig(out_dir / "speed_consistency.png", dpi=130)
    plt.close(fig)

    ranked = sorted(zip(participants, evens, odds), key=lambda t: (t[1] + t[2]) / 2)
    extremes = {"slowest": ranked[-5:][::-1], "fastest": ranked[:5]}
    return {"n_participants": len(evens), "r": r, "extremes": extremes}


def ingredient_consistency(rows, min_recipes=3, min_population_uses=5, fdr_q=0.10):
    """For each participant with >=min_recipes trials, and each noun used population-wide >=
    min_population_uses times, test whether that participant's per-recipe usage rate departs from
    the population rate (binomial test), then Benjamini-Hochberg correct across all tests.
    """
    from scipy import stats

    all_nouns = defaultdict(int)
    noun_total_trials = 0
    by_participant = defaultdict(list)
    for r in rows:
        noun_total_trials += 1
        for n in r["nouns"]:
            all_nouns[n] += 1
        by_participant[r["participant"]].append(r)

    pop_rate = {n: c / noun_total_trials for n, c in all_nouns.items()
                if c >= min_population_uses}

    tests = []
    for p, recs in by_participant.items():
        if len(recs) < min_recipes:
            continue
        for n, rate in pop_rate.items():
            k = sum(1 for r in recs if n in r["nouns"])
            trials_n = len(recs)
            pval = stats.binomtest(k, trials_n, rate).pvalue
            tests.append({"participant": p, "noun": n, "k": k, "n_trials": trials_n,
                          "pop_rate": rate, "pval": pval})

    if not tests:
        return []

    pvals = np.array([t["pval"] for t in tests])
    order = np.argsort(pvals)
    m = len(pvals)
    thresh = np.arange(1, m + 1) / m * fdr_q
    passed = pvals[order] <= thresh
    n_sig = int(np.max(np.where(passed)[0]) + 1) if passed.any() else 0
    sig_idx = set(order[:n_sig].tolist())

    flagged = [t for i, t in enumerate(tests) if i in sig_idx]
    return sorted(flagged, key=lambda t: t["pval"])


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(order_results, duration_flags, speed_result, ingredient_flags, out_path):
    lines = ["# Participant variability across recipes\n"]

    min_minor_cluster = 5
    lines.append("## 1. Task-order multimodality per recipe\n")
    lines.append("Clustering trials by step-order edit distance (k=2). `percentile` is the "
                  "fraction of random same-size bipartitions of the *same* real distance matrix "
                  "that are looser (higher within-cluster distance) than the split actually "
                  "found -- near 1.0 means this specific split is real structure, not what a "
                  "random split of the same trials would give you anyway. A high percentile with "
                  f"a minority cluster below {min_minor_cluster} trials is usually just one odd "
                  "outlier trial being separated, not a genuine second mode -- those are labeled "
                  "OUTLIER SPLIT, not MULTIMODAL.\n")
    for res in sorted(order_results, key=lambda r: -r["percentile"]):
        if res["percentile"] < 0.95:
            tag = "no clear structure"
        elif res["min_cluster_n"] < min_minor_cluster:
            tag = "outlier split (not genuine multimodality)"
        else:
            tag = "MULTIMODAL"
        lines.append(f"\n### {res['recipe']} (n={res['n_trials']}) -- {tag}")
        lines.append(f"- observed within-cluster distance: {res['observed_within_dist']:.2f} "
                     f"vs. null {res['null_mean']:.2f} +/- {res['null_std']:.2f} "
                     f"(percentile {res['percentile']:.2f})")
        for c in res["clusters"]:
            lines.append(f"  - cluster {c['cluster']} (n={c['n']}): modal order — "
                         f"`{c['modal_sequence']}`  [{c['modal_count']}/{c['n']} trials match]")

    lines.append("\n## 2. Step duration bimodality\n")
    if duration_flags:
        lines.append("Steps where a 2-component Gaussian fit beats 1-component by BIC "
                     "(delta_bic > 6, i.e. decisive):\n")
        lines.append("| recipe | step | n | delta BIC | mode means (ticks) |")
        lines.append("|---|---|---|---|---|")
        for f in sorted(duration_flags, key=lambda r: -r["delta_bic"])[:30]:
            means = ", ".join(f"{m:.1f}" for m in f["means"])
            lines.append(f"| {f['recipe']} | {f['verb']} {f['noun']} | {f['n']} | "
                         f"{f['delta_bic']:.1f} | {means} |")
    else:
        lines.append("No steps cleared the BIC margin.\n")

    lines.append("\n## 3. Personal traits across recipes\n")
    lines.append(f"Split-half speed consistency: r={speed_result['r']:.2f} "
                 f"(n={speed_result['n_participants']} participants with >=3 recipes). "
                 "Positive r means a participant's relative speed (normalized to that recipe's "
                 "median) carries over between recipes -- a genuine trait, not per-recipe noise.\n")
    if "extremes" in speed_result:
        lines.append("Consistently slowest (even-half, odd-half normalized speed):")
        for p, e, o in speed_result["extremes"]["slowest"]:
            lines.append(f"- {p}: {e:.2f}, {o:.2f}")
        lines.append("Consistently fastest:")
        for p, e, o in speed_result["extremes"]["fastest"]:
            lines.append(f"- {p}: {e:.2f}, {o:.2f}")

    lines.append("\n### Ingredient/noun usage consistency (FDR q=0.10)\n")
    if ingredient_flags:
        lines.append("| participant | noun | usage | population rate | p-value |")
        lines.append("|---|---|---|---|---|")
        for f in ingredient_flags[:40]:
            lines.append(f"| {f['participant']} | {f['noun']} | {f['k']}/{f['n_trials']} | "
                         f"{f['pop_rate']:.2f} | {f['pval']:.4f} |")
    else:
        lines.append("No participant-noun pairs survived FDR correction.\n")

    out_path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences", default="dataset/processed/breakfast/sequences.json")
    ap.add_argument("--labels", default="dataset/processed/breakfast/labels.json")
    ap.add_argument("--out-dir",
                    default="dataset/processed/breakfast/reports/participant_variability")
    ap.add_argument("--seed", type=int, default=RNG_SEED)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    raw_trials = load_trials(args.sequences, args.labels)
    for t in raw_trials:
        t["segments"] = collapse_segments(t["subtask_labels"])

    by_recipe = defaultdict(list)
    for t in raw_trials:
        by_recipe[t["recipe"]].append(t)

    print(f"{len(raw_trials)} trials, {len(by_recipe)} recipes, "
          f"{len({t['participant'] for t in raw_trials})} participants")

    order_results = []
    duration_flags = []
    for recipe, trials in sorted(by_recipe.items()):
        print(f"  {recipe}: n={len(trials)} order clustering + duration scan...")
        res = cluster_recipe(recipe, trials, out_dir, rng=rng)
        if res is not None:
            order_results.append(res)
        duration_flags += duration_bimodality(recipe, trials, out_dir, rng)

    rows = participant_features(raw_trials)
    speed_result = speed_consistency(rows, out_dir)
    ingredient_flags = ingredient_consistency(rows)

    report_path = out_dir.parent / "participant_variability.md"
    write_report(order_results, duration_flags, speed_result, ingredient_flags, report_path)
    print(f"wrote {report_path}")
    print(f"figures in {out_dir}")


if __name__ == "__main__":
    main()
