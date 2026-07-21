from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp
from scipy.optimize import linear_sum_assignment
from tqdm.auto import tqdm

from cook_ad.hsmm.params import _dirichlet_counts, _row_normalize

FLOOR = 1e-12


class RecipeParams(NamedTuple):
    init_counts: jnp.ndarray    # (K_recipe,)   Dirichlet pseudocounts, initial-recipe dist
    trans_counts: jnp.ndarray   # (K_recipe,K_recipe)  Dirichlet pseudocounts; diagonal IS used
                                 #   (self-transitions allowed -- most trials stay one recipe)
    emit_counts: jnp.ndarray    # (K_recipe,K_subtask)  P(subtask symbol | recipe), Dirichlet counts


def init_weak_limit_recipe_params(
    key, k_recipe, k_subtask, alpha_init=0.5, alpha_trans=0.5, alpha_emit=None
) -> RecipeParams:
    """Mirrors hsmm.params.init_weak_limit_params: alpha/K<1 sparsity on init/trans (weak-
    limit, unused recipe modes pushed to the floor) and alpha=width (per-category alpha=1) on
    the emission, since it's an ordinary closed-vocabulary categorical over subtask ids. No
    diagonal zeroing on trans_counts -- unlike the subtask HSMM, recipe self-transitions are
    not just allowed but expected (a trial mostly stays in one recipe).
    """
    if alpha_emit is None:
        alpha_emit = float(k_subtask)

    key_init, key_trans, key_emit = jax.random.split(key, 3)

    init_counts = alpha_init * jax.random.dirichlet(
        key_init, jnp.full((k_recipe,), alpha_init / k_recipe)
    )
    trans_counts = _dirichlet_counts(key_trans, alpha_trans, k_recipe, k_recipe)
    emit_counts = _dirichlet_counts(key_emit, alpha_emit, k_subtask, k_recipe)
    return RecipeParams(init_counts, trans_counts, emit_counts)


def save_params(recipe_params: RecipeParams, path):
    np.savez(path, **{name: np.asarray(value) for name, value in recipe_params._asdict().items()})


def load_params(path) -> RecipeParams:
    with np.load(path) as data:
        return RecipeParams(**{name: jnp.asarray(data[name]) for name in RecipeParams._fields})


def to_log_probs(recipe_params: RecipeParams):
    log_init = _row_normalize(recipe_params.init_counts, FLOOR)
    log_trans = _row_normalize(recipe_params.trans_counts, FLOOR)
    log_emit = _row_normalize(recipe_params.emit_counts, FLOOR)
    return log_init, log_trans, log_emit


def pad_segment_batch(seg_sequences, s_max=None):
    """seg_sequences: list of int lists (per-trial subtask-symbol sequences from
    segmentize.segment_all). Returns (obs_ids, mask: (N,S_max)), padded with dummy id 0
    (never read meaningfully, gated by mask), analogous to hsmm.em.pad_batch.
    """
    lengths = [len(seq) for seq in seg_sequences]
    if s_max is None:
        s_max = max(lengths)

    n = len(seg_sequences)
    obs_ids = np.zeros((n, s_max), dtype=np.int32)
    mask = np.zeros((n, s_max), dtype=bool)
    for i, seq in enumerate(seg_sequences):
        length = len(seq)
        obs_ids[i, :length] = seq
        mask[i, :length] = True

    return jnp.array(obs_ids), jnp.array(mask)


def _forward_backward(obs_ids, mask, log_init, log_trans, log_emit):
    """Plain discrete-HMM forward-backward for one trial's segment-symbol sequence (no
    duration model -- that's the subtask HSMM's job; here each "tick" is already a whole
    subtask segment). obs_ids/mask: (S,). Returns (gamma (S,K_recipe) log-posterior, xi_sum
    (K_recipe,K_recipe) expected transition counts, log_z scalar).

    Padded positions (mask=False) are handled the same way hsmm.messages handles right-
    censoring: alpha simply freezes past the true end (`jnp.where(mask[t], ...)`), and beta's
    boundary is force-set to 0 at the true final position `t_true-1` (not at the padded
    array's last index), so garbage computed beyond the true sequence never contaminates the
    real positions -- the force-set at t_true-1 is the firewall.
    """
    S = obs_ids.shape[0]
    k_recipe = log_init.shape[0]
    emit_ll = jnp.where(mask[:, None], log_emit[:, obs_ids].T, 0.0)  # (S,K_recipe)
    t_true = jnp.sum(mask)

    def fwd_step(alpha_prev, t):
        trans_term = logsumexp(alpha_prev[:, None] + log_trans, axis=0)
        alpha_new = trans_term + emit_ll[t]
        alpha_t = jnp.where(mask[t], alpha_new, alpha_prev)
        return alpha_t, alpha_t

    alpha0 = log_init + emit_ll[0]
    _, alpha_rest = jax.lax.scan(fwd_step, alpha0, jnp.arange(1, S))
    alpha = jnp.concatenate([alpha0[None, :], alpha_rest], axis=0)  # (S,K_recipe)

    emit_ll_ext = jnp.concatenate(
        [emit_ll, jnp.zeros((1, k_recipe), dtype=emit_ll.dtype)], axis=0
    )  # (S+1,K_recipe): row S is an unused dummy paired with the initial backward carry

    def bwd_step(beta_next, t):
        term = log_trans + emit_ll_ext[t + 1][None, :] + beta_next[None, :]  # (K_recipe,K_recipe)
        beta_t = logsumexp(term, axis=1)
        beta_t = jnp.where(t == t_true - 1, 0.0, beta_t)
        return beta_t, beta_t

    beta_init = jnp.zeros((k_recipe,), dtype=emit_ll.dtype)
    _, beta = jax.lax.scan(bwd_step, beta_init, jnp.arange(S), reverse=True)  # (S,K_recipe)

    log_z = logsumexp(alpha[t_true - 1])
    gamma = alpha + beta - log_z  # (S,K_recipe); entries at/after t_true are unused garbage

    xi_terms = (
        alpha[:-1, :, None] + log_trans[None, :, :] + emit_ll[1:, None, :] + beta[1:, None, :] - log_z
    )  # (S-1,K_recipe,K_recipe)
    valid_t = jnp.arange(S - 1) <= (t_true - 2)
    xi = jnp.where(valid_t[:, None, None], jnp.exp(xi_terms), 0.0)
    xi_sum = jnp.sum(xi, axis=0)

    return gamma, xi_sum, log_z


@jax.jit
def e_step(recipe_params, obs_ids, mask):
    log_init, log_trans, log_emit = to_log_probs(recipe_params)
    gamma, xi_sum, log_z = jax.vmap(_forward_backward, in_axes=(0, 0, None, None, None))(
        obs_ids, mask, log_init, log_trans, log_emit
    )
    gamma_masked = jnp.where(mask[:, :, None], jnp.exp(gamma), 0.0)
    n_subtask = log_emit.shape[1]
    obs_onehot = jax.nn.one_hot(obs_ids, n_subtask, dtype=gamma_masked.dtype)

    stats = {
        "init_counts": jnp.sum(gamma_masked[:, 0, :], axis=0),
        "trans_counts": jnp.sum(xi_sum, axis=0),
        "emit_counts": jnp.einsum("ntk,ntv->kv", gamma_masked, obs_onehot),
    }
    return stats, jnp.sum(log_z)


@jax.jit
def m_step(recipe_params, stats, alpha_init, alpha_trans, alpha_emit):
    k_recipe = recipe_params.init_counts.shape[0]
    n_subtask = recipe_params.emit_counts.shape[1]

    new_init = alpha_init / k_recipe + stats["init_counts"]
    new_trans = alpha_trans / k_recipe + stats["trans_counts"]
    new_emit = alpha_emit / n_subtask + stats["emit_counts"]

    return RecipeParams(new_init, new_trans, new_emit)


def run_em(
    key,
    seg_sequences,
    k_recipe,
    k_subtask,
    alpha_init=0.5,
    alpha_trans=0.5,
    alpha_emit=None,
    n_restarts=10,
    max_iters=100,
    tol=1e-4,
    progress=False,
):
    """Restart loop mirroring hsmm.em.run_em: fresh random init per restart, iterate E/M to
    convergence or max_iters, keep the best by final total log-likelihood.
    """
    if alpha_emit is None:
        alpha_emit = float(k_subtask)

    obs_ids, mask = pad_segment_batch(seg_sequences)

    best_params, best_loglik = None, -jnp.inf
    history = []

    restart_keys = jax.random.split(key, n_restarts)
    restart_bar = tqdm(restart_keys, desc="recipe restarts", disable=not progress)
    for restart_idx, restart_key in enumerate(restart_bar):
        p = init_weak_limit_recipe_params(
            restart_key, k_recipe, k_subtask, alpha_init, alpha_trans, alpha_emit
        )
        prev_loglik = -jnp.inf
        loglik = -jnp.inf
        restart_history = []
        iter_bar = tqdm(
            range(max_iters), desc=f"restart {restart_idx + 1}/{n_restarts}", leave=False, disable=not progress
        )
        for _ in iter_bar:
            stats, loglik = e_step(p, obs_ids, mask)
            p = m_step(p, stats, alpha_init, alpha_trans, alpha_emit)
            loglik_value = float(loglik)
            restart_history.append(loglik_value)
            iter_bar.set_postfix(loglik=f"{loglik_value:.1f}")
            if abs(loglik_value - float(prev_loglik)) < tol:
                break
            prev_loglik = loglik
        iter_bar.close()

        history.append(restart_history)
        if float(loglik) > float(best_loglik):
            best_params, best_loglik = p, loglik
        restart_bar.set_postfix(best_loglik=f"{float(best_loglik):.1f}")

    return best_params, best_loglik, history


def decode_recipe(recipe_params, obs_ids, mask):
    """Per-trial recipe = the majority vote of each segment's posterior-argmax recipe state
    (per user's locked choice: flat HMM with within-trial recipe transitions allowed, but a
    single per-trial recipe id is what gets scored against Breakfast's recipe_label)."""
    log_init, log_trans, log_emit = to_log_probs(recipe_params)
    gamma, _, _ = jax.vmap(_forward_backward, in_axes=(0, 0, None, None, None))(
        obs_ids, mask, log_init, log_trans, log_emit
    )
    per_segment = np.asarray(jnp.argmax(gamma, axis=-1))
    mask_np = np.asarray(mask)

    recipe_ids = np.zeros(obs_ids.shape[0], dtype=np.int64)
    for i in range(obs_ids.shape[0]):
        valid = per_segment[i][mask_np[i]]
        counts = np.bincount(valid)
        recipe_ids[i] = int(np.argmax(counts))
    return recipe_ids


def effective_k(recipe_ids, min_frac=0.02):
    """Number of recipe clusters holding a non-negligible share of trials -- the weak-limit
    report metric ('effective K, not nominal K'), read off the decode rather than raw
    pseudocounts. A plain distinct-label count would over-report: a handful of trials
    stray-assigned to an otherwise-unused nominal state (a few misclassifications out of
    hundreds) is noise, not a genuinely occupied mode, so clusters below `min_frac` of the
    dataset are excluded."""
    recipe_ids = np.asarray(recipe_ids)
    _, counts = np.unique(recipe_ids, return_counts=True)
    threshold = min_frac * len(recipe_ids)
    return int(np.sum(counts >= threshold))


def _contingency_table(labels_a, labels_b):
    a_vals, a_inv = np.unique(labels_a, return_inverse=True)
    b_vals, b_inv = np.unique(labels_b, return_inverse=True)
    table = np.zeros((len(a_vals), len(b_vals)), dtype=np.int64)
    np.add.at(table, (a_inv, b_inv), 1)
    return table, a_vals, b_vals


def adjusted_rand(labels_a, labels_b):
    """Hand-rolled Adjusted Rand Index (no scikit-learn dependency) from a contingency
    table. Standard sklearn convention: return 1.0 when the max-index/expected-index
    denominator is 0 (degenerate labelings, e.g. everything in one cluster on both sides)."""
    labels_a = np.asarray(labels_a)
    labels_b = np.asarray(labels_b)
    table, _, _ = _contingency_table(labels_a, labels_b)

    def comb2(n):
        return n * (n - 1) / 2.0

    sum_comb = np.sum(comb2(table))
    a_sums = table.sum(axis=1)
    b_sums = table.sum(axis=0)
    sum_comb_a = np.sum(comb2(a_sums))
    sum_comb_b = np.sum(comb2(b_sums))

    n = labels_a.shape[0]
    total_comb = comb2(n)
    expected = (sum_comb_a * sum_comb_b / total_comb) if total_comb > 0 else 0.0
    max_index = 0.5 * (sum_comb_a + sum_comb_b)
    denom = max_index - expected
    if denom == 0:
        return 1.0
    return (sum_comb - expected) / denom


def matched_accuracy(pred_clusters, true_labels):
    """Hungarian-matched cluster accuracy: best one-to-one mapping of predicted cluster ids
    to true labels (via scipy's linear_sum_assignment maximizing matched count), then the
    fraction of trials whose predicted cluster maps to their true label. Returns (accuracy,
    contingency_table, pred_vals, true_vals) so the mapping can be inspected."""
    pred = np.asarray(pred_clusters)
    true = np.asarray(true_labels)
    table, pred_vals, true_vals = _contingency_table(pred, true)

    row_ind, col_ind = linear_sum_assignment(-table)
    matched = table[row_ind, col_ind].sum()
    accuracy = matched / len(true)
    return accuracy, table, pred_vals, true_vals
