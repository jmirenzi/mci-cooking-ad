import numpy as np

# Every injection needs some segments before/after the perturbation for it to be a genuine
# anomaly (an out-of-order step is only out-of-order relative to context). Trajectories with
# fewer segments than this are skipped by the driver rather than injected into.
MIN_SEGMENTS = 4

ERROR_TYPES = ("substitution", "abandonment", "omission", "transposition", "repetition")

# A (state, token) cell counts as "no training evidence" if its accumulated E-step pseudocount
# sits within this fraction of the state's own (max - min) range above its floor -- i.e. near
# the Dirichlet prior alone, not because the fitted model ranks it least-likely. Scale-invariant
# by construction, so it works whether the row's dynamic range is wide (a real corpus fit) or
# narrow (a hand-set test fixture).
SUBSTITUTION_UNSEEN_BAND = 0.05

# Abandonment keeps a random fraction of the segment's own duration before cutting -- a person
# who quits partway through, not one who quits on the very first tick.
ABANDON_KEEP_FRAC = (0.05, 0.20)


def _seg_bounds(segments):
    """[(state, d), ...] -> [(start, end_exclusive, state, d), ...] over tick indices."""
    bounds = []
    pos = 0
    for state, d in segments:
        bounds.append((pos, pos + d, state, d))
        pos += d
    return bounds


def _result(verb_ids, noun_ids, t0, t1, error_type, tick_map, edited_ticks=()):
    """`tick_map[i]` is the original-trial tick that degraded tick `i` came from -- the same
    concatenation each injector below already applies to its own verb_ids/noun_ids, applied
    instead to np.arange(T). `edited_ticks` lists degraded indices whose CONTENT differs from
    tick_map's source tick; every injector but substitution reorders/copies/drops ticks without
    rewriting any surviving one, so tick_map alone would otherwise imply their content is
    unchanged everywhere, which is exactly true for those four and false for substitution's
    retagged segment."""
    return {
        "verb_ids": np.asarray(verb_ids, dtype=np.int64),
        "noun_ids": np.asarray(noun_ids, dtype=np.int64),
        "window": (int(t0), int(t1)),
        "error_type": error_type,
        "tick_map": np.asarray(tick_map, dtype=np.int64),
        "edited_ticks": np.asarray(list(edited_ticks), dtype=np.int64),
    }


def _unseen_candidates(counts_row, current_id, band=SUBSTITUTION_UNSEEN_BAND):
    """Token ids with essentially no accumulated training evidence for this state: pseudocount
    within `band` of the row's own floor, scaled by the row's own (max - min) range. Falls back
    to the single least-observed id (excluding `current_id`) if the band is empty -- which only
    happens for a state whose row carries almost no dynamic range at all."""
    counts_row = np.asarray(counts_row)
    lo, hi = float(counts_row.min()), float(counts_row.max())
    threshold = lo + band * (hi - lo)
    candidates = np.flatnonzero(counts_row <= threshold)
    candidates = candidates[candidates != current_id]
    if candidates.size == 0:
        order = np.argsort(counts_row)
        candidates = order[order != current_id][:1]
    return candidates


def _argmin_candidate(counts_row, current_id):
    """The single least-observed token id for this state, excluding `current_id` -- the
    deterministic worst case used by select='hardest', as opposed to a random draw from the
    whole near-zero-evidence band."""
    counts_row = np.asarray(counts_row)
    order = np.argsort(counts_row)
    return int(order[order != current_id][0])


def inject_substitution(traj, rng, hsmm_params, select="random", neighbours=None):
    """Swap an ENTIRE segment's verb or noun for a token with no training evidence linking it to
    that state -- a clean substitution ('spreading, but with mustard, for the whole step'), not a
    single mid-step flicker. The replacement is drawn from hsmm_params.verb_counts/noun_counts --
    the raw accumulated E-step pseudocounts already computed once during fitting -- rather than
    an argmax over the fitted probability row, which is what keeps a 'random' pick from being
    engineered to be the single worst point under the exact distribution that later scores it.
    Only the chosen channel changes; the other stays exactly as observed, which is what isolates
    the item channel from the action channel (or vice versa).

    'random' picks the channel at random and draws the replacement from the whole near-zero-
    evidence band (see _unseen_candidates) -- the honest-recall case. 'hardest' is deterministic:
    it takes whichever channel has the single least-observed cell for this state, and uses that
    exact token (_argmin_candidate) rather than a random member of the band -- the worst case a
    substitution of this kind can look like.

    'near' replaces with the token's NEAREST EMBEDDING NEIGHBOUR (`neighbours`, from
    kernel.nearest_neighbours) -- milk for water, not knife for water. Both 'random' and
    'hardest' pick a token with no training evidence for the state, i.e. the most distant one
    under a semantic kernel, so neither can produce a near substitution. The replacement here
    comes from the embeddings alone and never from hsmm_params, so two models pointed at one
    --traj-params source are graded on a byte-identical degraded stream.

    `neighbours` is {channel: (W,) int array}; only the channels present are eligible, which is
    how a noun-only near-substitution benchmark is requested."""
    verb_ids = np.array(traj["verb_ids"])
    noun_ids = np.array(traj["noun_ids"])
    bounds = _seg_bounds(traj["segments"])
    T = len(verb_ids)

    i = _pick_segment(rng, range(1, len(bounds) - 1), select=select)  # interior: skip the leading/trailing idle
    start, end, state, _ = bounds[i]

    noun_row, verb_row = hsmm_params.noun_counts[state], hsmm_params.verb_counts[state]
    if select == "near":
        if not neighbours:
            raise ValueError("select='near' needs a `neighbours` table (kernel.nearest_neighbours)")
        eligible = [c for c in ("noun", "verb") if c in neighbours]
        channel = eligible[0] if len(eligible) == 1 else ("noun" if rng.random() < 0.5 else "verb")
    elif select == "hardest":
        channel = "noun" if float(np.min(noun_row)) <= float(np.min(verb_row)) else "verb"
    else:
        channel = "noun" if rng.random() < 0.5 else "verb"

    row, current, target = (
        (noun_row, int(noun_ids[start]), noun_ids) if channel == "noun" else (verb_row, int(verb_ids[start]), verb_ids)
    )
    if select == "near":
        new_id = int(np.asarray(neighbours[channel])[current])
    elif select == "hardest":
        new_id = _argmin_candidate(row, current)
    else:
        new_id = int(rng.choice(_unseen_candidates(row, current)))
    target[start:end] = new_id

    edited = np.arange(start, end)
    result = _result(verb_ids, noun_ids, start, end - 1, "substitution", np.arange(T), edited_ticks=edited)
    result["channel"] = channel
    # The token pair, so a caller can partition the injections afterwards -- e.g. separating
    # genuine near substitutions (milk -> water) from annotation variants (egg -> eggs).
    result["orig_id"], result["new_id"] = current, new_id
    return result


def inject_abandonment(traj, rng, keep_frac=ABANDON_KEEP_FRAC, select="random"):
    """Truncate an interior segment to some fraction of its OWN duration (the user quits partway
    through the step, not on its very first tick, and not on the trailing idle at the end of the
    trial -- leaving that early is invisible as an anomaly). Only the retrospective left-tail
    duration channel can catch this; the live stall signal cannot.

    'random' draws the kept fraction uniformly from `keep_frac` (5-20% by default) -- the honest-
    recall case. 'hardest' is deterministic: it always keeps the bottom of that range (5%, floored
    at 1 tick) -- the shortest, most abrupt cutoff, and so the worst case for the duration
    channel's left tail."""
    verb_ids = np.array(traj["verb_ids"])
    noun_ids = np.array(traj["noun_ids"])
    bounds = _seg_bounds(traj["segments"])
    T = len(verb_ids)

    # interior (skip the leading/trailing idle) AND at least 2 ticks long -- a 1-tick segment
    # can't be truncated at all (keep>=1 would leave nothing to cut), which used to produce a
    # degraded stream byte-identical to the healthy one while still being scored as an anomaly.
    valid = [i for i, (_, _, _, d) in enumerate(bounds) if 1 <= i < len(bounds) - 1 and d >= 2]
    i = _pick_segment(rng, valid, select=select)
    start, end, _, d = bounds[i]
    frac = keep_frac[0] if select == "hardest" else rng.uniform(*keep_frac)
    keep = int(np.clip(round(d * frac), 1, d - 1))
    cut_lo, cut_hi = start + keep, end
    verb_ids = np.delete(verb_ids, np.arange(cut_lo, cut_hi))
    noun_ids = np.delete(noun_ids, np.arange(cut_lo, cut_hi))
    onset = start + keep - 1  # the segment's premature end tick (unshifted; cuts are after it)
    tick_map = np.concatenate([np.arange(0, cut_lo), np.arange(cut_hi, T)])
    return _result(verb_ids, noun_ids, onset, onset, "abandonment", tick_map)


def inject_omission(traj, rng, select="random"):
    """Drop an interior dependent segment entirely, so its predecessor transitions straight to
    its successor -- an out-of-order / missing-step transition anomaly."""
    verb_ids = np.array(traj["verb_ids"])
    noun_ids = np.array(traj["noun_ids"])
    bounds = _seg_bounds(traj["segments"])
    T = len(verb_ids)

    i = _pick_segment(rng, range(1, len(bounds) - 1), select=select)  # interior
    start, end, _, _ = bounds[i]
    verb_ids = np.delete(verb_ids, np.arange(start, end))
    noun_ids = np.delete(noun_ids, np.arange(start, end))
    tick_map = np.concatenate([np.arange(0, start), np.arange(end, T)])
    return _result(verb_ids, noun_ids, start, start, "omission", tick_map)  # boundary now at `start`


def inject_transposition(traj, rng, select="random"):
    """Swap two adjacent interior segments -- steps done in the wrong order. Anomaly spans the
    swapped pair (two out-of-order boundaries)."""
    verb_ids = np.array(traj["verb_ids"])
    noun_ids = np.array(traj["noun_ids"])
    bounds = _seg_bounds(traj["segments"])
    T = len(verb_ids)

    i = _pick_segment(rng, range(1, len(bounds) - 2), select=select)  # i and i+1 interior
    a_start, a_end, _, da = bounds[i]
    b_start, b_end, _, db = bounds[i + 1]

    def _swapped(arr):
        return np.concatenate([arr[:a_start], arr[b_start:b_end], arr[a_start:a_end], arr[b_end:]])

    verb_ids = _swapped(verb_ids)
    noun_ids = _swapped(noun_ids)
    tick_map = _swapped(np.arange(T))
    return _result(verb_ids, noun_ids, a_start, a_start + da + db - 1, "transposition", tick_map)


def inject_repetition(traj, rng, select="random"):
    """Duplicate an interior segment in place (the user repeats a step). Shows as an impossible
    re-entry transition or, once Viterbi merges the copy, an over-long stall."""
    verb_ids = np.array(traj["verb_ids"])
    noun_ids = np.array(traj["noun_ids"])
    bounds = _seg_bounds(traj["segments"])
    T = len(verb_ids)

    i = _pick_segment(rng, range(1, len(bounds) - 1), select=select)  # interior
    start, end, _, d = bounds[i]

    def _duplicated(arr):
        return np.concatenate([arr[:end], arr[start:end], arr[end:]])

    verb_ids = _duplicated(verb_ids)
    noun_ids = _duplicated(noun_ids)
    tick_map = _duplicated(np.arange(T))  # many-to-one: the copy's ticks map to their originals
    return _result(verb_ids, noun_ids, end, end + d - 1, "repetition", tick_map)  # the inserted copy


def _pick_segment(rng, valid_indices, select):
    """Pick a segment index from `valid_indices` -- already filtered to this injector's own
    structural constraints (interior, minimum duration, ...), so every candidate here is a
    legal pick. 'random' (honest recall) or 'hardest' (leftmost valid -- a deterministic
    placeholder; a truly-adversarial pick would score each candidate's induced surprise, noted
    as future work)."""
    valid_indices = list(valid_indices)
    if not valid_indices:
        raise ValueError("trajectory has no segment satisfying this injection's constraints")
    if select in ("random", "near"):
        return int(rng.choice(valid_indices))
    if select == "hardest":
        return valid_indices[0]
    raise ValueError(f"unknown select mode: {select!r}")


def inject(error_type, traj, rng, hsmm_params, select="random", neighbours=None):
    """Uniform dispatch. Only substitution needs hsmm_params (to read the state's accumulated
    verb/noun counts and find a token with no training evidence); the other four are purely
    structural."""
    if error_type == "substitution":
        return inject_substitution(traj, rng, hsmm_params, select=select, neighbours=neighbours)
    if error_type == "abandonment":
        return inject_abandonment(traj, rng, select=select)
    if error_type == "omission":
        return inject_omission(traj, rng, select=select)
    if error_type == "transposition":
        return inject_transposition(traj, rng, select=select)
    if error_type == "repetition":
        return inject_repetition(traj, rng, select=select)
    raise ValueError(f"unknown error type: {error_type!r}")
