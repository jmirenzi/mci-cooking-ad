from typing import NamedTuple

import jax.numpy as jnp

# Which HSMMParams count family each confirmation channel bumps. Duration (dur_r/dur_p) is
# deliberately absent: it is a pair of NB point estimates, not a conjugate Dirichlet count
# family, so it has no bounded-count online update in this design (a sufficient-statistic /
# duration-histogram representation would be needed -- future work, flagged in the plan).
COUNT_FIELD = {"verb": "verb_counts", "noun": "noun_counts", "trans": "trans_counts"}

DEFAULT_DELTA = 1.0
DEFAULT_MAX_BUMP = 5.0


class PreferenceEvent(NamedTuple):
    channel: str    # "verb" | "noun" | "trans"
    state: int      # believed subtask z*; for "trans" this is the FROM state
    token: int      # observed verb/noun id; for "trans" this is the TO state


def bounded_bump(live_counts, frozen_counts, row, col, delta, max_bump):
    """Add `delta` to the (row, col) cell of live_counts, capped so the live cell never
    exceeds the FROZEN baseline for that cell by more than `max_bump`.

    Bounding against frozen (not an absolute ceiling) is what makes a recurring, repeatedly-
    confirmed substitution stay elevated within a window: no matter how many times it is
    accepted, the live cell can rise at most `max_bump` pseudocounts above the reference, so
    the token stays well below the state's dominant token and keeps registering as surprising.
    One acceptance therefore does not blind the detector -- only a caretaker consolidating the
    change into the frozen baseline (state_manager.consolidate) does. Pure/functional.
    """
    ceiling = frozen_counts[row, col] + max_bump
    new_val = jnp.minimum(live_counts[row, col] + delta, ceiling)
    return live_counts.at[row, col].set(new_val)


def apply_preference(live, frozen, event, delta=DEFAULT_DELTA, max_bump=DEFAULT_MAX_BUMP):
    """Accommodate a confirmed preference: a single bounded count bump to the live model on
    the channel/cell the confirmation query was about. live/frozen: HSMMParams. Returns the
    updated live HSMMParams (frozen is untouched -- it moves only at consolidation)."""
    field = COUNT_FIELD[event.channel]
    new_counts = bounded_bump(
        getattr(live, field), getattr(frozen, field), event.state, event.token, delta, max_bump
    )
    return live._replace(**{field: new_counts})
