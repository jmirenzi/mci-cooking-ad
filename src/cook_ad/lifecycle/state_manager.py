from typing import NamedTuple

from cook_ad.hsmm.params import HSMMParams
from cook_ad.lifecycle import online_update


class DualModel(NamedTuple):
    frozen: HSMMParams   # slow reference; moves ONLY at consolidation, on confirmed preferences
    live: HSMMParams     # short-horizon copy; accommodates confirmed preferences within a window


def init_dual_model(frozen: HSMMParams) -> DualModel:
    """Copy-then-freeze: live starts identical to frozen. The count arrays are immutable jax
    arrays and every update produces a fresh array (`.at[...].set`), so sharing the reference
    is safe -- no defensive copy needed."""
    return DualModel(frozen=frozen, live=frozen)


def handle_confirmation(dual, event, outcome, delta=online_update.DEFAULT_DELTA,
                        max_bump=online_update.DEFAULT_MAX_BUMP):
    """Route a confirmation-query outcome (the confirmation oracle is supplied, not learned):
      "preference" -- user affirmed their surprising action -> bounded bump to LIVE only.
      "breakdown"  -- user recognized a mistake            -> update NOTHING, flag the incident.
    Returns (new_dual, record). Frozen never moves here -- only consolidate() moves it.
    """
    if outcome == "preference":
        new_live = online_update.apply_preference(dual.live, dual.frozen, event, delta, max_bump)
        return DualModel(dual.frozen, new_live), {"outcome": "preference", "event": event, "updated": True}
    if outcome == "breakdown":
        return dual, {"outcome": "breakdown", "event": event, "updated": False, "flagged": True}
    raise ValueError(f"unknown confirmation outcome: {outcome!r} (expected 'preference' or 'breakdown')")


def consolidate(dual, approved="all"):
    """Weekly consolidation as a RESET, not an increment (the locked precision-growth guard):
    approved live drift defines the new frozen baseline, then live is re-initialized from that
    new frozen. Precision has nowhere to compound -- frozen is re-based, live is short-horizon.

    approved:
      "all"                       accept all accumulated live drift (new frozen = live).
      list of PreferenceEvent     accept only those cells (copy live->frozen there); unapproved
                                  live drift is discarded when live resets to the new frozen.
    """
    if approved == "all":
        new_frozen = dual.live
    else:
        new_frozen = _apply_approved(dual.frozen, dual.live, approved)
    return DualModel(frozen=new_frozen, live=new_frozen)


def _apply_approved(frozen, live, approved):
    new_frozen = frozen
    for event in approved:
        field = online_update.COUNT_FIELD[event.channel]
        approved_val = getattr(live, field)[event.state, event.token]
        updated = getattr(new_frozen, field).at[event.state, event.token].set(approved_val)
        new_frozen = new_frozen._replace(**{field: updated})
    return new_frozen
