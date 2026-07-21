"""Recipe/subtask ground-truth labels.

VALIDATION ONLY. Nothing in this module is ever fed to EM/training -- it exists to
score recovered latent states (Phase 3 cascade, Phase 6 evaluation) against the
Breakfast annotations, kept in a file structurally separate from the integer
training sequences.
"""

from cook_ad.data.tick_expansion import expand_to_ticks


def build_trial_labels(trial_id, recipe_label, segments, fps, tick_seconds):
    """segments: list of (label, verb, noun, start_frame, end_frame) for one trial.

    Returns a per-tick raw action-label array aligned with the trial's verb_ids/noun_ids
    (same tick binning), so it can be compared index-for-index against inferred states.
    """
    label_segments = [(label, start, end) for label, verb, noun, start, end in segments]
    subtask_labels = expand_to_ticks(label_segments, fps, tick_seconds)
    return {
        "trial_id": trial_id,
        "recipe_label": recipe_label,
        "subtask_labels": subtask_labels,
    }
