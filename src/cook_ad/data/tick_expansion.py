import math


def expand_to_ticks(segments, fps, tick_seconds):
    """Bin frame-level (value, start_frame, end_frame) segments into per-tick values.

    Frames are 1-indexed and inclusive, contiguous across segments. A tick whose frame
    span straddles two segments is assigned the value with the larger frame overlap.
    """
    total_frames = segments[-1][2]
    frames_per_tick = fps * tick_seconds
    n_ticks = math.ceil(total_frames / frames_per_tick)

    frame_values = [None] * (total_frames + 1)
    for value, start, end in segments:
        for frame in range(start, end + 1):
            frame_values[frame] = value

    tick_values = []
    for tick in range(n_ticks):
        tick_start = int(tick * frames_per_tick) + 1
        tick_end = min(int((tick + 1) * frames_per_tick), total_frames)
        counts = {}
        for frame in range(tick_start, tick_end + 1):
            value = frame_values[frame]
            counts[value] = counts.get(value, 0) + 1
        majority_value = max(counts.items(), key=lambda kv: kv[1])[0]
        tick_values.append(majority_value)

    return tick_values


def expand_verb_noun_to_ticks(segments, fps, tick_seconds):
    """segments: list of (verb_id, noun_id, start_frame, end_frame) -> (verb_ids, noun_ids) per tick."""
    paired = [((verb_id, noun_id), start, end) for verb_id, noun_id, start, end in segments]
    tick_pairs = expand_to_ticks(paired, fps, tick_seconds)
    verb_ids = [v for v, n in tick_pairs]
    noun_ids = [n for v, n in tick_pairs]
    return verb_ids, noun_ids
