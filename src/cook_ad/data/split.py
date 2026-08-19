import json


def load_split(path):
    with open(path) as f:
        return json.load(f)


def _ids_for_part(split, part):
    key = f"{part}_trial_ids"
    if key not in split:
        raise ValueError(f"split file has no '{key}' key (part={part!r})")
    return set(split[key])


def filter_sequences(sequences, split, part):
    ids = _ids_for_part(split, part)
    return [s for s in sequences if s["trial_id"] in ids]


def filter_labels(labels, split, part):
    ids = _ids_for_part(split, part)
    return [entry for entry in labels if entry["trial_id"] in ids]
