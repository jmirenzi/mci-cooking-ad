"""Stage 4: fetch and put-back logistics, inserted by code (never by the LLM).

Over a whole session's core tokens: `take n` before the first use of every ingredient, tool or
container noun; after its last use, `wash n` (tools and containers only) with P = wash_p, then
`put n` with P = put_p. Both rates come from the `tidy` knob (knobs.tidy_rates). The number of
takes, and the probability of each put or wash, do not depend on step order, so mise_en_place
cannot move token counts through this pass.

Two exceptions: food made during the session (PREPARED: a pancake, the mixture) is never fetched
or put away, and a put-back right after the recipe's own `put n` (placing the bowl in the
microwave, the plate on the table) is dropped, since it would show as the same action twice.

A tool or container is in use until its phase ends, not just until a token names it: `add garlic`
and `cook garlic` happen in the pan without saying `pan`. With `phases` (one per token), its last
use moves to the last token of the phase of its last named use, so it is washed and put away
after the cooking, not after `wait pan`. Ingredients keep their last named use.
"""
LOGISTIC_TAGS = ("ingredient", "tool", "container")
PREPARED = ("food", "mixture", "liquid", "pancake", "omelette", "sandwich", "salad", "soup",
            "pizza", "dough", "burger", "pie")
WASHABLE_TAGS = ("tool", "container")
DEFAULT_VERBS = {"take": "take", "put": "put", "wash": "wash"}


def insert_logistics(tokens, tags, put_p, wash_p, rng, verbs=DEFAULT_VERBS, phases=None):
    """-> (tokens with logistics, provenance list: "core" or "logistics" per output token)."""
    first, last = {}, {}
    for i, (_, noun) in enumerate(tokens):
        if noun not in PREPARED and any(t in tags["nouns"].get(noun, ()) for t in LOGISTIC_TAGS):
            first.setdefault(noun, i)
            last[noun] = i
    if phases is not None:
        for noun in last:
            if any(t in tags["nouns"].get(noun, ()) for t in WASHABLE_TAGS):
                ph = phases[last[noun]]
                last[noun] = max(i for i in range(last[noun], len(tokens)) if phases[i] == ph)
    # Draw once per noun, in first-use order, whether or not the draw is used: the stream of
    # random numbers then does not depend on which rates moved.
    after = {}
    for noun in sorted(first, key=first.get):
        washable = any(t in tags["nouns"].get(noun, ()) for t in WASHABLE_TAGS)
        do_wash = rng.random() < wash_p and washable
        do_put = rng.random() < put_p and (do_wash or tokens[last[noun]] != (verbs["put"], noun))
        after.setdefault(last[noun], []).extend(
            ([(verbs["wash"], noun)] if do_wash else []) + ([(verbs["put"], noun)] if do_put else []))
    before = {}
    for noun, i in first.items():
        before.setdefault(i, []).append((verbs["take"], noun))

    out, prov = [], []
    for i, tok in enumerate(tokens):
        for t in sorted(before.get(i, [])):
            out.append(t)
            prov.append("logistics")
        out.append(tok)
        prov.append("core")
        for t in after.get(i, []):
            out.append(t)
            prov.append("logistics")
    return out, prov
