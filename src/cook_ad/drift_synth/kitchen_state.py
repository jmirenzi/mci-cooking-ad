"""Stage 4: fetch and put-back logistics, inserted by code (never by the LLM).

Over a whole session's core tokens: `take n` before the first use of every ingredient, tool or
container noun; after its last use, `wash n` (tools and containers only) with P = wash_p, then
`put n` with P = put_p. Both rates come from the `tidy` knob (knobs.tidy_rates). The number of
takes, and the probability of each put or wash, do not depend on step order, so mise_en_place
cannot move token counts through this pass.
"""
LOGISTIC_TAGS = ("ingredient", "tool", "container")
WASHABLE_TAGS = ("tool", "container")
DEFAULT_VERBS = {"take": "take", "put": "put", "wash": "wash"}


def insert_logistics(tokens, tags, put_p, wash_p, rng, verbs=DEFAULT_VERBS):
    """-> (tokens with logistics, provenance list: "core" or "logistics" per output token)."""
    first, last = {}, {}
    for i, (_, noun) in enumerate(tokens):
        if any(t in tags["nouns"].get(noun, ()) for t in LOGISTIC_TAGS):
            first.setdefault(noun, i)
            last[noun] = i
    # Draw once per noun, in first-use order, whether or not the draw is used: the stream of
    # random numbers then does not depend on which rates moved.
    after = {}
    for noun in sorted(first, key=first.get):
        washable = any(t in tags["nouns"].get(noun, ()) for t in WASHABLE_TAGS)
        do_wash = rng.random() < wash_p and washable
        do_put = rng.random() < put_p
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
