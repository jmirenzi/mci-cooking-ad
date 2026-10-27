"""Stage 4: fetch and put-back logistics, inserted by code (never by the LLM).

Over a whole session's core tokens: `take n` before the first use of every ingredient, tool or
container noun; after its last use, `wash n` (tools and containers only) with P = wash_p, then
`put n` with P = put_p. Both rates come from the `tidy` knob (knobs.tidy_rates). The number of
takes, and the probability of each put or wash, do not depend on step order, so mise_en_place
cannot move token counts through this pass.

Exceptions:
- Food made during the session (PREPARED: a pancake, the mixture, the zucchini noodles) is never
  fetched or put away, and water (UNFETCHED) comes from the tap.
- A CONSUMED item (a garlic clove, the banana, the bread) is fetched but never put back.
- A put-back right after the recipe's own `put n` (placing the bowl in the microwave, the plate on
  the table) is dropped, since it would show as the same action twice.
- A vessel is in use for every step that happens in it (`in_use`, from vessels.yaml), whether or
  not a token names it: `add garlic` is in the pan. `serves_in` is the vessel the finished dish
  stays in; it is fetched but never washed or put away.
"""
LOGISTIC_TAGS = ("ingredient", "tool", "container")
PREPARED = ("food", "mixture", "liquid", "pancake", "omelette", "sandwich", "salad", "soup",
            "pizza", "dough", "burger", "pie", "noodle")
UNFETCHED = ("water",)
CONSUMED = ("garlic", "courgette", "banana", "bread", "strawberry", "tomato", "corn", "chocolate",
            "lemon", "lime", "wrap")
WASHABLE_TAGS = ("tool", "container")
DEFAULT_VERBS = {"take": "take", "put": "put", "wash": "wash"}


def insert_logistics(tokens, tags, put_p, wash_p, rng, verbs=DEFAULT_VERBS, in_use=None, serves_in=None):
    """-> (tokens with logistics, provenance list: "core" or "logistics" per output token).
    `in_use`: per token, the vessel nouns of its step (named by the token or not)."""
    def logistic(noun):
        return (noun not in PREPARED and noun not in UNFETCHED
                and any(t in tags["nouns"].get(noun, ()) for t in LOGISTIC_TAGS))

    first, last = {}, {}
    for i, (_, noun) in enumerate(tokens):
        for n in [noun, *(in_use[i] if in_use else ())]:
            if logistic(n):
                first.setdefault(n, i)
                last[n] = i
    # Draw once per noun, in first-use order, whether or not the draw is used: the stream of
    # random numbers then does not depend on which rates moved.
    after = {}
    for noun in sorted(first, key=first.get):
        washable = any(t in tags["nouns"].get(noun, ()) for t in WASHABLE_TAGS)
        do_wash = rng.random() < wash_p and washable
        do_put = rng.random() < put_p and (do_wash or tokens[last[noun]] != (verbs["put"], noun))
        if noun == serves_in:
            do_wash = do_put = False
        elif noun in CONSUMED:
            do_put = False
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
