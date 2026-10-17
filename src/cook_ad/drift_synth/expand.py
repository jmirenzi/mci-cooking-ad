"""Stage 3: each recipe step -> core (verb, noun) tokens plus knob variants, by the LLM, once.

Per step, pass 1 asks for the core actions only, and pass 2 asks for a lower-fat variant (the
`fatty` knob's remove side: a substitute, or omit) whenever a core token uses a fat-tagged noun.
Then one call per recipe asks where (at most 2 steps, possibly none) a cook who likes sour
flavours would add one sour ingredient. Asked per step, the model put lemon and vinegar into
nearly every step, hot chocolate included; asked once with the whole dish in view, it can say
"nowhere".

The flavour pass (`add_flavour_variants`, 2026-10-17) adds the rest of the two-sided flavour
knobs: one call per recipe and knob for the sweet / salty / fatty insert sites (1-2 steps, always
some: how plausible they are is the affinity's job, `propose_affinity`), and, in code, an omit
for every core token whose noun is one of the knob's nouns (knobs.yaml `nouns`).

The JSON schemas enumerate the vocabulary, so the server cannot emit an out-of-vocab token.
`validate_expansion` checks what a schema cannot. A failing reply is retried up to 3 times with
the errors fed back; a step whose core pass still fails is logged and skipped, and a failed
variant pass just leaves that variant out. Nothing is invented as a fallback. Results are
committed to configs/drift_synth/expansions/<recipe>.json.
"""
import json
from pathlib import Path

from cook_ad.drift_synth.graphs import _id_key, parents
from cook_ad.drift_synth.kitchen_state import LOGISTIC_TAGS
from cook_ad.drift_synth.types import StepExpansion, Variant

VARIANT_KNOBS = {           # knob -> (allowed kinds, the noun tag its tokens must carry)
    "sour": (("insert", "omit"), "sour"),
    "sweet": (("insert", "omit"), "sweet"),
    "salty": (("insert", "omit"), "salty"),
    "fatty": (("insert", "substitute", "omit"), "fat"),
}
AFFINITY_LEVELS = {"rare": 0.1, "sometimes": 0.4, "common": 1.0}
# A pair with one of these verbs moves an ingredient, so its noun must be one (not the pan).
ADDING_VERBS = ("add", "pour", "sprinkle", "squeeze", "season", "scoop")
# Ingredient nouns a step may use without the recipe naming them: generic stuff and the dish
# being made (EPIC's prepared-food classes).
GENERIC_INGREDIENTS = ("food", "mixture", "liquid", "water", "ice", "pancake", "omelette",
                       "sandwich", "salad", "soup", "pizza", "dough", "burger", "pie", "cake")
MAX_SOUR_STEPS = 2
MAX_TOKENS = 12
MAX_TRIES = 3

SYSTEM = """You turn one step of a cooking recipe into the hand actions a camera would see, written
as (verb, noun) pairs from a fixed vocabulary (EPIC-KITCHENS classes).

Rules:
- One pair per visible action. Use the fewest pairs that show the step, usually 1 to 4, at most
  {max_tokens}. Never pad, never repeat a pair to fill space.
- The noun is the thing handled or the food acted on, never the destination:
  "Add salt to the pan" is "add salt", not "add pan".
- Use only the listed verbs and nouns, picking the class that covers the object: a mug is "cup",
  a skillet is "pan", cilantro is "coriander", a tortilla is "wrap", mozzarella is "cheese".
- Do NOT fetch or put away ingredients, tools or containers ("take X", or "put X" back in the
  fridge or cupboard): code adds those. "put" is right for placing something as part of the step.
- If the step waits on heat or time (microwaving, simmering, boiling, cooling, draining), include
  a "wait" pair on the appliance or the food.

Examples of the level wanted:
- "Chop 1/4 medium onion" -> cut onion
- "Add 1/2 tsp salt to the bowl" -> add salt
- "Whisk the egg mixture" -> mix egg
- "Heat 1 tbsp oil in a pan over medium heat" -> put pan, turn-on hob, pour oil, wait pan
- "Microwave the bowl for 2 minutes" -> open microwave, put bowl, close microwave, turn-on microwave, wait microwave, open microwave
- "Peel 2 garlic cloves" -> peel garlic
Answer only with the JSON object."""

CALORIE_SYSTEM = """A cooking simulator has a calorie-conscious cook who sometimes makes a step lower
in fat. You get one recipe step and its actions as (verb, noun) pairs; the pairs at the given
indices use a fat ingredient. Give the pairs that replace them in a lower-fat version (e.g. "pour
oil" -> "spray oil" is not lower fat; "sprinkle cheese" -> "sprinkle basil"; "add butter" -> "add
water"), using only the listed verbs and nouns. If the fat ingredient would simply be left out,
return an empty list. Answer only with the JSON object."""

SOUR_SYSTEM = """A cooking simulator has a cook who likes sour flavours. You get a whole recipe,
step by step with its actions as (verb, noun) pairs. Choose where this cook would add ONE sour
ingredient ({sour}) in ordinary home cooking: at most {max_steps} steps, each with the single pair
that adds it (e.g. "squeeze lemon" over a salad, "pour vinegar" into a pan sauce, "add yoghurt" to
a curry). Sweet dishes, desserts, baking and drinks usually get none: answer an empty list then.
Use only the listed verbs and nouns. Answer only with the JSON object."""


def _vocab_lists(vocab, sil=("stall", "idle")):
    return (sorted(v for v in vocab["verbs"] if v not in sil),
            sorted(n for n in vocab["nouns"] if n not in sil))


def _pairs_schema(vocab):
    verbs, nouns = _vocab_lists(vocab)
    pair = {"type": "object", "additionalProperties": False, "required": ["verb", "noun"],
            "properties": {"verb": {"type": "string", "enum": verbs},
                           "noun": {"type": "string", "enum": nouns}}}
    return {"type": "array", "items": pair}


def schema(vocab):
    """Pass 1: {"tokens": [{verb, noun}, ...]}."""
    return {"type": "object", "additionalProperties": False, "required": ["tokens"],
            "properties": {"tokens": _pairs_schema(vocab)}}


def calorie_schema(vocab):
    return {"type": "object", "additionalProperties": False, "required": ["tokens"],
            "properties": {"tokens": _pairs_schema(vocab)}}


def sour_schema(vocab, step_ids):
    site = {"type": "object", "additionalProperties": False, "required": ["step", "tokens"],
            "properties": {"step": {"type": "string", "enum": sorted(step_ids, key=_id_key)},
                           "tokens": _pairs_schema(vocab)}}
    return {"type": "object", "additionalProperties": False, "required": ["sites"],
            "properties": {"sites": {"type": "array", "items": site}}}


def _tagged(tags, tag):
    return sorted(n for n, t in tags["nouns"].items() if tag in t)


def _has(tags, noun, tag):
    return tag in tags["nouns"].get(noun, ())


def system_prompt(tags):
    return SYSTEM.format(max_tokens=MAX_TOKENS)


def _vocab_block(vocab):
    verbs, nouns = _vocab_lists(vocab)
    return "Verbs: " + ", ".join(verbs) + "\nNouns: " + ", ".join(nouns)


def user_prompt(step, recipe, vocab, exemplars, ingredients=()):
    par = parents(recipe)
    children = sorted((c for p, c in recipe.edges if p == step.step_id), key=_id_key)
    lines = [f"Recipe: {recipe.recipe_id}",
             f"Step ({step.phase}): {step.description}",
             "Comes after: " + ("; ".join(recipe.steps[p].description for p in sorted(par[step.step_id], key=_id_key)) or "(start)"),
             "Comes before: " + ("; ".join(recipe.steps[c].description for c in children) or "(end)"),
             "Ingredient nouns of this recipe: " + (", ".join(ingredients) or "(see step)"),
             "",
             "Real action sequences from kitchen videos, for the level of detail wanted:"]
    lines += [f"- {', '.join(f'{v} {n}' for v, n in ex)}" for ex in exemplars]
    return "\n".join(lines + ["", _vocab_block(vocab)])


def _step_block(step, recipe, tokens):
    return (f"Recipe: {recipe.recipe_id}\nStep: {step.description}\nActions: "
            + ", ".join(f"[{i}] {v} {n}" for i, (v, n) in enumerate(tokens)))


def parse(step_id, reply):
    """{"tokens": [...], "variants": [...]} (variants optional) -> StepExpansion."""
    toks = [(p["verb"], p["noun"]) for p in reply["tokens"]]
    variants = [Variant(v["knob"], v["kind"], step_id, [(p["verb"], p["noun"]) for p in v["tokens"]],
                        list(v["replaces"]) if v["kind"] != "insert" else None)
                for v in reply.get("variants", [])]
    return StepExpansion(step_id, toks, variants)


def validate_expansion(exp, vocab, tags, logistics_verbs=("take",), ingredients=None):
    """-> list of error strings (empty = valid). `ingredients`: if given, the only ingredient
    nouns the core tokens may use (plus GENERIC_INGREDIENTS); variants are exempt, since a sour
    insert or a lower-fat swap brings in a new ingredient by design."""
    errs = []
    nv, nn = vocab["verbs"], vocab["nouns"]

    def check_tokens(toks, where):
        for v, n in toks:
            if v not in nv or n not in nn:
                errs.append(f"{where}: '{v} {n}' is not in the vocabulary")
            elif v in ADDING_VERBS and not _has(tags, n, "ingredient"):
                errs.append(f"{where}: in '{v} {n}' the noun must be the ingredient moved, "
                            f"not the destination")

    if not 1 <= len(exp.tokens) <= MAX_TOKENS:
        errs.append(f"tokens: need 1 to {MAX_TOKENS} pairs, got {len(exp.tokens)}")
    check_tokens(exp.tokens, "tokens")
    for v, n in exp.tokens:
        if v in logistics_verbs and any(_has(tags, n, t) for t in LOGISTIC_TAGS):
            errs.append(f"tokens: '{v} {n}' is fetch logistics; remove it (code adds it)")
        if (ingredients is not None and _has(tags, n, "ingredient") and n not in ingredients
                and n not in GENERIC_INGREDIENTS and not _has(tags, n, "tool")):
            errs.append(f"tokens: '{v} {n}': {n} is not an ingredient of this recipe "
                        f"({', '.join(ingredients)})")
    if len(exp.tokens) != len(set(exp.tokens)) and len(exp.tokens) > 2:
        errs.append("tokens: a pair is repeated; give each action once")
    for i, var in enumerate(exp.variants):
        w = f"variant {i} ({var.knob} {var.kind})"
        kinds, tag = VARIANT_KNOBS.get(var.knob, ((), None))
        if var.kind not in kinds:
            errs.append(f"{w}: {var.knob} allows only {list(kinds)}")
            continue
        check_tokens(var.tokens, w)
        if var.kind == "insert":
            if not var.tokens:
                errs.append(f"{w}: an insert needs tokens")
            elif not any(_has(tags, n, tag) for _, n in var.tokens):
                errs.append(f"{w}: no inserted pair uses a {tag}-tagged noun")
            continue
        idx = var.replaces or []
        if not idx or any(not 0 <= j < len(exp.tokens) for j in idx):
            errs.append(f"{w}: replaces {idx} must be indices into the {len(exp.tokens)} tokens")
            continue
        if not any(_has(tags, exp.tokens[j][1], tag) for j in idx):
            errs.append(f"{w}: none of the replaced pairs uses a {tag}-tagged noun")
        if var.kind == "omit" and var.tokens:
            errs.append(f"{w}: an omit has no tokens")
        if var.kind == "substitute" and not var.tokens:
            errs.append(f"{w}: a substitute needs tokens")
        if var.kind == "substitute" and any(_has(tags, n, tag) for _, n in var.tokens):
            errs.append(f"{w}: the substitute still uses a {tag}-tagged noun")
    return errs


def _retry(ask, build, check):
    """Ask up to MAX_TRIES times; feed errors back. -> (result or None, [errors per attempt])."""
    history, feedback = [], ""
    for _ in range(MAX_TRIES):
        try:
            result = build(ask(feedback))
            errs = check(result)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            result, errs = None, [f"malformed reply: {e!r}"]
        history.append(errs)
        if not errs:
            return result, history
        feedback = "\n\nYour previous answer was rejected:\n- " + "\n- ".join(errs) + "\nFix these."
    return None, history


def expand_step(step, recipe, vocab, tags, exemplars, llm_cfg, ingredients=None):
    """Core pass and calorie pass for one step. -> (StepExpansion or None,
    {"core": [...], "fatty": [...]} attempt errors)."""
    from cook_ad.drift_synth.lexicon import step_verb
    from cook_ad.drift_synth.llm import ask_json, client_for
    if step_verb(step.description) == "take":
        # CC4D's "Take X" steps are fetch logistics themselves: no core action. Stage 4 inserts
        # `take X` before X's first use.
        return StepExpansion(step.step_id, [], []), {"core": [[]]}
    log = {}

    core_client = client_for(llm_cfg, schema(vocab), "expansion")
    user = user_prompt(step, recipe, vocab, exemplars, ingredients or ())
    base, log["core"] = _retry(
        lambda fb: ask_json(core_client, system_prompt(tags), user + fb),
        lambda r: parse(step.step_id, r),
        lambda e: validate_expansion(e, vocab, tags, ingredients=ingredients))
    if base is None:
        return None, log
    toks = base.tokens
    variants = []

    fat_idx = [i for i, (_, n) in enumerate(toks) if _has(tags, n, "fat")]
    if fat_idx:
        client = client_for(llm_cfg, calorie_schema(vocab), "calorie_variant")
        user = (_step_block(step, recipe, toks) + f"\nFat-ingredient pairs at indices: {fat_idx}\n\n"
                + _vocab_block(vocab))

        def build_cal(r):
            sub = [(p["verb"], p["noun"]) for p in r["tokens"]]
            return Variant("fatty", "substitute" if sub else "omit", step.step_id,
                           sub, list(fat_idx))
        var, log["fatty"] = _retry(
            lambda fb: ask_json(client, CALORIE_SYSTEM, user + fb), build_cal,
            lambda v: validate_expansion(StepExpansion(step.step_id, toks, [v]), vocab, tags))
        if var is not None:
            variants.append(var)
    return StepExpansion(step.step_id, toks, variants), log


def sour_sites(recipe, expansions, vocab, tags, llm_cfg):
    """One call per recipe: {step_id: sour insert Variant} (at most MAX_SOUR_STEPS, maybe none),
    and the attempt errors."""
    from cook_ad.drift_synth.llm import ask_json, client_for
    ids = sorted(expansions, key=_id_key)
    client = client_for(llm_cfg, sour_schema(vocab, ids), "sour_sites")
    sysmsg = SOUR_SYSTEM.format(sour=", ".join(_tagged(tags, "sour")), max_steps=MAX_SOUR_STEPS)
    user = (f"Recipe: {recipe.recipe_id}\n"
            + "\n".join(f"step {k}: {recipe.steps[k].description} | actions: "
                        + ", ".join(f"{v} {n}" for v, n in expansions[k].tokens) for k in ids)
            + "\n\n" + _vocab_block(vocab))

    def build(r):
        return {site["step"]: Variant("sour", "insert", site["step"],
                                      [(p["verb"], p["noun"]) for p in site["tokens"]], None)
                for site in r["sites"]}

    def check(sites):
        errs = []
        if len(sites) > MAX_SOUR_STEPS:
            errs.append(f"at most {MAX_SOUR_STEPS} steps, got {len(sites)}")
        for k, var in sites.items():
            errs += [f"step {k}: {e}" for e in validate_expansion(
                StepExpansion(k, expansions[k].tokens, [var]), vocab, tags)]
            if len(var.tokens) > 2:
                errs.append(f"step {k}: give at most two pairs")
        return errs
    return _retry(lambda fb: ask_json(client, sysmsg, user + fb), build, check)


# ---- the flavour pass: sweet / salty / fatty inserts, omits, affinity --------------------------

SITE_SYSTEM = """A cooking simulator has a cook who {likes}. You get a whole recipe, step by step
with its actions as (verb, noun) pairs. If this cook made the dish their way, where would they add
{what}? Choose the 1 or {max_steps} most natural steps, each with the single pair that adds it
(e.g. {example}). Always choose at least one step, even for a dish where it would be unusual: how
likely it is is decided elsewhere. Use only these nouns for the added ingredient: {nouns}, and only
the listed verbs. Answer only with the JSON object."""

SITE_KNOBS = {   # knob -> (likes, what, example)
    "sweet": ("likes food sweeter", "extra sugar or syrup",
              '"sprinkle sugar" over fruit, "pour syrup" on pancakes, "add sugar" to a sauce'),
    "salty": ("likes food saltier", "extra salt", '"sprinkle salt" over a salad, "add salt" to a pan'),
    "fatty": ("likes rich food", "extra fat",
              '"add butter" to a pan, "pour oil" over vegetables, "sprinkle cheese" on top'),
}

AFFINITY_SYSTEM = """A cooking simulator has cooks with flavour preferences who sometimes change a
recipe: adding something sour (lemon, lime, vinegar), something sweet (sugar, syrup), salt, or extra
fat (butter, oil, cheese, cream). For the recipe you get, say for each how often a home cook who
likes that flavour would add it to THIS dish:
- "common": a usual, natural addition (salt on a savoury dish, butter on toast, syrup on pancakes)
- "sometimes": plausible, some people do it
- "rare": odd but possible (lime in hot chocolate, salt in coffee)
Give a short reason for each. Answer only with the JSON object."""


def sites_schema(vocab, step_ids, nouns):
    verbs, _ = _vocab_lists(vocab)
    pair = {"type": "object", "additionalProperties": False, "required": ["verb", "noun"],
            "properties": {"verb": {"type": "string", "enum": verbs},
                           "noun": {"type": "string", "enum": sorted(nouns)}}}
    site = {"type": "object", "additionalProperties": False, "required": ["step", "pair"],
            "properties": {"step": {"type": "string", "enum": sorted(step_ids, key=_id_key)},
                           "pair": pair}}
    return {"type": "object", "additionalProperties": False, "required": ["sites"],
            "properties": {"sites": {"type": "array", "items": site, "minItems": 1,
                                     "maxItems": MAX_SOUR_STEPS}}}


def affinity_schema():
    from cook_ad.drift_synth.knobs import FLAVOUR
    one = {"type": "object", "additionalProperties": False, "required": ["reason", "level"],
           "properties": {"reason": {"type": "string"},
                          "level": {"type": "string", "enum": list(AFFINITY_LEVELS)}}}
    return {"type": "object", "additionalProperties": False, "required": list(FLAVOUR),
            "properties": {k: one for k in FLAVOUR}}


def _recipe_block(recipe, expansions, vocab=None):
    ids = sorted(expansions, key=_id_key)
    out = (f"Recipe: {recipe.recipe_id}\n"
           + "\n".join(f"step {k}: {recipe.steps[k].description} | actions: "
                       + ", ".join(f"{v} {n}" for v, n in expansions[k].tokens) for k in ids))
    return out + ("\n\n" + _vocab_block(vocab) if vocab else "")


def flavour_sites(recipe, expansions, knob, nouns, vocab, tags, llm_cfg):
    """One call per recipe and knob: {step_id: insert Variant} (1-2 steps), and the attempt errors."""
    from cook_ad.drift_synth.llm import ask_json, client_for
    ids = [k for k in sorted(expansions, key=_id_key) if expansions[k].tokens]
    client = client_for(llm_cfg, sites_schema(vocab, ids, nouns), f"{knob}_sites")
    likes, what, example = SITE_KNOBS[knob]
    sysmsg = SITE_SYSTEM.format(likes=likes, what=what, max_steps=MAX_SOUR_STEPS, example=example,
                                nouns=", ".join(sorted(nouns)))
    user = _recipe_block(recipe, expansions, vocab)

    def build(r):
        out = {}
        for site in r["sites"]:
            out.setdefault(site["step"], Variant(knob, "insert", site["step"], [], None)
                           ).tokens.append((site["pair"]["verb"], site["pair"]["noun"]))
        return out

    def check(sites):
        errs = []
        if not 1 <= len(sites) <= MAX_SOUR_STEPS:
            errs.append(f"choose 1 to {MAX_SOUR_STEPS} steps, got {len(sites)}")
        for k, var in sites.items():
            errs += [f"step {k}: {e}" for e in validate_expansion(
                StepExpansion(k, expansions[k].tokens, [var]), vocab, tags)]
            errs += [f"step {k}: '{v} {n}' must add one of {sorted(nouns)}"
                     for v, n in var.tokens if n not in nouns]
            if len(var.tokens) > 1:
                errs.append(f"step {k}: give one pair per step")
        return errs
    return _retry(lambda fb: ask_json(client, sysmsg, user + fb), build, check)


def code_omits(exp, knob, nouns):
    """An omit variant for every core token whose noun is one of the knob's nouns."""
    return [Variant(knob, "omit", exp.step_id, [], [j])
            for j, (_, n) in enumerate(exp.tokens) if n in nouns]


def add_flavour_variants(recipe, expansions, knobs, vocab, tags, llm_cfg):
    """-> ({step_id: StepExpansion} with every flavour knob's variants, {knob: attempt errors}
    for failed site passes). Keeps the core tokens, the sour inserts and the lower-fat variants;
    rebuilds everything this pass owns, so it can be rerun."""
    from cook_ad.drift_synth.knobs import FLAVOUR
    rename = {"calorie_conscious": "fatty"}
    out = {}
    for k, e in expansions.items():
        keep = [Variant(rename.get(v.knob, v.knob), v.kind, v.anchor_step, v.tokens, v.replaces)
                for v in e.variants]
        keep = [v for v in keep if (v.knob, v.kind) in
                {("sour", "insert"), ("fatty", "substitute"), ("fatty", "omit")}]
        for knob in FLAVOUR:
            if knob != "fatty":
                keep += code_omits(e, knob, set(knobs[knob]["nouns"]))
        out[k] = StepExpansion(e.step_id, e.tokens, keep)
    failures = {}
    for knob in SITE_KNOBS:
        sites, hist = flavour_sites(recipe, out, knob, set(knobs[knob]["nouns"]), vocab, tags, llm_cfg)
        if sites is None:
            failures[knob] = hist
        for k, var in (sites or {}).items():
            e = out[k]
            out[k] = StepExpansion(e.step_id, e.tokens, e.variants + [var])
    return out, failures


def propose_affinity(recipe, expansions, llm_cfg):
    """-> {knob: {"level", "affinity", "reason"}} for one recipe (one call)."""
    from cook_ad.drift_synth.llm import ask_json, client_for
    client = client_for(llm_cfg, affinity_schema(), "flavour_affinity")
    r = ask_json(client, AFFINITY_SYSTEM, _recipe_block(recipe, expansions))
    return {k: {"level": v["level"], "affinity": AFFINITY_LEVELS[v["level"]], "reason": v["reason"]}
            for k, v in r.items()}


# ---- few-shot exemplars from EPIC-100 ---------------------------------------------------------

class ExemplarIndex:
    """Windows of consecutive EPIC narrations (as class names), found by noun overlap."""

    def __init__(self, videos, width=8, stride=4):
        self.windows = []
        self.by_noun = {}
        for vid in sorted(videos):
            narr = videos[vid]
            for start in range(0, max(len(narr) - width + 1, 1), stride):
                win = _collapse([(v, n) for _, _, v, n in narr[start:start + width]])
                wid = len(self.windows)
                self.windows.append((vid, win))
                for n in {n for _, n in win}:
                    self.by_noun.setdefault(n, []).append(wid)

    @classmethod
    def from_epic(cls, root, **kw):
        from cook_ad.data.parse_epic import load_classes, load_narrations
        vk, nk, _ = load_classes(root)
        return cls({vid: [(t0, t1, vk[v], nk[n]) for t0, t1, v, n in e["narrations"]]
                    for vid, e in load_narrations(root).items()}, **kw)

    def best(self, nouns, k=4):
        """The k windows sharing the most distinct nouns with `nouns`, one per video, ties broken
        by window order (deterministic)."""
        nouns = set(nouns)
        score = {}
        for n in nouns:
            for wid in self.by_noun.get(n, ()):
                score[wid] = score.get(wid, 0) + 1
        out, seen = [], set()
        for wid in sorted(score, key=lambda w: (-score[w], w)):
            vid, win = self.windows[wid]
            if vid not in seen:
                out.append(win)
                seen.add(vid)
            if len(out) == k:
                break
        return out


def _collapse(tokens):
    """Consecutive repeats -> one (a run), as the stream would show them."""
    return [t for i, t in enumerate(tokens) if i == 0 or t != tokens[i - 1]]


def build_expansions(recipe, vocab, tags, index, llm_cfg, out_dir, step_nouns):
    """All steps of `recipe` (cached file wins). `step_nouns(description)` -> EPIC nouns for
    exemplar lookup. Returns ({step_id: StepExpansion}, {step_id: attempt errors} failures)."""
    from concurrent.futures import ThreadPoolExecutor

    from cook_ad.drift_synth.assets import load_expansions, save_expansions
    path = Path(out_dir) / f"{recipe.recipe_id}.json"
    fail_path = path.with_suffix(".failures.json")
    if path.exists():
        failures = json.loads(fail_path.read_text()) if fail_path.exists() else {}
        return load_expansions(path), failures
    steps = sorted(recipe.steps.values(), key=lambda s: _id_key(s.step_id))
    recipe_nouns = sorted({n for s in steps for n in step_nouns(s.description)})
    ingredients = [n for n in recipe_nouns if _has(tags, n, "ingredient")]

    def one(step):
        nouns = step_nouns(step.description) or recipe_nouns
        return step.step_id, expand_step(step, recipe, vocab, tags, index.best(nouns), llm_cfg,
                                         ingredients)

    with ThreadPoolExecutor(max_workers=llm_cfg.get("concurrency", 4)) as pool:
        results = list(pool.map(one, steps))
    expansions = {sid: exp for sid, (exp, _) in results if exp is not None}
    # A step is a failure if its core pass failed (skipped), or a variant pass failed (variant
    # left out); `log` keeps every attempt's errors for the report.
    failures = {sid: log for sid, (exp, log) in results
                if exp is None or any(h and h[-1] for h in log.values())}
    sites, hist = sour_sites(recipe, expansions, vocab, tags, llm_cfg)
    if sites is None:
        failures["recipe"] = {"sour": hist}
    for k, var in (sites or {}).items():
        e = expansions[k]
        expansions[k] = StepExpansion(e.step_id, e.tokens, e.variants + [var])
    path.parent.mkdir(parents=True, exist_ok=True)
    save_expansions(path, expansions)
    if failures:
        fail_path.write_text(json.dumps(failures, indent=1, sort_keys=True) + "\n")
    return expansions, failures
