"""Stage 2: tags per vocabulary item, proposed by the LLM, reviewed by hand.

Knobs act on tags, and a tag belongs to a vocabulary item (an EPIC verb or noun class), never to a
recipe. Each item is shown to the LLM with its EPIC class instances, because a class is often
broader than its name (`pepper` is bell and black pepper; `sauce` is soy, salsa, mayonnaise, ...).
`tags.yaml` is written with `reviewed: false`; `load_tags` refuses it until a human flips it.
"""
import ast
import csv
from pathlib import Path

import yaml

# Which tags make sense for which kind of item. A verb is never an ingredient; a noun is never
# fine-motor (the fine-motor knob acts on actions).
NOUN_TAGS = ("sour", "fat", "sweet", "salty", "heat", "cleaning", "ingredient", "tool", "container")
VERB_TAGS = ("fine_motor", "heat", "cleaning")

TAG_SYSTEM = """You tag kitchen vocabulary items for a cooking simulator. Each item is an EPIC-KITCHENS
class; the words in brackets are what the class covers. Give every item the tags that are true of
it in ordinary home cooking, and no others. An empty list is a fine answer.

Noun tags:
- sour: tastes sour or acidic and is added for that (lemon, lime, vinegar, yoghurt, pickles).
- fat: a fat or a fat-rich food a calorie-conscious cook would cut or swap (butter, oil, cheese, cream, bacon, mayonnaise).
- sweet: added for sweetness (sugar, honey, syrup, chocolate, jam).
- heat: an appliance or surface that heats (hob, oven, microwave, kettle, toaster).
- cleaning: used for cleaning (sponge, washing liquid, cloth, kitchen towel).
- ingredient: something eaten or drunk, or added to food or drink.
- tool: a utensil or device handled while cooking (knife, spoon, spatula, grater, blender).
- container: holds food or drink (bowl, plate, pan, pot, cup, jar, box, bag, can).
Verb tags:
- fine_motor: needs precise hand control and gets slower or repeated when dexterity drops (cut, peel, grate, crack, spread, whisk, fold, roll, tie, insert).
- heat: applies heat (cook, bake, fry, boil, turn on a hob).
- cleaning: cleans (wash, wipe, scrub, dry, rinse).
If a class mixes members, tag it by what most members are; do not tag `sauce` sour because salsa is.
Answer only with the JSON object."""


def class_instances(epic_annotations):
    """{"verbs": {key: [instances]}, "nouns": {...}} from EPIC-100's class CSVs."""
    root = Path(epic_annotations)
    out = {}
    for kind, name in (("verbs", "EPIC_100_verb_classes.csv"), ("nouns", "EPIC_100_noun_classes_v2.csv")):
        with open(root / name, newline="") as f:
            out[kind] = {r["key"]: ast.literal_eval(r["instances"]) for r in csv.DictReader(f)}
    return out


def _schema(items, allowed):
    return {"type": "object", "additionalProperties": False, "required": list(items),
            "properties": {k: {"type": "array", "items": {"type": "string", "enum": list(allowed)}}
                           for k in items}}


def _prompt(kind, items, instances):
    lines = [f"Tag these {kind}:"]
    for k in items:
        shown = [i.replace(":", " ") for i in instances.get(k, [])[:10]]
        lines.append(f"{k} [{', '.join(shown)}]")
    return "\n".join(lines)


def propose_tags(vocab, instances, llm_cfg, batch_size=40, sil=("stall", "idle")):
    """-> {"verbs": {verb: [tags]}, "nouns": {noun: [tags]}} over the whole vocab (SIL excluded)."""
    from cook_ad.drift_synth.llm import ask_json, client_for
    out = {}
    for kind, allowed in (("verbs", VERB_TAGS), ("nouns", NOUN_TAGS)):
        items = sorted(k for k in vocab[kind] if k not in sil)
        out[kind] = {}
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            client = client_for(llm_cfg, _schema(batch, allowed), f"tags_{kind}")
            reply = ask_json(client, TAG_SYSTEM, _prompt(kind, batch, instances[kind]))
            out[kind].update({k: sorted(set(reply[k]), key=allowed.index) for k in batch})
    return out


def write_tags(path, tags, tag_set):
    doc = {"reviewed": False, "tag_set": list(tag_set),
           "verbs": {k: v for k, v in sorted(tags["verbs"].items())},
           "nouns": {k: v for k, v in sorted(tags["nouns"].items())}}
    header = ("# LLM-proposed tags per EPIC vocabulary item (scripts/drift_synth/build_assets.py).\n"
              "# Review by hand: add/remove tags, then set `reviewed: true`. The generator refuses\n"
              "# this file until then. Knobs act on these tags (configs/drift_synth/knobs.yaml).\n")
    Path(path).write_text(header + yaml.safe_dump(doc, sort_keys=False, default_flow_style=None,
                                                  width=100))


def load_tags(path, require_reviewed=True):
    """-> {"verbs": {...}, "nouns": {...}}; raises unless the file says `reviewed: true`."""
    doc = yaml.safe_load(Path(path).read_text())
    if require_reviewed and doc.get("reviewed") is not True:
        raise RuntimeError(f"{path} has not been reviewed: check the tags, then set `reviewed: true`")
    return {"verbs": {k: list(v or []) for k, v in doc["verbs"].items()},
            "nouns": {k: list(v or []) for k, v in doc["nouns"].items()}}


def tagged(tags, kind, tag):
    return sorted(k for k, v in tags[kind].items() if tag in v)


def summary(tags):
    """Markdown: every tag's members, for the review."""
    lines = ["| kind | tag | n | items |", "|---|---|---|---|"]
    for kind, allowed in (("verbs", VERB_TAGS), ("nouns", NOUN_TAGS)):
        for t in allowed:
            items = tagged(tags, kind, t)
            lines.append(f"| {kind} | {t} | {len(items)} | {', '.join(items)} |")
        untagged = sorted(k for k, v in tags[kind].items() if not v)
        lines.append(f"| {kind} | (none) | {len(untagged)} | {', '.join(untagged)} |")
    return "\n".join(lines)
