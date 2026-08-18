"""Protocol drivers and response parsing for the LLM baseline.

Two protocols, both returning one Verdict per step so eval/element_metrics.py scores them
identically:

  * `run_incremental` (default) -- one request per step. Request i shows the steps 0..i and asks
    for a verdict on step i. The model sees a growing PREFIX and never the future, which is what
    makes its step-latency comparable to the HSMM's online channels (predictive occupancy, live
    stall). Costs ~1 request per step, i.e. ~7 per trial.

    It is PREFIX-ONLY: the model's own earlier answers are NOT fed back. Causality -- seeing only
    the prefix -- is what the comparison needs; conversational self-feedback is a separate
    property that was costing two things for no measurement benefit. First, error cascade: a false
    alarm at step 2 sat in context for steps 3..7 and could bias them, so a miss at step 5 could
    not be distinguished from contamination by an earlier mistake. Each step is now an independent
    test, which is what the metrics assume. Second, every request in a sweep is now a pure
    function of (trial, step index) rather than depending on a previous response, so the sweep is
    embarrassingly parallel -- schedulable concurrently, or submissible to an async batch endpoint,
    neither of which is possible when request i+1 contains response i.

    The message layout is chosen so request i's prompt is a strict token PREFIX of request i+1's:
    one system turn, then one user turn holding the numbered step list, which only ever grows by
    appending a line. Servers with prefix caching (vLLM APC, and the hosted providers' implicit
    caching) then reuse nearly all of it, and the shared system prompt is reused across the whole
    sweep.
  * `run_batch` -- one request for the whole trial, one verdict line per step. ~7x cheaper, but
    the model sees every step before judging any of them, so it is NOT causal and its latency
    column is not comparable to the incremental arm or to the HSMM. Reports must label it.
"""
import re
from typing import NamedTuple

from cook_ad.llm import textify
from cook_ad.synthetic import error_injection

NO_ANOMALY = re.compile(r"^\s*no\s+anomaly\s*[.!]?\s*$", re.IGNORECASE)

# The strict grammar the preprompt asks for:
#   <TYPE> Anomaly. Correct move would have been VERB NOUN for NUMBER seconds
STRICT = re.compile(
    r"^\s*(?P<type>\w+)\s+anomaly\s*[.:!]?\s*"
    r"(?:correct\s+move\s+would\s+have\s+been\s+"
    r"(?P<verb>\w+)\s+(?P<noun>\w+)\s+for\s+(?P<dur>\d+)\s+seconds?\s*[.!]?)?\s*$",
    re.IGNORECASE,
)

# Lenient fallback: the reply contains an anomaly type somewhere. Used only after STRICT fails,
# and the Verdict still records parse_ok=False so a model that cannot hold the format shows up as
# a parse_failure_rate rather than being silently rescued into a clean number.
LENIENT_TYPE = re.compile(
    r"\b(" + "|".join(error_injection.ERROR_TYPES) + r")\b", re.IGNORECASE
)
LENIENT_CORRECTION = re.compile(
    r"\b(?P<verb>\w+)\s+(?P<noun>\w+)\s+for\s+(?P<dur>\d+)\s+seconds?\b", re.IGNORECASE
)


class Verdict(NamedTuple):
    step_index: int
    is_anomaly: bool
    error_type: str | None                      # one of error_injection.ERROR_TYPES, or None
    correction: tuple[str, str, int] | None     # (verb, noun, duration_seconds)
    raw: str
    parse_ok: bool


def _canon_type(text):
    t = text.strip().lower()
    return t if t in error_injection.ERROR_TYPES else None


def _canon_correction(verb, noun, dur, vocab):
    """Keep a correction only if both tokens are in-vocabulary. An out-of-vocabulary correction
    cannot be compared against the pre-injection step without fuzzy matching, and putting a fuzzy
    threshold inside a metric would make correction_accuracy a tunable number."""
    if verb is None or noun is None:
        return None
    v, n = verb.strip(), noun.strip()
    if v not in vocab["verbs"] or n not in vocab["nouns"]:
        return None
    return (v, n, int(dur))


def parse_response(text, step_index, vocab):
    """Strict grammar first, then a lenient rescue, then a recorded parse failure."""
    raw = (text or "").strip()
    # Models often prepend chatter; judge on the first non-empty line, which is what the
    # preprompt asks for, before falling back to scanning the whole reply.
    first = next((ln for ln in raw.splitlines() if ln.strip()), "")

    for candidate in (first, raw):
        if NO_ANOMALY.match(candidate):
            return Verdict(step_index, False, None, None, raw, True)

    for candidate in (first, raw):
        m = STRICT.match(candidate)
        if m:
            etype = _canon_type(m.group("type"))
            if etype is not None:
                correction = _canon_correction(m.group("verb"), m.group("noun"), m.group("dur"), vocab) \
                    if m.group("dur") else None
                return Verdict(step_index, True, etype, correction, raw, True)

    m = LENIENT_TYPE.search(raw)
    if m:
        etype = _canon_type(m.group(1))
        c = LENIENT_CORRECTION.search(raw)
        correction = _canon_correction(c.group("verb"), c.group("noun"), c.group("dur"), vocab) if c else None
        return Verdict(step_index, True, etype, correction, raw, False)

    # Unparseable. NOT coerced to "No Anomaly": a model that cannot follow the format is a
    # finding about that model, and silently reading it as the majority class would inflate its
    # apparent healthy-trial specificity for free.
    return Verdict(step_index, False, None, None, raw, False)


def render_prefix(steps, upto):
    """The numbered step list through index `upto` inclusive -- the user turn's whole content.

    Grows by pure append as `upto` advances, which is what keeps request i's prompt a strict
    prefix of request i+1's.
    """
    return "\n".join(f"{s.index + 1}. {textify.render_step(s)}" for s in steps[: upto + 1])


def incremental_messages(system_prompt, steps):
    """Every request for one trial, built upfront. Possible only because the protocol is
    prefix-only -- with self-feedback, request i+1 could not be built until response i arrived."""
    return [
        [{"role": "system", "content": system_prompt},
         {"role": "user", "content": render_prefix(steps, i)}]
        for i in range(len(steps))
    ]


def run_incremental(client, system_prompt, steps, vocab):
    """One request per step, prefix-only (no self-feedback). len(steps) requests.

    Two turns per request -- system, then the step list so far -- rather than an alternating
    conversation. A single growing user turn is universally supported (consecutive same-role
    messages are not), and it keeps the append-only prefix property that makes the prompts
    cacheable.

    All requests for the trial are issued through client.complete_many, so they run concurrently
    when the client allows it. Order is preserved, so verdict k is still step k's.
    """
    replies = client.complete_many(incremental_messages(system_prompt, steps))
    return [parse_response(reply, step.index, vocab) for step, reply in zip(steps, replies)]


BATCH_INSTRUCTION = (
    "Here is the complete sequence of steps for one trial, numbered in order. Give your verdict "
    "for EVERY step, one per line, in order, each line formatted exactly as:\n\n"
    "    <step number>. <your verdict>\n\n"
    "where <your verdict> follows the response format you were given. Output exactly one line per "
    "step and nothing else."
)

BATCH_LINE = re.compile(r"^\s*(?P<idx>\d+)\s*[.):]\s*(?P<body>.+?)\s*$")


def run_batch(client, system_prompt, steps, vocab):
    """One request for the whole trial. Cheap but NON-CAUSAL -- the model sees every step before
    judging any of them, so latency from this protocol is not comparable to run_incremental or to
    the HSMM's online channels."""
    listing = "\n".join(f"{i + 1}. {textify.render_step(s)}" for i, s in enumerate(steps))
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{BATCH_INSTRUCTION}\n\n{listing}"},
    ]
    reply = client.complete(messages)

    by_index = {}
    for line in (reply or "").splitlines():
        m = BATCH_LINE.match(line)
        if not m:
            continue
        i = int(m.group("idx")) - 1
        if 0 <= i < len(steps) and i not in by_index:
            by_index[i] = m.group("body")

    # A step the model never answered for is a parse failure for that step, not an implicit
    # "No Anomaly" -- same reasoning as parse_response's final branch.
    return [
        parse_response(by_index[s.index], s.index, vocab) if s.index in by_index
        else Verdict(s.index, False, None, None, "<no line returned for this step>", False)
        for s in steps
    ]


PROTOCOLS = {"incremental": run_incremental, "batch": run_batch}


def run_trial(client, system_prompt, steps, vocab, protocol="incremental"):
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown protocol: {protocol!r} (expected one of {tuple(PROTOCOLS)})")
    return PROTOCOLS[protocol](client, system_prompt, steps, vocab)


def request_cost(steps, protocol="incremental"):
    """Uncached requests one trial costs under a protocol -- what --dry-run sums."""
    return len(steps) if protocol == "incremental" else 1
