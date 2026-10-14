"""class LLMVerdictDetector: the LLM judges each element itself (docs/llm.md §3).

unit "run": an element is a run of constant (verb, noun), judged once it has ended.
unit "tick": an element is one tick, judged as it happens. An anomalous verdict becomes an alarm
at the element's first tick. Protocols: "incremental" (the prefix so far, no feedback),
"conversational" (its own earlier answers fed back) and "batch" (the whole trial; not causal).
"""
from cook_ad.core import Alarm, Detection, Trial
from cook_ad.detectors.llm_base import LLMDetector
from cook_ad.llm import detect, textify

UNITS = {"run": "step", "tick": "tick"}  # our name -> the name the prompt text and llm/ use


class LLMVerdictDetector(LLMDetector):

    def __init__(self, client, vocab, lexicon, protocol="incremental", unit="run", **kwargs):
        if protocol not in detect.PROTOCOLS:
            raise ValueError(f"unknown protocol {protocol!r}")
        self.protocol, self.unit = protocol, UNITS[unit]
        self.causal = protocol != "batch"
        self.name = f"llm-verdict-{protocol}-{unit}"
        super().__init__(client, vocab, lexicon, prompt_unit=self.unit, **kwargs)

    def elements(self, verbs, nouns):
        return textify.elements_from_ids(verbs, nouns, self.lexicon, self.unit)

    def _alarm(self, verdict, elements, known_at):
        e = elements[verdict.step_index]
        return Alarm(int(e.tick_start), verdict.error_type or "", 1.0, int(known_at))

    # ---- one tick at a time

    def reset(self, recipe=None):
        self.t, self._v, self._n, self._judged = 0, [], [], 0
        self._messages = [{"role": "system", "content": self._prompt()}]

    def step(self, verb, noun):
        if not self.causal:
            raise NotImplementedError("the batch protocol needs the whole trial: use detect_trial()")
        self._v.append(verb)
        self._n.append(noun)
        self.t += 1
        # a tick is judged as soon as it is seen; a run once a different tick has followed it
        return self._judge_ended(self.t if self.unit == "tick" else self.t - 1)

    def finish(self):
        return self._judge_ended(self.t) if self.causal else []

    def _judge_ended(self, end):
        """Query every element not yet judged that is over by tick `end` (exclusive)."""
        elements = self.elements(self._v, self._n)
        known_at, alarms = self.t - 1, []
        while self._judged < len(elements) and elements[self._judged].tick_end <= end:
            v = self._query(elements, self._judged)
            if v.is_anomaly:
                alarms.append(self._alarm(v, elements, known_at))
            self._judged += 1
        return alarms

    def _query(self, elements, i):
        if self.protocol == "incremental":
            messages = [{"role": "system", "content": self._prompt()},
                        {"role": "user", "content": detect.render_prefix(elements, i, self.unit)}]
            return detect.parse_response(self.client.complete(messages), elements[i].index,
                                         self.vocab)
        line = textify.render_trial(elements[: i + 1], self.unit)[i]
        self._messages.append({"role": "user", "content": line})
        reply = self.client.complete(self._messages)
        self._messages.append({"role": "assistant", "content": reply})
        return detect.parse_response(reply, elements[i].index, self.vocab)

    # ---- whole trial

    def known_at(self, element, n):
        if not self.causal:
            return n - 1
        return element.tick_start if self.unit == "tick" else min(element.tick_end, n - 1)

    def detect_trial(self, trial: Trial) -> Detection:
        elements = self.elements(trial.verbs, trial.nouns)
        verdicts = detect.run_trial(self.client, self._prompt(), elements, self.vocab,
                                    self.protocol, self.unit)
        n = len(trial.verbs)
        alarms = [self._alarm(v, elements, self.known_at(elements[v.step_index], n))
                  for v in verdicts if v.is_anomaly]
        return Detection(sorted(alarms, key=lambda a: a.tick), trace=verdicts)
