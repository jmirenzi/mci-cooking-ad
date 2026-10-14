"""class LLMPredictiveDetector: the LLM predicts, arithmetic decides (docs/llm.md §4).

At each run boundary the LLM is asked how long the run that has just started will last and which
run comes next. Channels:
  p_symbol    the run that starts differs from the one predicted at the previous boundary;
  p_dur_long  the first tick past the predicted duration plus tolerance;
  p_dur_short the run ended short of it (tick = its last tick, known at the next boundary).
"""
from cook_ad.core import Alarm, Detection, Trial
from cook_ad.detectors.llm_base import LLMDetector
from cook_ad.eval.element_metrics import duration_tolerance
from cook_ad.llm import predict


class LLMPredictiveDetector(LLMDetector):
    """Causal: one request per run boundary, prefix only."""
    name = "llm-predictive"
    protocol = predict.PROTOCOL

    # ---- one tick at a time

    def reset(self, recipe=None):
        self.t, self._v, self._n = 0, [], []
        self._preds, self._start, self._long_fired = [], 0, False

    def step(self, verb, noun):
        new_run = self.t == 0 or (verb, noun) != (self._v[-1], self._n[-1])
        alarms = self._close_run(known_at=self.t) if new_run and self.t else []
        self._v.append(verb)
        self._n.append(noun)
        if new_run:
            alarms += self._boundary()
        alarms += self._over_run()
        self.t += 1
        return alarms

    def finish(self):
        return self._close_run(known_at=self.t - 1) if self.t else []

    def _boundary(self):
        """Ask for this run's duration and the next run; test this run against the last NEXT."""
        steps = self.steps(self._v, self._n)
        i = len(steps) - 1
        messages = [{"role": "system", "content": self._prompt()},
                    {"role": "user", "content": predict.predictive_prefix(steps, i)}]
        self._preds.append(predict.parse_predictions(self.client.complete(messages), self.vocab))
        self._start, self._long_fired = self.t, False
        if i == 0:
            return []
        _, score = predict.symbol_disagreement(self._preds[i - 1], (steps[i].verb, steps[i].noun))
        return [Alarm(self.t, predict.SYMBOL, score, self.t)] if score else []

    def _limit(self):
        point = self._preds[-1][0].duration
        return None if point is None else (point, duration_tolerance(point))

    def _over_run(self):
        lim = self._limit()
        if lim is None or self._long_fired or self.t - self._start + 1 <= lim[0] + lim[1]:
            return []
        self._long_fired = True
        return [Alarm(self.t, predict.DUR_LONG, 1.0, self.t)]

    def _close_run(self, known_at):
        """Under-run of the run that ended at tick self.t - 1."""
        lim = self._limit()
        if lim is None or self.t - self._start >= lim[0] - lim[1]:
            return []
        return [Alarm(self.t - 1, predict.DUR_SHORT, 1.0, known_at)]

    # ---- whole trial

    def detect_trial(self, trial: Trial) -> Detection:
        """All boundary requests at once (concurrently); the same requests as the step loop."""
        flags, stats = predict.run_predictive(self.client, self._prompt(),
                                              self.steps(trial.verbs, trial.nouns), self.vocab)
        n = len(trial.verbs)
        alarms = [Alarm(int(f.tick), f.channel, float(f.strength),
                        min(f.tick + 1, n - 1) if f.channel == predict.DUR_SHORT else int(f.tick))
                  for f in flags]
        return Detection(alarms, trace=stats)
