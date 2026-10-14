"""class LagHSMMDetector: Live, with the event channels conditioned on L more ticks (docs/detectors.md §4)."""
import math

import jax
import jax.numpy as jnp

from cook_ad.detectors.hsmm_live import LiveHSMMDetector
from cook_ad.hsmm import kernels

_backward_step = jax.jit(kernels.backward_step)


class LagHSMMDetector(LiveHSMMDetector):
    """Event alarms about tick t are raised at tick t + L. L = math.inf is the full smoother
    (a reference, not causal). Emission channels are forecasts, so they are final at t."""

    def __init__(self, hsmm, bad_sets, op, lag):
        self.L = lag
        self.causal = lag != math.inf
        self.name = f"hsmm-lag{lag}"
        super().__init__(hsmm, bad_sets, op)

    def beta(self, fr):
        """beta_t over o_{t+1 .. t+L}."""
        return self.hsmm.smooth(fr, self.L)

    def step(self, verb, noun):
        if not self.causal:
            raise NotImplementedError("L = inf needs the whole trial: use detect_trial()")
        return super().step(verb, noun)

    def _event_steps(self):
        s = self.t - self.L
        if s < 0:
            return []
        return self._event_alarms_at(s, self._window_beta(s, self.t), known_at=self.t)

    def _window_beta(self, s, last):
        """beta_s over o_{s+1 .. last}, by backward steps from the newest tick."""
        v = self.hsmm.current
        beta = jnp.zeros_like(self._history[s][1])
        for k in range(last, s, -1):
            beta = _backward_step(beta, self._history[k][2], v.log_A, v.log_h, v.log_stay)
        return beta

    def finish(self):
        """The last L ticks, with their windows cut at the end of the trial."""
        last = self.t - 1
        return [a for s in range(max(0, self.t - self.L), self.t)
                for a in self._event_alarms_at(s, self._window_beta(s, last), known_at=last)]
