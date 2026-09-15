"""
One Euro Filter (Casiez, Roussel, Vogel - CHI 2012, "1€ Filter: A Simple
Speed-based Low-pass Filter for Noisy Input in Interactive Systems").

Why this replaces the plain EMA that was here before: a fixed-alpha EMA has
one lag/smoothness tradeoff, chosen once, that has to compromise between
two conflicting needs - heavy smoothing to kill jitter while the hand is
nearly still, and low lag to stay responsive while the hand is moving
fast. Pick alpha for one case and the other suffers (too jittery at rest,
or too laggy in motion - which reads as "choppy" because the cursor is
visibly fighting a filter tuned for the wrong speed).

The One Euro Filter fixes this by making the cutoff frequency a function
of the signal's own estimated speed: near-zero speed -> low cutoff ->
heavy smoothing (kills jitter at rest); high speed -> higher cutoff ->
light smoothing (low lag while moving). `min_cutoff` sets the baseline
smoothing at rest; `beta` controls how aggressively the cutoff opens up as
speed increases (higher beta = less lag during fast movement, at the cost
of a bit more jitter tolerance during fast movement, where it's far less
noticeable anyway).
"""

import math


def _smoothing_factor(t_e: float, cutoff: float) -> float:
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)


class _LowPassFilter:
    def __init__(self, initial_value: float = 0.0):
        self.y = initial_value
        self.initialized = False

    def filter(self, value: float, alpha: float) -> float:
        if not self.initialized:
            self.y = value
            self.initialized = True
        else:
            self.y = alpha * value + (1 - alpha) * self.y
        return self.y


class OneEuroFilter1D:
    """Filters a single scalar signal over time. `min_cutoff` and `beta`
    are the two parameters worth tuning; `d_cutoff` (smoothing applied to
    the internal velocity estimate itself - this IS the "velocity
    smoothing" step, built into the filter rather than a separate stage)
    rarely needs changing from 1.0."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_filter = _LowPassFilter()
        self._dx_filter = _LowPassFilter(0.0)
        self._t_prev = None

    def filter(self, t: float, x: float) -> float:
        if self._t_prev is None:
            self._t_prev = t
            self._x_filter.filter(x, 1.0)
            return x

        t_e = max(1e-6, t - self._t_prev)
        self._t_prev = t

        prev_x = self._x_filter.y
        dx = (x - prev_x) / t_e
        a_d = _smoothing_factor(t_e, self.d_cutoff)
        edx = self._dx_filter.filter(dx, a_d)

        cutoff = self.min_cutoff + self.beta * abs(edx)
        a = _smoothing_factor(t_e, cutoff)
        return self._x_filter.filter(x, a)

    def reset(self):
        self._x_filter = _LowPassFilter()
        self._dx_filter = _LowPassFilter(0.0)
        self._t_prev = None


class OneEuroFilter2D:
    """Convenience wrapper: two independent OneEuroFilter1D instances (x, y)
    sharing the same timestamps, for filtering a 2D point like a fingertip
    position."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self._fx = OneEuroFilter1D(min_cutoff, beta, d_cutoff)
        self._fy = OneEuroFilter1D(min_cutoff, beta, d_cutoff)

    def filter(self, t: float, point: tuple) -> tuple:
        x, y = point
        return (self._fx.filter(t, x), self._fy.filter(t, y))

    def reset(self):
        self._fx.reset()
        self._fy.reset()
