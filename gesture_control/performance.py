"""Small rolling performance profiler used by the live HUD."""

from collections import defaultdict, deque
from contextlib import contextmanager
import time


class PerformanceMonitor:
    def __init__(self, window=60):
        self.window = window
        self._samples = defaultdict(lambda: deque(maxlen=window))

    @contextmanager
    def measure(self, name):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record(name, (time.perf_counter() - start) * 1000.0)

    def record(self, name, milliseconds):
        self._samples[name].append(float(milliseconds))

    def averages(self):
        return {
            name: sum(values) / len(values)
            for name, values in self._samples.items() if values
        }
