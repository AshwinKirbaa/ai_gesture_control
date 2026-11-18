"""Automatic cursor-noise calibration from captured landmark samples."""

import math


def recommend_dead_zone(points, percentile=0.95, safety_factor=1.35,
                        minimum=0.0005, maximum=0.01):
    """Estimate a dead zone from successive normalized fingertip positions.

    The user holds a pointing pose still; the selected percentile represents
    ordinary tracker noise. A small safety margin prevents residual drift.
    """
    if len(points) < 10:
        raise ValueError("at least 10 cursor samples are required")
    deltas = sorted(
        math.hypot(b[0] - a[0], b[1] - a[1])
        for a, b in zip(points, points[1:])
    )
    index = min(len(deltas) - 1, max(0, round((len(deltas) - 1) * percentile)))
    return max(minimum, min(maximum, deltas[index] * safety_factor))


class CursorCalibrator:
    def __init__(self, duration_seconds=3.0, minimum_samples=20):
        self.duration_seconds = duration_seconds
        self.minimum_samples = minimum_samples
        self.started_at = None
        self.last_sample_at = None
        self.active_seconds = 0.0
        self.points = []

    def add(self, point, now):
        if self.started_at is None:
            self.started_at = now
        if self.last_sample_at is not None:
            # Count only continuous valid tracking time. A missing/incorrect
            # pose pauses calibration instead of letting wall time expire.
            gap = max(0.0, now - self.last_sample_at)
            if gap <= 0.25:
                self.active_seconds += gap
        self.last_sample_at = now
        self.points.append((float(point[0]), float(point[1])))

    def remaining(self, now):
        return max(0.0, self.duration_seconds - self.active_seconds)

    def samples_remaining(self):
        return max(0, self.minimum_samples - len(self.points))

    def complete(self, now):
        return (
            self.started_at is not None
            and self.remaining(now) <= 0
            and self.samples_remaining() == 0
        )

    def result(self):
        return recommend_dead_zone(self.points)
