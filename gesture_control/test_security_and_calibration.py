"""Offline tests for strict authentication, calibration, and telemetry."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gesture_control.auth_bridge import AuthBridge
from gesture_control.auto_calibration import CursorCalibrator, recommend_dead_zone
from gesture_control.performance import PerformanceMonitor


class FakeFaceId:
    @staticmethod
    def detect_faces(_frame):
        return [{"embedding": [1.0]}]


def test_missing_owner_never_unlocks_for_an_arbitrary_face():
    auth = AuthBridge(FakeFaceId(), None, {"auth_check_every_n_frames": 1}, 0.5)
    for frame in range(10):
        status = auth.update(object(), now=float(frame + 1))
    assert status.locked
    assert "not enrolled" in status.reason


def test_calibration_uses_noise_percentile_with_limits():
    points = [(0.5 + i * 0.0001, 0.5) for i in range(20)]
    value = recommend_dead_zone(points)
    assert 0.0005 <= value <= 0.01
    assert value == 0.0005  # minimum avoids an unusably tiny threshold


def test_performance_monitor_reports_rolling_average():
    monitor = PerformanceMonitor(window=2)
    monitor.record("hands", 10)
    monitor.record("hands", 20)
    monitor.record("hands", 30)
    assert monitor.averages()["hands"] == 25


def test_calibration_pauses_when_valid_tracking_stops():
    calibrator = CursorCalibrator(duration_seconds=1.0, minimum_samples=3)
    calibrator.add((0.5, 0.5), now=0.0)
    # Wall time alone must not finish calibration after tracking disappears.
    assert not calibrator.complete(now=10.0)
    assert calibrator.remaining(now=10.0) == 1.0
    calibrator.add((0.5001, 0.5), now=10.0)
    calibrator.add((0.5002, 0.5), now=10.25)
    assert not calibrator.complete(now=10.25)


if __name__ == "__main__":
    test_missing_owner_never_unlocks_for_an_arbitrary_face()
    test_calibration_uses_noise_percentile_with_limits()
    test_performance_monitor_reports_rolling_average()
    test_calibration_pauses_when_valid_tracking_stops()
    print("PASS: strict authentication, calibration, and telemetry")
