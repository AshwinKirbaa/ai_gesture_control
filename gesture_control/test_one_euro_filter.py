"""
Offline sanity tests for one_euro_filter.py. No camera/display required.
"""

import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gesture_control.one_euro_filter import OneEuroFilter1D

DEFAULT_MIN_CUTOFF = 0.8
DEFAULT_BETA = 0.9


def test_reduces_jitter_at_rest():
    random.seed(42)
    f = OneEuroFilter1D(min_cutoff=DEFAULT_MIN_CUTOFF, beta=DEFAULT_BETA, d_cutoff=1.0)
    t, outputs = 0.0, []
    true_val = 0.5
    for _ in range(60):
        noisy = true_val + random.uniform(-0.01, 0.01)  # raw noise range ~0.02
        outputs.append(f.filter(t, noisy))
        t += 1 / 30.0
    settled = outputs[-20:]
    jitter = max(settled) - min(settled)
    assert jitter < 0.01, f"expected meaningfully reduced jitter, got {jitter:.5f}"
    print(f"PASS: at-rest jitter reduced to {jitter:.5f} (raw noise range was ~0.02)")


def test_low_lag_during_fast_motion():
    f = OneEuroFilter1D(min_cutoff=DEFAULT_MIN_CUTOFF, beta=DEFAULT_BETA, d_cutoff=1.0)
    t, lags = 0.0, []
    for i in range(30):
        true_x = i * 0.03  # a fast, steady sweep
        out = f.filter(t, true_x)
        lags.append(abs(true_x - out))
        t += 1 / 30.0
    late_lag = sum(lags[-10:]) / 10
    assert late_lag < 0.06, f"expected low lag once warmed up on fast motion, got {late_lag:.4f}"
    print(f"PASS: tracking lag during fast motion settled to {late_lag:.4f} (normalized units)")


def test_adapts_more_than_fixed_ema_would():
    """The whole point of One Euro over a plain EMA: it should smooth a
    still signal MORE than it smooths a fast-moving one, i.e. its
    effective responsiveness changes with speed - a fixed-alpha EMA can't
    do this at all (same alpha regardless of speed)."""
    # Still signal: measure how much filtered output moves per raw input step.
    f_still = OneEuroFilter1D(min_cutoff=DEFAULT_MIN_CUTOFF, beta=DEFAULT_BETA, d_cutoff=1.0)
    t = 0.0
    f_still.filter(t, 0.5)
    t += 1 / 30.0
    out_still = f_still.filter(t, 0.51)  # small step while "at rest"
    still_response = abs(out_still - 0.5)

    # Fast-moving signal: same-size relative step, but signal was already moving fast.
    f_fast = OneEuroFilter1D(min_cutoff=DEFAULT_MIN_CUTOFF, beta=DEFAULT_BETA, d_cutoff=1.0)
    t = 0.0
    x = 0.0
    for i in range(10):
        x += 0.03
        f_fast.filter(t, x)
        t += 1 / 30.0
    x += 0.03
    out_fast = f_fast.filter(t, x)
    fast_response = abs(out_fast - (x - 0.03))

    assert fast_response > still_response, (
        f"expected the filter to respond MORE readily once the signal is moving fast "
        f"(fast_response={fast_response:.4f}, still_response={still_response:.4f})"
    )
    print(f"PASS: filter opens up during fast motion (still_response={still_response:.4f} "
          f"< fast_response={fast_response:.4f})")


if __name__ == "__main__":
    test_reduces_jitter_at_rest()
    test_low_lag_during_fast_motion()
    test_adapts_more_than_fixed_ema_would()
    print("\nAll one_euro_filter tests passed.")
