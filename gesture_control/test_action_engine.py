"""
Offline sanity tests for the pure-math helpers in action_engine.py
(clamp_movement, accumulate_scroll_ticks). No display or pynput needed -
these are deliberately factored out as pure functions so they can be
tested without a screen.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gesture_control.action_engine import clamp_movement, accumulate_scroll_ticks


def test_speed_cap_limits_large_single_frame_jump():
    """A single spurious large jump (e.g. hand tracking briefly landing on
    the wrong spot) should be capped, not applied at full magnitude."""
    dx, dy = clamp_movement(desired_dx=500, desired_dy=0, prev_dx=0, prev_dy=0,
                             max_speed=60, max_accel=1000)
    assert abs(dx) <= 60 + 1e-6, f"expected speed-capped movement, got dx={dx}"
    print(f"PASS: a 500px jump was capped to {dx:.1f}px by the speed limiter")


def test_acceleration_cap_smooths_sudden_direction_change():
    """Even a movement within the speed cap should still be limited in how
    much it can differ from the previous frame's actual movement."""
    dx, dy = clamp_movement(desired_dx=50, desired_dy=0, prev_dx=-50, prev_dy=0,
                             max_speed=1000, max_accel=10)
    change = ((dx - (-50)) ** 2 + dy ** 2) ** 0.5
    assert change <= 10 + 1e-6, f"expected acceleration-capped change <= 10, got {change:.2f}"
    print(f"PASS: a sudden 100px direction reversal was limited to a {change:.1f}px change this frame")


def test_gentle_movement_passes_through_unclamped():
    """Normal, smooth movement well within both caps shouldn't be altered
    at all - the limiter should be invisible during ordinary use."""
    dx, dy = clamp_movement(desired_dx=5, desired_dy=3, prev_dx=4, prev_dy=2,
                             max_speed=60, max_accel=18)
    assert abs(dx - 5) < 1e-9 and abs(dy - 3) < 1e-9, f"expected passthrough, got ({dx}, {dy})"
    print("PASS: gentle in-bounds movement passes through the limiter unchanged")


def test_fast_swipe_ramps_up_over_several_frames_not_instantly():
    """A legitimately fast swipe should still be reachable - just reached
    over a couple of frames instead of snapping instantly to full speed."""
    prev_dx, prev_dy = 0.0, 0.0
    applied = []
    for _ in range(8):
        dx, dy = clamp_movement(desired_dx=200, desired_dy=0, prev_dx=prev_dx, prev_dy=prev_dy,
                                 max_speed=200, max_accel=30)
        applied.append(dx)
        prev_dx, prev_dy = dx, dy
    assert applied[0] < 200, "first frame should be limited, not instant full speed"
    assert applied[-1] > applied[0], "movement should ramp up over subsequent frames"
    assert applied[-1] >= 190, f"should eventually reach close to the desired speed, got {applied[-1]:.1f}"
    print(f"PASS: fast movement ramps {applied[0]:.0f} -> {applied[-1]:.0f} px over {len(applied)} frames instead of snapping")


def test_scroll_accumulates_fractional_remainder():
    """The old behavior truncated the fractional part every single frame,
    silently discarding most of the signal during slow scrolling. This
    checks the remainder is carried forward and eventually produces ticks
    even when each individual frame's delta is below 1.0."""
    accum = 0.0
    total_ticks = 0
    for _ in range(10):
        ticks, accum = accumulate_scroll_ticks(accum, 0.35)  # sub-1.0 delta every frame
        total_ticks += ticks
    # 10 frames * 0.35 = 3.5 total signal -> should produce 3 whole ticks, not 0
    assert total_ticks == 3, f"expected 3 accumulated ticks from 10x0.35, got {total_ticks}"
    print(f"PASS: 10 frames of 0.35/frame scroll delta correctly accumulated to {total_ticks} ticks (not 0)")


def test_scroll_accumulation_handles_negative_direction():
    accum = 0.0
    total_ticks = 0
    for _ in range(10):
        ticks, accum = accumulate_scroll_ticks(accum, -0.35)
        total_ticks += ticks
    assert total_ticks == -3, f"expected -3 accumulated ticks, got {total_ticks}"
    print(f"PASS: negative-direction scroll accumulates correctly ({total_ticks} ticks)")


if __name__ == "__main__":
    test_speed_cap_limits_large_single_frame_jump()
    test_acceleration_cap_smooths_sudden_direction_change()
    test_gentle_movement_passes_through_unclamped()
    test_fast_swipe_ramps_up_over_several_frames_not_instantly()
    test_scroll_accumulates_fractional_remainder()
    test_scroll_accumulation_handles_negative_direction()
    print("\nAll action_engine helper tests passed.")
