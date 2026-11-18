"""
Offline sanity tests for gesture_engine.py using hand-built synthetic
landmark sets. No camera, no MediaPipe, no display required - this only
exercises the pure geometry + state-machine logic, run with:

    python gesture_control/test_gesture_engine.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gesture_control.gesture_engine import (
    GestureEngine, finger_states, pinch_distance_normalized,
)


def _lm(x, y, z=0.0):
    return (x, y, z)


def make_hand(finger_tip_offsets):
    """Builds a plausible 21-point hand. finger_tip_offsets lets a test
    nudge specific tips (e.g. to simulate a fist or a pinch) while keeping
    everything else at a normal open-hand pose."""
    lm = [None] * 21
    lm[0] = _lm(0.50, 0.90)  # wrist

    # thumb
    lm[1] = _lm(0.45, 0.85)
    lm[2] = _lm(0.42, 0.80)
    lm[3] = _lm(0.40, 0.75)
    lm[4] = _lm(0.38, 0.70)  # tip

    # index
    lm[5] = _lm(0.48, 0.75)
    lm[6] = _lm(0.48, 0.65)
    lm[7] = _lm(0.48, 0.55)
    lm[8] = _lm(0.48, 0.45)  # tip

    # middle
    lm[9] = _lm(0.50, 0.75)
    lm[10] = _lm(0.50, 0.63)
    lm[11] = _lm(0.50, 0.51)
    lm[12] = _lm(0.50, 0.40)  # tip

    # ring
    lm[13] = _lm(0.52, 0.75)
    lm[14] = _lm(0.52, 0.65)
    lm[15] = _lm(0.52, 0.55)
    lm[16] = _lm(0.52, 0.45)  # tip

    # pinky
    lm[17] = _lm(0.54, 0.75)
    lm[18] = _lm(0.54, 0.68)
    lm[19] = _lm(0.54, 0.61)
    lm[20] = _lm(0.54, 0.54)  # tip

    for idx, (x, y) in finger_tip_offsets.items():
        lm[idx] = _lm(x, y)

    return lm


def curled_hand():
    """A fist: every tip pulled back near the wrist/MCP, closer to the
    wrist than its own pip joint."""
    return make_hand({
        4: (0.44, 0.83),   # thumb tip near thumb MCP, tucked in
        8: (0.48, 0.74),   # index tip near index MCP
        12: (0.50, 0.74),
        16: (0.52, 0.74),
        20: (0.54, 0.73),
    })


def pinch_hand():
    """Thumb tip and index tip brought together."""
    h = make_hand({})
    h[4] = _lm(0.475, 0.46)   # thumb tip moved right next to index tip
    return h


def test_finger_states_open_palm():
    h = make_hand({})
    states = finger_states(h)
    assert states == [True, True, True, True, True], f"expected all up, got {states}"
    print("PASS: open palm -> all fingers up")


def test_finger_states_fist():
    h = curled_hand()
    states = finger_states(h)
    assert states == [False, False, False, False, False], f"expected all down, got {states}"
    print("PASS: fist -> all fingers down")


def test_pinch_distance_shrinks():
    open_h = make_hand({})
    pinch_h = pinch_hand()
    d_open = pinch_distance_normalized(open_h, "thumb", "index")
    d_pinch = pinch_distance_normalized(pinch_h, "thumb", "index")
    assert d_pinch < d_open, f"pinch distance ({d_pinch}) should be smaller than open ({d_open})"
    assert d_pinch < 0.32, f"pinch distance {d_pinch} should be under the default enter threshold"
    print(f"PASS: pinch distance shrinks on pinch ({d_open:.2f} -> {d_pinch:.2f})")


def test_open_palm_gesture_start_hold_end_with_hold_ms():
    gestures_cfg = {
        "open_palm": {"type": "fingers", "hand": "any",
                       "pattern": [True, True, True, True, True], "hold_ms": 200},
    }
    reliability_cfg = {"smoothing_window_frames": 3, "static_gesture_debounce_frames": 2,
                        "default_action_cooldown_seconds": 0.1, "motion_gesture_cooldown_seconds": 0.1}
    engine = GestureEngine(gestures_cfg, reliability_cfg)

    open_hand = {"handedness": "Right", "landmarks": make_hand({})}
    t = 1000.0
    seen_states = []

    # Frames before hold_ms elapses: should NOT start yet.
    for i in range(3):
        events = engine.process([open_hand], now=t)
        seen_states.extend(e.state for e in events)
        t += 0.05  # 50ms per frame
    assert "start" not in seen_states, f"fired too early: {seen_states}"

    # Keep holding past 200ms total -> should start, then hold.
    fired_start = False
    for i in range(6):
        events = engine.process([open_hand], now=t)
        for e in events:
            if e.state == "start":
                fired_start = True
        t += 0.05
    assert fired_start, "open_palm never fired a 'start' event while held"
    print("PASS: open_palm respects hold_ms before firing 'start'")

    # Hand disappears -> should fire 'end' once the grace period elapses
    # (a hand briefly missing for less than that is treated as a normal
    # tracking blip, not a real disappearance - see _HandSlot).
    t += 0.3
    events = engine.process([], now=t)
    assert any(e.state == "end" for e in events), "expected 'end' event after the hand is gone past the grace period"
    print("PASS: gesture force-closes with 'end' once the hand is gone past the grace period")


def test_swipe_left_detected_from_motion():
    gestures_cfg = {
        "swipe_left": {"type": "swipe", "hand": "any", "direction": "left",
                        "min_velocity": 0.5, "window_ms": 300},
    }
    reliability_cfg = {"smoothing_window_frames": 3, "static_gesture_debounce_frames": 1,
                        "default_action_cooldown_seconds": 0.1, "motion_gesture_cooldown_seconds": 0.1}
    engine = GestureEngine(gestures_cfg, reliability_cfg)

    t = 2000.0
    fired = False
    # Move the whole hand from x=0.8 to x=0.2 over ~250ms (fast leftward swipe).
    xs = [0.80, 0.68, 0.56, 0.44, 0.32, 0.20]
    for x in xs:
        hand = {"handedness": "Right", "landmarks": make_hand({})}
        # shift every landmark's x by translating the base hand
        hand["landmarks"] = [(px - (0.50 - x), py, pz) for (px, py, pz) in hand["landmarks"]]
        events = engine.process([hand], now=t)
        if any(e.gesture_name == "swipe_left" and e.state == "start" for e in events):
            fired = True
        t += 0.05
    assert fired, "swipe_left was never detected during a fast leftward hand motion"
    print("PASS: swipe_left detected from leftward hand motion")


def test_cooldown_prevents_immediate_retrigger():
    gestures_cfg = {
        "fist": {"type": "fingers", "hand": "any",
                  "pattern": [False, False, False, False, False], "hold_ms": 0},
    }
    reliability_cfg = {"smoothing_window_frames": 2, "static_gesture_debounce_frames": 1,
                        "default_action_cooldown_seconds": 1.0, "motion_gesture_cooldown_seconds": 1.0}
    engine = GestureEngine(gestures_cfg, reliability_cfg)

    fist = {"handedness": "Right", "landmarks": curled_hand()}
    t = 3000.0
    starts = 0
    # Rapidly open/close the "fist-ness" flag by alternating fist / not-fist,
    # simulating a jittery false re-trigger attempt within the cooldown window.
    for i in range(10):
        events = engine.process([fist], now=t)
        starts += sum(1 for e in events if e.state == "start")
        t += 0.05
        events = engine.process([{"handedness": "Right", "landmarks": make_hand({})}], now=t)
        t += 0.05
    assert starts <= 1, f"cooldown should limit rapid retriggers, but saw {starts} starts"
    print(f"PASS: cooldown limited rapid re-triggers to {starts} start event(s)")


def test_swipe_works_with_realistic_default_smoothing():
    """Regression test: swipes were being run through the same majority-vote
    smoothing as held static gestures. With the actual default smoothing
    window (5 frames) a swipe - true for only 1-2 frames - could never win
    a majority vote and would never fire. This uses the real defaults from
    gesture_settings.json, not the loosened settings the earlier swipe test
    used, specifically to catch that regression."""
    gestures_cfg = {
        "swipe_left": {"type": "swipe", "hand": "any", "direction": "left",
                        "min_velocity": 0.85, "window_ms": 350},
    }
    reliability_cfg = {"smoothing_window_frames": 5, "static_gesture_debounce_frames": 4,
                        "default_action_cooldown_seconds": 0.35, "motion_gesture_cooldown_seconds": 0.5}
    engine = GestureEngine(gestures_cfg, reliability_cfg)

    t = 4000.0
    fired = False
    xs = [0.80, 0.68, 0.56, 0.44, 0.32, 0.20]
    for x in xs:
        hand = {"handedness": "Right", "landmarks": make_hand({})}
        hand["landmarks"] = [(px - (0.50 - x), py, pz) for (px, py, pz) in hand["landmarks"]]
        events = engine.process([hand], now=t)
        if any(e.gesture_name == "swipe_left" and e.state == "start" for e in events):
            fired = True
        t += 0.05
    assert fired, "swipe_left never fired under realistic default smoothing settings"
    print("PASS: swipe_left fires with the real default smoothing/debounce settings")


def test_gesture_state_survives_one_dropped_detection_frame():
    """Regression test: hand identity used to be keyed by MediaPipe's raw
    per-frame handedness label. A single dropped detection frame (very
    common during fast motion / motion blur) wiped all state for that hand
    - including the cooldown that had just been set - letting the same
    gesture misfire again immediately. This checks a started gesture's
    cooldown survives exactly one missed frame."""
    gestures_cfg = {
        "fist": {"type": "fingers", "hand": "any",
                  "pattern": [False, False, False, False, False], "hold_ms": 0},
    }
    reliability_cfg = {"smoothing_window_frames": 3, "static_gesture_debounce_frames": 1,
                        "default_action_cooldown_seconds": 1.0, "motion_gesture_cooldown_seconds": 1.0,
                        "hand_grace_period_seconds": 0.25}
    engine = GestureEngine(gestures_cfg, reliability_cfg)

    fist = {"handedness": "Right", "landmarks": curled_hand()}
    t = 5000.0
    starts = 0

    for _ in range(4):
        events = engine.process([fist], now=t)
        starts += sum(1 for e in events if e.state == "start")
        t += 0.03

    # Simulate one dropped detection frame (hand briefly not seen at all).
    t += 0.03
    engine.process([], now=t)

    # Hand reappears well within the grace period, still fisted.
    t += 0.03
    for _ in range(3):
        events = engine.process([fist], now=t)
        starts += sum(1 for e in events if e.state == "start")
        t += 0.03

    assert starts == 1, f"expected exactly 1 start across the dropped-frame gap, got {starts}"
    print("PASS: gesture cooldown survives a single dropped detection frame")


def test_swipe_motion_survives_one_dropped_frame_mid_swipe():
    """Regression test: motion history used to be wiped the instant a hand
    was absent from even one frame - exactly when a fast swipe is most
    likely to cause a brief tracking dropout. Checks a swipe still
    registers when one frame in the middle of the motion has no detection."""
    gestures_cfg = {
        "swipe_left": {"type": "swipe", "hand": "any", "direction": "left",
                        "min_velocity": 0.85, "window_ms": 350},
    }
    reliability_cfg = {"smoothing_window_frames": 5, "static_gesture_debounce_frames": 4,
                        "default_action_cooldown_seconds": 0.35, "motion_gesture_cooldown_seconds": 0.5,
                        "hand_grace_period_seconds": 0.25}
    engine = GestureEngine(gestures_cfg, reliability_cfg)

    t = 6000.0
    fired = False
    xs = [0.80, 0.68, None, 0.44, 0.32, 0.20]  # None = simulated dropped frame
    for x in xs:
        if x is None:
            engine.process([], now=t)
        else:
            hand = {"handedness": "Right", "landmarks": make_hand({})}
            hand["landmarks"] = [(px - (0.50 - x), py, pz) for (px, py, pz) in hand["landmarks"]]
            events = engine.process([hand], now=t)
            if any(e.gesture_name == "swipe_left" and e.state == "start" for e in events):
                fired = True
        t += 0.05
    assert fired, "swipe_left did not survive a single dropped frame mid-motion"
    print("PASS: swipe motion history survives one dropped frame mid-swipe")


def test_two_hands_tracked_independently_with_separate_state():
    """Two hands must get separate slots, separate gesture state, and
    separate pinch hysteresis. Regression test: pinch hysteresis used to
    be stored on the shared gesture-definition dict, so two hands
    evaluating the same pinch gesture overwrote each other's enter/exit
    state every frame."""
    gestures_cfg = {
        "pinch_any": {"type": "pinch", "hand": "any",
                       "finger_a": "thumb", "finger_b": "index",
                       "enter_threshold": 0.20, "exit_threshold": 0.38, "hold_ms": 0},
    }
    reliability_cfg = {"smoothing_window_frames": 3, "static_gesture_debounce_frames": 1,
                        "default_action_cooldown_seconds": 0.05, "motion_gesture_cooldown_seconds": 0.05,
                        "hand_grace_period_seconds": 0.25, "max_tracked_hands": 2}
    engine = GestureEngine(gestures_cfg, reliability_cfg)

    # Two hands well apart in frame: left one pinching, right one open.
    def shifted(lms, dx):
        return [(x + dx, y, z) for (x, y, z) in lms]

    t = 7000.0
    slot_ids_seen = set()
    for _ in range(5):
        hand_pinching = {"handedness": "Left", "landmarks": shifted(pinch_hand(), -0.25)}
        hand_open = {"handedness": "Right", "landmarks": shifted(make_hand({}), 0.25)}
        events = engine.process([hand_pinching, hand_open], now=t)
        for hd in engine.last_hand_data:
            slot_ids_seen.add(hd["slot_id"])
        t += 0.05

    assert len(slot_ids_seen) == 2, f"expected 2 distinct hand slots, got {slot_ids_seen}"

    # Exactly one of the two hands should be registering the pinch.
    active_slots = {e.hand_label for e in events if e.state in ("start", "hold")}
    assert len(active_slots) == 1, (
        f"exactly one hand should be pinching, but {len(active_slots)} slots "
        f"reported the gesture active - suggests shared state between hands"
    )
    print(f"PASS: two hands tracked as separate slots {sorted(slot_ids_seen)}; "
          f"only the pinching hand triggered the gesture")


def test_hand_filter_restricts_gesture_to_one_hand():
    """A gesture with hand:'Right' must never fire for a left hand - this
    is what makes a two-hand layout (different actions per hand) work."""
    gestures_cfg = {
        "right_only_fist": {"type": "fingers", "hand": "Right",
                             "pattern": [False, False, False, False, False], "hold_ms": 0},
    }
    reliability_cfg = {"smoothing_window_frames": 3, "static_gesture_debounce_frames": 1,
                        "default_action_cooldown_seconds": 0.05, "motion_gesture_cooldown_seconds": 0.05,
                        "hand_grace_period_seconds": 0.25}
    engine = GestureEngine(gestures_cfg, reliability_cfg)

    t = 8000.0
    left_events = []
    for _ in range(5):
        left_fist = {"handedness": "Left", "landmarks": curled_hand()}
        left_events += engine.process([left_fist], now=t)
        t += 0.05
    assert not any(e.gesture_name == "right_only_fist" for e in left_events), \
        "a hand:'Right' gesture must not fire for a Left hand"

    engine.reset()
    right_events = []
    for _ in range(5):
        right_fist = {"handedness": "Right", "landmarks": curled_hand()}
        right_events += engine.process([right_fist], now=t)
        t += 0.05
    assert any(e.gesture_name == "right_only_fist" and e.state == "start" for e in right_events), \
        "a hand:'Right' gesture should fire for a Right hand"
    print("PASS: hand filter correctly restricts a gesture to one specific hand")


def test_bimanual_spread_fires_when_hands_move_apart():
    """Two-hand spread gesture should fire when both hands move apart, not
    when they stay still or move together."""
    gestures_cfg = {
        "spread": {"type": "bimanual", "subtype": "two_hand_distance",
                    "direction": "spreading", "min_delta": 0.005, "hold_ms": 0},
    }
    reliability_cfg = {"smoothing_window_frames": 3, "static_gesture_debounce_frames": 1,
                        "default_action_cooldown_seconds": 0.1, "motion_gesture_cooldown_seconds": 0.1,
                        "hand_grace_period_seconds": 0.25, "max_tracked_hands": 2}
    engine = GestureEngine(gestures_cfg, reliability_cfg)

    t = 9000.0
    fired = False
    # Both hands start close, then spread apart over several frames.
    for i in range(8):
        sep = 0.10 + i * 0.04  # increasing separation
        left  = {"handedness": "Left",  "landmarks": make_hand({8: (0.50 - sep, 0.45)})}
        right = {"handedness": "Right", "landmarks": make_hand({8: (0.50 + sep, 0.45)})}
        events = engine.process([left, right], now=t)
        if any(e.gesture_name == "spread" and e.state == "start" for e in events):
            fired = True
        t += 0.05

    assert fired, "bimanual spread gesture never fired while hands moved apart"

    # Reset and try with hands staying still - should NOT fire.
    engine.reset()
    t += 1.0
    still_fired = False
    for _ in range(8):
        left  = {"handedness": "Left",  "landmarks": make_hand({8: (0.30, 0.45)})}
        right = {"handedness": "Right", "landmarks": make_hand({8: (0.70, 0.45)})}
        events = engine.process([left, right], now=t)
        if any(e.gesture_name == "spread" and e.state == "start" for e in events):
            still_fired = True
        t += 0.05
    assert not still_fired, "bimanual spread incorrectly fired when hands were stationary"
    print("PASS: bimanual spread fires during hand-spread motion, not when stationary")


def test_bimanual_both_open_palms_requires_both_hands():
    """two_hand_any subtype: gesture should fire only when BOTH hands
    satisfy their conditions, not just one."""
    gestures_cfg = {
        "both_open": {
            "type": "bimanual", "subtype": "two_hand_any", "hold_ms": 0,
            "left_condition":  {"type": "fingers", "pattern": [True, True, True, True, True]},
            "right_condition": {"type": "fingers", "pattern": [True, True, True, True, True]},
        },
    }
    reliability_cfg = {"smoothing_window_frames": 2, "static_gesture_debounce_frames": 1,
                        "default_action_cooldown_seconds": 0.1, "motion_gesture_cooldown_seconds": 0.1,
                        "hand_grace_period_seconds": 0.25, "max_tracked_hands": 2}
    engine = GestureEngine(gestures_cfg, reliability_cfg)

    t = 10000.0
    # Only left hand open, right fisted - should NOT fire.
    half_fired = False
    for _ in range(4):
        left  = {"handedness": "Left",  "landmarks": make_hand({})}       # open
        right = {"handedness": "Right", "landmarks": curled_hand()}         # fist
        events = engine.process([left, right], now=t)
        if any(e.gesture_name == "both_open" for e in events):
            half_fired = True
        t += 0.05
    assert not half_fired, "both_open fired with only one hand open"

    # Both open - SHOULD fire.
    engine.reset()
    t += 1.0
    both_fired = False
    for _ in range(4):
        left  = {"handedness": "Left",  "landmarks": make_hand({})}
        right = {"handedness": "Right", "landmarks": make_hand({})}
        events = engine.process([left, right], now=t)
        if any(e.gesture_name == "both_open" and e.state in ("start", "hold") for e in events):
            both_fired = True
        t += 0.05
    assert both_fired, "both_open never fired when both hands were open"
    print("PASS: two_hand_any requires both hands to satisfy their conditions")


if __name__ == "__main__":
    test_finger_states_open_palm()
    test_finger_states_fist()
    test_pinch_distance_shrinks()
    test_open_palm_gesture_start_hold_end_with_hold_ms()
    test_swipe_left_detected_from_motion()
    test_cooldown_prevents_immediate_retrigger()
    test_swipe_works_with_realistic_default_smoothing()
    test_gesture_state_survives_one_dropped_detection_frame()
    test_swipe_motion_survives_one_dropped_frame_mid_swipe()
    test_two_hands_tracked_independently_with_separate_state()
    test_hand_filter_restricts_gesture_to_one_hand()
    test_bimanual_spread_fires_when_hands_move_apart()
    test_bimanual_both_open_palms_requires_both_hands()
    print("\nAll gesture_engine tests passed.")
