"""Draws the calibration/status HUD onto the preview frame: auth status,
FPS, detected gestures per hand, last action fired, and pause state."""

import cv2

GREEN = (0, 200, 0)
RED = (0, 0, 255)
YELLOW = (0, 220, 220)
WHITE = (240, 240, 240)
GRAY = (170, 170, 170)
DOT_COLOR = (0, 0, 255)
LINE_COLOR = (255, 255, 255)
BOX_COLOR = (0, 255, 255)

# Standard 21-point MediaPipe hand skeleton connections (matches the
# official mediapipe.solutions.hands.HAND_CONNECTIONS set) - hardcoded here
# since main_gesture.py uses the Tasks API / legacy-fallback chain and
# either way we want this drawing code to work regardless of which backend
# ended up being used.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                  # palm base
]


def draw_hand_skeleton(frame, hands, per_hand_active: dict = None):
    """Draws the full 21-point landmark skeleton + a bounding box + a
    handedness/gesture label for every detected hand, the same way the
    MediaPipe demo apps do. `hands` should be GestureEngine.last_hand_data
    (each entry has "landmarks", "slot_id", "handedness") rather than the
    raw HandTracker output, so the displayed active-gesture label matches
    the same stable per-hand identity the control logic uses - not
    MediaPipe's raw per-frame Left/Right label, which can flicker.
    `per_hand_active` optionally maps slot_id -> list of currently-active
    gesture names, shown next to that hand's box for live calibration."""
    per_hand_active = per_hand_active or {}
    h, w = frame.shape[:2]

    for hand in hands:
        label = hand["handedness"]
        slot_id = hand.get("slot_id", label)
        lms = hand["landmarks"]
        pts = [(int(x * w), int(y * h)) for (x, y, _z) in lms]

        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], LINE_COLOR, 2)
        for (px, py) in pts:
            cv2.circle(frame, (px, py), 5, DOT_COLOR, -1)
            cv2.circle(frame, (px, py), 5, (0, 0, 0), 1)  # thin outline, easier to see on light backgrounds

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        pad = 20
        x1, y1 = max(0, min(xs) - pad), max(0, min(ys) - pad)
        x2, y2 = min(w - 1, max(xs) + pad), min(h - 1, max(ys) + pad)
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)

        active = per_hand_active.get(slot_id, [])
        gesture_text = active[0] if active else "-"
        cv2.putText(frame, f"{label}: {gesture_text}", (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, BOX_COLOR, 2)

    return frame


def draw_overlay(frame, *, auth_status, fps, active_gestures, last_action,
                  paused: bool, calibrate_mode: bool, owner_name: str):
    h, w = frame.shape[:2]
    y = 30

    if auth_status.locked:
        cv2.putText(frame, "LOCKED", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, RED, 2)
    else:
        label = f"UNLOCKED ({owner_name})"
        cv2.putText(frame, label, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, GREEN, 2)
    y += 28

    cv2.putText(frame, f"auth: {auth_status.reason}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, GRAY, 1)
    y += 24

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)
    y += 26

    if paused:
        cv2.putText(frame, "GESTURES PAUSED (fist to resume)", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 2)
        y += 26

    if calibrate_mode:
        cv2.putText(frame, "CALIBRATION MODE - actions are NOT executed", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, YELLOW, 2)
        y += 24

    if active_gestures:
        cv2.putText(frame, "gestures:", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)
        y += 20
        for g in active_gestures:
            cv2.putText(frame, f"  {g}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREEN, 1)
            y += 20

    if last_action:
        cv2.putText(frame, f"last action: {last_action}", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)

    cv2.putText(frame, "ESC = emergency stop  |  q = quit  |  r = reload config  |  p = pause",
                (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GRAY, 1)

    return frame
