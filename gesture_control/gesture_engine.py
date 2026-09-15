"""
Turns raw MediaPipe hand landmarks into named, debounced gesture events.

Landmark indices (standard MediaPipe hand model, 21 points per hand):
    0 wrist
    1-4   thumb  (4 = tip)
    5-8   index  (8 = tip)
    9-12  middle (12 = tip)
    13-16 ring   (16 = tip)
    17-20 pinky  (20 = tip)

Pipeline per hand, per frame:
    1. Match this frame's detections to a stable internal hand "slot" by
       nearest palm-center position, independent of MediaPipe's own
       per-frame Left/Right classification (see _HandSlot below for why).
    2. Compute finger-up states, pinch distances, palm center/orientation.
    3. Evaluate every configured gesture definition against this frame's
       raw hand data -> raw boolean match.
    4. STATIC gestures (fingers/pinch) are smoothed with a short
       majority-vote window (kills single-frame flicker) and require
       hold_ms of continuous match before firing "start". MOTION gestures
       (swipe) are evaluated directly on the raw edge instead - see
       _update_track for why they can't share the same smoothing path.
    5. A per-gesture cooldown after each "start" stops a jittery hand
       refiring immediately.
    6. Pinch-type gestures use separate enter/exit thresholds (hysteresis)
       so the boundary itself doesn't flicker.

Emits GestureEvent(gesture_name, hand_label, state, hand_data) each frame,
where state is "start" (just became active), "hold" (still active,
continuous gestures use this every frame), or "end" (just became inactive).
`hand_label` is the stable internal slot id (e.g. "hand_0"), not a raw
MediaPipe handedness string - see _HandSlot.
"""

import collections
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from gesture_control.one_euro_filter import OneEuroFilter2D

THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP = 4, 8, 12, 16, 20
THUMB_IP = 3
INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP = 6, 10, 14, 18
WRIST = 0
INDEX_MCP, MIDDLE_MCP, PINKY_MCP = 5, 9, 17

FINGER_TIPS = {"thumb": THUMB_TIP, "index": INDEX_TIP, "middle": MIDDLE_TIP,
                "ring": RING_TIP, "pinky": PINKY_TIP}


@dataclass
class GestureEvent:
    gesture_name: str
    hand_label: str
    state: str  # "start" | "hold" | "end"
    hand_data: Dict[str, Any]


# --------------------------------------------------------------------------
# Per-frame geometric feature extraction (pure functions, easy to unit test)
# --------------------------------------------------------------------------

def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def hand_scale(landmarks) -> float:
    """A stable reference length (wrist to middle-MCP) so pinch/other
    distances can be compared independent of hand size or camera distance."""
    return max(1e-6, _dist(landmarks[WRIST], landmarks[MIDDLE_MCP]))


def finger_states(landmarks) -> List[bool]:
    """Returns [thumb, index, middle, ring, pinky] up/down booleans.

    Index/middle/ring/pinky: up if the tip is further from the wrist than
    the pip joint is (works regardless of hand rotation, unlike a raw
    y-coordinate comparison which breaks if the hand is sideways/upside down).
    Thumb: up if the tip is further from the pinky-MCP than the thumb-IP
    joint is - i.e. the thumb is splayed away from the palm.
    """
    scale = hand_scale(landmarks)
    wrist = landmarks[WRIST]

    def extended(tip_idx, pip_idx):
        tip_d = _dist(landmarks[tip_idx], wrist)
        pip_d = _dist(landmarks[pip_idx], wrist)
        return (tip_d - pip_d) / scale > 0.10

    thumb_tip_d = _dist(landmarks[THUMB_TIP], landmarks[PINKY_MCP])
    thumb_ip_d = _dist(landmarks[THUMB_IP], landmarks[PINKY_MCP])
    thumb_up = (thumb_tip_d - thumb_ip_d) / scale > 0.05

    return [
        thumb_up,
        extended(INDEX_TIP, INDEX_PIP),
        extended(MIDDLE_TIP, MIDDLE_PIP),
        extended(RING_TIP, RING_PIP),
        extended(PINKY_TIP, PINKY_PIP),
    ]


def pinch_distance_normalized(landmarks, finger_a: str, finger_b: str) -> float:
    a = landmarks[FINGER_TIPS[finger_a]]
    b = landmarks[FINGER_TIPS[finger_b]]
    return _dist(a, b) / hand_scale(landmarks)


def palm_center(landmarks) -> Tuple[float, float]:
    xs = [landmarks[i][0] for i in (WRIST, INDEX_MCP, MIDDLE_MCP, PINKY_MCP)]
    ys = [landmarks[i][1] for i in (WRIST, INDEX_MCP, MIDDLE_MCP, PINKY_MCP)]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def palm_facing_camera(landmarks, handedness: str) -> bool:
    """Cross product of two in-palm vectors gives a rough palm normal;
    its sign tells us whether the palm or the back of the hand faces the
    camera. Mirrors flip handedness, so we fold that in too."""
    wrist = landmarks[WRIST]
    v1 = (landmarks[INDEX_MCP][0] - wrist[0], landmarks[INDEX_MCP][1] - wrist[1])
    v2 = (landmarks[PINKY_MCP][0] - wrist[0], landmarks[PINKY_MCP][1] - wrist[1])
    cross_z = v1[0] * v2[1] - v1[1] * v2[0]
    facing = cross_z < 0
    if handedness == "Left":
        facing = not facing
    return facing


# --------------------------------------------------------------------------
# Per-(hand, gesture) temporal state
# --------------------------------------------------------------------------

@dataclass
class _GestureTrack:
    raw_history: collections.deque = field(default_factory=lambda: collections.deque(maxlen=8))
    smoothed_active: bool = False
    active_since: Optional[float] = None
    started: bool = False
    cooldown_until: float = 0.0


class _HandMotionTrack:
    """Position history for one physical hand, used for swipe detection."""

    def __init__(self, maxlen=20):
        self.history = collections.deque(maxlen=maxlen)  # (t, x, y)

    def push(self, t, x, y):
        self.history.append((t, x, y))

    def displacement_since(self, window_s: float):
        if not self.history:
            return None
        now_t, now_x, now_y = self.history[-1]
        for t, x, y in reversed(self.history):
            if now_t - t >= window_s:
                dt = now_t - t
                if dt <= 0:
                    return None
                return (now_x - x, now_y - y, dt)
        # window longer than our whole buffer - use the oldest sample we have
        t0, x0, y0 = self.history[0]
        dt = now_t - t0
        if dt <= 0:
            return None
        return (now_x - x0, now_y - y0, dt)


class _HandSlot:
    """A stable identity for one physical hand across frames.

    MediaPipe re-classifies Left/Right independently every single frame,
    and that classification is noisy - it can flip for the same physical
    hand between consecutive frames, especially with only one hand in
    frame or an ambiguous angle. Earlier this engine used that raw label
    as the identity key for all per-hand state (gesture hold timers,
    cooldowns, motion history for swipes). Every time the label flickered,
    all of that state was wiped and restarted from scratch - which both
    broke swipe detection (motion history reset mid-swipe) and caused
    spurious re-clicks (a cooldown that had just been set got wiped,
    letting the same gesture fire again immediately).

    Instead, hands are now matched frame-to-frame by nearest palm-center
    position (see GestureEngine._assign_slots), and a slot survives a
    brief gap in detection (`grace_period_seconds`) rather than being
    torn down the instant a single frame misses it - a dropped detection
    frame during fast motion (exactly when a swipe happens, or exactly
    when motion blur is worst) is normal and shouldn't reset anything.
    """

    def __init__(self):
        self.center: Optional[Tuple[float, float]] = None
        self.handedness: str = "Right"
        self.last_seen: float = 0.0
        self.motion = _HandMotionTrack()
        self.point_filter: Optional[OneEuroFilter2D] = None


@dataclass
class _GestureTrack:
    raw_history: collections.deque = field(default_factory=lambda: collections.deque(maxlen=8))
    smoothed_active: bool = False
    active_since: Optional[float] = None
    started: bool = False
    cooldown_until: float = 0.0
    last_raw: bool = False  # used for motion-type edge detection


class GestureEngine:
    def __init__(self, gestures_cfg: dict, reliability_cfg: dict):
        self.gestures_cfg = gestures_cfg
        self.reliability_cfg = reliability_cfg
        self.smoothing_window = reliability_cfg.get("smoothing_window_frames", 5)
        self.default_cooldown = reliability_cfg.get("default_action_cooldown_seconds", 0.35)
        self.motion_cooldown = reliability_cfg.get("motion_gesture_cooldown_seconds", 0.5)
        self.debounce_frames = reliability_cfg.get("static_gesture_debounce_frames", 3)
        self.grace_period = reliability_cfg.get("hand_grace_period_seconds", 0.25)
        self.max_slots = reliability_cfg.get("max_tracked_hands", 2)
        self.max_match_distance = reliability_cfg.get("hand_match_max_distance", 0.35)

        # One Euro Filter params for cursor/scroll/drag point smoothing.
        # See one_euro_filter.py for why this replaced a plain fixed-alpha
        # EMA: min_cutoff sets baseline smoothing at rest, beta controls
        # how much lag gets traded away as the hand speeds up.
        self.one_euro_min_cutoff = reliability_cfg.get("one_euro_min_cutoff", 0.8)
        self.one_euro_beta = reliability_cfg.get("one_euro_beta", 0.9)
        self.one_euro_dcutoff = reliability_cfg.get("one_euro_dcutoff", 1.0)

        self._tracks: Dict[Tuple[str, str], _GestureTrack] = {}
        self._slots: Dict[str, _HandSlot] = {}
        self.last_hand_data: List[dict] = []  # resolved per-hand data from the most recent process() call

    def reset(self):
        self._tracks.clear()
        self._slots.clear()

    def process(self, hands: List[dict], now: Optional[float] = None) -> List[GestureEvent]:
        """hands: output of HandTracker.update() - list of
        {"handedness": ..., "landmarks": [(x,y,z), ...]}.
        Returns this frame's GestureEvents."""
        now = now if now is not None else time.time()
        events: List[GestureEvent] = []

        detections = []
        for hand in hands:
            lms = hand["landmarks"]
            cx, cy = palm_center(lms)
            detections.append({"handedness": hand["handedness"], "landmarks": lms, "center": (cx, cy)})

        assignment = self._assign_slots(detections)
        seen_slot_ids = set()
        this_frame_hand_data = []

        for di, det in enumerate(detections):
            sid = assignment[di]
            seen_slot_ids.add(sid)
            slot = self._slots.setdefault(sid, _HandSlot())
            slot.center = det["center"]
            slot.handedness = det["handedness"]
            slot.last_seen = now
            slot.motion.push(now, *det["center"])

            lms = det["landmarks"]
            fingers = finger_states(lms)
            index_tip = lms[INDEX_TIP]
            raw_point = (index_tip[0], index_tip[1])

            if slot.point_filter is None:
                slot.point_filter = OneEuroFilter2D(
                    min_cutoff=self.one_euro_min_cutoff,
                    beta=self.one_euro_beta,
                    d_cutoff=self.one_euro_dcutoff,
                )
            smoothed_point = slot.point_filter.filter(now, raw_point)

            hand_data = {
                "handedness": slot.handedness,
                "landmarks": lms,
                "fingers": fingers,
                "palm_center": det["center"],
                "point": smoothed_point,  # One-Euro-filtered - use for cursor/scroll
                "point_raw": raw_point,   # unsmoothed - use where lag would hurt
                "slot_id": sid,
            }
            this_frame_hand_data.append(hand_data)

            for gname, gdef in self.gestures_cfg.items():
                if not self._hand_matches_filter(slot.handedness, gdef.get("hand", "any")):
                    continue
                raw = self._evaluate_raw(gdef, lms, fingers, slot.motion, now)
                ev = self._update_track(gname, sid, gdef, raw, hand_data, now)
                if ev is not None:
                    events.append(ev)

        # Only evict a slot (closing any active gesture + wiping its motion/
        # smoothing history) once it's been genuinely absent for longer than
        # the grace period - not the instant one frame misses it.
        for sid in list(self._slots.keys()):
            if sid in seen_slot_ids:
                continue
            slot = self._slots[sid]
            if now - slot.last_seen > self.grace_period:
                keys_for_slot = [k for k in self._tracks if k[1] == sid]
                for key in keys_for_slot:
                    track = self._tracks[key]
                    if track.started:
                        gname, _ = key
                        events.append(GestureEvent(gname, sid, "end", {"handedness": slot.handedness}))
                    del self._tracks[key]
                del self._slots[sid]

        self.last_hand_data = this_frame_hand_data
        return events

    # -- hand identity matching ------------------------------------------

    def _assign_slots(self, detections: List[dict]) -> Dict[int, str]:
        """Greedy nearest-neighbor match of this frame's detections to
        existing slots by palm-center distance, so identity survives
        MediaPipe's per-frame Left/Right flicker (see _HandSlot)."""
        pairs = []
        for di, det in enumerate(detections):
            for sid, slot in self._slots.items():
                if slot.center is None:
                    continue
                d = _dist(det["center"], slot.center)
                if d <= self.max_match_distance:
                    pairs.append((d, di, sid))
        pairs.sort(key=lambda p: p[0])

        assignment: Dict[int, str] = {}
        used_slots = set()
        for d, di, sid in pairs:
            if di in assignment or sid in used_slots:
                continue
            assignment[di] = sid
            used_slots.add(sid)

        for di in range(len(detections)):
            if di in assignment:
                continue
            assignment[di] = self._alloc_slot_id(used_slots)
            used_slots.add(assignment[di])

        return assignment

    def _alloc_slot_id(self, already_used_this_frame: set) -> str:
        for i in range(self.max_slots):
            sid = f"hand_{i}"
            if sid not in self._slots and sid not in already_used_this_frame:
                return sid
        # Every slot id is taken (more simultaneous hands than max_slots,
        # or stale slots haven't expired yet) - reuse the least-recently-seen
        # one rather than growing unboundedly.
        candidates = [sid for sid in self._slots if sid not in already_used_this_frame]
        if candidates:
            return min(candidates, key=lambda sid: self._slots[sid].last_seen)
        return f"hand_{len(self._slots)}"

    # -- gesture matching -------------------------------------------------

    @staticmethod
    def _hand_matches_filter(label: str, wanted: str) -> bool:
        if wanted in ("any", None):
            return True
        return label.lower() == wanted.lower()

    def _evaluate_raw(self, gdef, lms, fingers, motion: _HandMotionTrack, now: float) -> bool:
        gtype = gdef.get("type")
        if gtype == "fingers":
            pattern = gdef.get("pattern", [None] * 5)
            return all(want is None or want == have for want, have in zip(pattern, fingers))

        if gtype == "pinch":
            dist = pinch_distance_normalized(lms, gdef.get("finger_a", "thumb"), gdef.get("finger_b", "index"))
            # hysteresis state is stored on the gesture def dict itself (cheap, fine here)
            is_on = gdef.get("_pinch_is_on", False)
            if is_on:
                is_on = dist < gdef.get("exit_threshold", 0.40)
            else:
                is_on = dist < gdef.get("enter_threshold", 0.20)
            gdef["_pinch_is_on"] = is_on
            return is_on

        if gtype == "swipe":
            window_s = gdef.get("window_ms", 350) / 1000.0
            disp = motion.displacement_since(window_s)
            if disp is None:
                return False
            dx, dy, dt = disp
            if dt <= 0:
                return False
            # Require real, sustained travel - not just a high velocity
            # reading over a tiny, jitter-scale distance (frame-to-frame
            # tracking noise can produce deceptively high instantaneous
            # "velocity" over almost no actual movement).
            min_displacement = gdef.get("min_displacement", 0.15)
            if math.hypot(dx, dy) < min_displacement:
                return False
            speed_x, speed_y = dx / dt, dy / dt
            direction = gdef.get("direction")
            min_v = gdef.get("min_velocity", 0.85)
            if direction == "left":
                return -speed_x > min_v and abs(speed_x) > abs(speed_y)
            if direction == "right":
                return speed_x > min_v and abs(speed_x) > abs(speed_y)
            if direction == "up":
                return -speed_y > min_v and abs(speed_y) > abs(speed_x)
            if direction == "down":
                return speed_y > min_v and abs(speed_y) > abs(speed_x)
            return False

        return False

    def _update_track(self, gname, sid, gdef, raw: bool, hand_data: dict, now: float) -> Optional[GestureEvent]:
        key = (gname, sid)
        track = self._tracks.setdefault(key, _GestureTrack(
            raw_history=collections.deque(maxlen=max(2, self.smoothing_window))
        ))
        is_motion_type = gdef.get("type") == "swipe"

        if is_motion_type:
            # A swipe is a brief, one-off velocity spike - true for maybe
            # 1-2 frames total. Running it through the same majority-vote
            # window used for held static gestures (default: needs a
            # majority of the last 5 frames to agree) means it would
            # almost never accumulate enough "true" frames to register at
            # all. The velocity calculation over window_ms is already the
            # robustness mechanism for motion gestures; this just needs a
            # clean rising-edge trigger + cooldown on top of that.
            fired = raw and not track.last_raw and now >= track.cooldown_until
            track.last_raw = raw
            if fired:
                track.cooldown_until = now + self.motion_cooldown
                return GestureEvent(gname, sid, "start", hand_data)
            return None

        track.raw_history.append(raw)
        votes_true = sum(1 for v in track.raw_history if v)
        smoothed = votes_true * 2 > len(track.raw_history)  # majority vote
        hold_ms = gdef.get("hold_ms", 0)

        if smoothed and not track.smoothed_active:
            track.smoothed_active = True
            track.active_since = now
        elif not smoothed and track.smoothed_active:
            track.smoothed_active = False
            track.active_since = None
            if track.started:
                track.started = False
                return GestureEvent(gname, sid, "end", hand_data)
            return None

        if track.smoothed_active:
            elapsed_ms = (now - track.active_since) * 1000.0 if track.active_since else 0.0
            held_long_enough = elapsed_ms >= hold_ms
            enough_frames = len(track.raw_history) >= min(self.debounce_frames, track.raw_history.maxlen)

            if not track.started and held_long_enough and enough_frames and now >= track.cooldown_until:
                track.started = True
                track.cooldown_until = now + self.default_cooldown
                return GestureEvent(gname, sid, "start", hand_data)

            if track.started:
                return GestureEvent(gname, sid, "hold", hand_data)

        return None
