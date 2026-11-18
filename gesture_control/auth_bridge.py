"""
Wraps the EXISTING face_id.py (InsightFace detection + recognition) with a
LOCKED/UNLOCKED state machine for gesture control. face_id.py itself is
untouched - this module only calls its public functions:
    face_id.get_face_app(), face_id.detect_faces(), face_id.best_match_similarity()

State machine:
    Owner recognized (N consecutive frames), regardless of who else is
        also in frame                                 -> UNLOCKED
    Owner NOT among the detected faces (M consecutive
        frames) - whether that's because a stranger is
        in frame instead, or no face at all            -> LOCKED
    Starts LOCKED by default.

The gate is "is the owner confirmed present", not "is anyone else also
present" - other people being visible in the background doesn't block
control as long as the owner is also there and recognized. Set
`authentication.strangers_allowed: false` to go back to the stricter
"any unrecognized face anywhere in frame locks control, even with the
owner also present" behavior instead.

Using separate "frames to unlock" vs "frames/timeout to lock" thresholds is
deliberate hysteresis: locking should be quick (safety-first - don't keep
control open a moment longer than necessary), unlocking should require a
sustained, confident look at the owner (avoid a single lucky frame handing
control to the wrong person).
"""

import time
from dataclasses import dataclass


@dataclass
class AuthStatus:
    locked: bool
    owner_similarity: float  # best similarity seen this frame, -1 if no face
    reason: str              # human-readable, for the calibration overlay


class AuthBridge:
    def __init__(self, face_id_module, owner_embeddings, auth_cfg: dict,
                 recognition_threshold: float, bypass_auth: bool = False):
        self.face_id = face_id_module
        self.owner_embeddings = owner_embeddings
        self.cfg = auth_cfg
        self.recognition_threshold = recognition_threshold
        self.strangers_allowed = auth_cfg.get("strangers_allowed", True)
        self.bypass_auth = bypass_auth

        self.owner_streak = 0
        self.unknown_streak = 0
        self.last_face_seen_time = 0.0
        self.locked = not bypass_auth
        self._frame_counter = 0
        self._last_status = (
            AuthStatus(False, 1.0, "auth disabled")
            if bypass_auth
            else AuthStatus(True, -1.0, "starting up (looking for face)")
        )

    def update(self, frame_bgr, now: float = None) -> AuthStatus:
        if self.bypass_auth:
            return self._last_status
        now = now if now is not None else time.time()
        self._frame_counter += 1

        check_every = max(1, self.cfg.get("auth_check_every_n_frames", 2))
        if self._frame_counter % check_every != 0:
            return self._last_status  # reuse last verdict on skipped frames

        detections = self.face_id.detect_faces(frame_bgr)

        if not detections:
            self.owner_streak = 0
            self.unknown_streak = 0
            timeout = self.cfg.get("owner_lost_timeout_seconds", 0.8)
            if self.owner_embeddings is None or (now - self.last_face_seen_time > timeout):
                self.locked = True
            status = AuthStatus(self.locked, -1.0, "no face detected - LOCKED")
            self._last_status = status
            return status

        self.last_face_seen_time = now

        # Case 1: No owner enrolled yet -> Face presence mode (requires any face detected)
        if self.owner_embeddings is None or len(self.owner_embeddings) == 0:
            self.owner_streak += 1
            if self.owner_streak >= 2:
                self.locked = False
                reason = f"face detected ({len(detections)}) - UNLOCKED"
            else:
                reason = "face detected, confirming..."
            status = AuthStatus(self.locked, 1.0, reason)
            self._last_status = status
            return status

        # Case 2: Owner enrolled -> Strict biometric owner recognition
        best_sim = -1.0
        any_owner = False
        any_unknown = False
        other_face_count = 0

        for det in detections:
            sim = self.face_id.best_match_similarity(det["embedding"], self.owner_embeddings)
            best_sim = max(best_sim, sim)
            if sim >= self.recognition_threshold:
                any_owner = True
            else:
                any_unknown = True
                other_face_count += 1

        if self.strangers_allowed:
            # Gate purely on "is the owner confirmed present"
            if any_owner:
                self.owner_streak += 1
                self.unknown_streak = 0
            else:
                self.owner_streak = 0
                self.unknown_streak = self.unknown_streak + 1 if any_unknown else 0
        else:
            # Stricter: ANY unrecognized face in frame locks control
            if any_unknown:
                self.unknown_streak += 1
                self.owner_streak = 0
            else:
                self.unknown_streak = 0
            if any_owner and not any_unknown:
                self.owner_streak += 1
            elif not any_owner:
                self.owner_streak = 0

        bystander_note = f", {other_face_count} other face(s)" if (any_owner and other_face_count) else ""

        if self.unknown_streak >= self.cfg.get("consecutive_unknown_frames_to_lock", 3):
            self.locked = True
            reason = f"face not recognized (sim={best_sim:.2f}) - LOCKED"
        elif self.owner_streak >= self.cfg.get("consecutive_owner_frames_to_unlock", 4):
            self.locked = False
            reason = f"owner confirmed ({self.owner_streak} frames, sim={best_sim:.2f}){bystander_note}"
        else:
            reason = f"confirming owner (sim={best_sim:.2f}){bystander_note}..."

        status = AuthStatus(self.locked, best_sim, reason)
        self._last_status = status
        return status

    def force_lock(self, reason: str = "manual/emergency lock"):
        self.locked = True
        self.owner_streak = 0
        self._last_status = AuthStatus(True, self._last_status.owner_similarity, reason)
