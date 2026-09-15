"""
Hand landmark tracking, wrapping MediaPipe.

Honest note on GPU/NPU (see README_GESTURE.md for the long version):
MediaPipe's Python "Tasks" API accepts a GPU delegate, but on Windows this
is unreliable - it frequently fails to initialize because Windows Python
wheels aren't built with the OpenGL/EGL backend the delegate needs. There is
no NPU delegate at all in MediaPipe's Python API. So this module: tries GPU
via the Tasks API if configured, falls back to CPU via the same API on any
failure, and falls back further to the older `mediapipe.solutions.hands`
API (always CPU) if the Tasks API or its model file are unavailable. CPU
hand tracking alone is still fast enough for real-time gesture control on
essentially any modern laptop.

Output format (backend-independent), one entry per detected hand:
    {
        "handedness": "Left" | "Right",
        "landmarks": [(x, y, z), ...]   # 21 points, x/y normalized 0..1, z relative depth
    }
"""

import os
import time
import urllib.request

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "hand_landmarker.task")


def _ensure_model_downloaded():
    if os.path.exists(MODEL_PATH):
        return True
    os.makedirs(MODEL_DIR, exist_ok=True)
    try:
        print(f"[HANDS] Downloading hand landmark model (first run only) from {MODEL_URL} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[HANDS] Model downloaded.")
        return True
    except Exception as e:
        print(f"[HANDS] Could not download hand landmark model ({e}). "
              f"Falling back to the legacy CPU hand-tracking backend.")
        return False


class HandTracker:
    """Facade over whichever backend actually works. Call update(frame_rgb)
    every frame; returns the list of hand dicts described in the module
    docstring."""

    def __init__(self, cfg_hands: dict):
        self.cfg = cfg_hands
        self.backend_name = None
        self._impl = None
        self._init_backend()

    # -- backend selection -------------------------------------------------

    def _init_backend(self):
        requested = self.cfg.get("backend", "tasks_gpu_then_cpu")

        if requested in ("tasks_gpu_then_cpu", "tasks_cpu"):
            want_gpu = requested == "tasks_gpu_then_cpu"
            if self._try_tasks_backend(use_gpu=want_gpu):
                return
            if want_gpu and self._try_tasks_backend(use_gpu=False):
                return

        # last resort: legacy solutions API, CPU only, no model download needed
        if self._init_legacy_backend():
            return

        raise RuntimeError(
            "Could not initialize any hand-tracking backend. This usually means either: "
            "(1) the Tasks API model download failed - check internet access to "
            f"{MODEL_URL}, or (2) your installed mediapipe version has dropped the legacy "
            "'mediapipe.solutions' API (Google removed it in some recent releases) - install "
            "the pinned version from requirements-gesture.txt (mediapipe==0.10.14) which still "
            "has it, or just fix internet access so the Tasks-API model can download instead."
        )

    def _try_tasks_backend(self, use_gpu: bool) -> bool:
        if not _ensure_model_downloaded():
            return False
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            delegate = (mp_python.BaseOptions.Delegate.GPU if use_gpu
                        else mp_python.BaseOptions.Delegate.CPU)
            base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH, delegate=delegate)
            options = mp_vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=self.cfg.get("max_num_hands", 2),
                min_hand_detection_confidence=self.cfg.get("min_detection_confidence", 0.6),
                min_tracking_confidence=self.cfg.get("min_tracking_confidence", 0.5),
                running_mode=mp_vision.RunningMode.VIDEO,
            )
            landmarker = mp_vision.HandLandmarker.create_from_options(options)

            # Cheap smoke test so a broken GPU delegate fails now, not on frame 1.
            self._impl = _TasksBackend(landmarker, mp)
            self.backend_name = f"mediapipe-tasks ({'GPU' if use_gpu else 'CPU'})"
            print(f"[HANDS] Using backend: {self.backend_name}")
            return True
        except Exception as e:
            which = "GPU" if use_gpu else "CPU"
            print(f"[HANDS] Tasks API ({which}) init failed, will fall back. Reason: {e}")
            return False

    def _init_legacy_backend(self) -> bool:
        try:
            import mediapipe as mp
            hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=self.cfg.get("max_num_hands", 2),
                model_complexity=self.cfg.get("model_complexity", 1),
                min_detection_confidence=self.cfg.get("min_detection_confidence", 0.6),
                min_tracking_confidence=self.cfg.get("min_tracking_confidence", 0.5),
            )
        except (AttributeError, ModuleNotFoundError, ImportError) as e:
            print(f"[HANDS] Legacy solutions API unavailable in this mediapipe install ({e}).")
            return False

        self._impl = _LegacyBackend(hands)
        self.backend_name = "mediapipe-legacy-solutions (CPU)"
        print(f"[HANDS] Using backend: {self.backend_name}")
        return True

    # -- per-frame API -------------------------------------------------

    def update(self, frame_rgb):
        return self._impl.process(frame_rgb)

    def close(self):
        if self._impl is not None:
            self._impl.close()


class _TasksBackend:
    def __init__(self, landmarker, mp):
        self.landmarker = landmarker
        self.mp = mp
        self._start_ts = time.time()

    def process(self, frame_rgb):
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=frame_rgb)
        ts_ms = int((time.time() - self._start_ts) * 1000)
        result = self.landmarker.detect_for_video(image, ts_ms)

        hands_out = []
        if result.hand_landmarks:
            for i, hand_lms in enumerate(result.hand_landmarks):
                label = "Right"
                if result.handedness and len(result.handedness) > i:
                    label = result.handedness[i][0].category_name
                pts = [(lm.x, lm.y, lm.z) for lm in hand_lms]
                hands_out.append({"handedness": label, "landmarks": pts})
        return hands_out

    def close(self):
        try:
            self.landmarker.close()
        except Exception:
            pass


class _LegacyBackend:
    def __init__(self, hands):
        self.hands = hands

    def process(self, frame_rgb):
        result = self.hands.process(frame_rgb)
        hands_out = []
        if result.multi_hand_landmarks:
            for i, hand_lms in enumerate(result.multi_hand_landmarks):
                label = "Right"
                if result.multi_handedness and len(result.multi_handedness) > i:
                    label = result.multi_handedness[i].classification[0].label
                pts = [(lm.x, lm.y, lm.z) for lm in hand_lms.landmark]
                hands_out.append({"handedness": label, "landmarks": pts})
        return hands_out

    def close(self):
        try:
            self.hands.close()
        except Exception:
            pass
