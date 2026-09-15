"""
Owner-gated hand-gesture laptop controller.

Reuses, unmodified, from the existing ai_security_camera project:
    - main.open_camera()              (camera open/retry/warm-up logic)
    - face_id.get_face_app()/detect_faces()/best_match_similarity()/load_owner_embeddings()
    - config.py                       (camera + recognition settings, unless overridden)

Run:
    python main_gesture.py                 # normal operation
    python main_gesture.py --calibrate      # shows gestures/auth/FPS, does NOT execute actions
    python main_gesture.py --config path.json   # use an alternate gesture config file

Controls while running:
    ESC = emergency stop (locks control, releases any held mouse/keys, exits)
    q   = quit normally
    r   = hot-reload gesture_settings.json without restarting
    p   = manually toggle gesture pause (same as the default 'fist' gesture)
"""

import argparse
import sys
import threading
import time

import cv2

# ---- existing project modules, imported (not modified) ----
import config as base_config
import face_id
import main as base_main  # reuse open_camera(); guarded by __main__ check, safe to import

# ---- new gesture-control modules ----
from gesture_control import config_loader
from gesture_control.action_engine import ACTIONS, release_all_inputs
from gesture_control.auth_bridge import AuthBridge
from gesture_control.gesture_engine import GestureEngine
from gesture_control.hand_tracker import HandTracker
from gesture_control.overlay import draw_overlay, draw_hand_skeleton

_emergency_stop = threading.Event()


def _try_start_global_esc_listener():
    """Best-effort global ESC hook via pynput, so emergency stop works even
    if the OpenCV preview window isn't focused. If pynput/permissions
    aren't available, we silently rely on the in-window ESC handling
    instead (still fully functional, just requires window focus)."""
    try:
        from pynput import keyboard

        def on_press(key):
            if key == keyboard.Key.esc:
                _emergency_stop.set()

        listener = keyboard.Listener(on_press=on_press, daemon=True)
        listener.start()
        return listener
    except Exception as e:
        print(f"[WARN] Global ESC listener unavailable ({e}). "
              f"ESC will still work while the preview window is focused.")
        return None


def _downscale_for_tracking(frame, target_width: int):
    """Hand landmark inference cost scales with pixel count, and the
    existing project's camera defaults (e.g. 1920x1080, meant for
    face-recognition accuracy at a distance) are far more pixels than
    MediaPipe needs for hand tracking. Feeding it a smaller copy is the
    single biggest lever for reducing per-frame latency and jitter; the
    original full-res `frame` is untouched and still used for display and
    face auth. MediaPipe's landmark outputs are normalized (0..1), so
    nothing downstream needs to know or care that this was resized."""
    h, w = frame.shape[:2]
    if w <= target_width:
        return frame
    scale = target_width / w
    return cv2.resize(frame, (target_width, int(h * scale)), interpolation=cv2.INTER_LINEAR)


def resolve_camera_settings(gcfg: dict):
    cam_cfg = gcfg["camera"]
    if cam_cfg.get("use_existing_project_camera_settings", True):
        return {
            "index": base_config.CAMERA_INDEX,
            "width": base_config.CAMERA_WIDTH,
            "height": base_config.CAMERA_HEIGHT,
            "backend": base_config.CAMERA_BACKEND,
            "mirror": base_config.MIRROR_PREVIEW,
        }
    return {
        "index": cam_cfg["index"],
        "width": cam_cfg["width"],
        "height": cam_cfg["height"],
        "backend": cam_cfg["backend"],
        "mirror": cam_cfg["mirror_preview"],
    }


class ActionController:
    """Turns GestureEvents into real actions via action_engine.ACTIONS,
    respecting the discrete/continuous/stateful contract. Also owns the
    'gesture pause' flag driven by the special system_pause_toggle action."""

    def __init__(self, mappings: dict, dry_run: bool):
        self.mappings = mappings
        self.dry_run = dry_run
        self._continuous_state = {}   # gesture_name -> prev_state dict
        self._stateful_state = {}     # gesture_name -> state dict
        self.paused = False
        self.last_action_label = ""

    def toggle_pause(self):
        self.paused = not self.paused
        print(f"[CONTROL] Gestures {'PAUSED' if self.paused else 'RESUMED'}.")

    def handle_events(self, events):
        active_labels = []
        for ev in events:
            mapping = self.mappings.get(ev.gesture_name)
            if mapping is None:
                continue
            action_name = mapping["action"]
            params = {k: v for k, v in mapping.items() if k != "action"}
            spec = ACTIONS.get(action_name)
            if spec is None:
                continue

            if ev.state in ("start", "hold"):
                active_labels.append(f"{ev.gesture_name} [{ev.hand_label}]")

            if action_name == "system_pause_toggle":
                if ev.state == "start":
                    self.toggle_pause()
                continue

            if self.paused:
                continue

            self._dispatch(ev, action_name, spec, params)

        return active_labels

    def _dispatch(self, ev, action_name, spec, params):
        kind = spec["kind"]
        try:
            if kind == "discrete" and ev.state == "start":
                if self.dry_run:
                    print(f"[CALIBRATE] would trigger discrete action: {action_name} {params}")
                else:
                    spec["on_trigger"](params)
                self.last_action_label = action_name

            elif kind == "continuous" and ev.state in ("start", "hold"):
                prev = self._continuous_state.setdefault(ev.gesture_name, {})
                if self.dry_run:
                    self.last_action_label = f"{action_name} (dry-run)"
                else:
                    self._continuous_state[ev.gesture_name] = spec["on_frame"](
                        ev.hand_data, params, prev
                    )
                    self.last_action_label = action_name
            elif kind == "continuous" and ev.state == "end":
                self._continuous_state.pop(ev.gesture_name, None)

            elif kind == "stateful" and ev.state == "start":
                if self.dry_run:
                    self._stateful_state[ev.gesture_name] = {}
                    self.last_action_label = f"{action_name} start (dry-run)"
                else:
                    self._stateful_state[ev.gesture_name] = spec["on_enter"](params)
                    self.last_action_label = f"{action_name} start"
            elif kind == "stateful" and ev.state == "hold":
                state = self._stateful_state.get(ev.gesture_name)
                if state is not None and not self.dry_run:
                    self._stateful_state[ev.gesture_name] = spec["on_frame"](
                        ev.hand_data, params, state
                    )
            elif kind == "stateful" and ev.state == "end":
                state = self._stateful_state.pop(ev.gesture_name, None)
                if state is not None and not self.dry_run:
                    spec["on_exit"](params, state)
                self.last_action_label = f"{action_name} end"

        except Exception as e:
            # An action failing (e.g. a missing optional dependency) should
            # never crash the control loop or leave the owner without a
            # working ESC/quit.
            print(f"[ACTION ERROR] '{action_name}' failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Owner-gated hand-gesture laptop controller.")
    parser.add_argument("--calibrate", action="store_true",
                         help="Show detection/gesture/auth overlay but do NOT execute actions.")
    parser.add_argument("--config", default=config_loader.DEFAULT_CONFIG_PATH,
                         help="Path to gesture_settings.json")
    args = parser.parse_args()

    gcfg = config_loader.load_config(args.config)

    base_config.ensure_dirs()
    owner_embeddings = face_id.load_owner_embeddings()
    if owner_embeddings is None:
        print("[ERROR] No trained owner data found. Run register_face.py first (from the "
              "existing ai_security_camera project).")
        sys.exit(1)
    face_id.get_face_app()

    recognition_threshold = gcfg["authentication"]["recognition_threshold"]
    if recognition_threshold is None:
        recognition_threshold = base_config.RECOGNITION_THRESHOLD

    auth = AuthBridge(face_id, owner_embeddings, gcfg["authentication"], recognition_threshold)
    gesture_engine = GestureEngine(gcfg["gestures"], gcfg["reliability"])
    controller = ActionController(gcfg["mappings"], dry_run=args.calibrate)
    hand_tracker = HandTracker(gcfg["hands"])

    cam_settings = resolve_camera_settings(gcfg)
    # Reuse the exact camera-open/retry/warm-up routine from the existing project.
    base_config.CAMERA_INDEX = cam_settings["index"]
    base_config.CAMERA_WIDTH = cam_settings["width"]
    base_config.CAMERA_HEIGHT = cam_settings["height"]
    base_config.CAMERA_BACKEND = cam_settings["backend"]
    cap = base_main.open_camera()

    esc_listener = None
    if gcfg["reliability"].get("emergency_esc_enabled", True):
        esc_listener = _try_start_global_esc_listener()

    processing_width = gcfg["hands"].get("processing_width", 640)

    print("[INFO] Gesture control running."
          + (" (CALIBRATION MODE - no actions will fire)" if args.calibrate else ""))

    fps_count, fps_t0, fps = 0, time.time(), 0.0

    try:
        while True:
            if _emergency_stop.is_set():
                print("[EMERGENCY] ESC pressed - stopping all control immediately.")
                break

            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue

            if cam_settings["mirror"]:
                frame = cv2.flip(frame, 1)

            now = time.time()
            auth_status = auth.update(frame, now)

            active_gesture_labels = []
            if not auth_status.locked:
                rgb = cv2.cvtColor(_downscale_for_tracking(frame, processing_width), cv2.COLOR_BGR2RGB)
                hands = hand_tracker.update(rgb)
                if cam_settings["mirror"]:
                    # MediaPipe's Left/Right classification assumes an
                    # un-mirrored camera frame. Since we flip the frame for
                    # a natural mirror preview before tracking, its labels
                    # come back inverted (a real right hand gets called
                    # "Left"). Correct that here so displayed labels and any
                    # hand-specific gesture filters are actually accurate.
                    for h in hands:
                        h["handedness"] = "Left" if h["handedness"] == "Right" else "Right"
                events = gesture_engine.process(hands, now)
                active_gesture_labels = controller.handle_events(events)

                per_hand_active = {}
                for ev in events:
                    if ev.state in ("start", "hold"):
                        per_hand_active.setdefault(ev.hand_label, []).append(ev.gesture_name)
            else:
                # Locked: no hand tracking, no gesture evaluation, no
                # actions - guarantees zero input control while locked.
                gesture_engine.reset()
                hands, per_hand_active = [], {}
                if controller.paused:
                    controller.paused = False  # don't strand pause state across a lock cycle

            fps_count += 1
            if fps_count >= 20:
                elapsed = time.time() - fps_t0
                fps = fps_count / elapsed if elapsed > 0 else 0.0
                fps_count, fps_t0 = 0, time.time()

            if hands:
                draw_hand_skeleton(frame, gesture_engine.last_hand_data, per_hand_active)

            draw_overlay(
                frame,
                auth_status=auth_status,
                fps=fps,
                active_gestures=active_gesture_labels,
                last_action=controller.last_action_label,
                paused=controller.paused,
                calibrate_mode=args.calibrate,
                owner_name=base_config.OWNER_NAME,
            )
            cv2.imshow("Gesture Control - press q to quit, ESC for emergency stop", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC, in-window fallback
                _emergency_stop.set()
                continue
            if key == ord("q"):
                print("[INFO] Quit signal received.")
                break
            if key == ord("r"):
                try:
                    gcfg = config_loader.load_config(args.config)
                    gesture_engine = GestureEngine(gcfg["gestures"], gcfg["reliability"])
                    controller.mappings = gcfg["mappings"]
                    print("[INFO] gesture_settings.json reloaded.")
                except config_loader.ConfigError as e:
                    print(f"[WARN] Reload failed, keeping previous config: {e}")
            if key == ord("p"):
                controller.toggle_pause()

    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.")
    finally:
        release_all_inputs()
        try:
            hand_tracker.close()
        except Exception:
            pass
        cap.release()
        cv2.destroyAllWindows()
        if esc_listener is not None:
            esc_listener.stop()
        print("[INFO] Gesture control stopped. All inputs released.")


if __name__ == "__main__":
    main()
