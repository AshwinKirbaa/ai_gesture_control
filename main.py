import os
import sys
import time
import threading
from datetime import datetime

import cv2

import config
import alert
import face_id


def get_backend_flag():
    return getattr(cv2, f"CAP_{config.CAMERA_BACKEND}", cv2.CAP_DSHOW)


def open_camera(retries: int = 5, delay: float = 1.0):
    backend_flag = get_backend_flag()
    for attempt in range(1, retries + 1):
        cap = cv2.VideoCapture(config.CAMERA_INDEX, backend_flag)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
            actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print(f"[INFO] Camera opened. Requested {config.CAMERA_WIDTH}x{config.CAMERA_HEIGHT}, "
                  f"actual {int(actual_w)}x{int(actual_h)}.")
            if int(actual_w) != config.CAMERA_WIDTH or int(actual_h) != config.CAMERA_HEIGHT:
                print("[WARN] Camera did not honor requested resolution - this is usually "
                      "the cause of a 'zoomed/cropped' preview. Try setting CAMERA_WIDTH/"
                      "CAMERA_HEIGHT in config.py to the 'actual' values printed above.")

            time.sleep(0.5)
            got_real_frame = False
            for i in range(60):
                ret, warm_frame = cap.read()
                if ret and warm_frame is not None and warm_frame.mean() > 5:
                    got_real_frame = True
                    break
            if not got_real_frame:
                print("[WARN] Camera is still returning black/empty frames after warm-up. "
                      "The feed may not work correctly - run test_camera.py to diagnose.")
            return cap
        print(f"[WARN] Camera open failed (attempt {attempt}/{retries}). Retrying...")
        cap.release()
        time.sleep(delay)
    print("[ERROR] Could not open webcam after multiple attempts. "
          "Check that it isn't in use by another app and try again.")
    sys.exit(1)


def save_capture(frame):
    """Returns (path, timestamp_str) or (None, None) if saving failed."""
    now = datetime.now()
    ts_str = now.strftime("%Y-%m-%d %H:%M:%S")
    fname = now.strftime("unknown_%Y%m%d_%H%M%S.jpg")
    path = os.path.join(config.CAPTURES_DIR, fname)
    try:
        os.makedirs(config.CAPTURES_DIR, exist_ok=True)
        ok = cv2.imwrite(path, frame)
        if not ok:
            print(f"[WARN] cv2.imwrite reported failure for {path}")
            return None, None
        return path, ts_str
    except Exception as e:
        print(f"[WARN] Could not save capture: {e}")
        return None, None


def main():
    config.ensure_dirs()

    owner_embeddings = face_id.load_owner_embeddings()
    if owner_embeddings is None:
        print("[ERROR] No trained owner data found. Run register_face.py first.")
        sys.exit(1)
    print(f"[INFO] Loaded {len(owner_embeddings)} owner face sample(s).")

    face_id.get_face_app()  # load model once up front, not on first frame

    cap = open_camera()
    print("[INFO] Security camera running. Press 'q' to quit.")
    if config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID:
        print("[INFO] Telegram alerts: CONFIGURED.")
    else:
        print("[WARN] Telegram alerts: NOT CONFIGURED (set TELEGRAM_BOT_TOKEN / "
              "TELEGRAM_CHAT_ID in config.py). Alerts will only print to console.")

    unknown_streak = 0
    last_alert_time = 0.0
    consecutive_read_failures = 0
    fps_frame_count = 0
    fps_window_start = time.time()
    frame_counter = 0
    cached_detections = []

    try:
        while True:
            ret, frame = cap.read()

            if not ret or frame is None:
                consecutive_read_failures += 1
                print("[WARN] Failed to read frame from camera.")
                if consecutive_read_failures > 30:
                    print("[ERROR] Too many consecutive frame read failures. "
                          "Attempting to reopen camera...")
                    cap.release()
                    cap = open_camera()
                    consecutive_read_failures = 0
                time.sleep(0.1)
                continue
            consecutive_read_failures = 0

            if config.MIRROR_PREVIEW:
                frame = cv2.flip(frame, 1)

            frame_counter += 1
            run_detection = (frame_counter % config.DETECTION_EVERY_N_FRAMES == 0)

            if run_detection:
                detections = face_id.detect_faces(frame)
                cached_detections = detections
            else:
                # Reuse the last computed boxes/labels - keeps the preview
                # feeling responsive without paying full detection cost
                # every single frame.
                detections = cached_detections

            fps_frame_count += 1
            if fps_frame_count >= 30:
                elapsed = time.time() - fps_window_start
                fps = fps_frame_count / elapsed if elapsed > 0 else 0
                print(f"[PERF] {fps:.1f} fps (detection runs 1 in every "
                      f"{config.DETECTION_EVERY_N_FRAMES} frames)")
                fps_frame_count = 0
                fps_window_start = time.time()

            status_text = "NO FACE"
            status_color = (200, 200, 200)
            any_owner = False
            any_unknown = False

            if len(detections) > 0:
                for det in detections:
                    x, y, w, h = det["box"]
                    similarity = face_id.best_match_similarity(det["embedding"], owner_embeddings)
                    is_owner = similarity >= config.RECOGNITION_THRESHOLD

                    if is_owner:
                        any_owner = True
                        name_label = config.OWNER_NAME
                        box_color = (0, 200, 0)
                    else:
                        any_unknown = True
                        name_label = "Unknown"
                        box_color = (0, 0, 255)

                    cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

                    label_text = name_label
                    if config.SHOW_CONFIDENCE_ON_SCREEN:
                        label_text += f" ({similarity:.2f})"
                    cv2.putText(frame, label_text, (x, max(15, y - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

                if any_unknown:
                    status_text = "UNKNOWN PERSON"
                    status_color = (0, 0, 255)
                elif any_owner:
                    status_text = f"OWNER DETECTED ({config.OWNER_NAME})"
                    status_color = (0, 200, 0)

                # Streak/alert logic only advances on real detection frames -
                # skipped (cached-reuse) frames just redraw, they don't count
                # toward or reset the streak.
                if run_detection:
                    unknown_streak = unknown_streak + 1 if any_unknown else 0

                if config.SHOW_CONFIDENCE_ON_SCREEN:
                    cv2.putText(frame, f"unknown streak: {unknown_streak}/{config.CONSECUTIVE_UNKNOWN_FRAMES}",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                if run_detection and unknown_streak >= config.CONSECUTIVE_UNKNOWN_FRAMES:
                    now = time.time()
                    if now - last_alert_time >= config.ALERT_COOLDOWN_SECONDS:
                        try:
                            capture_path, ts_str = save_capture(frame)
                            if capture_path:
                                print(f"[EVENT] Unknown person confirmed at {ts_str}. Saved: {capture_path}")
                                # Send in the background so a slow/retrying upload
                                # never freezes the live camera window.
                                threading.Thread(
                                    target=alert.send_telegram_alert,
                                    args=(capture_path, ts_str),
                                    daemon=True,
                                ).start()
                            last_alert_time = now
                        except Exception as e:
                            print(f"[WARN] Unknown-person handling failed (continuing): {e}")
                            last_alert_time = now
                    else:
                        remaining = config.ALERT_COOLDOWN_SECONDS - (now - last_alert_time)
                        print(f"[INFO] Unknown person still present, but alert is on cooldown "
                              f"({remaining:.0f}s remaining before next alert is allowed).")
                    unknown_streak = 0
            else:
                if run_detection:
                    unknown_streak = 0

            cv2.putText(frame, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)
            cv2.imshow("AI Security Camera - press q to quit", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] Quit signal received.")
                break

    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.")
    except Exception as e:
        import traceback
        print(f"[FATAL] Unexpected error: {e}")
        traceback.print_exc()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Camera released. Program stopped.")


if __name__ == "__main__":
    main()
