"""
Registers the owner's face (InsightFace edition):
1. Opens the webcam.
2. Captures REGISTER_SAMPLE_COUNT face embeddings - move your head (tilt,
   turn side to side) during capture for much better angle coverage.
3. Also imports any photos dropped in owner_photos_to_import/.
4. Saves all embeddings to disk for main.py to compare against.

Run this once before main.py (re-run any time to re-register).
"""

import os
import sys
import time
import argparse

import cv2

import config
import face_id


def import_external_photos(embeddings_list) -> int:
    """Scan config.IMPORT_PHOTOS_DIR for any number of existing owner photos
    and add their face embeddings to the training set."""
    if not os.path.isdir(config.IMPORT_PHOTOS_DIR):
        return 0

    files = [f for f in os.listdir(config.IMPORT_PHOTOS_DIR)
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not files:
        return 0

    print(f"[INFO] Found {len(files)} photo(s) to import from {config.IMPORT_PHOTOS_DIR}")
    count = 0
    for fname in files:
        path = os.path.join(config.IMPORT_PHOTOS_DIR, fname)
        img = cv2.imread(path)
        if img is None:
            print(f"[WARN] Could not read {fname}, skipping.")
            continue

        detections = face_id.detect_faces(img)
        if not detections:
            print(f"[WARN] No face detected in {fname}, skipping.")
            continue

        det = max(detections, key=lambda d: d["box"][2] * d["box"][3])
        embeddings_list.append(det["embedding"])

        x, y, w, h = det["box"]
        crop = img[y:y + h, x:x + w]
        out_path = os.path.join(config.OWNER_SAMPLES_DIR, f"import_{count:03d}.jpg")
        cv2.imwrite(out_path, crop)
        count += 1

    print(f"[INFO] Imported {count} usable face(s) from external photos.")
    return count


def capture_samples(embeddings_list) -> int:
    backend_flag = getattr(cv2, f"CAP_{config.CAMERA_BACKEND}", cv2.CAP_DSHOW)
    cap = cv2.VideoCapture(config.CAMERA_INDEX, backend_flag)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Check CAMERA_INDEX / camera permissions.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

    time.sleep(0.5)
    for i in range(60):
        ret, warm_frame = cap.read()
        if ret and warm_frame is not None and warm_frame.mean() > 5:
            break
    else:
        print("[WARN] Camera is still returning black/empty frames after warm-up. "
              "Run test_camera.py to diagnose.")

    print("[INFO] Look at the camera. Capturing samples...")
    print("[INFO] Move your head slightly - tilt, turn side to side - for better accuracy.")
    print("[INFO] Press 'q' to stop early.")

    count = 0
    fail_count = 0
    last_capture_time = 0

    while count < config.REGISTER_SAMPLE_COUNT:
        ret, frame = cap.read()
        if not ret or frame is None:
            fail_count += 1
            if fail_count > 30:
                print("[ERROR] Camera stopped returning frames.")
                break
            continue
        fail_count = 0

        if config.MIRROR_PREVIEW:
            frame = cv2.flip(frame, 1)

        detections = face_id.detect_faces(frame)

        if detections:
            det = max(detections, key=lambda d: d["box"][2] * d["box"][3])
            x, y, w, h = det["box"]
            now = time.time()
            if now - last_capture_time > 0.15:  # small delay between captures
                embeddings_list.append(det["embedding"])
                crop = frame[y:y + h, x:x + w]
                out_path = os.path.join(config.OWNER_SAMPLES_DIR, f"sample_{count:03d}.jpg")
                cv2.imwrite(out_path, crop)
                count += 1
                last_capture_time = now

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(frame, f"Captured: {count}/{config.REGISTER_SAMPLE_COUNT}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Register Owner Face - press q to stop", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return count


def reset_owner_data():
    """Delete all previously captured/imported owner samples and embeddings,
    so the next registration starts completely clean."""
    import shutil
    if os.path.isdir(config.OWNER_SAMPLES_DIR):
        shutil.rmtree(config.OWNER_SAMPLES_DIR)
    os.makedirs(config.OWNER_SAMPLES_DIR, exist_ok=True)
    if os.path.exists(config.EMBEDDINGS_PATH):
        os.remove(config.EMBEDDINGS_PATH)
    print("[INFO] Cleared old owner samples and embeddings.")


def main():
    parser = argparse.ArgumentParser(description="Register the owner's face.")
    parser.add_argument("--skip-capture", action="store_true",
                         help="Skip live webcam capture; only use photos from "
                              "owner_photos_to_import/.")
    parser.add_argument("--reset", action="store_true",
                         help="Delete all previously captured/imported owner samples "
                              "and embeddings before registering again.")
    args = parser.parse_args()

    config.ensure_dirs()

    if args.reset:
        reset_owner_data()

    face_id.get_face_app()  # trigger model load/download up front with a clear message

    embeddings_list = []
    import_external_photos(embeddings_list)

    if not args.skip_capture:
        capture_samples(embeddings_list)

    if len(embeddings_list) < 5:
        print(f"[ERROR] Only {len(embeddings_list)} usable owner sample(s). Need at least 5. "
              f"Add more photos to {config.IMPORT_PHOTOS_DIR} or run without --skip-capture.")
        sys.exit(1)

    arr = face_id.save_owner_embeddings(embeddings_list)
    print(f"[INFO] Saved {len(arr)} owner embeddings to {config.EMBEDDINGS_PATH}")
    print("[DONE] Owner registered successfully. You can now run main.py")


if __name__ == "__main__":
    main()
