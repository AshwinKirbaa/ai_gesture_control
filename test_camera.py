"""
Standalone camera diagnostic - no face detection, no recognition.
Tries several OpenCV camera backends, reads a frame from each after a
warm-up, reports average brightness, and saves the frame to disk so you
can see which backend/setting actually gives a real (non-black) image.

Usage: python test_camera.py
"""

import cv2
import numpy as np
import os

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera_test_output")
os.makedirs(OUT_DIR, exist_ok=True)

BACKENDS = {
    "DSHOW": cv2.CAP_DSHOW,
    "MSMF": cv2.CAP_MSMF,
    "ANY": cv2.CAP_ANY,
}

CAMERA_INDEX = 0


def try_backend(name, backend_flag):
    print(f"\n=== Trying backend: {name} ===")
    cap = cv2.VideoCapture(CAMERA_INDEX, backend_flag)
    if not cap.isOpened():
        print(f"[{name}] FAILED to open.")
        return

    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"[{name}] Opened. Reported resolution: {int(w)}x{int(h)}")

    # Warm up
    for _ in range(20):
        cap.read()

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print(f"[{name}] Could not read a frame (ret={ret}).")
        return

    brightness = float(np.mean(frame))
    out_path = os.path.join(OUT_DIR, f"test_{name}.jpg")
    cv2.imwrite(out_path, frame)
    print(f"[{name}] Frame captured. Mean brightness: {brightness:.1f} (near 0 = black/broken)")
    print(f"[{name}] Saved to: {out_path}")

    if brightness < 5:
        print(f"[{name}] >>> LIKELY BLACK/BROKEN <<<")
    else:
        print(f"[{name}] >>> LOOKS OK - this backend works <<<")


def main():
    print("Camera diagnostic starting. This will try 3 backends in turn.")
    for name, flag in BACKENDS.items():
        try_backend(name, flag)
    print(f"\nDone. Open the images in: {OUT_DIR}")
    print("Use whichever backend produced a normal (non-black) image.")


if __name__ == "__main__":
    main()
