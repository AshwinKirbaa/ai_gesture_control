"""
Shared face-detection logic used by both main.py and register_face.py.

Detects faces across in-plane head tilts (rotation) and left/right profile
turns, using only OpenCV's bundled Haar cascades (no extra downloads).
Returns upright face crops ready for recognition/training, plus a display
box mapped back to the original frame's coordinates.
"""

import cv2
import numpy as np

import config


def get_face_cascade():
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        raise RuntimeError("Could not load frontal face cascade. Check OpenCV installation.")
    return cascade


def get_profile_cascade():
    path = cv2.data.haarcascades + "haarcascade_profileface.xml"
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        raise RuntimeError("Could not load profile face cascade. Check OpenCV installation.")
    return cascade


def get_eye_cascade():
    path = cv2.data.haarcascades + "haarcascade_eye.xml"
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        raise RuntimeError("Could not load eye cascade. Check OpenCV installation.")
    return cascade


def get_clahe():
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def _non_max_suppression_indices(boxes, overlap_thresh=0.3):
    """Standard NMS: given [(x,y,w,h), ...], return indices to keep,
    removing duplicate detections of the same physical face found by
    different angles/cascades."""
    if len(boxes) == 0:
        return []
    arr = np.array(boxes, dtype=float)
    x1, y1 = arr[:, 0], arr[:, 1]
    x2, y2 = arr[:, 0] + arr[:, 2], arr[:, 1] + arr[:, 3]
    areas = arr[:, 2] * arr[:, 3]
    idxs = list(np.argsort(y2))
    keep = []
    while idxs:
        last = idxs[-1]
        keep.append(last)
        rest = idxs[:-1]
        if not rest:
            break
        xx1 = np.maximum(x1[last], x1[rest])
        yy1 = np.maximum(y1[last], y1[rest])
        xx2 = np.minimum(x2[last], x2[rest])
        yy2 = np.minimum(y2[last], y2[rest])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        overlap = (w * h) / areas[rest]
        idxs = [rest[i] for i in range(len(rest)) if overlap[i] <= overlap_thresh]
    return keep


def detect_faces(gray_full, cascade, profile_cascade, eye_cascade):
    """
    Detect faces in a full-resolution grayscale frame across multiple head
    tilts and both left/right profile turns.

    Returns a list of dicts: {'crop': upright FACE_SIZE grayscale patch,
    'box': (x,y,w,h) in gray_full's coordinates, for drawing}.
    """
    h_full, w_full = gray_full.shape[:2]
    scale = config.DETECTION_WIDTH / w_full
    if scale < 1.0:
        small = cv2.resize(gray_full, (0, 0), fx=scale, fy=scale)
    else:
        small = gray_full
        scale = 1.0
    h_small, w_small = small.shape[:2]
    min_size = (max(15, int(config.MIN_FACE_SIZE[0] * scale)),
                max(15, int(config.MIN_FACE_SIZE[1] * scale)))

    candidates = []  # {'crop': upright patch from `small`-scale image, 'box_small': (x,y,w,h)}

    # ---- Frontal faces at multiple in-plane rotation angles (head tilt) ----
    for angle in config.DETECTION_ANGLES:
        if angle == 0:
            rotated = small
            m_inv = None
        else:
            center = (w_small / 2.0, h_small / 2.0)
            m = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(small, m, (w_small, h_small))
            m_inv = cv2.invertAffineTransform(m)

        faces = cascade.detectMultiScale(
            rotated, scaleFactor=config.DETECTION_SCALE_FACTOR,
            minNeighbors=config.DETECTION_MIN_NEIGHBORS, minSize=min_size
        )
        for (x, y, w, h) in faces:
            crop = rotated[y:y + h, x:x + w]
            if crop.size == 0:
                continue

            ratio = w / float(h)
            if not (0.7 <= ratio <= 1.4):
                continue

            if config.REQUIRE_EYE_CHECK:
                upper = crop[:max(1, int(h * 0.6)), :]
                eyes = eye_cascade.detectMultiScale(upper, scaleFactor=1.1, minNeighbors=6,
                                                      minSize=(15, 15))
                if len(eyes) < 1:
                    continue

            if m_inv is None:
                box_small = (x, y, w, h)
            else:
                cx, cy = x + w / 2.0, y + h / 2.0
                pt = m_inv.dot(np.array([cx, cy, 1.0]))
                box_small = (int(pt[0] - w / 2.0), int(pt[1] - h / 2.0), w, h)

            candidates.append({'crop': crop, 'box_small': box_small})

    # ---- Side-profile faces (head turned left/right), both directions ----
    if config.ENABLE_PROFILE_DETECTION:
        for flip in (False, True):
            src = cv2.flip(small, 1) if flip else small
            faces = profile_cascade.detectMultiScale(
                src, scaleFactor=config.DETECTION_SCALE_FACTOR,
                minNeighbors=config.DETECTION_MIN_NEIGHBORS, minSize=min_size
            )
            for (x, y, w, h) in faces:
                crop = src[y:y + h, x:x + w]
                if crop.size == 0:
                    continue
                ratio = w / float(h)
                if not (0.65 <= ratio <= 1.2):  # profiles are naturally narrower
                    continue
                if flip:
                    box_small = (w_small - (x + w), y, w, h)
                    crop = cv2.flip(crop, 1)  # un-flip so orientation matches `small`
                else:
                    box_small = (x, y, w, h)
                candidates.append({'crop': crop, 'box_small': box_small})

    if not candidates:
        return []

    # ---- Merge duplicate detections of the same face across angles/cascades ----
    boxes = [c['box_small'] for c in candidates]
    keep = _non_max_suppression_indices(boxes, overlap_thresh=0.3)
    candidates = [candidates[i] for i in keep]

    # ---- Map back to full-resolution display coords; standardize crop size ----
    results = []
    for c in candidates:
        x, y, w, h = c['box_small']
        x, y = max(0, x), max(0, y)
        display_box = (int(x / scale), int(y / scale), int(w / scale), int(h / scale))
        crop_resized = cv2.resize(c['crop'], config.FACE_SIZE)
        results.append({'crop': crop_resized, 'box': display_box})
    return results
