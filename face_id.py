"""
Face detection + recognition using InsightFace (SCRFD detector + ArcFace-family
embeddings). Handles head tilt, turn, and profile natively - no manual
rotation scanning needed, unlike the earlier Haar-cascade approach.
"""

import numpy as np

import config

_app = None


def get_face_app():
    """Lazily loads the InsightFace model (downloads it automatically on
    first run - needs internet once). Only loads the detection + recognition
    sub-models, skipping age/gender/landmark models we don't need, for speed.
    """
    global _app
    if _app is None:
        from insightface.app import FaceAnalysis
        print(f"[INFO] Loading InsightFace model '{config.INSIGHTFACE_MODEL_NAME}' "
              f"(first run downloads it automatically - needs internet)...")
        app = FaceAnalysis(
            name=config.INSIGHTFACE_MODEL_NAME,
            providers=["CPUExecutionProvider"],
            allowed_modules=["detection", "recognition"],
        )
        app.prepare(ctx_id=-1, det_size=config.DET_SIZE, det_thresh=config.DET_THRESH)
        _app = app
        print("[INFO] InsightFace model ready.")
    return _app


def detect_faces(frame_bgr):
    """Detects every face in a BGR frame at (almost) any pose/angle.
    Returns a list of dicts: {'box': (x, y, w, h), 'embedding': np.ndarray(512,)}
    """
    app = get_face_app()
    try:
        faces = app.get(frame_bgr)
    except Exception as e:
        print(f"[WARN] InsightFace detection error on this frame, skipping: {e}")
        return []

    results = []
    h_frame, w_frame = frame_bgr.shape[:2]
    for f in faces:
        try:
            x1, y1, x2, y2 = f.bbox.astype(int)
        except Exception:
            continue
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w_frame, x2)
        y2 = min(h_frame, y2)
        w, h = x2 - x1, y2 - y1
        if w < config.MIN_FACE_PIXELS or h < config.MIN_FACE_PIXELS:
            continue
        if f.embedding is None:
            continue
        results.append({"box": (x1, y1, w, h), "embedding": f.embedding})
    return results


def cosine_similarity(a, b):
    a_norm = a / (np.linalg.norm(a) + 1e-6)
    b_norm = b / (np.linalg.norm(b) + 1e-6)
    return float(np.dot(a_norm, b_norm))


def best_match_similarity(embedding, owner_embeddings):
    """Best (highest) cosine similarity between `embedding` and every stored
    owner sample. Returns -1.0 if there are no stored owner embeddings."""
    if owner_embeddings is None or len(owner_embeddings) == 0:
        return -1.0
    sims = owner_embeddings @ (embedding / (np.linalg.norm(embedding) + 1e-6))
    norms = np.linalg.norm(owner_embeddings, axis=1) + 1e-6
    sims = sims / norms
    return float(np.max(sims))


def load_owner_embeddings():
    import os
    if not os.path.exists(config.EMBEDDINGS_PATH):
        return None
    try:
        return np.load(config.EMBEDDINGS_PATH)
    except Exception as e:
        print(f"[ERROR] Could not load owner embeddings file (corrupted?): {e}")
        return None


def save_owner_embeddings(embeddings_list):
    arr = np.stack(embeddings_list, axis=0)
    np.save(config.EMBEDDINGS_PATH, arr)
    return arr
