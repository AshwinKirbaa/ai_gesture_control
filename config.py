import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------- Camera ----------
CAMERA_INDEX = 0          # 0 = default built-in laptop webcam
CAMERA_WIDTH = 1920       # match your camera's native resolution (check Windows
CAMERA_HEIGHT = 1080      # Camera app > Settings > Advanced camera options > Media type)
CAMERA_BACKEND = "DSHOW"  # "DSHOW", "MSMF", or "ANY" - run test_camera.py to find
                           # which one gives a real (non-black) image on your laptop
MIRROR_PREVIEW = True      # Flip horizontally so it behaves like a normal selfie
                           # camera. Applied consistently to both registration
                           # and live detection.

# ---------- Paths (Windows-compatible via os.path.join) ----------
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "known_faces")
OWNER_SAMPLES_DIR = os.path.join(KNOWN_FACES_DIR, "owner")   # raw face crops, kept
                                                              # for your own reference only
CAPTURES_DIR = os.path.join(BASE_DIR, "captures")
MODEL_DIR = os.path.join(BASE_DIR, "models")
EMBEDDINGS_PATH = os.path.join(MODEL_DIR, "owner_embeddings.npy")
IMPORT_PHOTOS_DIR = os.path.join(BASE_DIR, "owner_photos_to_import")

OWNER_NAME = "Ashwin Kirbaa"

# ---------- InsightFace (detection + recognition) ----------
# SCRFD detector handles head tilt/turn/profile natively - no manual rotation
# scanning needed. ArcFace-family embeddings are far more accurate than LBPH.
INSIGHTFACE_MODEL_NAME = "buffalo_sc"  # buffalo_s and buffalo_sc use identical
                                        # detection/recognition weights - buffalo_sc
                                        # just skips downloading unused extra models.
                                        # The real speed lever below is DET_SIZE.
DET_SIZE = (256, 256)      # detector input size - THIS is what actually controls
                           # speed (lower = much faster, less compute) AND range/
                           # accuracy (higher = detects farther/smaller faces and
                           # harder poses better). You have fps headroom (10-15fps
                           # at 160), so raised for better range - lower back to
                           # (160,160) if this feels too slow again.
DET_THRESH = 0.45          # detector confidence threshold - LOWER catches more
                           # borderline/hard-pose faces (farther away, tilted)
                           # but slightly raises false-positive risk. Default is 0.5.
MIN_FACE_PIXELS = 30       # ignore detections smaller than this (noise filtering)
DETECTION_EVERY_N_FRAMES = 2   # run full detection on 1 out of every N camera
                               # frames; other frames reuse the last result.
                               # Lets you keep DET_SIZE high (good tilt/distance
                               # detection) while roughly multiplying effective
                               # speed by N. 1 = detect every frame (most
                               # responsive, slowest). Raise to 3 if still slow.

# Cosine similarity between a live face and the closest stored owner sample.
# Typical values: same person 0.45-0.85+, different person usually < 0.30.
# RAISE if strangers are being accepted as owner; LOWER if the owner is
# being rejected too often.
RECOGNITION_THRESHOLD = 0.38
SHOW_CONFIDENCE_ON_SCREEN = True

# ---------- Detection stability ----------
CONSECUTIVE_UNKNOWN_FRAMES = 8     # must see "unknown" this many frames in a row

# ---------- Alerting ----------
ALERT_COOLDOWN_SECONDS = 60        # min seconds between two alerts

# ---------- Telegram bot (phone alert channel) ----------
# Fill these in after creating a bot with @BotFather (see README instructions).
TELEGRAM_BOT_TOKEN = "8770466270:AAHuyjdrnODaPCKiJ7WxjqmMBDKB0ORIg4U"
TELEGRAM_CHAT_ID = "7878998325"

# ---------- Registration ----------
REGISTER_SAMPLE_COUNT = 100   # aim for 30-150; more angles/lighting = better accuracy


def ensure_dirs():
    for d in (KNOWN_FACES_DIR, OWNER_SAMPLES_DIR, CAPTURES_DIR, MODEL_DIR, IMPORT_PHOTOS_DIR):
        os.makedirs(d, exist_ok=True)
