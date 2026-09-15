import time
import requests
import cv2
import config


def send_telegram_alert(image_path: str, timestamp_str: str) -> bool:
    """Send the captured photo + timestamp to the owner's phone via Telegram.
    Never raises - returns True/False and logs to console instead, so the
    main loop can never crash because of a network/alert failure.

    The image is re-encoded smaller/more compressed before upload - the raw
    1920x1080 capture can be large enough to time out on a slow connection.
    Retries once on transient network failure.
    """
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        print(f"[ALERT] Telegram not configured. Unknown person at {timestamp_str}. "
              f"Image saved at: {image_path}")
        return False

    # Re-encode a smaller/compressed copy in memory for a faster upload -
    # the full-resolution original stays on disk in captures/ untouched.
    img = cv2.imread(image_path)
    if img is not None:
        h, w = img.shape[:2]
        target_w = 960
        if w > target_w:
            scale = target_w / w
            img = cv2.resize(img, (target_w, int(h * scale)))
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        photo_bytes = buf.tobytes() if ok else None
    else:
        photo_bytes = None

    url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendPhoto"
    caption = f"Unknown person detected at {timestamp_str}"

    for attempt in range(1, 3):  # try up to 2 times
        try:
            if photo_bytes is not None:
                files = {"photo": ("alert.jpg", photo_bytes, "image/jpeg")}
                response = requests.post(
                    url, data={"chat_id": config.TELEGRAM_CHAT_ID, "caption": caption},
                    files=files, timeout=(10, 30),
                )
            else:
                with open(image_path, "rb") as photo_file:
                    response = requests.post(
                        url, data={"chat_id": config.TELEGRAM_CHAT_ID, "caption": caption},
                        files={"photo": photo_file}, timeout=(10, 30),
                    )
            if response.status_code == 200:
                print("[ALERT] Telegram alert sent successfully.")
                return True
            print(f"[ALERT] Telegram send failed ({response.status_code}): {response.text}")
            return False
        except requests.exceptions.RequestException as e:
            print(f"[ALERT] Network error while sending alert (attempt {attempt}/2): {e}")
            if attempt < 2:
                time.sleep(2)
                continue
            return False
        except FileNotFoundError:
            print(f"[ALERT] Image file not found: {image_path}")
            return False
        except Exception as e:
            print(f"[ALERT] Unexpected error while sending alert: {e}")
            return False
    return False
