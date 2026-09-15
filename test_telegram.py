"""
Standalone test for the Telegram alert path only - no camera involved.
Run this to isolate why alerts aren't sending.

Usage: python test_telegram.py
"""

import sys
import requests
import config


def main():
    print("=== Telegram Alert Diagnostic ===")

    if not config.TELEGRAM_BOT_TOKEN:
        print("[FAIL] TELEGRAM_BOT_TOKEN is empty in config.py")
        sys.exit(1)
    if not config.TELEGRAM_CHAT_ID:
        print("[FAIL] TELEGRAM_CHAT_ID is empty in config.py")
        sys.exit(1)

    print(f"[INFO] Bot token starts with: {config.TELEGRAM_BOT_TOKEN[:10]}...")
    print(f"[INFO] Chat ID: {config.TELEGRAM_CHAT_ID}")

    # 1) Verify the bot token itself is valid
    url_me = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/getMe"
    try:
        r = requests.get(url_me, timeout=10)
        print(f"[getMe] status={r.status_code} body={r.text}")
        if r.status_code != 200:
            print("[FAIL] Bot token appears invalid. Re-check it from @BotFather.")
            sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"[FAIL] Network error reaching Telegram API: {e}")
        print("Check your internet connection / firewall / antivirus blocking api.telegram.org")
        sys.exit(1)

    # 2) Try sending an actual text message
    url_msg = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        r = requests.post(url_msg, data={
            "chat_id": config.TELEGRAM_CHAT_ID,
            "text": "Test alert from AI Security Camera - if you see this, alerts are working."
        }, timeout=10)
        print(f"[sendMessage] status={r.status_code} body={r.text}")
        if r.status_code == 200:
            print("[SUCCESS] Check your Telegram app now.")
        else:
            print("[FAIL] sendMessage failed. Common cause: you haven't sent /start to your "
                  "bot yet, or the chat_id is wrong (must be YOUR chat id, not the bot's).")
    except requests.exceptions.RequestException as e:
        print(f"[FAIL] Network error sending message: {e}")


if __name__ == "__main__":
    main()
