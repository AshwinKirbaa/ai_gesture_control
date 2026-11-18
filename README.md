# AI Gesture Control

Control a Windows laptop with hand gestures while face recognition ensures that only the registered owner can operate it.

The project combines MediaPipe hand tracking, configurable gesture recognition, InsightFace owner authentication, and OS-level mouse/keyboard automation. It also retains the original AI security-camera mode with optional Telegram alerts.

## Highlights

- Owner-gated mouse and keyboard control
- Real-time tracking of up to two hands
- Configurable finger poses, pinches, and directional swipes
- Cursor smoothing with a One Euro Filter
- Speed and acceleration limits to reduce cursor jumps
- Gesture debounce, hold duration, hysteresis, and cooldowns
- Automatic GPU-to-CPU MediaPipe fallback
- Calibration mode that performs no OS actions
- Automatic cursor-noise calibration and per-stage performance timings
- Emergency stop that releases held inputs
- JSON-based gesture remapping without changing Python code
- Offline unit tests for gesture detection, filtering, and actions

## Default controls

| Gesture | Action |
| --- | --- |
| Point with index finger | Move cursor |
| Thumb-index pinch | Left click |
| Thumb-middle pinch | Right click |
| Two-finger gesture | Drag |
| Open palm | Play/pause media |
| Thumb and pinky pose | Lock Windows |
| Thumb pose | Vertical scroll |
| Pinky pose | Switch application (`Alt+Tab`) |
| Swipe left/right | Previous/next media |
| Swipe up/down | Volume up/down |

All mappings can be changed in `gesture_control/config/gesture_settings.json`.

## How owner authentication works

The application starts in a locked state. It unlocks only after recognizing the registered owner for the configured number of consecutive checks. If the owner disappears or is not recognized, control locks again and any held mouse buttons or keys are released.

By default, other people may remain in frame as long as the owner is also recognized. Set `authentication.strangers_allowed` to `false` for stricter behavior where any unknown face locks control.

When locked, the hand-tracking and action pipeline does not run.

## Project structure

```text
ai_gesture_control/
├── main_gesture.py                      # Gesture-control entry point
├── main.py                              # Security-camera entry point
├── register_face.py                     # Owner face enrollment
├── config.py                            # Shared camera and face settings
├── face_id.py                           # InsightFace integration
├── requirements.txt                     # Face recognition dependencies
├── requirements-gesture.txt             # Gesture-control dependencies
└── gesture_control/
    ├── action_engine.py                  # Mouse/keyboard action registry
    ├── auth_bridge.py                    # Owner authentication state machine
    ├── config_loader.py                  # JSON configuration loader
    ├── gesture_engine.py                 # Gesture recognition and debounce
    ├── hand_tracker.py                   # MediaPipe backend and fallback
    ├── one_euro_filter.py                # Adaptive cursor smoothing
    ├── overlay.py                        # Preview and calibration HUD
    ├── config/gesture_settings.json      # Gesture and action configuration
    └── models/hand_landmarker.task       # MediaPipe hand model
```

## Requirements

- Windows 10 or 11
- Python 3.10 or 3.11 recommended
- A webcam
- Internet access for installation and the first InsightFace model download

MediaPipe attempts to use its GPU delegate first and automatically falls back to CPU if GPU initialization is unavailable. MediaPipe's Python API does not provide a laptop NPU backend.

## Installation

1. Clone the repository:

   ```powershell
   git clone https://github.com/AshwinKirbaa/ai_gesture_control.git
   cd ai_gesture_control
   ```

2. Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install the base and gesture-control dependencies:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt -r requirements-gesture.txt
   ```

The optional `screen-brightness-control` dependency is included for brightness gesture mappings.

## Register the owner

Gesture control will not start until owner face embeddings exist.

```powershell
python register_face.py
```

Look at the camera and slowly vary your head angle and lighting. Press `q` to stop early.

You can also place clear owner photos in `owner_photos_to_import/` and register from photos only:

```powershell
python register_face.py --skip-capture
```

To discard previous enrollment data and start again:

```powershell
python register_face.py --reset
```

Face samples and embeddings are stored locally and excluded from Git.

## Run

Start normal gesture control:

```powershell
python main_gesture.py
```

Start in calibration mode to inspect tracking and authentication without triggering mouse or keyboard actions:

```powershell
python main_gesture.py --calibrate
```

Automatically measure tracking jitter while holding one pointing hand still
and save a tuned cursor dead zone:

```powershell
python main_gesture.py --auto-calibrate
```

Use a different configuration file:

```powershell
python main_gesture.py --config path\to\gesture_settings.json
```

### Runtime keys

| Key | Behavior |
| --- | --- |
| `Esc` | Emergency stop, release inputs, and exit |
| `q` | Exit normally |
| `r` | Reload the gesture JSON configuration |
| `p` | Pause or resume gesture execution |

## Customize gestures

Edit `gesture_control/config/gesture_settings.json`, then press `r` in the running preview to reload it.

Gesture definitions support:

- `fingers`: a five-value pattern for thumb through pinky
- `pinch`: two fingertips with separate enter and exit thresholds
- `swipe`: a direction, velocity threshold, displacement, and time window

Each gesture can specify a hand, minimum hold duration, action, and action parameters. Available actions include clicks, cursor movement, scrolling, dragging, media control, volume, brightness, screenshots, common editing shortcuts, window control, application launching, custom keys, and hotkeys.

For example, this mapping makes a fist lock Windows:

```json
{
  "fist": {
    "action": "lock_windows"
  }
}
```

To add a new action, define it in `gesture_control/action_engine.py` and register it with `@register(...)` or `register_stateful(...)`.

## Tuning

Useful settings in `gesture_settings.json` include:

| Setting | Effect |
| --- | --- |
| `hands.processing_width` | Lower values improve speed; higher values improve distant-hand accuracy |
| `hands.min_detection_confidence` | Raises or lowers the hand detection requirement |
| `mouse.relative_sensitivity` | Changes cursor movement scale |
| `mouse.dead_zone` | Ignores small tracking movements and hand tremor; tune with `--auto-calibrate` |
| `mouse.max_speed_px_per_frame` | Limits large cursor jumps |
| `mouse.max_accel_px_per_frame` | Controls how quickly cursor speed may change |
| `reliability.one_euro_beta` | Higher values reduce lag during fast movement |
| `reliability.hand_grace_period_seconds` | Preserves hand state across brief tracking loss |

Use `--calibrate` after changing gesture definitions. The HUD reports rolling
capture, authentication, hand tracking, gesture, and action timings so the
slow stage is visible directly.

## Telegram configuration

Telegram credentials are not stored in source code. Copy `.env.example` to a
local `.env` if your launcher loads environment files, or set them in the
shell before running security-camera mode:

```powershell
$env:TELEGRAM_BOT_TOKEN = "your-new-token"
$env:TELEGRAM_CHAT_ID = "your-chat-id"
python main.py
```

If a token has previously been committed, revoke it through @BotFather before
using Telegram alerts again.

## Tests

The core math and gesture logic can be tested without a camera:

```powershell
python gesture_control/test_one_euro_filter.py
python gesture_control/test_action_engine.py
python gesture_control/test_gesture_engine.py
```

## Camera troubleshooting

If the preview is black, cropped, or cannot open the webcam, close other camera applications and run:

```powershell
python test_camera.py
```

Inspect the images created in `camera_test_output/`, then update `CAMERA_BACKEND`, `CAMERA_WIDTH`, and `CAMERA_HEIGHT` in `config.py`.

## Security-camera mode

The original unknown-person monitoring mode remains available:

```powershell
python main.py
```

It uses the same webcam and owner enrollment. Telegram settings are used only by security-camera mode, not by gesture control.

## Privacy and safety

Face embeddings, enrollment photos, and camera captures are sensitive biometric data. Keep them private, secure the computer account, and delete them when no longer required. Always test custom gestures in calibration mode before allowing them to control the operating system.

This is an experimental accessibility and automation project, not a security guarantee. Recognition and gesture accuracy vary with lighting, pose, camera quality, distance, and hardware.

## Additional technical notes

See [`README_GESTURE.md`](README_GESTURE.md) for the longer implementation history, reliability details, and design notes.

## License

No license file is currently included. Unless a license is added, the source remains under the repository owner's default copyright.
