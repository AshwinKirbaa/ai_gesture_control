# Hand-Gesture Laptop Controller (owner-gated)

Adds gesture-based mouse/keyboard control on top of your existing
`ai_security_camera` face-recognition project. **Nothing in the original
project was modified** - `main.py`, `face_id.py`, `config.py`,
`face_utils.py`, `alert.py`, and `register_face.py` are untouched and your
security-camera mode (`python main.py`) still works exactly as before.
Telegram alerting is not used by this feature.

## What was added

```
ai_security_camera/
├── main_gesture.py                 <- new entry point for gesture control
├── requirements-gesture.txt        <- extra deps (mediapipe, pyautogui, pynput, ...)
└── gesture_control/
    ├── config/gesture_settings.json  <- YOUR editable gesture->action mappings
    ├── config_loader.py
    ├── action_engine.py             <- the OS-action registry (extend here)
    ├── gesture_engine.py            <- landmarks -> debounced gesture events
    ├── hand_tracker.py              <- MediaPipe wrapper (GPU->CPU->legacy fallback)
    ├── auth_bridge.py               <- LOCKED/UNLOCKED state machine over face_id.py
    ├── overlay.py                   <- on-screen calibration HUD
    └── test_gesture_engine.py       <- offline unit tests, no camera needed
```

`main_gesture.py` imports `open_camera()` from your existing `main.py` and
the detection/recognition functions from your existing `face_id.py` rather
than re-implementing them, so the camera-handling and face-matching
behavior you already tuned (resolution, backend, `RECOGNITION_THRESHOLD`,
etc.) carries over automatically.

## Install

```powershell
pip install -r requirements.txt -r requirements-gesture.txt
```

You already have owner face data registered from the existing project
(`models/owner_embeddings.npy`). If not, run `python register_face.py` first
- gesture control refuses to start without it.

## Run

```powershell
python main_gesture.py                       # normal operation
python main_gesture.py --calibrate            # see gestures/auth/FPS live, no actions fire
python main_gesture.py --config my_config.json
```

While running: **ESC = emergency stop** (locks control, releases any held
mouse button/keys, exits). `q` = normal quit. `r` = hot-reload
`gesture_settings.json` without restarting. `p` = manually pause/resume
gesture execution (same as the default fist gesture).

## Security model

```
Owner recognized (N consecutive frames), regardless of who
    else is also visible in the frame        -> UNLOCKED
Owner NOT among the detected faces for M
    consecutive frames (a stranger instead,
    or no face at all)                       -> LOCKED
```

The gate is "is the owner confirmed present" - not "is anyone else also
present". By default (`authentication.strangers_allowed: true`), other
people visible in the background don't block control as long as you're
also there and recognized; only your absence locks it. Set that to
`false` if you'd rather have the stricter behavior where any unrecognized
face anywhere in frame locks control immediately, even with you also
present.

Tunable in `gesture_settings.json` under `authentication`. When locked, the
main loop **does not run hand tracking or the gesture/action pipeline at
all** - it's not just "actions are blocked", the input pipeline never
executes, so there is no code path from a locked camera to a mouse/keyboard
event. `RECOGNITION_THRESHOLD` is reused from your existing `config.py` by
default (`authentication.recognition_threshold: null`); set a number there
to override just for gesture control without touching the security-camera
mode's threshold.

## Customizing gestures and actions (no Python required)

Edit `gesture_control/config/gesture_settings.json`. Two sections matter:

- **`gestures`**: defines *how to detect* a gesture (finger pattern, pinch
  pair + thresholds, or a directional swipe).
- **`mappings`**: defines *what a detected gesture does* - just a gesture
  name pointing at an action name (see the full action list in
  `action_engine.py`'s `ACTIONS` dict, and the catalog note at the bottom
  of the JSON file). Change the right-hand side of any mapping to
  reassign it, e.g.:

```json
"fist": { "action": "lock_windows" }
```

Gesture definition types:

| type      | fields                                                                 |
|-----------|-------------------------------------------------------------------------|
| `fingers` | `pattern`: 5 booleans (thumb..pinky) or `null` for "don't care"        |
| `pinch`   | `finger_a`, `finger_b`, `enter_threshold`, `exit_threshold` (hysteresis)|
| `swipe`   | `direction` (left/right/up/down), `min_velocity`, `window_ms`          |

All gestures also take `hand` (`"any"`, `"Left"`, or `"Right"`) and
`hold_ms` (minimum continuous hold before it fires, 0 = instant).

### Adding a brand-new action

Open `action_engine.py`, write a function, decorate it with
`@register("your_name", kind="discrete")` (or `"continuous"` for
held/streamed actions like cursor movement, or use
`register_stateful(...)` for enter/move/exit actions like drag). Then
reference `"your_name"` from `mappings` in the JSON file. See the module
docstring there for the exact function signatures.

## Reliability features implemented

- **Smoothing**: majority vote over a short rolling window of raw
  per-frame matches (`reliability.smoothing_window_frames`) before a
  gesture is considered "detected" at all.
- **Debounce + hold duration**: a gesture must stay smoothed-true for
  `hold_ms` before it fires a `start` event - one mechanism handles both
  "don't fire on a single flickery frame" and "require deliberate hold".
- **Cooldown**: after any gesture fires, it can't re-fire for
  `default_action_cooldown_seconds` (or `motion_gesture_cooldown_seconds`
  for swipes) - covered by `test_gesture_engine.py`'s cooldown test.
- **Hysteresis**: pinch gestures use separate enter/exit distance
  thresholds so the boundary itself doesn't flicker.
- **Cursor jitter control**: `move_cursor`/`drag` use an exponential
  relative-delta model (`mouse.relative_sensitivity`) rather than raw
  absolute landmark position, which is far less jittery than absolute
  mapping; `mouse.mode` can be switched to `"absolute"` if you prefer that.
- **Stuck-hand safety**: if a hand disappears mid-gesture, the engine
  force-fires an `end` event so continuous/stateful actions (scroll, drag)
  never get stuck active.
- **Emergency stop**: ESC (global hook via `pynput`, plus an in-window
  fallback) immediately releases any held mouse buttons and exits.
- **Auth-lock safety**: on lock, the gesture engine's state is reset and
  the gesture-pause flag doesn't carry across a lock/unlock cycle.

Run `python gesture_control/test_gesture_engine.py` any time you change
gesture logic - it exercises finger-state detection, pinch distance, hold
timing, swipe detection, and cooldown behavior without needing a camera.

## GPU / NPU - what's actually true here

You asked to use the laptop's GPU and NPU. Being straightforward about
what's realistic with these libraries:

- **GPU**: MediaPipe's Python **Tasks API** (`HandLandmarker`) does accept
  a GPU delegate, and `hand_tracker.py` tries it first
  (`hands.backend: "tasks_gpu_then_cpu"` in the config). In practice the
  GPU delegate on **Windows** is unreliable - the Windows Python wheel
  usually isn't built with the OpenGL/EGL backend the delegate needs, so
  it commonly fails to initialize. The code detects that failure and
  **automatically falls back to CPU** rather than crashing. If your setup
  does support it, you'll see `[HANDS] Using backend: mediapipe-tasks
  (GPU)` in the console; otherwise you'll see `(CPU)` and everything still
  works.
- **NPU**: MediaPipe's Python API has **no NPU delegate at all** - it's a
  C++/GPU/CPU stack, and even MediaPipe's own NPU support (Qualcomm
  Hexagon, via LiteRT) is C++/Android/Java-first with no exposed Python
  path. There isn't a way to hand hand-tracking to a laptop NPU through
  this library from Python today, and I didn't want to imply otherwise
  with a fake "NPU mode" toggle that silently does nothing.
- In practice this doesn't hurt usability: MediaPipe hand landmarks on CPU
  alone typically run at 25-30+ fps on a normal laptop, well above what's
  needed for responsive gesture control. Your face recognition
  (InsightFace, via the existing `face_id.py`) already runs on CPU too
  (`CPUExecutionProvider`) and that's left as-is per your instructions not
  to unnecessarily rewrite the existing project; swapping it to
  `onnxruntime-directml` for GPU acceleration is a separate, optional
  change to `face_id.py` if you want it later.

## Fixing tracking accuracy and mouse lag/jitter

Two things were added/fixed based on real-world testing:

**1. Visual hand-skeleton overlay** (so you can actually see what the
system sees). Every frame now draws the full 21-point landmark skeleton,
a bounding box, and the currently-active gesture name over each detected
hand - not just the text HUD. This runs automatically; nothing to enable.

**2. Cursor lag/jitter root causes and fixes:**

| Cause | Fix |
|---|---|
| Hand tracking ran on the full capture resolution (e.g. 1920x1080, sized for face-recognition accuracy, not real-time tracking) every single frame | `hands.processing_width` (default 640) downscales a *copy* of the frame just for hand tracking; display and face auth still use full resolution |
| Raw per-frame fingertip jitter was amplified directly by `relative_sensitivity` | Added real exponential smoothing (`reliability.cursor_smoothing_alpha`) on the point used for cursor/scroll/drag - this was documented in an earlier comment but not actually implemented before; it is now |
| Screen size was queried via `pyautogui.size()` on every single frame even in relative mode, which doesn't need it | Cached once; relative mode no longer queries it at all |
| `pyautogui.moveRel`/`scroll` have more call overhead than needed for a per-frame hot path | Cursor/scroll/drag now use `pynput.mouse.Controller` instead |
| Tiny hand tremor still moved the cursor even when you meant to hold still | Added `mouse.dead_zone` - movement below this threshold is ignored entirely |
| Editing `mouse.*` settings in the config file did nothing | Bug: those settings were never wired into the actual mapping. Fixed - they now apply as defaults, overridable per-gesture |

If it's still too jittery/laggy after these fixes, the levers to reach for
first are `hands.processing_width` (try 480), `reliability.cursor_smoothing_alpha`
(try 0.25), and `mouse.dead_zone` (raise it).


While inspecting the project, `config.py` had a live Telegram bot token
and chat ID committed in plaintext:

```python
TELEGRAM_BOT_TOKEN = "8770466270:AAHuyjdrnODaPCKiJ7WxjqmMBDKB0ORIg4U"
TELEGRAM_CHAT_ID = "7878998325"
```

This gesture-control feature doesn't use or touch that. But independent of
this project, it's worth rotating that token via @BotFather (revoke +
regenerate) and loading it from an environment variable or a
`.gitignore`'d file instead, especially before sharing this project folder
with anyone or pushing it anywhere.

## Priority fixes implemented (One Euro Filter, acceleration limiter, etc.)

Implemented per your priority ranking:

| Feature | Priority | What changed |
|---|---|---|
| One Euro Filter | Must add | Replaced the fixed-alpha EMA entirely. Adaptive: heavy smoothing while the hand is nearly still, light smoothing (low lag) while it's moving fast - a plain EMA can only pick one fixed tradeoff. See `one_euro_filter.py`; params under `reliability.one_euro_*`. |
| Velocity smoothing | Must add | Built into the One Euro Filter itself (`d_cutoff` smooths the internal velocity estimate) rather than a separate stage - this is the standard, correct way to do it per the original paper. |
| Acceleration limiter | Must add | New `clamp_movement()` in `action_engine.py`, applied to `move_cursor` and `drag`. Two caps: a per-frame speed ceiling (protects against a single bad tracking frame jumping the cursor) and a per-frame *change-in-movement* ceiling (this is what actually removes the choppy feel - jitter shows up as small random accelerations frame to frame, and this caps exactly that, while still letting a real fast swipe ramp up to full speed over a couple of frames). Config: `mouse.max_speed_px_per_frame`, `mouse.max_accel_px_per_frame`. |
| Fractional scroll accumulation | Add | `accumulate_scroll_ticks()` carries the fractional remainder of each frame's scroll delta forward instead of truncating it away every frame (which made slow scrolling feel sticky/stepped). |
| Dedicated swipe detector | Add | Already architecturally separate from the static-gesture (fingers/pinch) path - see the swipe-specific rising-edge logic in `gesture_engine.py`. Hardened further with a `min_displacement` check so a high *instantaneous* velocity reading over almost no real travel (jitter can produce this) doesn't false-trigger a swipe. |
| 480px processing | Test | `hands.processing_width` default lowered from 640 to 480 to test the latency/accuracy tradeoff. Raise back to 640 if hand detection feels less accurate at a distance. |
| More EMA smoothing | Don't rely on this | Not added - the EMA-based `cursor_smoothing_alpha` setting was removed entirely in favor of the One Euro Filter, which is a strictly better fit for this problem (adaptive vs. fixed). |

All of the above pure-math pieces (the filter itself, the acceleration
limiter, the scroll accumulator) are unit tested offline with no camera or
display needed:
```
python gesture_control/test_one_euro_filter.py
python gesture_control/test_action_engine.py
python gesture_control/test_gesture_engine.py
```

If the cursor still feels choppy after this, the two dials to reach for
first are `reliability.one_euro_beta` (raise for less lag during fast
movement) and `mouse.max_accel_px_per_frame` (raise if fast swipes feel
sluggish to reach full speed; lower if it still feels jerky at rest).

## Fixing swipe/scroll not working, misclicks, and random play/pause

These three symptoms came from real bugs, not just wrong sensitivity numbers:

**"Swipe and scroll don't work"** — swipe gestures were being run through the
same majority-vote smoothing built for held static gestures (fingers/pinch).
A swipe is a brief, one-off velocity spike - true for maybe 1-2 frames total
- so with the default 5-frame smoothing window it could essentially never
win a majority vote and would rarely if ever fire. Fixed: motion (swipe)
gestures now use a direct rising-edge trigger + cooldown instead of the
majority-vote path; the velocity-over-time-window calculation is already
the robustness mechanism they need. Covered by
`test_swipe_works_with_realistic_default_smoothing()`.

**"Always misclicked when I activate hand gesture" / "play/pause toggles
randomly"** — two compounding causes:
1. The default pinch enter-threshold (0.32, in hand-scale units) was loose
   enough that ordinary hand motion - just opening/closing fingers, or
   moving into a pointing pose - would transiently bring thumb and index
   close enough to register as a deliberate pinch. Tightened to 0.20
   enter / 0.38 exit, plus a longer required hold (90ms) and stricter
   debounce (4 frames), so a real pinch has to be deliberate and sustained.
2. Bigger issue: every hand's gesture state (hold timers, cooldowns, swipe
   motion history) was keyed by MediaPipe's raw per-frame Left/Right
   classification. That classification is noisy and can flip for the same
   physical hand between consecutive frames. Every flip wiped all state for
   that "hand" - including a cooldown that had *just* been set - so the
   same gesture could immediately re-fire right after it had just fired.
   Fixed: hands are now tracked by a stable internal identity matched via
   palm-center proximity frame-to-frame, completely independent of
   MediaPipe's own left/right labeling, and that identity now survives a
   brief (configurable, default 250ms) gap in detection instead of being
   torn down by a single missed frame. Covered by
   `test_gesture_state_survives_one_dropped_detection_frame()` and
   `test_swipe_motion_survives_one_dropped_frame_mid_swipe()` (a dropped
   detection frame during fast motion - exactly when a swipe happens - used
   to wipe the very motion history needed to detect it).

A smaller correctness fix alongside this: MediaPipe's handedness labels
assume a non-mirrored frame. Since the frame is flipped for a natural
mirror preview before tracking, labels came back inverted (your real right
hand was reported as "Left"). `main_gesture.py` now corrects this, so the
skeleton overlay's Left/Right labels are actually accurate.

If swipes/scroll still feel inconsistent after this, the levers are
`gestures.swipe_*.min_velocity` (lower = easier to trigger) and
`reliability.hand_grace_period_seconds` (raise slightly if your webcam
drops more than one consecutive frame during fast motion). If clicks still
fire too easily, lower `gestures.pinch_*.enter_threshold` further (e.g.
0.15) or raise `hold_ms`.

## Development flow this followed

1. Inspected the existing project (`main.py`, `face_id.py`, `config.py`,
   `face_utils.py`, `alert.py`) to find a non-invasive integration point.
2. Built hand tracking (`hand_tracker.py`) with a tested GPU->CPU->legacy
   fallback chain.
3. Built the configurable gesture engine (`gesture_engine.py`), unit
   tested offline against synthetic landmarks.
4. Built the action engine (`action_engine.py`) as a name->function
   registry, decoupled from the gesture engine entirely.
5. Integrated existing face authentication via `auth_bridge.py`, which
   only calls `face_id.py`'s existing public functions.
6. Added `gesture_settings.json` (config) and the calibration overlay +
   `--calibrate` dry-run mode.
7. `main_gesture.py` wires it all together and reuses `main.py`'s
   `open_camera()`.
