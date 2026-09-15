"""
Action engine: a name -> action registry. Gestures never call pyautogui /
pynput directly - they trigger an action *name* from gesture_settings.json,
and this module is the only place that touches the OS input APIs.

To add a brand-new action:
    1. Write a function following one of the three shapes below.
    2. Register it with @register("your_action_name", kind="discrete"|"continuous"|"stateful").
    3. Reference "your_action_name" from the "mappings" section of
       gesture_settings.json. No other file needs to change.

Three action "kinds":
    discrete    - fires once per gesture activation (e.g. left_click).
                  Signature: fn(params: dict) -> None
    continuous  - runs every frame while the gesture is held
                  (e.g. move_cursor, scroll). Signature:
                  fn(hand_data: dict, params: dict, prev_state: dict) -> dict
                  Return a dict that gets passed back in as prev_state next
                  frame (use it to store things like "last cursor pos").
    stateful    - has an explicit begin/update/end lifecycle (e.g. drag).
                  on_enter(params) -> state
                  on_frame(hand_data, params, state) -> state
                  on_exit(params, state) -> None

pyautogui/pynput are imported lazily so this module (and the gesture engine
that depends on it for validation) can still be imported and unit-tested
on a machine without a display, without pyautogui raising at import time.
"""

import math
import subprocess
import sys
import time

ACTIONS = {}  # name -> {"kind": ..., callbacks...}


def register(name, kind):
    def deco(obj):
        if kind == "discrete":
            ACTIONS[name] = {"kind": "discrete", "on_trigger": obj}
        elif kind == "continuous":
            ACTIONS[name] = {"kind": "continuous", "on_frame": obj}
        return obj
    return deco


def register_stateful(name, on_enter, on_frame, on_exit):
    """Non-decorator form, used for drag which needs three callbacks."""
    ACTIONS[name] = {
        "kind": "stateful",
        "on_enter": on_enter,
        "on_frame": on_frame,
        "on_exit": on_exit,
    }


def _pyautogui():
    import pyautogui
    pyautogui.FAILSAFE = False  # we do our own safety handling (auth lock + ESC)
    pyautogui.PAUSE = 0
    return pyautogui


def _pynput_keyboard():
    from pynput.keyboard import Controller, Key
    return Controller(), Key


def _pynput_mouse():
    from pynput.mouse import Controller, Button
    return Controller(), Button


# --------------------------------------------------------------------------
# Discrete (one-shot) actions
# --------------------------------------------------------------------------

@register("left_click", kind="discrete")
def _left_click(params):
    _pyautogui().click(button="left")


@register("right_click", kind="discrete")
def _right_click(params):
    _pyautogui().click(button="right")


@register("middle_click", kind="discrete")
def _middle_click(params):
    _pyautogui().click(button="middle")


@register("double_click", kind="discrete")
def _double_click(params):
    _pyautogui().doubleClick()


@register("copy", kind="discrete")
def _copy(params):
    _pyautogui().hotkey("ctrl", "c")


@register("paste", kind="discrete")
def _paste(params):
    _pyautogui().hotkey("ctrl", "v")


@register("cut", kind="discrete")
def _cut(params):
    _pyautogui().hotkey("ctrl", "x")


@register("undo", kind="discrete")
def _undo(params):
    _pyautogui().hotkey("ctrl", "z")


@register("redo", kind="discrete")
def _redo(params):
    _pyautogui().hotkey("ctrl", "y")


@register("alt_tab", kind="discrete")
def _alt_tab(params):
    _pyautogui().hotkey("alt", "tab")


@register("win_shortcut", kind="discrete")
def _win_shortcut(params):
    """params: {"keys": ["win", "d"]} etc."""
    keys = params.get("keys", ["win", "d"])
    _pyautogui().hotkey(*keys)


@register("custom_hotkey", kind="discrete")
def _custom_hotkey(params):
    """params: {"keys": ["ctrl", "shift", "esc"]}"""
    keys = params.get("keys", [])
    if keys:
        _pyautogui().hotkey(*keys)


@register("key_press", kind="discrete")
def _key_press(params):
    """params: {"key": "enter"}"""
    key = params.get("key")
    if key:
        _pyautogui().press(key)


@register("key_combo", kind="discrete")
def _key_combo(params):
    """params: {"keys": ["ctrl", "shift", "n"]}"""
    keys = params.get("keys", [])
    if keys:
        _pyautogui().hotkey(*keys)


@register("close_window", kind="discrete")
def _close_window(params):
    _pyautogui().hotkey("alt", "f4")


@register("minimize_window", kind="discrete")
def _minimize_window(params):
    _pyautogui().hotkey("win", "down")


@register("maximize_window", kind="discrete")
def _maximize_window(params):
    _pyautogui().hotkey("win", "up")


@register("volume_up", kind="discrete")
def _volume_up(params):
    _pyautogui().press("volumeup")


@register("volume_down", kind="discrete")
def _volume_down(params):
    _pyautogui().press("volumedown")


@register("volume_mute", kind="discrete")
def _volume_mute(params):
    _pyautogui().press("volumemute")


@register("media_play_pause", kind="discrete")
def _media_play_pause(params):
    _pyautogui().press("playpause")


@register("media_next", kind="discrete")
def _media_next(params):
    _pyautogui().press("nexttrack")


@register("media_previous", kind="discrete")
def _media_previous(params):
    _pyautogui().press("prevtrack")


@register("brightness_up", kind="discrete")
def _brightness_up(params):
    _adjust_brightness(params.get("step", 10))


@register("brightness_down", kind="discrete")
def _brightness_down(params):
    _adjust_brightness(-params.get("step", 10))


def _adjust_brightness(delta):
    """Uses the optional 'screen-brightness-control' package. Laptop
    brightness has no pyautogui/pynput key event on most hardware (it's
    handled by vendor ACPI drivers, not the keyboard HID layer), so this
    needs a dedicated Windows-capable library instead."""
    try:
        import screen_brightness_control as sbc
        current = sbc.get_brightness(display=0)[0]
        sbc.set_brightness(max(0, min(100, current + delta)))
    except Exception as e:
        print(f"[ACTION] brightness change skipped - install 'screen-brightness-control' "
              f"(pip install screen-brightness-control). Error: {e}")


@register("screenshot", kind="discrete")
def _screenshot(params):
    import os
    from datetime import datetime
    out_dir = params.get("output_dir", "gesture_screenshots")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, datetime.now().strftime("screenshot_%Y%m%d_%H%M%S.png"))
    _pyautogui().screenshot(path)
    print(f"[ACTION] Screenshot saved: {path}")


@register("lock_windows", kind="discrete")
def _lock_windows(params):
    if sys.platform.startswith("win"):
        subprocess.run(["rundll32.exe", "user32.dll,LockWorkStation"])
    else:
        print("[ACTION] lock_windows is Windows-only; skipped on this platform.")


@register("open_app", kind="discrete")
def _open_app(params):
    """params: {"path": "notepad.exe"} or {"path": "C:\\...\\app.exe", "args": [...]}"""
    path = params.get("path")
    if not path:
        print("[ACTION] open_app requires a 'path' param.")
        return
    args = params.get("args", [])
    try:
        subprocess.Popen([path, *args], shell=False)
    except Exception as e:
        print(f"[ACTION] Failed to open '{path}': {e}")


@register("system_pause_toggle", kind="discrete")
def _system_pause_toggle(params):
    """Special action: doesn't touch the OS at all. main_gesture.py checks
    for this action name and flips its own 'gestures paused' flag - this
    is the 'Fist -> Lock Control' style gesture from the spec. It pauses
    gesture->action execution without dropping owner authentication, so
    the owner can put their hand down without triggering things."""
    pass  # handled specially in main_gesture.py's controller loop


# --------------------------------------------------------------------------
# Continuous (per-frame, held-gesture) actions
# --------------------------------------------------------------------------

_screen_size_cache = None


def _get_screen_size():
    """Cached - screen resolution doesn't change mid-session, so querying
    it every single frame (as earlier code did) is pure wasted latency."""
    global _screen_size_cache
    if _screen_size_cache is None:
        _screen_size_cache = _pyautogui().size()
    return _screen_size_cache


def clamp_movement(desired_dx, desired_dy, prev_dx, prev_dy, max_speed, max_accel):
    """Acceleration limiter for cursor/drag movement, kept as a pure
    function so it's unit-testable without a display or pynput.

    Two independent caps, applied in order:
    1. Speed cap - the desired movement this frame can't exceed max_speed
       pixels, regardless of what caused it (protects against a single
       spurious large jump, e.g. from hand-tracking briefly landing on the
       wrong spot).
    2. Acceleration cap - the CHANGE in movement from last frame's actual
       applied movement can't exceed max_accel pixels, so even a
       legitimately fast swipe ramps up smoothly over a couple of frames
       instead of snapping instantly to full speed. This is what actually
       kills the "choppy" feel - jitter reads as small random accelerations
       frame to frame, and this caps exactly that.

    Returns the (dx, dy) to actually apply this frame.
    """
    mag = math.hypot(desired_dx, desired_dy)
    if max_speed is not None and mag > max_speed > 0:
        scale = max_speed / mag
        desired_dx *= scale
        desired_dy *= scale

    if max_accel is not None and max_accel > 0:
        dvx = desired_dx - prev_dx
        dvy = desired_dy - prev_dy
        dmag = math.hypot(dvx, dvy)
        if dmag > max_accel:
            scale = max_accel / dmag
            desired_dx = prev_dx + dvx * scale
            desired_dy = prev_dy + dvy * scale

    return desired_dx, desired_dy


def accumulate_scroll_ticks(accum: float, delta: float):
    """Fractional scroll accumulation, kept as a pure function so it's
    unit-testable. Scroll amounts are whole ticks, but the computed delta
    each frame is rarely a whole number - truncating it every frame (the
    old behavior) silently throws away the fractional remainder every
    single frame, which is most of the signal during slow/steady scrolling
    and makes it feel sticky/stepped. Instead, carry the leftover fraction
    forward and only emit a tick once the accumulated total crosses a
    whole number.

    Returns (ticks_to_emit: int, new_accum: float).
    """
    total = accum + delta
    ticks = int(total)  # truncates toward zero, consistent for +/-
    return ticks, total - ticks


@register("move_cursor", kind="continuous")
def _move_cursor(hand_data, params, prev_state):
    mouse, _Button = _pynput_mouse()
    mode = params.get("mode", "relative")
    nx, ny = hand_data["point"]  # normalized 0..1, One-Euro-filtered in gesture_engine

    if mode == "absolute":
        screen_w, screen_h = _get_screen_size()
        margin = params.get("absolute_margin", 0.12)
        # Map the usable central region of the frame to the full screen so
        # the owner doesn't have to reach the physical frame edges.
        ux = min(1.0, max(0.0, (nx - margin) / (1 - 2 * margin)))
        uy = min(1.0, max(0.0, (ny - margin) / (1 - 2 * margin)))
        mouse.position = (ux * screen_w, uy * screen_h)
    else:
        sensitivity = params.get("relative_sensitivity", 1400)
        dead_zone = params.get("dead_zone", 0.0015)  # normalized-units; ignores hand-tremor-scale noise
        max_speed = params.get("max_speed_px_per_frame", 60)
        max_accel = params.get("max_accel_px_per_frame", 18)

        last = prev_state.get("last_point")
        prev_dx, prev_dy = prev_state.get("last_applied_move", (0.0, 0.0))

        if last is not None:
            raw_dx = nx - last[0]
            raw_dy = ny - last[1]
            if abs(raw_dx) < dead_zone:
                raw_dx = 0.0
            if abs(raw_dy) < dead_zone:
                raw_dy = 0.0

            desired_dx = raw_dx * sensitivity
            desired_dy = raw_dy * sensitivity
            apply_dx, apply_dy = clamp_movement(
                desired_dx, desired_dy, prev_dx, prev_dy, max_speed, max_accel
            )
            if apply_dx or apply_dy:
                mouse.move(apply_dx, apply_dy)
            prev_state["last_applied_move"] = (apply_dx, apply_dy)
        else:
            prev_state["last_applied_move"] = (0.0, 0.0)

        prev_state["last_point"] = (nx, ny)

    return prev_state


@register("scroll_vertical", kind="continuous")
def _scroll_vertical(hand_data, params, prev_state):
    mouse, _Button = _pynput_mouse()
    dead_zone = params.get("dead_zone", 0.002)
    last_y = prev_state.get("last_y")
    ny = hand_data["point"][1]
    if last_y is not None:
        raw = last_y - ny
        if abs(raw) >= dead_zone:
            delta = raw * params.get("scroll_sensitivity", 25)
            ticks, new_accum = accumulate_scroll_ticks(prev_state.get("scroll_accum", 0.0), delta)
            prev_state["scroll_accum"] = new_accum
            if ticks:
                mouse.scroll(0, ticks)
    prev_state["last_y"] = ny
    return prev_state


@register("scroll_horizontal", kind="continuous")
def _scroll_horizontal(hand_data, params, prev_state):
    mouse, _Button = _pynput_mouse()
    dead_zone = params.get("dead_zone", 0.002)
    last_x = prev_state.get("last_x")
    nx = hand_data["point"][0]
    if last_x is not None:
        raw = nx - last_x
        if abs(raw) >= dead_zone:
            delta = raw * params.get("scroll_sensitivity", 25)
            ticks, new_accum = accumulate_scroll_ticks(prev_state.get("scroll_accum", 0.0), delta)
            prev_state["scroll_accum"] = new_accum
            if ticks:
                mouse.scroll(ticks, 0)
    prev_state["last_x"] = nx
    return prev_state


# --------------------------------------------------------------------------
# Stateful actions (explicit enter / frame / exit lifecycle)
# --------------------------------------------------------------------------

def _drag_enter(params):
    mouse, Button = _pynput_mouse()
    mouse.press(Button.left)
    return {"last_point": None}


def _drag_frame(hand_data, params, state):
    mouse, _Button = _pynput_mouse()
    sensitivity = params.get("relative_sensitivity", 1400)
    dead_zone = params.get("dead_zone", 0.0015)
    max_speed = params.get("max_speed_px_per_frame", 60)
    max_accel = params.get("max_accel_px_per_frame", 18)

    nx, ny = hand_data["point"]
    last = state.get("last_point")
    prev_dx, prev_dy = state.get("last_applied_move", (0.0, 0.0))

    if last is not None:
        raw_dx = nx - last[0]
        raw_dy = ny - last[1]
        if abs(raw_dx) < dead_zone:
            raw_dx = 0.0
        if abs(raw_dy) < dead_zone:
            raw_dy = 0.0

        desired_dx = raw_dx * sensitivity
        desired_dy = raw_dy * sensitivity
        apply_dx, apply_dy = clamp_movement(desired_dx, desired_dy, prev_dx, prev_dy, max_speed, max_accel)
        if apply_dx or apply_dy:
            mouse.move(apply_dx, apply_dy)
        state["last_applied_move"] = (apply_dx, apply_dy)
    else:
        state["last_applied_move"] = (0.0, 0.0)

    state["last_point"] = (nx, ny)
    return state


def _drag_exit(params, state):
    mouse, Button = _pynput_mouse()
    mouse.release(Button.left)


register_stateful("drag", _drag_enter, _drag_frame, _drag_exit)


def release_all_inputs():
    """Emergency-stop helper: releases any mouse buttons / modifier keys
    that might be physically 'held down' by a stateful action, so an
    ESC press or an auth lock never leaves the OS in a stuck-input state.
    Covers both pyautogui (used for discrete clicks) and pynput (used for
    drag/move), since either could theoretically be mid-press."""
    try:
        pag = _pyautogui()
        for btn in ("left", "middle", "right"):
            try:
                pag.mouseUp(button=btn)
            except Exception:
                pass
    except Exception as e:
        print(f"[ACTION] release_all_inputs: pyautogui unavailable ({e})")

    try:
        mouse, Button = _pynput_mouse()
        for btn in (Button.left, Button.middle, Button.right):
            try:
                mouse.release(btn)
            except Exception:
                pass
    except Exception as e:
        print(f"[ACTION] release_all_inputs: pynput unavailable ({e})")
