"""
Loads gesture_control/config/gesture_settings.json into a plain nested
dict, with light validation and safe defaults. Nothing here touches the
existing project's config.py - it's read separately (see main_gesture.py)
so both configs stay independent but can share the camera/auth values.
"""

import json
import os

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "config", "gesture_settings.json"
)

REQUIRED_TOP_LEVEL_KEYS = (
    "camera", "authentication", "hands", "reliability", "mouse",
    "gestures", "mappings",
)


class ConfigError(Exception):
    pass


def load_config(path: str = DEFAULT_CONFIG_PATH) -> dict:
    if not os.path.exists(path):
        raise ConfigError(f"Gesture config not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        try:
            cfg = json.load(f)
        except json.JSONDecodeError as e:
            raise ConfigError(f"gesture_settings.json is not valid JSON: {e}")

    missing = [k for k in REQUIRED_TOP_LEVEL_KEYS if k not in cfg]
    if missing:
        raise ConfigError(f"gesture_settings.json is missing required section(s): {missing}")

    _validate_mappings(cfg)
    _apply_mouse_defaults(cfg)
    return cfg


def _apply_mouse_defaults(cfg: dict):
    """The top-level 'mouse' section is meant to be the shared default for
    any gesture mapped to move_cursor/drag, with per-mapping params (if
    present) overriding it. Without this merge, editing 'mouse' in the
    config file would silently do nothing - the mapping's own params dict
    is all action_engine ever sees."""
    mouse_defaults = cfg.get("mouse", {})
    relevant_keys = ("mode", "relative_sensitivity", "absolute_margin", "dead_zone",
                      "max_speed_px_per_frame", "max_accel_px_per_frame")
    forwardable = {k: v for k, v in mouse_defaults.items() if k in relevant_keys}

    for mapping in cfg["mappings"].values():
        if mapping.get("action") in ("move_cursor", "drag"):
            for k, v in forwardable.items():
                mapping.setdefault(k, v)


def _validate_mappings(cfg: dict):
    from gesture_control.action_engine import ACTIONS  # local import avoids a cycle

    gesture_names = set(cfg["gestures"].keys())
    problems = []

    for gesture_name, mapping in cfg["mappings"].items():
        if gesture_name not in gesture_names:
            problems.append(
                f"mappings references gesture '{gesture_name}' which is not defined in 'gestures'"
            )
        action_name = mapping.get("action")
        if action_name not in ACTIONS:
            problems.append(
                f"mappings['{gesture_name}'] uses unknown action '{action_name}'. "
                f"See action_engine.ACTIONS for valid names."
            )

    if problems:
        raise ConfigError("Invalid gesture_settings.json:\n  - " + "\n  - ".join(problems))


def save_config(cfg: dict, path: str = DEFAULT_CONFIG_PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
