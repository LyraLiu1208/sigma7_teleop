from __future__ import annotations

from typing import Any

import numpy as np


TELEOP_MODE_POSITION_ONLY = "position_only"
TELEOP_MODE_POSITION_ORIENTATION = "position_orientation"
TELEOP_MODE_VALUES = (TELEOP_MODE_POSITION_ONLY, TELEOP_MODE_POSITION_ORIENTATION)


def rotation_about_z(angle: float) -> np.ndarray:
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def nominal_hole_rotation(scene_config: dict[str, Any]) -> np.ndarray:
    hole = scene_config.get("hole", {})
    return rotation_about_z(float(hole.get("rotation", 0.0)))


def build_target_orientations(
    scene_config: dict[str, Any],
    *,
    total_steps: int,
    teleop_mode: str,
) -> np.ndarray | None:
    if teleop_mode == TELEOP_MODE_POSITION_ONLY:
        return None
    if teleop_mode != TELEOP_MODE_POSITION_ORIENTATION:
        raise ValueError(f"Unsupported teleop_mode {teleop_mode!r}.")
    nominal_rotation = nominal_hole_rotation(scene_config)
    return np.repeat(nominal_rotation[None, :, :], total_steps + 1, axis=0)


def validate_teleop_mode(teleop_mode: str) -> str:
    mode = str(teleop_mode)
    if mode not in TELEOP_MODE_VALUES:
        raise ValueError(f"Unsupported teleop_mode {teleop_mode!r}; expected one of {TELEOP_MODE_VALUES!r}.")
    return mode
