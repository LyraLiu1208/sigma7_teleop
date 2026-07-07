from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from stiffness_copilot_mujoco.learning.task_state import TASK_STATE_DIM


STATE_DIM = 54
ACTION_DIM = 26
CONTACT_DIM = 11
CONTACT_FORCE_DIM = 3
STIFFNESS_MATRIX_SHAPE = (3, 3)
STIFFNESS_CHOLESKY_DIM = 6

REQUIRED_LEARNING_KEYS = (
    "state",
    "action",
    "contact_state",
    "reward_proxy",
    "timestamp",
    "episode_id",
    "task_state",
    "contact_force_world",
    "stiffness_matrix_target",
    "stiffness_cholesky_target",
    "label_neighbor_contact_ratio",
    "label_valid_sector_count",
    "environment_stiffness_eigvals",
    "raw_robot_stiffness_eigvals",
    "label_normalization_mask",
)

OPTIONAL_LEARNING_KEYS = (
    "mode_id",
)


@dataclass(frozen=True)
class LearningDatasetSchema:
    state_dim: int = STATE_DIM
    action_dim: int = ACTION_DIM
    contact_dim: int = CONTACT_DIM
    task_state_dim: int = TASK_STATE_DIM
    contact_force_dim: int = CONTACT_FORCE_DIM
    stiffness_matrix_shape: tuple[int, int] = STIFFNESS_MATRIX_SHAPE
    stiffness_cholesky_dim: int = STIFFNESS_CHOLESKY_DIM


def _require_2d(name: str, value: np.ndarray, width: int) -> None:
    if value.ndim != 2 or value.shape[1] != width:
        raise ValueError(f"{name} must have shape [N, {width}], got {value.shape}.")


def validate_learning_dataset(arrays: dict[str, np.ndarray], metadata: dict[str, Any] | None = None) -> None:
    missing = [key for key in REQUIRED_LEARNING_KEYS if key not in arrays]
    if missing:
        raise ValueError(f"Learning dataset missing keys: {missing}.")

    state = np.asarray(arrays["state"])
    action = np.asarray(arrays["action"])
    contact = np.asarray(arrays["contact_state"])
    reward = np.asarray(arrays["reward_proxy"])
    timestamp = np.asarray(arrays["timestamp"])
    episode_id = np.asarray(arrays["episode_id"])
    task_state = np.asarray(arrays["task_state"])
    contact_force = np.asarray(arrays["contact_force_world"])
    stiffness_matrix = np.asarray(arrays["stiffness_matrix_target"])
    stiffness_cholesky = np.asarray(arrays["stiffness_cholesky_target"])
    label_neighbor_contact_ratio = np.asarray(arrays["label_neighbor_contact_ratio"])
    label_valid_sector_count = np.asarray(arrays["label_valid_sector_count"])
    environment_stiffness_eigvals = np.asarray(arrays["environment_stiffness_eigvals"])
    raw_robot_stiffness_eigvals = np.asarray(arrays["raw_robot_stiffness_eigvals"])
    label_normalization_mask = np.asarray(arrays["label_normalization_mask"])
    hole_clearance_delta = np.asarray(arrays["hole_clearance_delta"]) if "hole_clearance_delta" in arrays else None
    mode_id = np.asarray(arrays["mode_id"]) if "mode_id" in arrays else None

    _require_2d("state", state, STATE_DIM)
    _require_2d("action", action, ACTION_DIM)
    _require_2d("contact_state", contact, CONTACT_DIM)
    expected_task_state_dim = int(metadata.get("task_state_dim", TASK_STATE_DIM)) if metadata is not None else TASK_STATE_DIM
    _require_2d("task_state", task_state, expected_task_state_dim)
    _require_2d("contact_force_world", contact_force, CONTACT_FORCE_DIM)
    _require_2d("stiffness_cholesky_target", stiffness_cholesky, STIFFNESS_CHOLESKY_DIM)
    _require_2d("environment_stiffness_eigvals", environment_stiffness_eigvals, 3)
    _require_2d("raw_robot_stiffness_eigvals", raw_robot_stiffness_eigvals, 3)
    if stiffness_matrix.ndim != 3 or stiffness_matrix.shape[1:] != STIFFNESS_MATRIX_SHAPE:
        raise ValueError(f"stiffness_matrix_target must have shape [N, 3, 3], got {stiffness_matrix.shape}.")

    length = state.shape[0]
    if length == 0:
        raise ValueError("Learning dataset must contain at least one sample.")

    length_checks = (
        ("action", action),
        ("contact_state", contact),
        ("reward_proxy", reward),
        ("timestamp", timestamp),
        ("episode_id", episode_id),
        ("task_state", task_state),
        ("contact_force_world", contact_force),
        ("stiffness_matrix_target", stiffness_matrix),
        ("stiffness_cholesky_target", stiffness_cholesky),
        ("label_neighbor_contact_ratio", label_neighbor_contact_ratio),
        ("label_valid_sector_count", label_valid_sector_count),
        ("environment_stiffness_eigvals", environment_stiffness_eigvals),
        ("raw_robot_stiffness_eigvals", raw_robot_stiffness_eigvals),
        ("label_normalization_mask", label_normalization_mask),
    )
    for name, value in length_checks:
        if value.shape[0] != length:
            raise ValueError(f"{name} length {value.shape[0]} does not match state length {length}.")
    if hole_clearance_delta is not None and hole_clearance_delta.shape[0] != length:
        raise ValueError(f"hole_clearance_delta length {hole_clearance_delta.shape[0]} does not match state length {length}.")
    if mode_id is not None:
        if mode_id.ndim != 1:
            raise ValueError(f"mode_id must have shape [N], got {mode_id.shape}.")
        if mode_id.shape[0] != length:
            raise ValueError(f"mode_id length {mode_id.shape[0]} does not match state length {length}.")
        if np.any(mode_id < 0):
            raise ValueError("mode_id must contain non-negative integer ids.")

    for name, value in arrays.items():
        numeric = np.asarray(value)
        if np.issubdtype(numeric.dtype, np.number) and not np.all(np.isfinite(numeric)):
            raise ValueError(f"{name} contains non-finite values.")

    if not np.allclose(stiffness_matrix, np.swapaxes(stiffness_matrix, -1, -2), atol=1e-8):
        raise ValueError("stiffness_matrix_target must be symmetric.")
    eigvals = np.linalg.eigvalsh(stiffness_matrix)
    if np.any(eigvals <= 0.0):
        raise ValueError("stiffness_matrix_target must be positive definite.")
    if np.any(eigvals < -1e-8) or np.any(eigvals > 1.0 + 1e-8):
        raise ValueError("stiffness_matrix_target eigenvalues must be in [0, 1].")

    if metadata is not None:
        if int(metadata.get("num_samples", -1)) != length:
            raise ValueError("metadata num_samples does not match array length.")
