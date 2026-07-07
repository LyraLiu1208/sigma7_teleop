from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from stiffness_copilot_mujoco.controllers.impedance import TaskSpaceImpedanceGains, load_task_space_impedance_gains
from stiffness_copilot_mujoco.sim.scene import ROOT


DEFAULT_TRACK_A_CONTROLLERS_YAML = ROOT / "configs" / "track_a_controllers.yaml"
DEFAULT_TRACK_A_GAIN_CONFIG = ROOT / "configs" / "controllers" / "fixed_impedance.yaml"


def _as_tuple_floats(value: Any, *, length: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple.")
    result = tuple(float(item) for item in value)
    if len(result) != length:
        raise ValueError(f"{name} must have length {length}, got {len(result)}.")
    return result


def _as_spd_matrix(value: Any, *, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3), got {matrix.shape}.")
    matrix = 0.5 * (matrix + matrix.T)
    eigvals = np.linalg.eigvalsh(matrix)
    if np.any(eigvals <= 0.0):
        raise ValueError(f"{name} must be positive definite, got eigenvalues {eigvals}.")
    return matrix


@dataclass(frozen=True)
class TrackAControllerEntry:
    controller_id: str
    position_stiffness_matrix: np.ndarray
    orientation_stiffness: tuple[float, float, float]
    nullspace_stiffness: tuple[float, float, float, float, float, float, float]
    damping_ratio: float
    position_twist_gain: float
    orientation_twist_gain: float
    integration_dt: float
    task_inertia_pinv_rcond: float
    description: str
    intended_role: str
    aliases: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "TrackAControllerEntry":
        controller_id = str(mapping["controller_id"])
        aliases = tuple(str(alias) for alias in mapping.get("aliases", ()))
        return cls(
            controller_id=controller_id,
            position_stiffness_matrix=_as_spd_matrix(mapping["position_stiffness_matrix"], name=f"{controller_id}.position_stiffness_matrix"),
            orientation_stiffness=_as_tuple_floats(mapping["orientation_stiffness"], length=3, name=f"{controller_id}.orientation_stiffness"),
            nullspace_stiffness=_as_tuple_floats(mapping["nullspace_stiffness"], length=7, name=f"{controller_id}.nullspace_stiffness"),
            damping_ratio=float(mapping["damping_ratio"]),
            position_twist_gain=float(mapping["position_twist_gain"]),
            orientation_twist_gain=float(mapping["orientation_twist_gain"]),
            integration_dt=float(mapping["integration_dt"]),
            task_inertia_pinv_rcond=float(mapping["task_inertia_pinv_rcond"]),
            description=str(mapping["description"]),
            intended_role=str(mapping["intended_role"]),
            aliases=aliases,
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "controller_id": self.controller_id,
            "position_stiffness_matrix": self.position_stiffness_matrix.tolist(),
            "orientation_stiffness": list(self.orientation_stiffness),
            "nullspace_stiffness": list(self.nullspace_stiffness),
            "damping_ratio": float(self.damping_ratio),
            "position_twist_gain": float(self.position_twist_gain),
            "orientation_twist_gain": float(self.orientation_twist_gain),
            "integration_dt": float(self.integration_dt),
            "task_inertia_pinv_rcond": float(self.task_inertia_pinv_rcond),
            "description": self.description,
            "intended_role": self.intended_role,
            "aliases": list(self.aliases),
        }


def load_track_a_controllers_registry(path: Path = DEFAULT_TRACK_A_CONTROLLERS_YAML) -> dict[str, TrackAControllerEntry]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a mapping with key 'controllers'.")
    controllers = payload.get("controllers")
    if not isinstance(controllers, list) or not controllers:
        raise ValueError(f"{path} must contain a non-empty 'controllers' list.")
    registry: dict[str, TrackAControllerEntry] = {}
    for raw_entry in controllers:
        if not isinstance(raw_entry, dict):
            raise ValueError(f"{path} contains a non-mapping controller entry: {raw_entry!r}")
        entry = TrackAControllerEntry.from_mapping(raw_entry)
        if entry.controller_id in registry:
            raise ValueError(f"Duplicate Track A controller_id {entry.controller_id!r} in {path}.")
        registry[entry.controller_id] = entry
        for alias in entry.aliases:
            if alias in registry:
                raise ValueError(f"Duplicate Track A controller alias {alias!r} in {path}.")
            registry[alias] = entry
    return registry


def get_track_a_controller(controller_id: str, *, controllers_yaml: Path = DEFAULT_TRACK_A_CONTROLLERS_YAML) -> TrackAControllerEntry:
    registry = load_track_a_controllers_registry(controllers_yaml)
    try:
        return registry[controller_id]
    except KeyError as exc:
        raise KeyError(
            f"Unknown Track A controller_id {controller_id!r}. Available: {', '.join(sorted(set(registry)))}"
        ) from exc


def load_track_a_controller_runtime(
    controller_id: str,
    *,
    controllers_yaml: Path = DEFAULT_TRACK_A_CONTROLLERS_YAML,
    gain_config: Path = DEFAULT_TRACK_A_GAIN_CONFIG,
) -> tuple[TrackAControllerEntry, str, TaskSpaceImpedanceGains]:
    entry = get_track_a_controller(controller_id, controllers_yaml=controllers_yaml)
    selected_profile, gains = load_task_space_impedance_gains(gain_config, controller_id)
    matrix = np.asarray(entry.position_stiffness_matrix, dtype=float)
    gains_matrix = np.diag(np.asarray(gains.position_stiffness, dtype=float))
    if not np.allclose(gains_matrix, matrix, atol=1e-9, rtol=0.0):
        raise ValueError(
            f"Gain config {gain_config} for controller_id {controller_id!r} does not match registry matrix. "
            f"registry={matrix.tolist()}, gains={gains_matrix.tolist()}"
        )
    return entry, selected_profile, gains


__all__ = [
    "DEFAULT_TRACK_A_CONTROLLERS_YAML",
    "DEFAULT_TRACK_A_GAIN_CONFIG",
    "TrackAControllerEntry",
    "get_track_a_controller",
    "load_track_a_controller_runtime",
    "load_track_a_controllers_registry",
]
