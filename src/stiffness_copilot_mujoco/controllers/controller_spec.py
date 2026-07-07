from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from stiffness_copilot_mujoco.controllers.impedance import (
    TRACK_A_BASELINE_CONTROLLER_PROFILE,
    TRACK_A_DATA_COLLECTION_CONTROLLER_PROFILE,
    TaskSpaceImpedanceGains,
    load_task_space_impedance_gains,
)


TRACK_A_COLLECTION_CONTROLLER_ROLE = "collection_controller"
TRACK_A_BASELINE_CONTROLLER_ROLE = "baseline_controller"
TRACK_A_TASK_SPACE_CONTROLLER_KIND = "task_space_impedance"
TRACK_A_CONTROLLER_FORCE_ACCOUNTING = "full_step"
TRACK_A_CONTROLLER_TERMINATION_CONDITION = "episode_spec_terminal"
TRACK_A_CONTROLLER_UPDATE_MODE = "every_sim_step"
TRACK_A_CONTROLLER_UPDATE_PERIOD_STEPS = 1


@dataclass(frozen=True)
class ControllerSpec:
    controller_role: str
    controller_kind: str
    requested_profile: str
    selected_profile: str
    gain_config_path: str
    gains: TaskSpaceImpedanceGains
    position_stiffness_matrix: np.ndarray | None = None
    position_stiffness_matrix_source: str | None = None
    control_update_mode: str = TRACK_A_CONTROLLER_UPDATE_MODE
    control_update_period_steps: int = TRACK_A_CONTROLLER_UPDATE_PERIOD_STEPS
    force_accounting_mode: str = TRACK_A_CONTROLLER_FORCE_ACCOUNTING
    termination_condition: str = TRACK_A_CONTROLLER_TERMINATION_CONDITION

    def to_dict(self, *, simulation_dt_seconds: float | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "controller_role": self.controller_role,
            "controller_kind": self.controller_kind,
            "requested_profile": self.requested_profile,
            "selected_profile": self.selected_profile,
            "gain_config_path": self.gain_config_path,
            "control_update_mode": self.control_update_mode,
            "control_update_period_steps": int(self.control_update_period_steps),
            "force_accounting_mode": self.force_accounting_mode,
            "termination_condition": self.termination_condition,
            "gains": {
                "position_stiffness": list(self.gains.position_stiffness),
                "orientation_stiffness": list(self.gains.orientation_stiffness),
                "nullspace_stiffness": list(self.gains.nullspace_stiffness),
                "damping_ratio": float(self.gains.damping_ratio),
                "position_twist_gain": float(self.gains.position_twist_gain),
                "orientation_twist_gain": float(self.gains.orientation_twist_gain),
                "integration_dt": float(self.gains.integration_dt),
                "task_inertia_pinv_rcond": float(self.gains.task_inertia_pinv_rcond),
            },
        }
        if self.position_stiffness_matrix is not None:
            payload["position_stiffness_matrix"] = np.asarray(self.position_stiffness_matrix, dtype=float).tolist()
            payload["position_stiffness_matrix_source"] = self.position_stiffness_matrix_source
            payload["position_stiffness_mode"] = "explicit_matrix"
        else:
            payload["position_stiffness_matrix"] = None
            payload["position_stiffness_matrix_source"] = None
            payload["position_stiffness_mode"] = "diagonal_from_gains"
        if simulation_dt_seconds is not None:
            payload["simulation_dt_seconds"] = float(simulation_dt_seconds)
            payload["control_rate_hz"] = float(1.0 / simulation_dt_seconds) if simulation_dt_seconds > 0.0 else None
        return payload


def _load_controller_spec(
    config_path: Path,
    *,
    requested_profile: str,
    controller_role: str,
    position_stiffness_matrix: np.ndarray | None = None,
    position_stiffness_matrix_source: str | None = None,
) -> ControllerSpec:
    selected_profile, gains = load_task_space_impedance_gains(config_path, requested_profile)
    return ControllerSpec(
        controller_role=controller_role,
        controller_kind=TRACK_A_TASK_SPACE_CONTROLLER_KIND,
        requested_profile=requested_profile,
        selected_profile=selected_profile,
        gain_config_path=str(config_path),
        gains=gains,
        position_stiffness_matrix=None if position_stiffness_matrix is None else np.asarray(position_stiffness_matrix, dtype=float),
        position_stiffness_matrix_source=position_stiffness_matrix_source,
    )


def load_track_a_baseline_controller_spec(config_path: Path) -> ControllerSpec:
    return _load_controller_spec(
        config_path,
        requested_profile=TRACK_A_BASELINE_CONTROLLER_PROFILE,
        controller_role=TRACK_A_BASELINE_CONTROLLER_ROLE,
    )


def load_track_a_collection_controller_spec(
    config_path: Path,
    *,
    position_stiffness_matrix: np.ndarray | None = None,
    position_stiffness_matrix_source: str | None = None,
) -> ControllerSpec:
    return _load_controller_spec(
        config_path,
        requested_profile=TRACK_A_DATA_COLLECTION_CONTROLLER_PROFILE,
        controller_role=TRACK_A_COLLECTION_CONTROLLER_ROLE,
        position_stiffness_matrix=position_stiffness_matrix,
        position_stiffness_matrix_source=position_stiffness_matrix_source,
    )


__all__ = [
    "ControllerSpec",
    "TRACK_A_BASELINE_CONTROLLER_ROLE",
    "TRACK_A_COLLECTION_CONTROLLER_ROLE",
    "TRACK_A_CONTROLLER_FORCE_ACCOUNTING",
    "TRACK_A_CONTROLLER_TERMINATION_CONDITION",
    "TRACK_A_CONTROLLER_UPDATE_MODE",
    "TRACK_A_CONTROLLER_UPDATE_PERIOD_STEPS",
    "TRACK_A_TASK_SPACE_CONTROLLER_KIND",
    "load_track_a_baseline_controller_spec",
    "load_track_a_collection_controller_spec",
]
