from __future__ import annotations

from pathlib import Path
from typing import Any

from stiffness_copilot_mujoco.controllers.impedance import (
    TRACK_A_BASELINE_CONTROLLER_PROFILE,
    TRACK_A_DATA_COLLECTION_CONTROLLER_PROFILE,
    TaskSpaceImpedanceGains,
    gains_from_mapping,
    load_gain_profiles,
    load_task_space_impedance_gains,
)


BASELINE_TO_PROFILE = {
    "low": "soft",
    "high": "stiff",
}

TRACK_A_BASELINE_CONTROLLER_ROLE = "baseline_controller"
TRACK_A_DATA_COLLECTION_CONTROLLER_ROLE = "data_collection_controller"


def profile_for_baseline(baseline: str) -> str:
    try:
        return BASELINE_TO_PROFILE[baseline]
    except KeyError as exc:
        raise KeyError(f"Unknown fixed-stiffness baseline {baseline!r}. Use 'low' or 'high'.") from exc


def load_baseline_gains(config_path: Path, baseline: str) -> tuple[str, str, TaskSpaceImpedanceGains]:
    profile = profile_for_baseline(baseline)
    selected, gains = load_task_space_impedance_gains(config_path, profile)
    return baseline, selected, gains


def load_track_a_baseline_controller_gains(
    config_path: Path,
) -> tuple[str, str, TaskSpaceImpedanceGains]:
    return load_task_space_impedance_gains(config_path, TRACK_A_BASELINE_CONTROLLER_PROFILE)


def load_track_a_data_collection_controller_gains(
    config_path: Path,
) -> tuple[str, str, TaskSpaceImpedanceGains]:
    return load_task_space_impedance_gains(config_path, TRACK_A_DATA_COLLECTION_CONTROLLER_PROFILE)


__all__ = [
    "BASELINE_TO_PROFILE",
    "TaskSpaceImpedanceGains",
    "gains_from_mapping",
    "load_baseline_gains",
    "load_track_a_baseline_controller_gains",
    "load_track_a_data_collection_controller_gains",
    "load_gain_profiles",
    "load_task_space_impedance_gains",
    "profile_for_baseline",
    "TRACK_A_BASELINE_CONTROLLER_PROFILE",
    "TRACK_A_BASELINE_CONTROLLER_ROLE",
    "TRACK_A_DATA_COLLECTION_CONTROLLER_PROFILE",
    "TRACK_A_DATA_COLLECTION_CONTROLLER_ROLE",
]
