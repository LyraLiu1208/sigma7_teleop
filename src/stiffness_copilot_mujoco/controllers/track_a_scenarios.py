from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from stiffness_copilot_mujoco.sim.scene import ROOT


DEFAULT_TRACK_A_SCENARIOS_YAML = ROOT / "configs" / "track_a_scenarios.yaml"


@dataclass(frozen=True)
class TrackAControlledContactProfile:
    hole_xy_radius: float
    teleop_noise_xy_amplitude: float
    teleop_noise_cycles: float
    teleop_noise_phase_x: float
    teleop_noise_phase_y: float
    clearance_delta: float
    friction_scale: float
    peg_tilt_x: float
    peg_tilt_y: float
    hole_yaw_offset: float = 0.0

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "TrackAControlledContactProfile":
        return cls(
            hole_xy_radius=float(mapping["hole_xy_radius"]),
            teleop_noise_xy_amplitude=float(mapping["teleop_noise_xy_amplitude"]),
            teleop_noise_cycles=float(mapping["teleop_noise_cycles"]),
            teleop_noise_phase_x=float(mapping["teleop_noise_phase_x"]),
            teleop_noise_phase_y=float(mapping["teleop_noise_phase_y"]),
            clearance_delta=float(mapping["clearance_delta"]),
            friction_scale=float(mapping["friction_scale"]),
            peg_tilt_x=float(mapping["peg_tilt_x"]),
            peg_tilt_y=float(mapping["peg_tilt_y"]),
            hole_yaw_offset=float(mapping.get("hole_yaw_offset", 0.0)),
        )

    def to_metadata(self) -> dict[str, float]:
        return {
            "hole_xy_radius": float(self.hole_xy_radius),
            "teleop_noise_xy_amplitude": float(self.teleop_noise_xy_amplitude),
            "teleop_noise_cycles": float(self.teleop_noise_cycles),
            "teleop_noise_phase_x": float(self.teleop_noise_phase_x),
            "teleop_noise_phase_y": float(self.teleop_noise_phase_y),
            "clearance_delta": float(self.clearance_delta),
            "friction_scale": float(self.friction_scale),
            "peg_tilt_x": float(self.peg_tilt_x),
            "peg_tilt_y": float(self.peg_tilt_y),
            "hole_yaw_offset": float(self.hole_yaw_offset),
        }


@dataclass(frozen=True)
class TrackAScenarioEntry:
    scenario_id: str
    status: str
    main_controller_id: str | None
    best_observed_controller_id: str | None
    candidate_controller_id: str | None
    high_success_requirement_met: bool | None
    confirmation_pending: bool | None
    observation_controller_id: str | None
    teleop_mode: str | None
    residual_dim: int | None
    residual_parameterization: str | None
    scene: str | None
    setting: str | None
    profile_name: str | None
    contact_profile: str | None
    contact_condition_name: str | None
    controlled_contact_profile: TrackAControlledContactProfile | None = None
    notes: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, scenario_id: str, mapping: dict[str, Any]) -> "TrackAScenarioEntry":
        notes = tuple(str(note) for note in mapping.get("notes", ()))
        controlled_contact_profile = mapping.get("controlled_contact_profile")
        return cls(
            scenario_id=scenario_id,
            status=str(mapping["status"]),
            main_controller_id=None if mapping.get("main_controller_id") is None else str(mapping["main_controller_id"]),
            best_observed_controller_id=None
            if mapping.get("best_observed_controller_id") is None
            else str(mapping["best_observed_controller_id"]),
            candidate_controller_id=None
            if mapping.get("candidate_controller_id") is None
            else str(mapping["candidate_controller_id"]),
            high_success_requirement_met=None
            if mapping.get("high_success_requirement_met") is None
            else bool(mapping["high_success_requirement_met"]),
            confirmation_pending=None
            if mapping.get("confirmation_pending") is None
            else bool(mapping["confirmation_pending"]),
            observation_controller_id=None
            if mapping.get("observation_controller_id") is None
            else str(mapping["observation_controller_id"]),
            teleop_mode=None if mapping.get("teleop_mode") is None else str(mapping["teleop_mode"]),
            residual_dim=None if mapping.get("residual_dim") is None else int(mapping["residual_dim"]),
            residual_parameterization=None
            if mapping.get("residual_parameterization") is None
            else str(mapping["residual_parameterization"]),
            scene=None if mapping.get("scene") is None else str(mapping["scene"]),
            setting=None if mapping.get("setting") is None else str(mapping["setting"]),
            profile_name=None if mapping.get("profile_name") is None else str(mapping["profile_name"]),
            contact_profile=None if mapping.get("contact_profile") is None else str(mapping["contact_profile"]),
            contact_condition_name=None if mapping.get("contact_condition_name") is None else str(mapping["contact_condition_name"]),
            controlled_contact_profile=(
                None
                if controlled_contact_profile is None
                else TrackAControlledContactProfile.from_mapping(dict(controlled_contact_profile))
            ),
            notes=notes,
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "status": self.status,
            "main_controller_id": self.main_controller_id,
            "best_observed_controller_id": self.best_observed_controller_id,
            "candidate_controller_id": self.candidate_controller_id,
            "high_success_requirement_met": self.high_success_requirement_met,
            "confirmation_pending": self.confirmation_pending,
            "observation_controller_id": self.observation_controller_id,
            "teleop_mode": self.teleop_mode,
            "residual_dim": self.residual_dim,
            "residual_parameterization": self.residual_parameterization,
            "scene": self.scene,
            "setting": self.setting,
            "profile_name": self.profile_name,
            "contact_profile": self.contact_profile,
            "contact_condition_name": self.contact_condition_name,
            "controlled_contact_profile": None
            if self.controlled_contact_profile is None
            else self.controlled_contact_profile.to_metadata(),
            "notes": list(self.notes),
        }


def load_track_a_scenarios_registry(path: Path = DEFAULT_TRACK_A_SCENARIOS_YAML) -> dict[str, TrackAScenarioEntry]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a mapping with key 'scenarios'.")
    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, dict) or not scenarios:
        raise ValueError(f"{path} must contain a non-empty 'scenarios' mapping.")
    registry: dict[str, TrackAScenarioEntry] = {}
    for scenario_id, raw_entry in scenarios.items():
        if not isinstance(raw_entry, dict):
            raise ValueError(f"{path} contains a non-mapping scenario entry for {scenario_id!r}: {raw_entry!r}")
        if scenario_id in registry:
            raise ValueError(f"Duplicate Track A scenario_id {scenario_id!r} in {path}.")
        registry[scenario_id] = TrackAScenarioEntry.from_mapping(str(scenario_id), raw_entry)
    return registry


def get_track_a_scenario(scenario_id: str, *, scenarios_yaml: Path = DEFAULT_TRACK_A_SCENARIOS_YAML) -> TrackAScenarioEntry:
    registry = load_track_a_scenarios_registry(scenarios_yaml)
    try:
        return registry[scenario_id]
    except KeyError as exc:
        raise KeyError(
            f"Unknown Track A scenario_id {scenario_id!r}. Available: {', '.join(sorted(registry))}"
        ) from exc


__all__ = [
    "DEFAULT_TRACK_A_SCENARIOS_YAML",
    "TrackAControlledContactProfile",
    "TrackAScenarioEntry",
    "get_track_a_scenario",
    "load_track_a_scenarios_registry",
]
