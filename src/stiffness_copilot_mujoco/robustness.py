from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np

from stiffness_copilot_mujoco.rollouts.fixed_impedance import RolloutPerturbation
from stiffness_copilot_mujoco.scenes import SceneName, scene_names


@dataclass(frozen=True)
class RobustnessPreset:
    scene: str
    setting_id: str
    hole_xy_radius: float
    hole_yaw_max_deg: float
    teleop_noise_xy_amplitude: float
    teleop_noise_cycles_min: float
    teleop_noise_cycles_max: float
    clearance_delta_min: float
    clearance_delta_max: float
    friction_scale_min: float
    friction_scale_max: float
    peg_tilt_max_deg: float

    def to_metadata(self) -> dict[str, float | str]:
        return {
            "scene": self.scene,
            "setting_id": self.setting_id,
            "hole_xy_radius": self.hole_xy_radius,
            "hole_yaw_max_deg": self.hole_yaw_max_deg,
            "teleop_noise_xy_amplitude": self.teleop_noise_xy_amplitude,
            "teleop_noise_cycles_min": self.teleop_noise_cycles_min,
            "teleop_noise_cycles_max": self.teleop_noise_cycles_max,
            "clearance_delta_min": self.clearance_delta_min,
            "clearance_delta_max": self.clearance_delta_max,
            "friction_scale_min": self.friction_scale_min,
            "friction_scale_max": self.friction_scale_max,
            "peg_tilt_max_deg": self.peg_tilt_max_deg,
        }

    @classmethod
    def from_metadata(cls, values: dict[str, object]) -> RobustnessPreset:
        setting_id = values.get("setting_id")
        if setting_id is None:
            legacy_name = values.get("difficulty", "unknown")
            setting_id = f"legacy_{values.get('scene', 'unknown')}_{legacy_name}"
        return cls(
            scene=str(values["scene"]),
            setting_id=str(setting_id),
            hole_xy_radius=float(values["hole_xy_radius"]),
            hole_yaw_max_deg=float(values["hole_yaw_max_deg"]),
            teleop_noise_xy_amplitude=float(values["teleop_noise_xy_amplitude"]),
            teleop_noise_cycles_min=float(values["teleop_noise_cycles_min"]),
            teleop_noise_cycles_max=float(values["teleop_noise_cycles_max"]),
            clearance_delta_min=float(values["clearance_delta_min"]),
            clearance_delta_max=float(values["clearance_delta_max"]),
            friction_scale_min=float(values["friction_scale_min"]),
            friction_scale_max=float(values["friction_scale_max"]),
            peg_tilt_max_deg=float(values["peg_tilt_max_deg"]),
        )


@dataclass(frozen=True)
class ControlledContactProfile:
    profile_name: str
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
    contact_condition_name: str | None = None

    def to_metadata(self) -> dict[str, Any]:
        legacy_field_mapping = {
            "randomization_vector": {
                "0": "hole_xy_offset_x",
                "1": "hole_xy_offset_y",
                "2": "hole_yaw_offset",
                "3": "clearance_delta",
                "4": "friction_scale",
                "5": "peg_tilt_x",
                "6": "peg_tilt_y",
                "7": "teleop_noise_xy_amplitude",
                "8": "teleop_noise_cycles",
            },
            "legacy_note": (
                "The randomization vector is retained for compatibility, but hole_xy_offset now means "
                "global table-level object placement rather than local insertion error."
            ),
        }
        return {
            "profile_name": self.profile_name,
            "contact_condition_name": self.contact_condition_name,
            "hole_xy_offset_semantics": "global_object_placement",
            "hole_xy_offset_units": "m",
            "hole_xy_offset_distribution": f"uniform_disk(radius={self.hole_xy_radius:g})",
            "trajectory_follows_randomized_hole": True,
            "contact_generation_parameters_fixed": True,
            "fixed_hole_yaw_offset": float(self.hole_yaw_offset),
            "fixed_teleop_noise_xy_amplitude": float(self.teleop_noise_xy_amplitude),
            "fixed_teleop_noise_cycles": float(self.teleop_noise_cycles),
            "fixed_teleop_noise_phase_x": float(self.teleop_noise_phase_x),
            "fixed_teleop_noise_phase_y": float(self.teleop_noise_phase_y),
            "fixed_clearance_delta": float(self.clearance_delta),
            "fixed_friction_scale": float(self.friction_scale),
            "fixed_peg_tilt_x": float(self.peg_tilt_x),
            "fixed_peg_tilt_y": float(self.peg_tilt_y),
            "legacy_field_mapping": legacy_field_mapping,
        }


CONTROLLED_CONTACT_PROFILE_CANDIDATES: dict[str, ControlledContactProfile] = {
    "circle_calibrated_v1_global_hole_fixed_contact_light": ControlledContactProfile(
        profile_name="circle_calibrated_v1_global_hole_fixed_contact",
        contact_condition_name="light",
        hole_xy_radius=0.02,
        teleop_noise_xy_amplitude=0.0010,
        teleop_noise_cycles=1.0,
        teleop_noise_phase_x=0.0,
        teleop_noise_phase_y=1.5707963267948966,
        clearance_delta=-0.0004,
        friction_scale=1.15,
        peg_tilt_x=0.0087,
        peg_tilt_y=-0.0087,
    ),
    "circle_calibrated_v1_global_hole_fixed_contact_medium": ControlledContactProfile(
        profile_name="circle_calibrated_v1_global_hole_fixed_contact",
        contact_condition_name="medium",
        hole_xy_radius=0.02,
        teleop_noise_xy_amplitude=0.0012,
        teleop_noise_cycles=1.0,
        teleop_noise_phase_x=0.7853981634,
        teleop_noise_phase_y=-0.7853981634,
        clearance_delta=-0.0005,
        friction_scale=1.25,
        peg_tilt_x=0.0131,
        peg_tilt_y=-0.0131,
    ),
    "circle_calibrated_v1_global_hole_fixed_contact_strong": ControlledContactProfile(
        profile_name="circle_calibrated_v1_global_hole_fixed_contact",
        contact_condition_name="strong",
        hole_xy_radius=0.02,
        teleop_noise_xy_amplitude=0.0015,
        teleop_noise_cycles=1.0,
        teleop_noise_phase_x=1.5707963268,
        teleop_noise_phase_y=-1.5707963268,
        clearance_delta=-0.0006,
        friction_scale=1.35,
        peg_tilt_x=0.0175,
        peg_tilt_y=-0.0175,
    ),
}


def make_controlled_contact_profile(
    *,
    profile_name: str,
    hole_xy_radius: float,
    teleop_noise_xy_amplitude: float,
    teleop_noise_cycles: float,
    teleop_noise_phase_x: float,
    teleop_noise_phase_y: float,
    clearance_delta: float,
    friction_scale: float,
    peg_tilt_x: float,
    peg_tilt_y: float,
    hole_yaw_offset: float = 0.0,
    contact_condition_name: str | None = None,
) -> ControlledContactProfile:
    return ControlledContactProfile(
        profile_name=profile_name,
        contact_condition_name=contact_condition_name,
        hole_xy_radius=hole_xy_radius,
        teleop_noise_xy_amplitude=teleop_noise_xy_amplitude,
        teleop_noise_cycles=teleop_noise_cycles,
        teleop_noise_phase_x=teleop_noise_phase_x,
        teleop_noise_phase_y=teleop_noise_phase_y,
        clearance_delta=clearance_delta,
        friction_scale=friction_scale,
        peg_tilt_x=peg_tilt_x,
        peg_tilt_y=peg_tilt_y,
        hole_yaw_offset=hole_yaw_offset,
    )


DEFAULT_ROBUSTNESS_PRESETS: dict[str, RobustnessPreset] = {
    "circle": RobustnessPreset(
        scene="circle",
        setting_id="circle_calibrated_v1",
        hole_xy_radius=0.0030,
        hole_yaw_max_deg=1.5,
        teleop_noise_xy_amplitude=0.0010,
        teleop_noise_cycles_min=0.5,
        teleop_noise_cycles_max=1.5,
        clearance_delta_min=-0.0008,
        clearance_delta_max=0.0003,
        friction_scale_min=0.45,
        friction_scale_max=1.70,
        peg_tilt_max_deg=1.5,
    ),
    "polygon_circle_logic_v1": RobustnessPreset(
        scene="polygon_circle_logic_v1",
        setting_id="polygon_circle_logic_v1",
        hole_xy_radius=0.0030,
        hole_yaw_max_deg=1.5,
        teleop_noise_xy_amplitude=0.0010,
        teleop_noise_cycles_min=0.5,
        teleop_noise_cycles_max=1.5,
        clearance_delta_min=-0.0008,
        clearance_delta_max=0.0003,
        friction_scale_min=0.45,
        friction_scale_max=1.70,
        peg_tilt_max_deg=1.5,
    ),
    "star_circle_logic_v1": RobustnessPreset(
        scene="star_circle_logic_v1",
        setting_id="star_circle_logic_v1",
        hole_xy_radius=0.0030,
        hole_yaw_max_deg=1.5,
        teleop_noise_xy_amplitude=0.0010,
        teleop_noise_cycles_min=0.5,
        teleop_noise_cycles_max=1.5,
        clearance_delta_min=-0.0008,
        clearance_delta_max=0.0003,
        friction_scale_min=0.45,
        friction_scale_max=1.70,
        peg_tilt_max_deg=1.5,
    ),
}


def get_robustness_preset(scene: str) -> RobustnessPreset:
    if scene not in scene_names():
        raise KeyError(f"Unknown scene {scene!r}.")
    return DEFAULT_ROBUSTNESS_PRESETS[scene]


def metadata_setting_id(values: dict[str, object], *, default: str = "unknown") -> str:
    return str(values.get("setting_id") or values.get("difficulty") or default)


def metadata_robustness_preset(values: dict[str, object]) -> RobustnessPreset:
    preset_values = values.get("robustness_preset") or values.get("difficulty_preset")
    if not isinstance(preset_values, dict):
        raise ValueError("Metadata must contain robustness_preset or legacy difficulty_preset.")
    return RobustnessPreset.from_metadata(preset_values)


def sample_robustness_perturbations(
    *,
    scene: SceneName | str,
    episodes: int,
    seed: int,
    preset: RobustnessPreset | None = None,
) -> tuple[RolloutPerturbation, ...]:
    if episodes <= 0:
        raise ValueError("episodes must be positive.")
    selected = preset or get_robustness_preset(str(scene))
    if selected.scene != str(scene):
        raise ValueError(f"Preset scene {selected.scene!r} does not match requested scene {scene!r}.")
    rng = np.random.default_rng(seed)
    perturbations: list[RolloutPerturbation] = []
    for _ in range(episodes):
        radius = float(rng.uniform(0.0, selected.hole_xy_radius))
        angle = float(rng.uniform(-np.pi, np.pi))
        tilt_radius = float(rng.uniform(0.0, np.deg2rad(selected.peg_tilt_max_deg)))
        tilt_angle = float(rng.uniform(-np.pi, np.pi))
        perturbations.append(
            RolloutPerturbation(
                hole_xy_offset=(radius * float(np.cos(angle)), radius * float(np.sin(angle))),
                hole_yaw_offset=float(
                    rng.uniform(-np.deg2rad(selected.hole_yaw_max_deg), np.deg2rad(selected.hole_yaw_max_deg))
                ),
                teleop_noise_xy_amplitude=float(selected.teleop_noise_xy_amplitude),
                teleop_noise_cycles=float(
                    rng.uniform(selected.teleop_noise_cycles_min, selected.teleop_noise_cycles_max)
                ),
                teleop_noise_phase_x=float(rng.uniform(-np.pi, np.pi)),
                teleop_noise_phase_y=float(rng.uniform(-np.pi, np.pi)),
                clearance_delta=float(rng.uniform(selected.clearance_delta_min, selected.clearance_delta_max)),
                friction_scale=float(rng.uniform(selected.friction_scale_min, selected.friction_scale_max)),
                peg_tilt_x=tilt_radius * float(np.cos(tilt_angle)),
                peg_tilt_y=tilt_radius * float(np.sin(tilt_angle)),
            )
        )
    return tuple(perturbations)


def sample_uniform_disk_offsets(*, episodes: int, seed: int, radius: float) -> tuple[tuple[float, float], ...]:
    if episodes <= 0:
        raise ValueError("episodes must be positive.")
    if radius < 0.0:
        raise ValueError("radius must be non-negative.")
    rng = np.random.default_rng(seed)
    offsets: list[tuple[float, float]] = []
    for _ in range(episodes):
        r = float(radius) * float(np.sqrt(rng.uniform(0.0, 1.0)))
        angle = float(rng.uniform(-np.pi, np.pi))
        offsets.append((r * float(np.cos(angle)), r * float(np.sin(angle))))
    return tuple(offsets)


def sample_controlled_contact_perturbations(
    *,
    episodes: int,
    seed: int,
    profile: ControlledContactProfile,
) -> tuple[RolloutPerturbation, ...]:
    hole_offsets = sample_uniform_disk_offsets(episodes=episodes, seed=seed, radius=profile.hole_xy_radius)
    return tuple(
        RolloutPerturbation(
            hole_xy_offset=offset,
            hole_yaw_offset=float(profile.hole_yaw_offset),
            teleop_noise_xy_amplitude=float(profile.teleop_noise_xy_amplitude),
            teleop_noise_cycles=float(profile.teleop_noise_cycles),
            teleop_noise_phase_x=float(profile.teleop_noise_phase_x),
            teleop_noise_phase_y=float(profile.teleop_noise_phase_y),
            clearance_delta=float(profile.clearance_delta),
            friction_scale=float(profile.friction_scale),
            peg_tilt_x=float(profile.peg_tilt_x),
            peg_tilt_y=float(profile.peg_tilt_y),
        )
        for offset in hole_offsets
    )


def apply_perturbation_to_scene_config(base_config: dict, perturbation: RolloutPerturbation | None) -> dict:
    config = copy.deepcopy(base_config)
    if perturbation is None:
        return config

    hole = config["hole"]
    hole_pos = list(hole["pos"])
    hole_pos[0] = float(hole_pos[0]) + float(perturbation.hole_xy_offset[0])
    hole_pos[1] = float(hole_pos[1]) + float(perturbation.hole_xy_offset[1])
    hole["pos"] = hole_pos
    hole["rotation"] = float(hole.get("rotation", 0.0)) + float(perturbation.hole_yaw_offset)

    clearance_delta = float(perturbation.clearance_delta)
    if abs(clearance_delta) > 0.0:
        if "wall_center_radius" in hole:
            hole["wall_center_radius"] = float(hole["wall_center_radius"]) + clearance_delta
        if "outer_radius" in hole:
            hole["outer_radius"] = float(hole["outer_radius"]) + clearance_delta
        if "inner_radius" in hole:
            hole["inner_radius"] = float(hole["inner_radius"]) + clearance_delta
        hole["completion_lateral_tolerance"] = max(
            0.0005,
            float(hole.get("completion_lateral_tolerance", 0.002)) + clearance_delta,
        )

    contact = config.setdefault("contact", {})
    friction = np.asarray(contact.get("friction", [0.9, 0.02, 0.002]), dtype=float)
    contact["friction"] = (friction * float(perturbation.friction_scale)).tolist()
    return config


__all__ = [
    "CONTROLLED_CONTACT_PROFILE_CANDIDATES",
    "DEFAULT_ROBUSTNESS_PRESETS",
    "ControlledContactProfile",
    "RobustnessPreset",
    "apply_perturbation_to_scene_config",
    "get_robustness_preset",
    "make_controlled_contact_profile",
    "metadata_robustness_preset",
    "metadata_setting_id",
    "sample_controlled_contact_perturbations",
    "sample_robustness_perturbations",
    "sample_uniform_disk_offsets",
]
