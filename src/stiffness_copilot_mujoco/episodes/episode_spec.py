from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


from stiffness_copilot_mujoco.episodes.teleop_proxy import (
    TELEOP_MODE_POSITION_ONLY,
    TELEOP_MODE_POSITION_ORIENTATION,
    TELEOP_MODE_VALUES,
    validate_teleop_mode,
)
from stiffness_copilot_mujoco.runtime_defaults import default_native_launcher


EPISODE_SPEC_SCHEMA_VERSION = "episode_spec_v2"
EPISODE_TRAJECTORY_SOURCE_OPEN_LOOP_FAMILY = "open_loop_family_episode_spec"
EPISODE_TRAJECTORY_SOURCE_EPISODE_SPEC_REPLAY = "episode_spec_replay"
EPISODE_TRAJECTORY_SOURCE_FIXED_PHASE_LEGACY_DEBUG = "fixed_phase_schedule_legacy_debug"

PHASE_ID_TO_NAME = {
    0: "approach_hold",
    1: "descend",
    2: "insert",
    3: "final_hold",
}
PHASE_NAME_TO_ID = {value: key for key, value in PHASE_ID_TO_NAME.items()}


def _json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _stable_digest(payload: dict[str, object]) -> str:
    canonical = json.dumps(_json_ready(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _as_float_array(values: Any, *, ndim: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have ndim {ndim}, observed {array.ndim}.")
    return array


def _as_int_array(values: Any, *, ndim: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=int)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have ndim {ndim}, observed {array.ndim}.")
    return array


@dataclass(frozen=True)
class EpisodeSpec:
    episode_spec_id: str
    episode_id: int
    seed: int
    scene: str
    setting_id: str
    profile_name: str
    contact_condition_name: str | None
    nominal_hole_position: np.ndarray
    nominal_hole_xy: np.ndarray
    actual_hole_position: np.ndarray
    hole_xy_offset: np.ndarray
    hole_yaw_offset: float
    hole_xy_radius: float
    hole_xy_offset_semantics: str
    hole_xy_offset_distribution: str
    trajectory_follows_randomized_hole: bool
    contact_generation_parameters_fixed: bool
    fixed_contact_condition: dict[str, Any]
    trajectory_source: str
    trajectory_family: str
    trajectory_family_id: int
    trajectory_parameters: np.ndarray
    target_offsets: np.ndarray
    phase_ids: np.ndarray
    total_steps: int
    actual_hole_xy: np.ndarray
    trajectory_center_xy: np.ndarray
    trajectory_minus_hole_xy: np.ndarray
    sample_stride: int | None = None
    image_stride: int | None = None
    native_launcher_required: str = default_native_launcher()
    teleop_mode: str = TELEOP_MODE_POSITION_ONLY
    target_rotations: np.ndarray | None = None

    @property
    def phase_names(self) -> list[str]:
        return [PHASE_ID_TO_NAME.get(int(phase_id), "unknown") for phase_id in np.asarray(self.phase_ids, dtype=int)]

    def phase_name_at_step(self, step: int) -> str:
        if step < 0 or step >= int(self.phase_ids.shape[0]):
            raise IndexError(f"step {step} out of range for phase_ids length {self.phase_ids.shape[0]}.")
        return PHASE_ID_TO_NAME.get(int(self.phase_ids[step]), "unknown")

    def target_offset_at_step(self, step: int) -> np.ndarray:
        if step < 0 or step >= int(self.target_offsets.shape[0]):
            raise IndexError(f"step {step} out of range for target_offsets length {self.target_offsets.shape[0]}.")
        return np.asarray(self.target_offsets[step], dtype=float)

    def target_position_at_step(self, step: int, *, reference_position: np.ndarray | None = None) -> np.ndarray:
        reference = self.actual_hole_position if reference_position is None else np.asarray(reference_position, dtype=float)
        if reference.shape != (3,):
            raise ValueError(f"reference_position must have shape (3,), observed {reference.shape}.")
        return reference + self.target_offset_at_step(step)

    def target_rotation_at_step(self, step: int) -> np.ndarray:
        if self.target_rotations is None:
            raise ValueError(f"EpisodeSpec {self.episode_spec_id!r} does not define target_rotations.")
        if step < 0 or step >= int(self.target_rotations.shape[0]):
            raise IndexError(f"step {step} out of range for target_rotations length {self.target_rotations.shape[0]}.")
        rotation = np.asarray(self.target_rotations[step], dtype=float)
        if rotation.shape != (3, 3):
            raise ValueError(f"target_rotations[{step}] must have shape (3, 3), observed {rotation.shape}.")
        return rotation

    def to_perturbation_kwargs(self) -> dict[str, float | tuple[float, float]]:
        return {
            "hole_xy_offset": (float(self.hole_xy_offset[0]), float(self.hole_xy_offset[1])),
            "hole_yaw_offset": float(self.hole_yaw_offset),
            "teleop_noise_xy_amplitude": float(self.fixed_contact_condition["teleop_noise_xy_amplitude"]),
            "teleop_noise_cycles": float(self.fixed_contact_condition["teleop_noise_cycles"]),
            "teleop_noise_phase_x": float(self.fixed_contact_condition["teleop_noise_phase_x"]),
            "teleop_noise_phase_y": float(self.fixed_contact_condition["teleop_noise_phase_y"]),
            "clearance_delta": float(self.fixed_contact_condition["clearance_delta"]),
            "friction_scale": float(self.fixed_contact_condition["friction_scale"]),
            "peg_tilt_x": float(self.fixed_contact_condition["peg_tilt_x"]),
            "peg_tilt_y": float(self.fixed_contact_condition["peg_tilt_y"]),
        }

    def validate(self) -> None:
        teleop_mode = validate_teleop_mode(self.teleop_mode) if isinstance(self.teleop_mode, str) else self.teleop_mode
        if self.target_offsets.ndim != 2 or self.target_offsets.shape[1] != 3:
            raise ValueError(f"target_offsets must have shape [T, 3], observed {self.target_offsets.shape}.")
        if self.target_rotations is not None:
            if self.target_rotations.ndim != 3 or self.target_rotations.shape[1:] != (3, 3):
                raise ValueError(
                    f"target_rotations must have shape [T, 3, 3], observed {self.target_rotations.shape}."
                )
            if self.target_rotations.shape[0] != self.target_offsets.shape[0]:
                raise ValueError("target_rotations and target_offsets length mismatch.")
        if teleop_mode == TELEOP_MODE_POSITION_ORIENTATION and self.target_rotations is None:
            raise ValueError("teleop_mode position_orientation requires target_rotations.")
        if teleop_mode not in TELEOP_MODE_VALUES:
            raise ValueError(f"Unsupported teleop_mode {self.teleop_mode!r}.")
        if self.phase_ids.ndim != 1:
            raise ValueError(f"phase_ids must have shape [T], observed {self.phase_ids.shape}.")
        if self.target_offsets.shape[0] != self.phase_ids.shape[0]:
            raise ValueError("target_offsets and phase_ids length mismatch.")
        if int(self.total_steps) != int(self.target_offsets.shape[0] - 1):
            raise ValueError(
                f"total_steps {self.total_steps} does not match target_offsets length {self.target_offsets.shape[0]}."
            )
        if self.nominal_hole_position.shape != (3,):
            raise ValueError(
                f"nominal_hole_position must have shape (3,), observed {self.nominal_hole_position.shape}."
            )
        if self.nominal_hole_xy.shape != (2,):
            raise ValueError(f"nominal_hole_xy must have shape (2,), observed {self.nominal_hole_xy.shape}.")
        if self.actual_hole_position.shape != (3,):
            raise ValueError(
                f"actual_hole_position must have shape (3,), observed {self.actual_hole_position.shape}."
            )
        if self.hole_xy_offset.shape != (2,):
            raise ValueError(f"hole_xy_offset must have shape (2,), observed {self.hole_xy_offset.shape}.")
        if self.actual_hole_xy.shape != (2,):
            raise ValueError(f"actual_hole_xy must have shape (2,), observed {self.actual_hole_xy.shape}.")
        if self.trajectory_center_xy.shape != (2,):
            raise ValueError(f"trajectory_center_xy must have shape (2,), observed {self.trajectory_center_xy.shape}.")
        if self.trajectory_minus_hole_xy.shape != (2,):
            raise ValueError(
                f"trajectory_minus_hole_xy must have shape (2,), observed {self.trajectory_minus_hole_xy.shape}."
            )
        expected_actual_xy = np.asarray(self.nominal_hole_xy, dtype=float) + np.asarray(self.hole_xy_offset, dtype=float)
        expected_actual_position = np.asarray(self.nominal_hole_position, dtype=float).copy()
        expected_actual_position[:2] = expected_actual_xy
        if not np.allclose(expected_actual_xy, self.actual_hole_xy, atol=1e-9, rtol=0.0):
            raise ValueError("actual_hole_xy is inconsistent with nominal_hole_xy + hole_xy_offset.")
        if not np.allclose(expected_actual_position, self.actual_hole_position, atol=1e-9, rtol=0.0):
            raise ValueError("actual_hole_position is inconsistent with nominal_hole_position + hole_xy_offset.")
        if not np.allclose(self.actual_hole_xy, self.trajectory_center_xy, atol=1e-9, rtol=0.0):
            raise ValueError("trajectory_center_xy must match actual_hole_xy.")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        payload: dict[str, object] = {
            "episode_spec_id": self.episode_spec_id,
            "episode_id": int(self.episode_id),
            "seed": int(self.seed),
            "scene": self.scene,
            "setting_id": self.setting_id,
            "profile_name": self.profile_name,
            "contact_condition_name": self.contact_condition_name,
            "nominal_hole_position": self.nominal_hole_position.tolist(),
            "nominal_hole_xy": self.nominal_hole_xy.tolist(),
            "actual_hole_position": self.actual_hole_position.tolist(),
            "hole_xy_offset": self.hole_xy_offset.tolist(),
            "hole_yaw_offset": float(self.hole_yaw_offset),
            "hole_xy_radius": float(self.hole_xy_radius),
            "hole_xy_offset_semantics": self.hole_xy_offset_semantics,
            "hole_xy_offset_distribution": self.hole_xy_offset_distribution,
            "trajectory_follows_randomized_hole": bool(self.trajectory_follows_randomized_hole),
            "contact_generation_parameters_fixed": bool(self.contact_generation_parameters_fixed),
            "fixed_contact_condition": _json_ready(self.fixed_contact_condition),
            "trajectory_source": self.trajectory_source,
            "trajectory_family": self.trajectory_family,
            "trajectory_family_id": int(self.trajectory_family_id),
            "trajectory_parameters": self.trajectory_parameters.tolist(),
            "teleop_mode": self.teleop_mode,
            "target_rotations": None if self.target_rotations is None else self.target_rotations.tolist(),
            "target_offsets": self.target_offsets.tolist(),
            "phase_ids": self.phase_ids.tolist(),
            "total_steps": int(self.total_steps),
            "actual_hole_xy": self.actual_hole_xy.tolist(),
            "trajectory_center_xy": self.trajectory_center_xy.tolist(),
            "trajectory_minus_hole_xy": self.trajectory_minus_hole_xy.tolist(),
            "sample_stride": None if self.sample_stride is None else int(self.sample_stride),
            "image_stride": None if self.image_stride is None else int(self.image_stride),
            "native_launcher_required": self.native_launcher_required,
            "episode_spec_schema_version": EPISODE_SPEC_SCHEMA_VERSION,
        }
        return payload

    def to_json(self) -> str:
        return json.dumps(_json_ready(self.to_dict()), sort_keys=True)

    @classmethod
    def create(
        cls,
        *,
        episode_id: int,
        seed: int,
        scene: str,
        setting_id: str,
        profile_name: str,
        contact_condition_name: str | None,
        nominal_hole_position: np.ndarray,
        nominal_hole_xy: np.ndarray,
        hole_xy_offset: np.ndarray,
        hole_yaw_offset: float,
        hole_xy_radius: float,
        hole_xy_offset_semantics: str,
        hole_xy_offset_distribution: str,
        trajectory_follows_randomized_hole: bool,
        contact_generation_parameters_fixed: bool,
        fixed_contact_condition: dict[str, Any],
        trajectory_source: str,
        trajectory_family: str,
        trajectory_family_id: int,
        trajectory_parameters: np.ndarray,
        target_offsets: np.ndarray,
        phase_ids: np.ndarray,
        total_steps: int,
        sample_stride: int | None = None,
        image_stride: int | None = None,
        native_launcher_required: str | None = None,
        teleop_mode: str = TELEOP_MODE_POSITION_ONLY,
        target_rotations: np.ndarray | None = None,
    ) -> "EpisodeSpec":
        nominal_hole_position = _as_float_array(nominal_hole_position, ndim=1, name="nominal_hole_position")
        nominal_hole_xy = _as_float_array(nominal_hole_xy, ndim=1, name="nominal_hole_xy")
        hole_xy_offset = _as_float_array(hole_xy_offset, ndim=1, name="hole_xy_offset")
        trajectory_parameters = _as_float_array(trajectory_parameters, ndim=1, name="trajectory_parameters")
        target_rotations_array = (
            None if target_rotations is None else _as_float_array(target_rotations, ndim=3, name="target_rotations")
        )
        target_offsets = _as_float_array(target_offsets, ndim=2, name="target_offsets")
        phase_ids = _as_int_array(phase_ids, ndim=1, name="phase_ids")
        actual_hole_xy = nominal_hole_xy + hole_xy_offset
        actual_hole_position = nominal_hole_position.copy()
        actual_hole_position[:2] = actual_hole_xy
        trajectory_center_xy = actual_hole_xy.copy()
        trajectory_minus_hole_xy = trajectory_center_xy - actual_hole_xy
        candidate = cls(
            episode_spec_id="",
            episode_id=int(episode_id),
            seed=int(seed),
            scene=scene,
            setting_id=setting_id,
            profile_name=profile_name,
            contact_condition_name=contact_condition_name,
            nominal_hole_position=nominal_hole_position,
            nominal_hole_xy=nominal_hole_xy,
            actual_hole_position=actual_hole_position,
            hole_xy_offset=hole_xy_offset,
            hole_yaw_offset=float(hole_yaw_offset),
            hole_xy_radius=float(hole_xy_radius),
            hole_xy_offset_semantics=hole_xy_offset_semantics,
            hole_xy_offset_distribution=hole_xy_offset_distribution,
            trajectory_follows_randomized_hole=bool(trajectory_follows_randomized_hole),
            contact_generation_parameters_fixed=bool(contact_generation_parameters_fixed),
            fixed_contact_condition=dict(fixed_contact_condition),
            trajectory_source=trajectory_source,
            trajectory_family=trajectory_family,
            trajectory_family_id=int(trajectory_family_id),
            trajectory_parameters=trajectory_parameters,
            teleop_mode=teleop_mode,
            target_rotations=target_rotations_array,
            target_offsets=target_offsets,
            phase_ids=phase_ids,
            total_steps=int(total_steps),
            actual_hole_xy=actual_hole_xy,
            trajectory_center_xy=trajectory_center_xy,
            trajectory_minus_hole_xy=trajectory_minus_hole_xy,
            sample_stride=None if sample_stride is None else int(sample_stride),
            image_stride=None if image_stride is None else int(image_stride),
            native_launcher_required=default_native_launcher() if native_launcher_required is None else native_launcher_required,
        )
        spec_id = _stable_digest(candidate._id_payload())
        return dataclass_replace(candidate, episode_spec_id=spec_id)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EpisodeSpec":
        nominal_hole_position = _as_float_array(
            data.get("nominal_hole_position", data["nominal_hole_xy"]),
            ndim=1,
            name="nominal_hole_position",
        )
        nominal_hole_xy = _as_float_array(data["nominal_hole_xy"], ndim=1, name="nominal_hole_xy")
        hole_xy_offset = _as_float_array(data["hole_xy_offset"], ndim=1, name="hole_xy_offset")
        trajectory_parameters = _as_float_array(data["trajectory_parameters"], ndim=1, name="trajectory_parameters")
        target_offsets = _as_float_array(data["target_offsets"], ndim=2, name="target_offsets")
        phase_ids = _as_int_array(data["phase_ids"], ndim=1, name="phase_ids")
        target_rotations = None
        if data.get("target_rotations") is not None:
            target_rotations = _as_float_array(data["target_rotations"], ndim=3, name="target_rotations")
        actual_hole_xy = _as_float_array(data.get("actual_hole_xy", nominal_hole_xy + hole_xy_offset), ndim=1, name="actual_hole_xy")
        actual_hole_position = _as_float_array(
            data.get(
                "actual_hole_position",
                np.array([actual_hole_xy[0], actual_hole_xy[1], nominal_hole_position[2]], dtype=float),
            ),
            ndim=1,
            name="actual_hole_position",
        )
        trajectory_center_xy = _as_float_array(
            data.get("trajectory_center_xy", actual_hole_xy), ndim=1, name="trajectory_center_xy"
        )
        trajectory_minus_hole_xy = _as_float_array(
            data.get("trajectory_minus_hole_xy", trajectory_center_xy - actual_hole_xy),
            ndim=1,
            name="trajectory_minus_hole_xy",
        )
        spec = cls(
            episode_spec_id=str(data["episode_spec_id"]),
            episode_id=int(data["episode_id"]),
            seed=int(data["seed"]),
            scene=str(data["scene"]),
            setting_id=str(data["setting_id"]),
            profile_name=str(data["profile_name"]),
            contact_condition_name=data.get("contact_condition_name"),
            nominal_hole_position=nominal_hole_position,
            nominal_hole_xy=nominal_hole_xy,
            actual_hole_position=actual_hole_position,
            hole_xy_offset=hole_xy_offset,
            hole_yaw_offset=float(data["hole_yaw_offset"]),
            hole_xy_radius=float(data["hole_xy_radius"]),
            hole_xy_offset_semantics=str(data["hole_xy_offset_semantics"]),
            hole_xy_offset_distribution=str(data["hole_xy_offset_distribution"]),
            trajectory_follows_randomized_hole=bool(data["trajectory_follows_randomized_hole"]),
            contact_generation_parameters_fixed=bool(data["contact_generation_parameters_fixed"]),
            fixed_contact_condition=dict(data["fixed_contact_condition"]),
            trajectory_source=str(data["trajectory_source"]),
            trajectory_family=str(data["trajectory_family"]),
            trajectory_family_id=int(data["trajectory_family_id"]),
            trajectory_parameters=trajectory_parameters,
            teleop_mode=TELEOP_MODE_POSITION_ONLY
            if data.get("teleop_mode") is None
            else str(data.get("teleop_mode", TELEOP_MODE_POSITION_ONLY)),
            target_rotations=target_rotations,
            target_offsets=target_offsets,
            phase_ids=phase_ids,
            total_steps=int(data["total_steps"]),
            actual_hole_xy=actual_hole_xy,
            trajectory_center_xy=trajectory_center_xy,
            trajectory_minus_hole_xy=trajectory_minus_hole_xy,
            sample_stride=None if data.get("sample_stride") is None else int(data["sample_stride"]),
            image_stride=None if data.get("image_stride") is None else int(data["image_stride"]),
            native_launcher_required=str(data.get("native_launcher_required", default_native_launcher())),
        )
        spec.validate()
        expected_id = _stable_digest(spec._id_payload(schema_version=str(data.get("episode_spec_schema_version", EPISODE_SPEC_SCHEMA_VERSION))))
        if spec.episode_spec_id != expected_id:
            raise ValueError(
                f"EpisodeSpec id mismatch: stored {spec.episode_spec_id!r}, expected {expected_id!r}."
            )
        return spec

    def _id_payload(self, *, schema_version: str | None = None) -> dict[str, object]:
        payload = {
            "episode_id": int(self.episode_id),
            "seed": int(self.seed),
            "scene": self.scene,
            "setting_id": self.setting_id,
            "profile_name": self.profile_name,
            "contact_condition_name": self.contact_condition_name,
            "nominal_hole_position": self.nominal_hole_position.tolist(),
            "nominal_hole_xy": self.nominal_hole_xy.tolist(),
            "actual_hole_position": self.actual_hole_position.tolist(),
            "hole_xy_offset": self.hole_xy_offset.tolist(),
            "hole_yaw_offset": float(self.hole_yaw_offset),
            "hole_xy_radius": float(self.hole_xy_radius),
            "hole_xy_offset_semantics": self.hole_xy_offset_semantics,
            "hole_xy_offset_distribution": self.hole_xy_offset_distribution,
            "trajectory_follows_randomized_hole": bool(self.trajectory_follows_randomized_hole),
            "contact_generation_parameters_fixed": bool(self.contact_generation_parameters_fixed),
            "fixed_contact_condition": _json_ready(self.fixed_contact_condition),
            "trajectory_source": self.trajectory_source,
            "trajectory_family": self.trajectory_family,
            "trajectory_family_id": int(self.trajectory_family_id),
            "trajectory_parameters": self.trajectory_parameters.tolist(),
        }
        version = EPISODE_SPEC_SCHEMA_VERSION if schema_version is None else str(schema_version)
        if version != "episode_spec_v1":
            payload["teleop_mode"] = self.teleop_mode
            payload["target_rotations"] = None if self.target_rotations is None else self.target_rotations.tolist()
        payload.update(
            {
                "target_offsets": self.target_offsets.tolist(),
                "phase_ids": self.phase_ids.tolist(),
                "total_steps": int(self.total_steps),
                "actual_hole_xy": self.actual_hole_xy.tolist(),
                "trajectory_center_xy": self.trajectory_center_xy.tolist(),
                "trajectory_minus_hole_xy": self.trajectory_minus_hole_xy.tolist(),
                "sample_stride": None if self.sample_stride is None else int(self.sample_stride),
                "image_stride": None if self.image_stride is None else int(self.image_stride),
                "native_launcher_required": self.native_launcher_required,
                "episode_spec_schema_version": version,
            }
        )
        return payload


def dataclass_replace(spec: EpisodeSpec, **changes: object) -> EpisodeSpec:
    payload = spec.__dict__.copy()
    payload.update(changes)
    return EpisodeSpec(**payload)


def write_episode_specs_jsonl(path: Path, specs: Iterable[EpisodeSpec]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for spec in specs:
            handle.write(json.dumps(_json_ready(spec.to_dict()), sort_keys=True) + "\n")


def load_episode_specs_jsonl(path: Path) -> list[EpisodeSpec]:
    with path.open("r", encoding="utf-8") as handle:
        return [EpisodeSpec.from_dict(json.loads(line)) for line in handle if line.strip()]


def select_episode_spec(
    specs: Iterable[EpisodeSpec],
    *,
    episode_spec_id: str | None = None,
    episode_id: int | None = None,
) -> EpisodeSpec:
    spec_list = list(specs)
    if episode_spec_id is not None:
        matches = [spec for spec in spec_list if spec.episode_spec_id == episode_spec_id]
        if not matches:
            raise KeyError(f"EpisodeSpec id {episode_spec_id!r} was not found.")
        if len(matches) > 1:
            raise ValueError(f"EpisodeSpec id {episode_spec_id!r} is ambiguous.")
        return matches[0]
    if episode_id is not None:
        matches = [spec for spec in spec_list if spec.episode_id == episode_id]
        if not matches:
            raise KeyError(f"EpisodeSpec episode_id {episode_id!r} was not found.")
        if len(matches) > 1:
            raise ValueError(f"EpisodeSpec episode_id {episode_id!r} is ambiguous.")
        return matches[0]
    if not spec_list:
        raise ValueError("No EpisodeSpec entries available.")
    return spec_list[0]
