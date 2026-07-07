from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from stiffness_copilot_mujoco.sim.scene import (
    CANONICAL_EYE_IN_HAND_CAMERA_ATTACHMENT_PARENT,
    CANONICAL_EYE_IN_HAND_CAMERA_MOUNT_TYPE,
    CANONICAL_EYE_IN_HAND_CAMERA_POSE_VERSION,
    DEFAULT_EYE_IN_HAND_CAMERA_NAME,
    canonical_eye_in_hand_camera_pose,
)


VISION_INPUT_MODES = ("state_only", "image_only", "image_plus_state")
REQUIRED_RENDERER_MODE = "mujoco_native"
DEBUG_RENDERER_NOT_VALID_FOR_DINOV3_TRAINING = "debug_renderer_not_valid_for_dinov3_training"
MISSING_RGB_NOT_VALID_FOR_DINOV3_TRAINING = "rgb_images_missing_for_dinov3_training"


@dataclass(frozen=True)
class VisionResidualDataset:
    task_state: np.ndarray | None
    rgb_images: np.ndarray | None
    residual_group_target: np.ndarray
    episode_id: np.ndarray
    timestamp: np.ndarray
    sample_step: np.ndarray | None
    phase_id: np.ndarray | None
    trajectory_family_id: np.ndarray | None
    trajectory_parameters: np.ndarray | None
    contact_force_world: np.ndarray | None
    metadata: dict[str, Any]
    train_episode_ids: np.ndarray | None
    val_episode_ids: np.ndarray | None

    @property
    def sample_count(self) -> int:
        return int(self.residual_group_target.shape[0])

    @property
    def task_state_dim(self) -> int | None:
        if self.task_state is None:
            return None
        return int(self.task_state.shape[1])

    @property
    def image_shape(self) -> tuple[int, int, int] | None:
        if self.rgb_images is None:
            return None
        shape = self.rgb_images.shape
        return int(shape[1]), int(shape[2]), int(shape[3])


def _metadata_from_npz(data: np.lib.npyio.NpzFile) -> dict[str, Any]:
    if "metadata" not in data.files:
        return {}
    return json.loads(str(data["metadata"]))


def infer_training_data_valid(metadata: dict[str, Any], *, rgb_images_present: bool) -> tuple[bool, str | None]:
    renderer_mode = metadata.get("renderer_mode")
    fallback_used = bool(metadata.get("fallback_used", False))
    rgb_enabled = bool(metadata.get("rgb_enabled", rgb_images_present))
    if not rgb_enabled or not rgb_images_present:
        return False, MISSING_RGB_NOT_VALID_FOR_DINOV3_TRAINING
    if renderer_mode != REQUIRED_RENDERER_MODE or fallback_used:
        return False, DEBUG_RENDERER_NOT_VALID_FOR_DINOV3_TRAINING
    return True, None


def _validate_eye_in_hand_camera_metadata(metadata: dict[str, Any]) -> None:
    camera_pose = metadata.get("eye_in_hand_camera_pose")
    if not isinstance(camera_pose, dict):
        raise ValueError("Vision dataset metadata field 'eye_in_hand_camera_pose' is required for camera-safe training.")
    camera_name = str(
        metadata.get("eye_in_hand_camera_name")
        or camera_pose.get("camera_name")
        or metadata.get("rgb_camera_name")
        or DEFAULT_EYE_IN_HAND_CAMERA_NAME
    )
    expected_pose = canonical_eye_in_hand_camera_pose(camera_name)
    required_exact_values = {
        "eye_in_hand_camera_pose_version": CANONICAL_EYE_IN_HAND_CAMERA_POSE_VERSION,
        "eye_in_hand_camera_canonical": True,
        "eye_in_hand_camera_name": expected_pose["camera_name"],
        "eye_in_hand_camera_attachment_parent": CANONICAL_EYE_IN_HAND_CAMERA_ATTACHMENT_PARENT,
        "eye_in_hand_camera_mount_type": CANONICAL_EYE_IN_HAND_CAMERA_MOUNT_TYPE,
    }
    for key, expected in required_exact_values.items():
        observed = metadata.get(key)
        if observed != expected:
            raise ValueError(f"Vision dataset metadata field {key!r} must be {expected!r}, observed {observed!r}.")
    for key in ("camera_name", "attachment_parent", "mount_type"):
        observed = camera_pose.get(key)
        expected = expected_pose[key]
        if observed != expected:
            raise ValueError(
                f"Vision dataset metadata field 'eye_in_hand_camera_pose.{key}' must be {expected!r}, observed {observed!r}."
            )
    for key in ("pos", "forward", "up"):
        observed = np.asarray(camera_pose.get(key), dtype=float)
        expected = np.asarray(expected_pose[key], dtype=float)
        if observed.shape != expected.shape or not np.allclose(observed, expected, atol=1e-12, rtol=0.0):
            raise ValueError(
                f"Vision dataset metadata field 'eye_in_hand_camera_pose.{key}' must be {expected.tolist()!r}, "
                f"observed {observed.tolist()!r}."
            )
    if not np.isclose(float(camera_pose.get("fovy")), float(expected_pose["fovy"]), atol=1e-12, rtol=0.0):
        raise ValueError(
            f"Vision dataset metadata field 'eye_in_hand_camera_pose.fovy' must be {expected_pose['fovy']!r}, "
            f"observed {camera_pose.get('fovy')!r}."
        )


def load_vision_residual_dataset(
    path: Path,
    *,
    input_mode: str = "image_plus_state",
    require_native_renderer: bool = True,
) -> VisionResidualDataset:
    if input_mode not in VISION_INPUT_MODES:
        raise ValueError(f"Unsupported input_mode {input_mode!r}. Expected one of {VISION_INPUT_MODES}.")
    candidate = path
    if path.is_dir():
        if (path / "eligible_residual_bc.npz").exists():
            candidate = path / "eligible_residual_bc.npz"
        else:
            npz_files = sorted(path.glob("*.npz"))
            if len(npz_files) == 1:
                candidate = npz_files[0]
            else:
                raise FileNotFoundError(f"Could not locate a dataset npz under {path}.")
    with np.load(candidate, allow_pickle=False) as data:
        metadata = _metadata_from_npz(data)
        if input_mode != "state_only":
            _validate_eye_in_hand_camera_metadata(metadata)
        # Image-only inference must never consume privileged state features.
        task_state = data["task_state"].astype(np.float32) if input_mode != "image_only" else None
        rgb_images = data["rgb_images"].astype(np.uint8) if input_mode != "state_only" and "rgb_images" in data.files else None
        if input_mode != "state_only" and rgb_images is None:
            raise ValueError(f"{candidate} is missing rgb_images required for input_mode={input_mode!r}.")
        if task_state is None and input_mode != "image_only":
            raise ValueError(f"{candidate} is missing task_state required for input_mode={input_mode!r}.")
        if input_mode != "state_only" and require_native_renderer:
            renderer_mode = metadata.get("renderer_mode")
            fallback_used = bool(metadata.get("fallback_used", False))
            if renderer_mode != REQUIRED_RENDERER_MODE or fallback_used:
                raise ValueError(
                    f"{candidate} is not an active native-rendered vision dataset. "
                    f"Expected renderer_mode={REQUIRED_RENDERER_MODE!r} and fallback_used=False, "
                    f"observed renderer_mode={renderer_mode!r}, fallback_used={fallback_used!r}."
                )
        if input_mode != "state_only":
            training_data_valid = metadata.get("training_data_valid")
            training_data_valid_reason = metadata.get("training_data_valid_reason")
            inferred_valid, inferred_reason = infer_training_data_valid(metadata, rgb_images_present=rgb_images is not None)
            if training_data_valid is None:
                training_data_valid = inferred_valid
                training_data_valid_reason = inferred_reason
            if not bool(training_data_valid):
                reason = training_data_valid_reason or inferred_reason or DEBUG_RENDERER_NOT_VALID_FOR_DINOV3_TRAINING
                raise ValueError(
                    f"{candidate} is not valid for DINOv3 training. "
                    f"training_data_valid={training_data_valid!r}, reason={reason!r}."
                )
        residual_group_target = data["residual_group_target"].astype(np.float32)
        episode_id = data["episode_id"].astype(np.int64)
        timestamp = data["timestamp"].astype(np.float32) if "timestamp" in data.files else np.arange(residual_group_target.shape[0], dtype=np.float32)
        sample_step = data["sample_step"].astype(np.int32) if "sample_step" in data.files else None
        phase_id = data["phase_id"].astype(np.int8) if "phase_id" in data.files else None
        trajectory_family_id = data["trajectory_family_id"].astype(np.int8) if "trajectory_family_id" in data.files else None
        trajectory_parameters = data["trajectory_parameters"].astype(np.float32) if "trajectory_parameters" in data.files else None
        contact_force_world = data["contact_force_world"].astype(np.float32) if "contact_force_world" in data.files else None
        train_episode_ids = data["train_episode_ids"].astype(np.int64) if "train_episode_ids" in data.files else None
        val_episode_ids = data["val_episode_ids"].astype(np.int64) if "val_episode_ids" in data.files else None

    n = residual_group_target.shape[0]
    if task_state is not None and task_state.shape[0] != n:
        raise ValueError("task_state length does not match residual_group_target length.")
    if rgb_images is not None and rgb_images.shape[0] != n:
        raise ValueError("rgb_images length does not match residual_group_target length.")
    if episode_id.shape[0] != n or timestamp.shape[0] != n:
        raise ValueError("episode_id or timestamp length does not match residual_group_target length.")
    for name, array in (
        ("sample_step", sample_step),
        ("phase_id", phase_id),
        ("trajectory_family_id", trajectory_family_id),
        ("trajectory_parameters", trajectory_parameters),
        ("contact_force_world", contact_force_world),
    ):
        if array is not None and array.shape[0] != n:
            raise ValueError(f"{name} length does not match residual_group_target length.")
    if train_episode_ids is not None and val_episode_ids is not None:
        if np.intersect1d(train_episode_ids, val_episode_ids).size:
            raise ValueError("train and validation episode ids overlap.")

    return VisionResidualDataset(
        task_state=task_state,
        rgb_images=rgb_images,
        residual_group_target=residual_group_target,
        episode_id=episode_id,
        timestamp=timestamp,
        sample_step=sample_step,
        phase_id=phase_id,
        trajectory_family_id=trajectory_family_id,
        trajectory_parameters=trajectory_parameters,
        contact_force_world=contact_force_world,
        metadata=metadata,
        train_episode_ids=train_episode_ids,
        val_episode_ids=val_episode_ids,
    )


__all__ = ["VISION_INPUT_MODES", "VisionResidualDataset", "load_vision_residual_dataset"]
