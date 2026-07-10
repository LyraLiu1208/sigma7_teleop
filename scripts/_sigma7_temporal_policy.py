from __future__ import annotations

import json
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
LOCAL_SRC_ROOT = ROOT / "src"
LEGACY_MUJOCO_ROOT = ROOT.parent / "stiffness_copilot_mujoco"
LEGACY_SRC_ROOT = LEGACY_MUJOCO_ROOT / "src"

for candidate in (LOCAL_SRC_ROOT, LEGACY_SRC_ROOT):
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from stiffness_copilot_mujoco.learning.frozen_vision_backbone import (  # noqa: E402
    FrozenBackboneLoadConfig,
    FrozenBackbonePreprocessConfig,
    build_frozen_dinov3_backbone,
)
from stiffness_copilot_mujoco.learning.residual_stiffness import BaseStiffnessSpec  # noqa: E402
from stiffness_copilot_mujoco.learning.vision_residual_stiffness import (  # noqa: E402
    DEFAULT_HEAD_ACTIVATION,
    DEFAULT_HEAD_DROPOUT,
    DEFAULT_HEAD_HIDDEN_DIMS,
    DEFAULT_HEAD_OUTPUT_BOUNDING,
    IMAGE_ONLY_RESIDUAL_BC_METHOD_NAME,
    LegacyVisionResidualHeadV1,
    LINEAR_OR_MLP_V1_HEAD_TYPE,
    MLP_V2_HEAD_TYPE,
    VisionResidualHeadV2,
    VisionResidualBCPolicy,
    _bounded_residual,
    _npz_arrays_to_state_dict,
    _state_dict_to_npz_arrays,
    describe_residual_policy_contract,
    load_image_only_residual_bc_policy,
)
from stiffness_copilot_mujoco.sim.scene import (  # noqa: E402
    CANONICAL_EYE_IN_HAND_CAMERA_ATTACHMENT_PARENT,
    CANONICAL_EYE_IN_HAND_CAMERA_MOUNT_TYPE,
    CANONICAL_EYE_IN_HAND_CAMERA_POSE_VERSION,
    canonical_eye_in_hand_camera_pose,
)
from stiffness_copilot_mujoco.learning.stiffness_labels import cholesky_params_to_matrix, spd_project  # noqa: E402


TEMPORAL_VISION_POLICY_SCHEMA_VERSION = "sigma7_temporal_vision_residual_bc_policy_v1"
TEMPORAL_IMAGE_ONLY_METHOD_NAME = "image_only_temporal_residual_bc"
TEMPORAL_INPUT_MODE = "image_only_temporal_feature_concat"
TEMPORAL_CONTEXT_MODE = "feature_concat_previous_frames"
TEMPORAL_CONTEXT_PADDING = "repeat_first_feature"
TEMPORAL_RUNTIME_KIND = "temporal_feature_concat"
SINGLE_FRAME_RUNTIME_KIND = "single_frame"


def _json_ready(value: Any) -> Any:
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


def read_policy_metadata(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        return json.loads(str(data["metadata"]))


def is_temporal_policy_metadata(metadata: dict[str, Any]) -> bool:
    return str(metadata.get("schema_version", "")) == TEMPORAL_VISION_POLICY_SCHEMA_VERSION


def is_temporal_policy_path(path: Path) -> bool:
    return is_temporal_policy_metadata(read_policy_metadata(path))


def build_history_concat_features(
    *,
    features: np.ndarray,
    episode_ids: np.ndarray,
    sample_step: np.ndarray | None,
    history_steps: int,
) -> np.ndarray:
    if history_steps < 1:
        raise ValueError("history_steps must be at least 1.")
    features = np.asarray(features, dtype=np.float32)
    n, feature_dim = features.shape
    if history_steps == 1:
        return features.copy()

    output = np.zeros((n, feature_dim * history_steps), dtype=np.float32)
    episode_ids = np.asarray(episode_ids, dtype=np.int64)
    order_key = np.arange(n, dtype=np.int64) if sample_step is None else np.asarray(sample_step, dtype=np.int64)
    for episode_id in np.unique(episode_ids):
        idx = np.flatnonzero(episode_ids == episode_id)
        order = idx[np.argsort(order_key[idx], kind="stable")]
        episode_features = features[order]
        window_parts: list[np.ndarray] = []
        for lag in range(history_steps):
            shifted = np.empty_like(episode_features)
            if lag == 0:
                shifted[:] = episode_features
            else:
                shifted[:lag] = episode_features[0]
                shifted[lag:] = episode_features[:-lag]
            window_parts.append(shifted)
        output[order] = np.concatenate(window_parts, axis=1)
    return output


def encode_rgb_batches(
    encoder: Any,
    rgb_images: np.ndarray,
    *,
    batch_size: int = 64,
    progress_prefix: str | None = None,
) -> np.ndarray:
    images = np.asarray(rgb_images, dtype=np.uint8)
    features: list[np.ndarray] = []
    total_batches = max(1, int(np.ceil(images.shape[0] / batch_size)))
    for batch_index, start in enumerate(range(0, images.shape[0], batch_size), start=1):
        stop = min(start + batch_size, images.shape[0])
        if progress_prefix:
            print(f"{progress_prefix} encode_batch={batch_index}/{total_batches} samples={start}:{stop}", flush=True)
        encoded = encoder.encode(images[start:stop])
        if isinstance(encoded, torch.Tensor):
            encoded = encoded.detach().cpu().numpy()
        features.append(np.asarray(encoded, dtype=np.float32))
    return np.concatenate(features, axis=0)


def _build_head(
    *,
    head_type: str,
    feature_dim: int,
    output_dim: int,
    head_hidden_dims: tuple[int, ...],
    head_dropout: float,
) -> torch.nn.Module:
    if head_type == MLP_V2_HEAD_TYPE:
        return VisionResidualHeadV2(
            feature_dim=feature_dim,
            output_dim=output_dim,
            hidden_dims=head_hidden_dims,
            dropout=head_dropout,
        )
    if head_type == LINEAR_OR_MLP_V1_HEAD_TYPE:
        if len(head_hidden_dims) != 1:
            raise ValueError("linear_or_mlp_v1 temporal policies must provide exactly one hidden dim.")
        return LegacyVisionResidualHeadV1(feature_dim, head_hidden_dims[0], output_dim)
    raise ValueError(f"Unsupported temporal head_type {head_type!r}.")


def _load_frozen_backbone_from_metadata(metadata: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    preprocess_config = metadata.get("preprocessing_config") or {}
    if not isinstance(preprocess_config, dict):
        raise ValueError("Temporal policy metadata field 'preprocessing_config' must be a mapping.")
    load_config = FrozenBackboneLoadConfig(
        dinov3_repo=Path(str(metadata["dinov3_repo"])),
        dinov3_checkpoint=Path(str(metadata["dinov3_checkpoint"])),
        dinov3_entrypoint=str(metadata["dinov3_entrypoint"]),
        preprocess_config=FrozenBackbonePreprocessConfig(
            resize_height=int(preprocess_config.get("resize_height", 224)),
            resize_width=int(preprocess_config.get("resize_width", 224)),
            mean=tuple(float(value) for value in preprocess_config.get("mean", (0.485, 0.456, 0.406))),
            std=tuple(float(value) for value in preprocess_config.get("std", (0.229, 0.224, 0.225))),
            interpolation=str(preprocess_config.get("interpolation", "bilinear")),
            antialias=bool(preprocess_config.get("antialias", True)),
            scale_to_unit_interval=bool(preprocess_config.get("scale_to_unit_interval", True)),
        ),
        seed=int(metadata.get("backbone_seed", metadata.get("seed", 0))),
    )
    return build_frozen_dinov3_backbone(load_config)


@dataclass(frozen=True)
class TemporalVisionResidualBCPolicy:
    encoder: Any
    head: torch.nn.Module
    head_type: str
    head_hidden_dims: tuple[int, ...]
    head_dropout: float
    head_activation: str
    output_bounding: str
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    base_spec: BaseStiffnessSpec
    metadata: dict[str, Any]
    history_steps: int
    feature_dim_per_frame: int

    @classmethod
    def load(cls, path: Path) -> "TemporalVisionResidualBCPolicy":
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            validate_temporal_track_a_policy_metadata(metadata)
            encoder, backbone_metadata = _load_frozen_backbone_from_metadata(metadata)
            metadata = dict(metadata)
            metadata.setdefault("backbone_metadata", backbone_metadata)
            head_type = str(metadata.get("head_type") or MLP_V2_HEAD_TYPE)
            feature_dim = int(metadata["feature_dim"])
            output_dim = int(metadata["output_dim"])
            head_hidden_dims = tuple(int(value) for value in metadata.get("head_hidden_dims", ()))
            if not head_hidden_dims:
                if head_type == LINEAR_OR_MLP_V1_HEAD_TYPE:
                    head_hidden_dims = (int(metadata.get("hidden_dim", 32)),)
                else:
                    head_hidden_dims = DEFAULT_HEAD_HIDDEN_DIMS
            head_dropout = float(metadata.get("head_dropout", 0.0))
            head = _build_head(
                head_type=head_type,
                feature_dim=feature_dim,
                output_dim=output_dim,
                head_hidden_dims=head_hidden_dims,
                head_dropout=head_dropout,
            )
            state_dict = _npz_arrays_to_state_dict("head_state__", data)
            if not state_dict:
                raise ValueError("Temporal policy archive is missing head_state__ tensors.")
            head.load_state_dict(state_dict, strict=True)
            head.eval()
            history_steps = int(metadata["history_steps"])
            feature_dim_per_frame = int(metadata["feature_dim_per_frame"])
            if feature_dim != feature_dim_per_frame * history_steps:
                raise ValueError(
                    "Temporal policy metadata is inconsistent: feature_dim must equal "
                    f"feature_dim_per_frame * history_steps, observed {feature_dim} vs "
                    f"{feature_dim_per_frame} * {history_steps}."
                )
            return cls(
                encoder=encoder,
                head=head,
                head_type=head_type,
                head_hidden_dims=head_hidden_dims,
                head_dropout=head_dropout,
                head_activation=str(metadata.get("head_activation", DEFAULT_HEAD_ACTIVATION)),
                output_bounding=str(metadata.get("output_bounding", DEFAULT_HEAD_OUTPUT_BOUNDING)),
                x_mean=np.asarray(data["x_mean"], dtype=np.float32),
                x_std=np.asarray(data["x_std"], dtype=np.float32),
                y_mean=np.asarray(data["y_mean"], dtype=np.float32),
                y_std=np.asarray(data["y_std"], dtype=np.float32),
                base_spec=BaseStiffnessSpec.from_metadata(metadata["base_stiffness_spec"]),
                metadata=metadata,
                history_steps=history_steps,
                feature_dim_per_frame=feature_dim_per_frame,
            )

    @property
    def input_dim(self) -> int:
        return int(self.x_mean.shape[0])

    @property
    def output_dim(self) -> int:
        return int(self.y_mean.shape[0])

    def predict_feature_window(
        self,
        concat_feature: np.ndarray,
        *,
        residual_scale: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x = np.asarray(concat_feature, dtype=np.float32).reshape(-1)
        if x.shape != self.x_mean.shape:
            raise ValueError(f"Temporal feature window must have shape {self.x_mean.shape}, observed {x.shape}.")
        x_norm = (x - self.x_mean) / self.x_std
        with torch.no_grad():
            raw = self.head(torch.as_tensor(x_norm, dtype=torch.float32)).detach().cpu().numpy()
        raw = np.asarray(raw, dtype=np.float32) * self.y_std + self.y_mean
        scaled = raw * float(residual_scale)
        bounded = _bounded_residual(scaled, self.base_spec.residual_bounds)
        theta_delta = self.base_spec.expand_group_delta(bounded, clip=True)
        theta = self.base_spec.theta_base + theta_delta
        matrix = spd_project(cholesky_params_to_matrix(theta))
        return raw, bounded, matrix, theta, theta_delta


class SingleFrameResidualPolicyRuntime:
    def __init__(self, policy: VisionResidualBCPolicy) -> None:
        self._policy = policy
        self.metadata = policy.metadata
        self.base_spec = policy.base_spec
        self.policy_kind = SINGLE_FRAME_RUNTIME_KIND

    def reset(self) -> None:
        return None

    def predict_image_only(
        self,
        rgb_image: np.ndarray,
        *,
        residual_scale: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self._policy.predict_image_only(rgb_image, residual_scale=residual_scale)


class TemporalFeatureConcatPolicyRuntime:
    def __init__(self, policy: TemporalVisionResidualBCPolicy) -> None:
        self._policy = policy
        self.metadata = policy.metadata
        self.base_spec = policy.base_spec
        self.policy_kind = TEMPORAL_RUNTIME_KIND
        self._feature_buffer: deque[np.ndarray] = deque(maxlen=int(policy.history_steps))

    def reset(self) -> None:
        self._feature_buffer.clear()

    def _encode_single_feature(self, rgb_image: np.ndarray) -> np.ndarray:
        encoded = self._policy.encoder.encode(np.asarray(rgb_image, dtype=np.uint8))
        if isinstance(encoded, torch.Tensor):
            encoded = encoded.detach().cpu().numpy()
        feature = np.asarray(encoded, dtype=np.float32)
        if feature.ndim == 2:
            if feature.shape[0] != 1:
                raise ValueError(f"Expected a single encoded feature vector, observed {feature.shape}.")
            feature = feature[0]
        if feature.shape != (self._policy.feature_dim_per_frame,):
            raise ValueError(
                f"Temporal encoder feature must have shape {(self._policy.feature_dim_per_frame,)}, observed {feature.shape}."
            )
        return feature

    def predict_image_only(
        self,
        rgb_image: np.ndarray,
        *,
        residual_scale: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        feature = self._encode_single_feature(rgb_image)
        if not self._feature_buffer:
            for _ in range(int(self._policy.history_steps)):
                self._feature_buffer.append(feature.copy())
        else:
            self._feature_buffer.append(feature.copy())
        newest_to_oldest = list(reversed(self._feature_buffer))
        concat_feature = np.concatenate(newest_to_oldest, axis=0)
        return self._policy.predict_feature_window(concat_feature, residual_scale=residual_scale)


def load_sigma7_residual_policy_runtime(path: Path) -> SingleFrameResidualPolicyRuntime | TemporalFeatureConcatPolicyRuntime:
    metadata = read_policy_metadata(path)
    if is_temporal_policy_metadata(metadata):
        return TemporalFeatureConcatPolicyRuntime(TemporalVisionResidualBCPolicy.load(path))
    return SingleFrameResidualPolicyRuntime(load_image_only_residual_bc_policy(path))


def validate_temporal_track_a_policy_metadata(metadata: dict[str, Any]) -> list[str]:
    hard_failures: list[str] = []
    required_exact_values = {
        "schema_version": TEMPORAL_VISION_POLICY_SCHEMA_VERSION,
        "method_name": TEMPORAL_IMAGE_ONLY_METHOD_NAME,
        "input_mode": TEMPORAL_INPUT_MODE,
        "uses_task_state_input": False,
        "uses_contact_force_input": False,
        "uses_clearance_input": False,
        "uses_trajectory_phase_input": False,
        "is_residual_policy": True,
        "is_full_stiffness_policy": False,
        "renderer_mode": "mujoco_native",
        "fallback_used": False,
        "eye_in_hand_camera_pose_version": CANONICAL_EYE_IN_HAND_CAMERA_POSE_VERSION,
        "eye_in_hand_camera_canonical": True,
        "eye_in_hand_camera_name": "eye_in_hand_rgb",
        "eye_in_hand_camera_attachment_parent": CANONICAL_EYE_IN_HAND_CAMERA_ATTACHMENT_PARENT,
        "eye_in_hand_camera_mount_type": CANONICAL_EYE_IN_HAND_CAMERA_MOUNT_TYPE,
        "stiffness_representation": "full_spd_matrix",
        "smoothing_required": True,
        "temporal_context_mode": TEMPORAL_CONTEXT_MODE,
        "temporal_context_padding": TEMPORAL_CONTEXT_PADDING,
    }
    for key, expected in required_exact_values.items():
        observed = metadata.get(key)
        if observed != expected:
            hard_failures.append(f"policy metadata field {key!r} must be {expected!r}, observed {observed!r}")
    for key in ("controllers_yaml", "reference_controller_id", "reference_stiffness_matrix", "collection_controller_id", "collection_stiffness_matrix"):
        if key not in metadata:
            hard_failures.append(f"policy metadata field {key!r} is required")
    try:
        history_steps = int(metadata.get("history_steps", 0))
    except Exception:
        history_steps = 0
    if history_steps < 1:
        hard_failures.append(f"policy metadata field 'history_steps' must be >= 1, observed {metadata.get('history_steps')!r}")
    try:
        feature_dim_per_frame = int(metadata.get("feature_dim_per_frame", 0))
        feature_dim = int(metadata.get("feature_dim", 0))
    except Exception:
        feature_dim_per_frame = 0
        feature_dim = 0
    if feature_dim_per_frame <= 0:
        hard_failures.append(
            f"policy metadata field 'feature_dim_per_frame' must be positive, observed {metadata.get('feature_dim_per_frame')!r}"
        )
    if history_steps >= 1 and feature_dim_per_frame > 0 and feature_dim != history_steps * feature_dim_per_frame:
        hard_failures.append(
            "policy metadata fields 'feature_dim', 'feature_dim_per_frame', and 'history_steps' are inconsistent: "
            f"{feature_dim} != {history_steps} * {feature_dim_per_frame}"
        )
    base_spec = metadata.get("base_stiffness_spec")
    if not isinstance(base_spec, dict):
        hard_failures.append("policy metadata field 'base_stiffness_spec' must be a mapping")
    else:
        try:
            base_spec_obj = BaseStiffnessSpec.from_metadata(base_spec)
            expected_contract = describe_residual_policy_contract(base_spec_obj)
        except Exception as exc:
            hard_failures.append(f"policy metadata field 'base_stiffness_spec' is invalid: {exc}")
            expected_contract = None
        if expected_contract is not None:
            for key, expected in {
                "output_space": expected_contract["output_space"],
                "output_dim": expected_contract["output_dim"],
                "residual_parameterization": expected_contract["residual_parameterization"],
                "residual_affects": expected_contract["residual_affects"],
                "residual_unaffected": expected_contract["residual_unaffected"],
            }.items():
                observed = metadata.get(key)
                if observed != expected:
                    hard_failures.append(f"policy metadata field {key!r} must be {expected!r}, observed {observed!r}")
    reference_matrix = np.asarray(metadata.get("reference_stiffness_matrix"), dtype=float)
    collection_matrix = np.asarray(metadata.get("collection_stiffness_matrix"), dtype=float)
    if reference_matrix.shape != (3, 3):
        hard_failures.append(
            f"policy metadata field 'reference_stiffness_matrix' must have shape (3, 3), observed {reference_matrix.shape}"
        )
    if collection_matrix.shape != (3, 3):
        hard_failures.append(
            f"policy metadata field 'collection_stiffness_matrix' must have shape (3, 3), observed {collection_matrix.shape}"
        )
    if isinstance(base_spec, dict) and reference_matrix.shape == (3, 3):
        base_matrix = np.asarray(base_spec.get("base_matrix"), dtype=float)
        if base_matrix.shape == (3, 3) and not np.allclose(base_matrix, reference_matrix, atol=1e-9, rtol=0.0):
            hard_failures.append("reference_stiffness_matrix must match base_stiffness_spec.base_matrix")
    if reference_matrix.shape == (3, 3) and collection_matrix.shape == (3, 3):
        if not np.allclose(reference_matrix, collection_matrix, atol=1e-9, rtol=0.0):
            hard_failures.append("collection_stiffness_matrix must match reference_stiffness_matrix")
    residual_bound = float(metadata.get("residual_bound", 0.0) or 0.0)
    if not np.isclose(residual_bound, 0.35, atol=1e-12, rtol=0.0):
        hard_failures.append(f"policy metadata field 'residual_bound' must be 0.35, observed {metadata.get('residual_bound')!r}")
    pose = metadata.get("eye_in_hand_camera_pose")
    if not isinstance(pose, dict):
        hard_failures.append("policy metadata field 'eye_in_hand_camera_pose' must be a mapping")
    else:
        expected_pose = canonical_eye_in_hand_camera_pose(str(metadata.get("eye_in_hand_camera_name") or "eye_in_hand_rgb"))
        for key in ("camera_name", "attachment_parent", "mount_type"):
            if pose.get(key) != expected_pose[key]:
                hard_failures.append(
                    f"policy metadata field 'eye_in_hand_camera_pose.{key}' must be {expected_pose[key]!r}, observed {pose.get(key)!r}"
                )
        for key in ("pos", "forward", "up"):
            observed = np.asarray(pose.get(key), dtype=float)
            expected = np.asarray(expected_pose[key], dtype=float)
            if observed.shape != expected.shape or not np.allclose(observed, expected, atol=1e-12, rtol=0.0):
                hard_failures.append(
                    f"policy metadata field 'eye_in_hand_camera_pose.{key}' must be {expected.tolist()!r}, "
                    f"observed {observed.tolist()!r}"
                )
        if not np.isclose(float(pose.get("fovy", 0.0)), float(expected_pose["fovy"]), atol=1e-12, rtol=0.0):
            hard_failures.append(
                f"policy metadata field 'eye_in_hand_camera_pose.fovy' must be {expected_pose['fovy']!r}, observed {pose.get('fovy')!r}"
            )
    return hard_failures


def validate_sigma7_track_a_residual_policy_metadata(metadata: dict[str, Any]) -> list[str]:
    if is_temporal_policy_metadata(metadata):
        return validate_temporal_track_a_policy_metadata(metadata)
    from stiffness_copilot_mujoco.evaluation.track_a_episode_runner import validate_track_a_v2_policy_metadata  # noqa: E402

    return validate_track_a_v2_policy_metadata(metadata)


def save_temporal_image_only_residual_bc_policy(
    path: Path,
    *,
    encoder: Any,
    head: torch.nn.Module,
    head_type: str,
    head_hidden_dims: tuple[int, ...] | list[int],
    head_dropout: float,
    head_activation: str,
    output_bounding: str,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    base_spec: BaseStiffnessSpec,
    metadata: dict[str, Any],
) -> None:
    full_metadata = dict(metadata)
    full_metadata["schema_version"] = TEMPORAL_VISION_POLICY_SCHEMA_VERSION
    full_metadata["method_name"] = TEMPORAL_IMAGE_ONLY_METHOD_NAME
    full_metadata["input_mode"] = TEMPORAL_INPUT_MODE
    full_metadata["base_stiffness_spec"] = base_spec.to_metadata()
    if encoder is not None and hasattr(encoder, "to_metadata"):
        full_metadata.update(encoder.to_metadata())
    full_metadata["head_type"] = str(head_type)
    full_metadata["head_hidden_dims"] = [int(value) for value in head_hidden_dims]
    full_metadata["head_dropout"] = float(head_dropout)
    full_metadata["head_activation"] = str(head_activation)
    full_metadata["output_bounding"] = str(output_bounding)
    full_metadata["feature_dim"] = int(np.asarray(x_mean).shape[0])
    full_metadata["output_dim"] = int(np.asarray(y_mean).shape[0])
    validate_failures = validate_temporal_track_a_policy_metadata(full_metadata)
    if validate_failures:
        raise ValueError("Temporal policy metadata validation failed: " + "; ".join(validate_failures))
    arrays: dict[str, np.ndarray] = {
        "x_mean": np.asarray(x_mean, dtype=np.float32),
        "x_std": np.asarray(x_std, dtype=np.float32),
        "y_mean": np.asarray(y_mean, dtype=np.float32),
        "y_std": np.asarray(y_std, dtype=np.float32),
        "metadata": np.asarray(json.dumps(_json_ready(full_metadata), sort_keys=True)),
    }
    arrays.update(_state_dict_to_npz_arrays("head_state__", head.state_dict()))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


__all__ = [
    "SINGLE_FRAME_RUNTIME_KIND",
    "TEMPORAL_CONTEXT_MODE",
    "TEMPORAL_CONTEXT_PADDING",
    "TEMPORAL_IMAGE_ONLY_METHOD_NAME",
    "TEMPORAL_INPUT_MODE",
    "TEMPORAL_RUNTIME_KIND",
    "TEMPORAL_VISION_POLICY_SCHEMA_VERSION",
    "SingleFrameResidualPolicyRuntime",
    "TemporalFeatureConcatPolicyRuntime",
    "TemporalVisionResidualBCPolicy",
    "build_history_concat_features",
    "encode_rgb_batches",
    "is_temporal_policy_metadata",
    "is_temporal_policy_path",
    "load_sigma7_residual_policy_runtime",
    "read_policy_metadata",
    "save_temporal_image_only_residual_bc_policy",
    "validate_sigma7_track_a_residual_policy_metadata",
    "validate_temporal_track_a_policy_metadata",
]
