from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from stiffness_copilot_mujoco.learning.residual_stiffness import BaseStiffnessSpec, PARAM_NAMES
from stiffness_copilot_mujoco.learning.stiffness_labels import cholesky_params_to_matrix, spd_project
from stiffness_copilot_mujoco.runtime_defaults import DEFAULT_DINOV3_CHECKPOINT, DEFAULT_DINOV3_REPO


VISION_POLICY_SCHEMA_VERSION = "vision_residual_bc_policy_v2"
IMAGE_ONLY_RESIDUAL_BC_METHOD_NAME = "image_only_residual_bc"
SMALL_CNN_BACKBONE_TYPE = "small_cnn_trainable"
LINEAR_OR_MLP_V1_HEAD_TYPE = "linear_or_mlp_v1"
MLP_V2_HEAD_TYPE = "mlp_v2"
IMAGE_ONLY_RESIDUAL_BC_OUTPUT_SPACE = "circle_shared_lateral_residual_1d"
POLYGON_IMAGE_ONLY_RESIDUAL_BC_OUTPUT_SPACE = "polygon_independent_lateral_residual_2d"
STAR_IMAGE_ONLY_RESIDUAL_BC_OUTPUT_SPACE = "star_independent_lateral_with_xy_coupling_residual_3d"
FULL_IMAGE_ONLY_RESIDUAL_BC_OUTPUT_SPACE = "full_spd_cholesky_residual_6d"
IMAGE_ONLY_RESIDUAL_BC_RESIDUAL_BOUND = 0.35
IMAGE_ONLY_RESIDUAL_BC_RENDERER_MODE = "mujoco_native"
IMAGE_ONLY_RESIDUAL_BC_FALLBACK_USED = False
IMAGE_ONLY_RESIDUAL_BC_STIFFNESS_REPRESENTATION = "full_spd_matrix"
IMAGE_ONLY_RESIDUAL_BC_RESIDUAL_PARAMETERIZATION = "constrained_shared_lateral_scalar"
IMAGE_ONLY_RESIDUAL_BC_RESIDUAL_AFFECTS = ["K_xx", "K_yy"]
IMAGE_ONLY_RESIDUAL_BC_RESIDUAL_UNAFFECTED = ["K_zz", "off_diagonal_terms"]
POLYGON_IMAGE_ONLY_RESIDUAL_BC_RESIDUAL_PARAMETERIZATION = "constrained_independent_lateral_scalars"
POLYGON_IMAGE_ONLY_RESIDUAL_BC_RESIDUAL_AFFECTS = ["alpha_x", "alpha_y"]
POLYGON_IMAGE_ONLY_RESIDUAL_BC_RESIDUAL_UNAFFECTED = ["K_zz", "off_diagonal_terms"]
STAR_IMAGE_ONLY_RESIDUAL_BC_RESIDUAL_PARAMETERIZATION = "independent_lateral_with_xy_coupling_3d"
STAR_IMAGE_ONLY_RESIDUAL_BC_RESIDUAL_AFFECTS = ["alpha_x", "alpha_y", "l21"]
STAR_IMAGE_ONLY_RESIDUAL_BC_RESIDUAL_UNAFFECTED = ["K_zz", "off_diagonal_terms"]
FULL_IMAGE_ONLY_RESIDUAL_BC_RESIDUAL_PARAMETERIZATION = "full_spd_cholesky_delta_6d"
FULL_IMAGE_ONLY_RESIDUAL_BC_RESIDUAL_AFFECTS = list(PARAM_NAMES)
FULL_IMAGE_ONLY_RESIDUAL_BC_RESIDUAL_UNAFFECTED: list[str] = []
IMAGE_ONLY_RESIDUAL_BC_SMOOTHING_REQUIRED = True
DEFAULT_HEAD_HIDDEN_DIMS = (256, 128)
DEFAULT_HEAD_DROPOUT = 0.05
DEFAULT_HEAD_ACTIVATION = "GELU"
DEFAULT_HEAD_OUTPUT_BOUNDING = "tanh_residual_bound"


def describe_residual_policy_contract(base_spec: BaseStiffnessSpec) -> dict[str, Any]:
    active_groups = tuple(tuple(int(idx) for idx in group) for group in base_spec.active_groups)
    active_group_names = tuple(str(name) for name in base_spec.active_group_names)
    if active_groups == ((0, 1),) and active_group_names == ("alpha_lateral_shared",):
        return {
            "output_space": IMAGE_ONLY_RESIDUAL_BC_OUTPUT_SPACE,
            "output_dim": 1,
            "residual_parameterization": IMAGE_ONLY_RESIDUAL_BC_RESIDUAL_PARAMETERIZATION,
            "residual_affects": IMAGE_ONLY_RESIDUAL_BC_RESIDUAL_AFFECTS,
            "residual_unaffected": IMAGE_ONLY_RESIDUAL_BC_RESIDUAL_UNAFFECTED,
        }
    if active_groups == ((0,), (1,)) and active_group_names == ("alpha_x", "alpha_y"):
        return {
            "output_space": POLYGON_IMAGE_ONLY_RESIDUAL_BC_OUTPUT_SPACE,
            "output_dim": 2,
            "residual_parameterization": POLYGON_IMAGE_ONLY_RESIDUAL_BC_RESIDUAL_PARAMETERIZATION,
            "residual_affects": POLYGON_IMAGE_ONLY_RESIDUAL_BC_RESIDUAL_AFFECTS,
            "residual_unaffected": POLYGON_IMAGE_ONLY_RESIDUAL_BC_RESIDUAL_UNAFFECTED,
        }
    if active_groups == ((0,), (1,), (3,)) and active_group_names == ("alpha_x", "alpha_y", "l21"):
        return {
            "output_space": STAR_IMAGE_ONLY_RESIDUAL_BC_OUTPUT_SPACE,
            "output_dim": 3,
            "residual_parameterization": STAR_IMAGE_ONLY_RESIDUAL_BC_RESIDUAL_PARAMETERIZATION,
            "residual_affects": STAR_IMAGE_ONLY_RESIDUAL_BC_RESIDUAL_AFFECTS,
            "residual_unaffected": STAR_IMAGE_ONLY_RESIDUAL_BC_RESIDUAL_UNAFFECTED,
        }
    if active_groups == ((0,), (1,), (2,), (3,), (4,), (5,)) and active_group_names == tuple(PARAM_NAMES):
        return {
            "output_space": FULL_IMAGE_ONLY_RESIDUAL_BC_OUTPUT_SPACE,
            "output_dim": 6,
            "residual_parameterization": FULL_IMAGE_ONLY_RESIDUAL_BC_RESIDUAL_PARAMETERIZATION,
            "residual_affects": FULL_IMAGE_ONLY_RESIDUAL_BC_RESIDUAL_AFFECTS,
            "residual_unaffected": FULL_IMAGE_ONLY_RESIDUAL_BC_RESIDUAL_UNAFFECTED,
        }
    raise ValueError(
        "Unsupported residual contract for base_stiffness_spec; expected either the circle shared-lateral "
        "1D contract, the polygon independent-lateral 2D contract, the star constrained 3D contract, or the "
        "full 6D cholesky contract."
    )


def _metadata_value(metadata: dict[str, Any], key: str) -> Any:
    if key not in metadata:
        raise ValueError(f"Vision policy metadata is missing required field {key!r}.")
    return metadata[key]


def validate_image_only_residual_bc_metadata(metadata: dict[str, Any]) -> None:
    base_spec = _metadata_value(metadata, "base_stiffness_spec")
    if not isinstance(base_spec, dict):
        raise ValueError("Vision policy metadata field 'base_stiffness_spec' must be a mapping.")
    base_matrix = np.asarray(base_spec.get("base_matrix"), dtype=float)
    if base_matrix.shape != (3, 3):
        raise ValueError(
            f"Vision policy metadata field 'base_stiffness_spec.base_matrix' must have shape (3, 3), observed {base_matrix.shape}."
        )
    base_spec_obj = BaseStiffnessSpec.from_metadata(base_spec)
    expected_contract = describe_residual_policy_contract(base_spec_obj)
    required_exact_values = {
        "schema_version": VISION_POLICY_SCHEMA_VERSION,
        "method_name": IMAGE_ONLY_RESIDUAL_BC_METHOD_NAME,
        "input_mode": "image_only",
        "uses_task_state_input": False,
        "uses_contact_force_input": False,
        "uses_clearance_input": False,
        "uses_trajectory_phase_input": False,
        "output_space": expected_contract["output_space"],
        "output_dim": expected_contract["output_dim"],
        "is_residual_policy": True,
        "is_full_stiffness_policy": False,
        "renderer_mode": IMAGE_ONLY_RESIDUAL_BC_RENDERER_MODE,
        "fallback_used": IMAGE_ONLY_RESIDUAL_BC_FALLBACK_USED,
        "stiffness_representation": IMAGE_ONLY_RESIDUAL_BC_STIFFNESS_REPRESENTATION,
        "residual_parameterization": expected_contract["residual_parameterization"],
        "residual_affects": expected_contract["residual_affects"],
        "residual_unaffected": expected_contract["residual_unaffected"],
        "smoothing_required": IMAGE_ONLY_RESIDUAL_BC_SMOOTHING_REQUIRED,
    }
    for key, expected in required_exact_values.items():
        observed = _metadata_value(metadata, key)
        if observed != expected:
            raise ValueError(f"Vision policy metadata field {key!r} must be {expected!r}, observed {observed!r}.")
    for key in ("controllers_yaml", "reference_controller_id", "collection_controller_id", "reference_controller_profile"):
        value = _metadata_value(metadata, key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Vision policy metadata field {key!r} must be a non-empty string.")
    for key in ("reference_controller_id", "collection_controller_id"):
        if str(metadata[key]) != str(metadata["reference_controller_id"]):
            raise ValueError(
                f"Vision policy metadata field {key!r} must match reference_controller_id, observed {metadata[key]!r}."
            )
    if str(metadata["reference_controller_profile"]) != str(metadata["reference_controller_id"]):
        raise ValueError("Vision policy metadata field 'reference_controller_profile' must match reference_controller_id.")
    for key in ("reference_stiffness_matrix", "collection_stiffness_matrix"):
        matrix = np.asarray(_metadata_value(metadata, key), dtype=float)
        if matrix.shape != (3, 3):
            raise ValueError(f"Vision policy metadata field {key!r} must have shape (3, 3), observed {matrix.shape}.")
    residual_bounds = np.asarray(base_spec.get("residual_bounds"), dtype=float)
    if residual_bounds.shape != (int(expected_contract["output_dim"]),):
        raise ValueError(
            "Vision policy metadata field 'base_stiffness_spec.residual_bounds' must have shape "
            f"({int(expected_contract['output_dim'])},), observed {residual_bounds.shape}."
        )
    if not np.allclose(residual_bounds, IMAGE_ONLY_RESIDUAL_BC_RESIDUAL_BOUND, atol=1e-12, rtol=0.0):
        raise ValueError(
            "Vision policy metadata field 'base_stiffness_spec.residual_bounds' must equal "
            f"{IMAGE_ONLY_RESIDUAL_BC_RESIDUAL_BOUND!r} in every dimension, observed {residual_bounds!r}."
        )
    reference_matrix = np.asarray(metadata["reference_stiffness_matrix"], dtype=float)
    collection_matrix = np.asarray(metadata["collection_stiffness_matrix"], dtype=float)
    if not np.allclose(reference_matrix, base_matrix, atol=1e-9, rtol=0.0):
        raise ValueError("reference_stiffness_matrix must match base_stiffness_spec.base_matrix.")
    if not np.allclose(collection_matrix, reference_matrix, atol=1e-9, rtol=0.0):
        raise ValueError("collection_stiffness_matrix must match reference_stiffness_matrix.")
    if bool(metadata.get("controller_policy_consistency_passed", False)) is not True:
        raise ValueError("Vision policy metadata field 'controller_policy_consistency_passed' must be True.")
    residual_bound = float(_metadata_value(metadata, "residual_bound"))
    if not np.isclose(residual_bound, IMAGE_ONLY_RESIDUAL_BC_RESIDUAL_BOUND, atol=1e-12, rtol=0.0):
        raise ValueError(
            f"Vision policy metadata field 'residual_bound' must be {IMAGE_ONLY_RESIDUAL_BC_RESIDUAL_BOUND!r}, "
            f"observed {residual_bound!r}."
        )
    head_type = str(metadata.get("head_type") or "")
    if head_type:
        if head_type not in {LINEAR_OR_MLP_V1_HEAD_TYPE, MLP_V2_HEAD_TYPE}:
            raise ValueError(f"Unsupported head_type {head_type!r}.")
        hidden_dims = metadata.get("head_hidden_dims")
        if hidden_dims is not None:
            if not isinstance(hidden_dims, (list, tuple)) or not hidden_dims:
                raise ValueError("Vision policy metadata field 'head_hidden_dims' must be a non-empty sequence.")
            if any(int(value) <= 0 for value in hidden_dims):
                raise ValueError("Vision policy metadata field 'head_hidden_dims' must contain positive integers.")
        if head_type == MLP_V2_HEAD_TYPE:
            if hidden_dims is None or len(hidden_dims) < 2:
                raise ValueError("mlp_v2 head metadata must include at least two hidden dims.")
            head_dropout = float(metadata.get("head_dropout", DEFAULT_HEAD_DROPOUT))
            if not (0.0 <= head_dropout < 1.0):
                raise ValueError("head_dropout must be in [0, 1).")
            head_activation = str(metadata.get("head_activation") or "")
            if head_activation and head_activation.upper() != DEFAULT_HEAD_ACTIVATION:
                raise ValueError(
                    f"head_activation must be {DEFAULT_HEAD_ACTIVATION!r} for mlp_v2, observed {head_activation!r}."
                )
            output_bounding = str(metadata.get("output_bounding") or "")
            if output_bounding and output_bounding not in {"tanh_residual_bound", "clip_residual_bound"}:
                raise ValueError(f"Unsupported output_bounding {output_bounding!r}.")
    backbone_type = str(metadata.get("backbone_type") or "")
    if backbone_type:
        backbone_frozen = metadata.get("backbone_frozen")
        if backbone_type == SMALL_CNN_BACKBONE_TYPE:
            if backbone_frozen not in (None, False):
                raise ValueError("small-CNN backbone metadata must not mark the backbone as frozen.")
        elif backbone_type == "dinov3_small_frozen":
            if backbone_frozen is not True:
                raise ValueError("dinov3 backbone metadata must set backbone_frozen=True.")
            for key in ("dinov3_repo", "dinov3_checkpoint", "dinov3_entrypoint"):
                value = metadata.get(key)
                if not isinstance(value, str) or not value:
                    raise ValueError(f"Vision policy metadata field {key!r} must be a non-empty string for DINOv3 policies.")
        else:
            raise ValueError(f"Unsupported backbone_type {backbone_type!r}.")


def load_image_only_residual_bc_policy(path: Path) -> "VisionResidualBCPolicy":
    policy = VisionResidualBCPolicy.load(path)
    if policy.input_mode != "image_only":
        raise ValueError(f"Expected image-only policy input mode, observed {policy.input_mode!r}.")
    if policy.encoder is None:
        raise ValueError("Image-only residual BC policy is missing its image encoder.")
    return policy


def _resolve_bundled_dinov3_asset(configured: str, bundled: Path) -> Path:
    configured_path = Path(str(configured)).expanduser()
    if configured_path.exists():
        return configured_path
    if bundled.exists():
        return bundled
    return configured_path


def _as_hidden_dims(
    hidden_dim: int | None = None,
    hidden_dims: tuple[int, ...] | list[int] | None = None,
    *,
    default: tuple[int, ...] = DEFAULT_HEAD_HIDDEN_DIMS,
) -> tuple[int, ...]:
    if hidden_dims is not None:
        dims = tuple(int(value) for value in hidden_dims)
    elif hidden_dim is not None:
        dims = (int(hidden_dim),)
    else:
        dims = default
    if not dims or any(value <= 0 for value in dims):
        raise ValueError("hidden_dims must contain at least one positive integer.")
    return dims


def _state_dict_to_npz_arrays(prefix: str, state_dict: dict[str, Any]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for key, value in state_dict.items():
        tensor = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
        arrays[f"{prefix}{key}"] = np.asarray(tensor, dtype=np.float32)
    return arrays


def _npz_arrays_to_state_dict(prefix: str, data: np.lib.npyio.NpzFile) -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {}
    for key in data.files:
        if key.startswith(prefix):
            state_key = key[len(prefix) :]
            state_dict[state_key] = torch.as_tensor(data[key], dtype=torch.float32)
    return state_dict


class LegacyVisionResidualHeadV1(torch.nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class VisionResidualHeadV2(torch.nn.Module):
    def __init__(
        self,
        feature_dim: int,
        output_dim: int,
        *,
        hidden_dims: tuple[int, ...] = DEFAULT_HEAD_HIDDEN_DIMS,
        dropout: float = DEFAULT_HEAD_DROPOUT,
    ) -> None:
        super().__init__()
        hidden_dims = _as_hidden_dims(hidden_dims=hidden_dims)
        layers: list[torch.nn.Module] = [torch.nn.LayerNorm(feature_dim)]
        in_dim = feature_dim
        for index, hidden_dim in enumerate(hidden_dims):
            layers.append(torch.nn.Linear(in_dim, hidden_dim))
            if index < len(hidden_dims) - 1:
                layers.append(torch.nn.GELU())
                if dropout > 0.0:
                    layers.append(torch.nn.Dropout(p=float(dropout)))
            in_dim = hidden_dim
        layers.append(torch.nn.GELU())
        if dropout > 0.0:
            layers.append(torch.nn.Dropout(p=float(dropout)))
        layers.append(torch.nn.Linear(in_dim, output_dim))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def _build_head_module(
    *,
    head_type: str,
    feature_dim: int,
    output_dim: int,
    hidden_dim: int | None = None,
    hidden_dims: tuple[int, ...] | list[int] | None = None,
    dropout: float = DEFAULT_HEAD_DROPOUT,
) -> torch.nn.Module:
    if head_type == LINEAR_OR_MLP_V1_HEAD_TYPE:
        dims = _as_hidden_dims(hidden_dim=hidden_dim, hidden_dims=hidden_dims, default=(32,))
        if len(dims) != 1:
            raise ValueError("linear_or_mlp_v1 expects exactly one hidden dimension.")
        return LegacyVisionResidualHeadV1(feature_dim, dims[0], output_dim)
    if head_type == MLP_V2_HEAD_TYPE:
        dims = _as_hidden_dims(hidden_dim=hidden_dim, hidden_dims=hidden_dims, default=DEFAULT_HEAD_HIDDEN_DIMS)
        return VisionResidualHeadV2(feature_dim, output_dim, hidden_dims=dims, dropout=dropout)
    raise ValueError(f"Unsupported head_type {head_type!r}.")


def _bounded_residual(raw: np.ndarray, residual_bound: np.ndarray | float) -> np.ndarray:
    bound = np.asarray(residual_bound, dtype=np.float32)
    return bound * np.tanh(raw / np.maximum(bound, 1e-12))


def _normalize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std < 1e-8] = 1.0
    return (values - mean) / std, mean, std


def _he_init(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    fan_in = float(np.prod(shape[1:])) if len(shape) > 1 else float(shape[0])
    scale = np.sqrt(2.0 / max(fan_in, 1.0))
    return rng.normal(0.0, scale, size=shape).astype(np.float32)


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _conv2d_nhwc(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    *,
    stride: int = 2,
    padding: int = 1,
) -> np.ndarray:
    if x.ndim != 4:
        raise ValueError(f"Expected NHWC batch, got {x.shape}.")
    if weight.ndim != 4:
        raise ValueError(f"Expected conv weight shape [O, I, K, K], got {weight.shape}.")
    batch, height, width, channels = x.shape
    out_channels, in_channels, kernel_h, kernel_w = weight.shape
    if channels != in_channels:
        raise ValueError(f"Input channels {channels} do not match weight channels {in_channels}.")
    x_nchw = np.transpose(x, (0, 3, 1, 2))
    x_pad = np.pad(x_nchw, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant")
    windows = np.lib.stride_tricks.sliding_window_view(x_pad, (kernel_h, kernel_w), axis=(2, 3))
    windows = windows[:, :, ::stride, ::stride, :, :]
    out = np.tensordot(windows, weight, axes=([1, 4, 5], [1, 2, 3])).transpose(0, 3, 1, 2)
    out += bias[None, :, None, None]
    return np.transpose(_relu(out), (0, 2, 3, 1))


@dataclass(frozen=True)
class VisionEncoderSpec:
    conv_weights: tuple[np.ndarray, ...]
    conv_biases: tuple[np.ndarray, ...]
    stride: int
    padding: int
    output_channels: int
    seed: int

    @classmethod
    def random(
        cls,
        *,
        input_channels: int = 3,
        conv_channels: tuple[int, int, int] = (8, 16, 16),
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        seed: int = 0,
    ) -> "VisionEncoderSpec":
        rng = np.random.default_rng(seed)
        weights = []
        biases = []
        in_channels = input_channels
        for out_channels in conv_channels:
            weights.append(_he_init(rng, (out_channels, in_channels, kernel_size, kernel_size)))
            biases.append(np.zeros(out_channels, dtype=np.float32))
            in_channels = out_channels
        return cls(
            conv_weights=tuple(weights),
            conv_biases=tuple(biases),
            stride=stride,
            padding=padding,
            output_channels=conv_channels[-1],
            seed=seed,
        )

    def encode(self, rgb_images: np.ndarray, *, batch_size: int = 16) -> np.ndarray:
        images = np.asarray(rgb_images, dtype=np.float32)
        if images.ndim == 3:
            images = images[None, ...]
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError(f"rgb_images must have shape [N, H, W, 3], got {images.shape}.")
        features: list[np.ndarray] = []
        for start in range(0, images.shape[0], batch_size):
            stop = min(start + batch_size, images.shape[0])
            x = images[start:stop] / 255.0
            for weight, bias in zip(self.conv_weights, self.conv_biases, strict=True):
                x = _conv2d_nhwc(x, weight, bias, stride=self.stride, padding=self.padding)
            pooled = x.mean(axis=(1, 2))
            features.append(pooled.astype(np.float32, copy=False))
        return np.concatenate(features, axis=0)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "seed": int(self.seed),
            "stride": int(self.stride),
            "padding": int(self.padding),
            "output_channels": int(self.output_channels),
            "conv_channels": [int(weight.shape[0]) for weight in self.conv_weights],
            "kernel_size": int(self.conv_weights[0].shape[2]) if self.conv_weights else None,
        }


@dataclass(frozen=True)
class VisionResidualBCPolicy:
    input_mode: str
    encoder: Any | None
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

    @classmethod
    def load(cls, path: Path) -> "VisionResidualBCPolicy":
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            encoder = None
            backbone_type = str(metadata.get("backbone_type") or "")
            if backbone_type == "dinov3_small_frozen":
                from stiffness_copilot_mujoco.learning.frozen_vision_backbone import (  # local import to avoid cycle
                    FrozenBackboneLoadConfig,
                    FrozenBackbonePreprocessConfig,
                    build_frozen_dinov3_backbone,
                )

                preprocess_config = metadata.get("preprocessing_config") or {}
                if not isinstance(preprocess_config, dict):
                    raise ValueError("DINOv3 policy metadata preprocessing_config must be a mapping.")
                resolved_dinov3_repo = _resolve_bundled_dinov3_asset(
                    str(metadata["dinov3_repo"]),
                    DEFAULT_DINOV3_REPO,
                )
                resolved_dinov3_checkpoint = _resolve_bundled_dinov3_asset(
                    str(metadata["dinov3_checkpoint"]),
                    DEFAULT_DINOV3_CHECKPOINT,
                )
                load_config = FrozenBackboneLoadConfig(
                    dinov3_repo=resolved_dinov3_repo,
                    dinov3_checkpoint=resolved_dinov3_checkpoint,
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
                encoder, backbone_metadata = build_frozen_dinov3_backbone(load_config)
                metadata = dict(metadata)
                metadata["dinov3_repo"] = str(resolved_dinov3_repo)
                metadata["dinov3_checkpoint"] = str(resolved_dinov3_checkpoint)
                metadata.setdefault("backbone_metadata", backbone_metadata)
            elif "encoder_seed" in metadata:
                encoder = VisionEncoderSpec.random(
                    seed=int(metadata["encoder_seed"]),
                    conv_channels=tuple(int(value) for value in metadata.get("encoder_conv_channels", (8, 16, 16))),
                    kernel_size=int(metadata.get("encoder_kernel_size", 3)),
                    stride=int(metadata.get("encoder_stride", 2)),
                    padding=int(metadata.get("encoder_padding", 1)),
                )
                conv_weights = []
                conv_biases = []
                layer_idx = 0
                while f"encoder_conv_{layer_idx}_weight" in data.files:
                    conv_weights.append(data[f"encoder_conv_{layer_idx}_weight"].astype(np.float32))
                    conv_biases.append(data[f"encoder_conv_{layer_idx}_bias"].astype(np.float32))
                    layer_idx += 1
                encoder = VisionEncoderSpec(
                    conv_weights=tuple(conv_weights),
                    conv_biases=tuple(conv_biases),
                    stride=int(metadata.get("encoder_stride", 2)),
                    padding=int(metadata.get("encoder_padding", 1)),
                    output_channels=int(metadata.get("encoder_output_channels", conv_weights[-1].shape[0] if conv_weights else 0)),
                    seed=int(metadata.get("encoder_seed", 0)),
                )

            head_type = str(metadata.get("head_type") or LINEAR_OR_MLP_V1_HEAD_TYPE)
            feature_dim = int(metadata.get("feature_dim", 0))
            output_dim = int(metadata.get("output_dim", 0))
            head_hidden_dims = tuple(int(value) for value in metadata.get("head_hidden_dims", ()))
            head_dropout = float(metadata.get("head_dropout", 0.0))
            head_activation = str(metadata.get("head_activation", "ReLU"))
            output_bounding = str(metadata.get("output_bounding", "tanh_residual_bound"))

            if head_type == MLP_V2_HEAD_TYPE:
                if feature_dim <= 0 or output_dim <= 0:
                    raise ValueError("mlp_v2 head metadata must include positive feature_dim and output_dim.")
                if not head_hidden_dims:
                    raise ValueError("mlp_v2 head metadata must include head_hidden_dims.")
                head = VisionResidualHeadV2(
                    feature_dim,
                    output_dim,
                    hidden_dims=head_hidden_dims,
                    dropout=head_dropout,
                )
                state_dict = _npz_arrays_to_state_dict("head_state__", data)
                if not state_dict:
                    raise ValueError("mlp_v2 policy archive is missing head_state__ tensors.")
                head.load_state_dict(state_dict, strict=True)
            elif head_type == LINEAR_OR_MLP_V1_HEAD_TYPE:
                if feature_dim <= 0 or output_dim <= 0:
                    raise ValueError("linear_or_mlp_v1 head metadata must include positive feature_dim and output_dim.")
                if not head_hidden_dims:
                    head_hidden_dims = (int(metadata.get("head_hidden_dim", metadata.get("hidden_dim", 32))),)
                if len(head_hidden_dims) != 1:
                    raise ValueError("linear_or_mlp_v1 head metadata must include exactly one hidden dim.")
                head = LegacyVisionResidualHeadV1(feature_dim, head_hidden_dims[0], output_dim)
                if "w1" in data.files and "b1" in data.files and "w2" in data.files and "b2" in data.files:
                    with torch.no_grad():
                        head.net[0].weight.copy_(torch.as_tensor(data["w1"].T, dtype=torch.float32))
                        head.net[0].bias.copy_(torch.as_tensor(data["b1"], dtype=torch.float32))
                        head.net[2].weight.copy_(torch.as_tensor(data["w2"].T, dtype=torch.float32))
                        head.net[2].bias.copy_(torch.as_tensor(data["b2"], dtype=torch.float32))
                else:
                    state_dict = _npz_arrays_to_state_dict("head_state__", data)
                    if state_dict:
                        head.load_state_dict(state_dict, strict=True)
            else:
                raise ValueError(f"Unsupported head_type {head_type!r}.")

            policy = cls(
                input_mode=str(metadata["input_mode"]),
                encoder=encoder,
                head=head,
                head_type=head_type,
                head_hidden_dims=head_hidden_dims,
                head_dropout=head_dropout,
                head_activation=head_activation,
                output_bounding=output_bounding,
                x_mean=data["x_mean"].astype(np.float32),
                x_std=data["x_std"].astype(np.float32),
                y_mean=data["y_mean"].astype(np.float32),
                y_std=data["y_std"].astype(np.float32),
                base_spec=BaseStiffnessSpec.from_metadata(metadata["base_stiffness_spec"]),
                metadata=metadata,
            )
            policy.head.eval()
            validate_image_only_residual_bc_metadata(policy.metadata)
            return policy

    @property
    def input_dim(self) -> int:
        return int(self.x_mean.shape[0])

    @property
    def output_dim(self) -> int:
        return int(self.y_mean.shape[0])

    @property
    def hidden_dims(self) -> tuple[int, ...]:
        return self.head_hidden_dims

    def _features(self, *, task_state: np.ndarray | None, rgb_images: np.ndarray | None) -> np.ndarray:
        parts: list[np.ndarray] = []
        single_sample = False
        if self.input_mode == "image_only":
            if task_state is not None:
                raise ValueError("image_only residual policy must not receive task_state.")
        else:
            if task_state is None:
                raise ValueError("task_state is required for this policy.")
            state = np.asarray(task_state, dtype=np.float32)
            if state.ndim == 1:
                single_sample = True
                state = state[None, ...]
            parts.append(state)
        if self.input_mode != "state_only":
            if rgb_images is None:
                raise ValueError("rgb_images is required for this policy.")
            if self.encoder is None:
                raise ValueError("Vision encoder is missing from the loaded policy.")
            images = np.asarray(rgb_images)
            if images.ndim == 3:
                single_sample = True
            encoded = self.encoder.encode(images)
            if isinstance(encoded, torch.Tensor):
                encoded = encoded.detach().cpu().numpy()
            parts.append(np.asarray(encoded, dtype=np.float32))
        if not parts:
            raise ValueError("No features were provided.")
        features = np.concatenate(parts, axis=1)
        return features[0] if single_sample else features

    def _raw_head_output(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float32)
        x_norm = (x - self.x_mean) / self.x_std
        with torch.no_grad():
            tensor = torch.as_tensor(x_norm, dtype=torch.float32)
            self.head.eval()
            raw = self.head(tensor).detach().cpu().numpy()
        return np.asarray(raw, dtype=np.float32)

    def predict_image_only(
        self,
        rgb_images: np.ndarray,
        *,
        residual_scale: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.input_mode != "image_only":
            raise ValueError(f"predict_image_only is only valid for image-only policies, observed {self.input_mode!r}.")
        raw = self.predict_group_delta_raw(task_state=None, rgb_images=rgb_images)
        scaled = np.asarray(raw, dtype=np.float32) * float(residual_scale)
        bounded = _bounded_residual(scaled, self.base_spec.residual_bounds)
        theta_delta = self.base_spec.expand_group_delta(bounded, clip=True)
        theta = self.base_spec.theta_base + theta_delta
        matrix = spd_project(cholesky_params_to_matrix(theta))
        return raw, bounded, matrix, theta, theta_delta

    def predict_group_delta_raw(self, *, task_state: np.ndarray | None, rgb_images: np.ndarray | None) -> np.ndarray:
        x = self._features(task_state=task_state, rgb_images=rgb_images)
        if x.shape != self.x_mean.shape:
            raise ValueError(f"Feature vector must have shape {self.x_mean.shape}, got {x.shape}.")
        raw = self._raw_head_output(x)
        return raw * self.y_std + self.y_mean

    def predict(
        self,
        *,
        task_state: np.ndarray | None,
        rgb_images: np.ndarray | None,
        residual_scale: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raw = self.predict_group_delta_raw(task_state=task_state, rgb_images=rgb_images)
        scaled = np.asarray(raw, dtype=np.float32) * float(residual_scale)
        bounded = _bounded_residual(scaled, self.base_spec.residual_bounds)
        theta_delta = self.base_spec.expand_group_delta(bounded, clip=True)
        theta = self.base_spec.theta_base + theta_delta
        matrix = spd_project(cholesky_params_to_matrix(theta))
        return matrix, theta, theta_delta, bounded

    def _legacy_single_hidden_layer_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.head_type != LINEAR_OR_MLP_V1_HEAD_TYPE:
            raise ValueError("Legacy w1/b1/w2/b2 access is only available for linear_or_mlp_v1 policies.")
        first = self.head.net[0]
        last = self.head.net[2]
        if not isinstance(first, torch.nn.Linear) or not isinstance(last, torch.nn.Linear):
            raise ValueError("Loaded legacy head does not expose the expected linear layers.")
        return (
            first.weight.detach().cpu().numpy().T.copy(),
            first.bias.detach().cpu().numpy().copy(),
            last.weight.detach().cpu().numpy().T.copy(),
            last.bias.detach().cpu().numpy().copy(),
        )

    @property
    def w1(self) -> np.ndarray:
        return self._legacy_single_hidden_layer_state()[0]

    @property
    def b1(self) -> np.ndarray:
        return self._legacy_single_hidden_layer_state()[1]

    @property
    def w2(self) -> np.ndarray:
        return self._legacy_single_hidden_layer_state()[2]

    @property
    def b2(self) -> np.ndarray:
        return self._legacy_single_hidden_layer_state()[3]


def train_head_with_adam(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    head_type: str = LINEAR_OR_MLP_V1_HEAD_TYPE,
    hidden_dim: int | None = None,
    hidden_dims: tuple[int, ...] | list[int] | None = None,
    head_dropout: float = DEFAULT_HEAD_DROPOUT,
    epochs: int = 20,
    lr: float = 0.01,
    weight_decay: float = 0.0,
    seed: int = 0,
    validation_patience: int | None = None,
    x_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    y_mean: np.ndarray | None = None,
    y_std: np.ndarray | None = None,
    residual_bound: float | None = None,
    progress_prefix: str | None = None,
    progress_every: int | None = None,
) -> tuple[torch.nn.Module, list[tuple[int, float, float]], dict[str, Any]]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    x_train = np.asarray(x_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    if x_val is None:
        x_val = x_train
    if y_val is None:
        y_val = y_train
    x_val = np.asarray(x_val, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.float32)
    input_dim = int(x_train.shape[1])
    output_dim = int(y_train.shape[1])
    dims = _as_hidden_dims(hidden_dim=hidden_dim, hidden_dims=hidden_dims, default=DEFAULT_HEAD_HIDDEN_DIMS if head_type == MLP_V2_HEAD_TYPE else (32,))
    head = _build_head_module(
        head_type=head_type,
        feature_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        hidden_dims=dims,
        dropout=head_dropout,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    loss_fn = torch.nn.MSELoss(reduction="mean")
    x_train_t = torch.as_tensor(x_train, dtype=torch.float32)
    y_train_t = torch.as_tensor(y_train, dtype=torch.float32)
    x_val_t = torch.as_tensor(x_val, dtype=torch.float32)
    y_val_t = torch.as_tensor(y_val, dtype=torch.float32)

    best_state = copy.deepcopy(head.state_dict())
    best_epoch = 0
    best_val_loss = float("inf")
    best_train_loss = float("inf")
    epochs_without_improvement = 0
    early_stop_reason = "max_epochs"
    history: list[tuple[int, float, float]] = []

    def _predict(module: torch.nn.Module, features: torch.Tensor) -> np.ndarray:
        module.eval()
        with torch.no_grad():
            return module(features).detach().cpu().numpy().astype(np.float32)

    for epoch in range(1, epochs + 1):
        head.train()
        optimizer.zero_grad(set_to_none=True)
        train_pred = head(x_train_t)
        train_loss_t = loss_fn(train_pred, y_train_t)
        train_loss_t.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
        optimizer.step()

        head.eval()
        with torch.no_grad():
            val_pred = head(x_val_t)
            val_loss_t = loss_fn(val_pred, y_val_t)
        train_loss = float(train_loss_t.detach().cpu().item())
        val_loss = float(val_loss_t.detach().cpu().item())
        history.append((epoch, train_loss, val_loss))

        if val_loss + 1e-9 < best_val_loss:
            best_val_loss = val_loss
            best_train_loss = train_loss
            best_epoch = epoch
            best_state = copy.deepcopy(head.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if validation_patience is not None and validation_patience > 0 and epochs_without_improvement >= validation_patience:
                early_stop_reason = "validation_patience"
                if progress_prefix:
                    print(
                        f"{progress_prefix} epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} early_stop=True",
                        flush=True,
                    )
                break

        should_log = epoch == 1 or epoch == epochs or epoch % max(1, epochs // 5) == 0
        if progress_every is not None and progress_every > 0:
            should_log = should_log or (epoch % progress_every == 0)
        if progress_prefix and should_log:
            print(f"{progress_prefix} epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}", flush=True)

    head.load_state_dict(best_state, strict=True)
    head.eval()
    train_pred_norm = _predict(head, x_train_t)
    val_pred_norm = _predict(head, x_val_t)
    if y_mean is None:
        y_mean = np.zeros(output_dim, dtype=np.float32)
    if y_std is None:
        y_std = np.ones(output_dim, dtype=np.float32)
    y_mean = np.asarray(y_mean, dtype=np.float32)
    y_std = np.asarray(y_std, dtype=np.float32)
    if residual_bound is None:
        residual_bound = IMAGE_ONLY_RESIDUAL_BC_RESIDUAL_BOUND
    residual_bound = float(residual_bound)
    train_target_raw = y_train * y_std + y_mean
    val_target_raw = y_val * y_std + y_mean
    train_pred_raw = train_pred_norm * y_std + y_mean
    val_pred_raw = val_pred_norm * y_std + y_mean
    train_pred_bounded = _bounded_residual(train_pred_raw, residual_bound)
    val_pred_bounded = _bounded_residual(val_pred_raw, residual_bound)
    train_target_bounded = _bounded_residual(train_target_raw, residual_bound)
    val_target_bounded = _bounded_residual(val_target_raw, residual_bound)

    def _mae(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
        return np.mean(np.abs(pred - target), axis=0)

    def _sign_accuracy(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
        return np.mean(np.sign(pred) == np.sign(target), axis=0)

    def _safe_stats(values: np.ndarray) -> tuple[list[float], list[float]]:
        return values.mean(axis=0).tolist(), values.std(axis=0).tolist()

    train_pred_mean, train_pred_std = _safe_stats(train_pred_bounded)
    val_pred_mean, val_pred_std = _safe_stats(val_pred_bounded)
    train_label_mean, train_label_std = _safe_stats(train_target_bounded)
    val_label_mean, val_label_std = _safe_stats(val_target_bounded)
    saturation_rate = float(np.mean(np.abs(val_pred_bounded) >= (0.95 * residual_bound)))
    per_dim_saturation_rate = np.mean(np.abs(val_pred_bounded) >= (0.95 * residual_bound), axis=0).tolist()
    summary: dict[str, Any] = {
        "head_type": head_type,
        "head_hidden_dims": list(dims),
        "head_dropout": float(head_dropout),
        "head_activation": DEFAULT_HEAD_ACTIVATION if head_type == MLP_V2_HEAD_TYPE else "ReLU",
        "output_bounding": DEFAULT_HEAD_OUTPUT_BOUNDING,
        "best_epoch": int(best_epoch),
        "best_train_loss": float(best_train_loss),
        "best_val_loss": float(best_val_loss),
        "train_loss": float(history[-1][1]),
        "val_loss": float(history[-1][2]),
        "early_stop_reason": early_stop_reason,
        "train_label_target_mean": train_label_mean,
        "train_label_target_std": train_label_std,
        "val_label_target_mean": val_label_mean,
        "val_label_target_std": val_label_std,
        "train_prediction_mean": train_pred_mean,
        "train_prediction_std": train_pred_std,
        "val_prediction_mean": val_pred_mean,
        "val_prediction_std": val_pred_std,
        "val_per_dim_mae": _mae(val_pred_bounded, val_target_bounded).tolist(),
        "val_per_dim_sign_accuracy": _sign_accuracy(val_pred_bounded, val_target_bounded).tolist(),
        "train_per_dim_mae": _mae(train_pred_bounded, train_target_bounded).tolist(),
        "train_per_dim_sign_accuracy": _sign_accuracy(train_pred_bounded, train_target_bounded).tolist(),
        "prediction_saturation_rate": saturation_rate,
        "prediction_saturation_rate_per_dim": per_dim_saturation_rate,
        "residual_bound": residual_bound,
    }
    return head, history, summary


def save_vision_residual_policy(
    path: Path,
    *,
    input_mode: str,
    encoder: Any | None,
    head: torch.nn.Module | None = None,
    head_type: str | None = None,
    head_hidden_dims: tuple[int, ...] | list[int] | None = None,
    head_dropout: float | None = None,
    head_activation: str | None = None,
    output_bounding: str | None = None,
    w1: np.ndarray | None = None,
    b1: np.ndarray | None = None,
    w2: np.ndarray | None = None,
    b2: np.ndarray | None = None,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    base_spec: BaseStiffnessSpec,
    metadata: dict[str, Any],
) -> None:
    full_metadata = dict(metadata)
    full_metadata["schema_version"] = VISION_POLICY_SCHEMA_VERSION
    full_metadata["input_mode"] = input_mode
    full_metadata["base_stiffness_spec"] = base_spec.to_metadata()
    if head is None:
        if any(value is None for value in (w1, b1, w2, b2)):
            raise ValueError("Provide either a head module or legacy w1/b1/w2/b2 arrays.")
        head = LegacyVisionResidualHeadV1(int(np.asarray(x_mean).shape[0]), int(np.asarray(w1).shape[1]), int(np.asarray(b2).shape[0]))
        with torch.no_grad():
            assert isinstance(head, LegacyVisionResidualHeadV1)
            head.net[0].weight.copy_(torch.as_tensor(np.asarray(w1).T, dtype=torch.float32))
            head.net[0].bias.copy_(torch.as_tensor(np.asarray(b1), dtype=torch.float32))
            head.net[2].weight.copy_(torch.as_tensor(np.asarray(w2).T, dtype=torch.float32))
            head.net[2].bias.copy_(torch.as_tensor(np.asarray(b2), dtype=torch.float32))
        head_type = head_type or LINEAR_OR_MLP_V1_HEAD_TYPE

    if isinstance(encoder, VisionEncoderSpec):
        full_metadata["backbone_type"] = SMALL_CNN_BACKBONE_TYPE
        full_metadata["backbone_frozen"] = False
        full_metadata["model_source_type"] = "local_explicit"
        full_metadata["encoder_seed"] = int(encoder.seed)
        full_metadata["encoder_stride"] = int(encoder.stride)
        full_metadata["encoder_padding"] = int(encoder.padding)
        full_metadata["encoder_output_channels"] = int(encoder.output_channels)
        full_metadata["encoder_conv_channels"] = [int(weight.shape[0]) for weight in encoder.conv_weights]
        full_metadata["encoder_kernel_size"] = int(encoder.conv_weights[0].shape[2]) if encoder.conv_weights else None
    elif encoder is not None and hasattr(encoder, "to_metadata"):
        backbone_metadata = encoder.to_metadata()
        full_metadata.update(backbone_metadata)
        full_metadata.setdefault("backbone_type", backbone_metadata.get("backbone_type", "dinov3_small_frozen"))
        full_metadata.setdefault("backbone_frozen", backbone_metadata.get("backbone_frozen", True))
    else:
        if "backbone_type" not in full_metadata:
            raise ValueError("A policy encoder or explicit backbone metadata is required.")
    if head_type is None:
        head_type = str(full_metadata.get("head_type") or LINEAR_OR_MLP_V1_HEAD_TYPE)
    head_type = str(head_type)
    if head_type not in {LINEAR_OR_MLP_V1_HEAD_TYPE, MLP_V2_HEAD_TYPE}:
        raise ValueError(f"Unsupported head_type {head_type!r}.")
    if head_hidden_dims is None:
        if head_type == MLP_V2_HEAD_TYPE:
            head_hidden_dims = DEFAULT_HEAD_HIDDEN_DIMS
        else:
            head_hidden_dims = (int(np.asarray(w1).shape[1]) if w1 is not None else 32,)
    head_hidden_dims = _as_hidden_dims(hidden_dims=head_hidden_dims)
    if head_dropout is None:
        head_dropout = DEFAULT_HEAD_DROPOUT if head_type == MLP_V2_HEAD_TYPE else 0.0
    if head_activation is None:
        head_activation = DEFAULT_HEAD_ACTIVATION if head_type == MLP_V2_HEAD_TYPE else "ReLU"
    if output_bounding is None:
        output_bounding = DEFAULT_HEAD_OUTPUT_BOUNDING
    full_metadata["head_type"] = head_type
    full_metadata["head_hidden_dims"] = [int(value) for value in head_hidden_dims]
    full_metadata["head_dropout"] = float(head_dropout)
    full_metadata["head_activation"] = str(head_activation)
    full_metadata["output_bounding"] = str(output_bounding)
    full_metadata["feature_dim"] = int(np.asarray(x_mean).shape[0])
    full_metadata["output_dim"] = int(np.asarray(y_mean).shape[0])
    full_metadata["head_output_dim"] = int(np.asarray(y_mean).shape[0])
    validate_image_only_residual_bc_metadata(full_metadata)
    arrays: dict[str, np.ndarray] = {
        "x_mean": np.asarray(x_mean, dtype=np.float32),
        "x_std": np.asarray(x_std, dtype=np.float32),
        "y_mean": np.asarray(y_mean, dtype=np.float32),
        "y_std": np.asarray(y_std, dtype=np.float32),
        "metadata": json.dumps(full_metadata, sort_keys=True),
    }
    if isinstance(encoder, VisionEncoderSpec):
        for idx, (weight, bias) in enumerate(zip(encoder.conv_weights, encoder.conv_biases, strict=True)):
            arrays[f"encoder_conv_{idx}_weight"] = np.asarray(weight, dtype=np.float32)
            arrays[f"encoder_conv_{idx}_bias"] = np.asarray(bias, dtype=np.float32)
    if isinstance(head, LegacyVisionResidualHeadV1):
        arrays["w1"] = head.net[0].weight.detach().cpu().numpy().T.copy()
        arrays["b1"] = head.net[0].bias.detach().cpu().numpy().copy()
        arrays["w2"] = head.net[2].weight.detach().cpu().numpy().T.copy()
        arrays["b2"] = head.net[2].bias.detach().cpu().numpy().copy()
    state_dict = head.state_dict()
    arrays.update(_state_dict_to_npz_arrays("head_state__", state_dict))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


__all__ = [
    "IMAGE_ONLY_RESIDUAL_BC_METHOD_NAME",
    "IMAGE_ONLY_RESIDUAL_BC_OUTPUT_SPACE",
    "POLYGON_IMAGE_ONLY_RESIDUAL_BC_OUTPUT_SPACE",
    "IMAGE_ONLY_RESIDUAL_BC_RESIDUAL_BOUND",
    "SMALL_CNN_BACKBONE_TYPE",
    "LINEAR_OR_MLP_V1_HEAD_TYPE",
    "MLP_V2_HEAD_TYPE",
    "DEFAULT_HEAD_HIDDEN_DIMS",
    "DEFAULT_HEAD_DROPOUT",
    "DEFAULT_HEAD_ACTIVATION",
    "DEFAULT_HEAD_OUTPUT_BOUNDING",
    "VISION_POLICY_SCHEMA_VERSION",
    "POLYGON_IMAGE_ONLY_RESIDUAL_BC_RESIDUAL_PARAMETERIZATION",
    "POLYGON_IMAGE_ONLY_RESIDUAL_BC_RESIDUAL_AFFECTS",
    "POLYGON_IMAGE_ONLY_RESIDUAL_BC_RESIDUAL_UNAFFECTED",
    "LegacyVisionResidualHeadV1",
    "VisionResidualHeadV2",
    "VisionEncoderSpec",
    "VisionResidualBCPolicy",
    "describe_residual_policy_contract",
    "load_image_only_residual_bc_policy",
    "validate_image_only_residual_bc_metadata",
    "save_vision_residual_policy",
    "train_head_with_adam",
]
