from __future__ import annotations

import contextlib
import ast
import importlib
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from stiffness_copilot_mujoco.learning.residual_stiffness import BaseStiffnessSpec
from stiffness_copilot_mujoco.learning.vision_residual_stiffness import describe_residual_policy_contract
from stiffness_copilot_mujoco.scenes import get_scene_spec, scene_names


DEFAULT_FROZEN_VISION_RESIZE = (224, 224)
DEFAULT_FROZEN_VISION_MEAN = (0.485, 0.456, 0.406)
DEFAULT_FROZEN_VISION_STD = (0.229, 0.224, 0.225)
DEFAULT_SCENE_HEAD_HIDDEN_DIM = 64
DEFAULT_DINOV3_BACKBONE_TYPE = "dinov3_small_frozen"
DEFAULT_HEAD_TYPE = "scene_specific_mlp_head_v1"
DEFAULT_RESIDUAL_PARAMETERIZATION = "scene_specific_residual_head_v1"
DEFAULT_MODEL_SOURCE_TYPE = "local_explicit"


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


def _freeze_module(module: nn.Module) -> nn.Module:
    module.requires_grad_(False)
    module.eval()
    return module


@contextlib.contextmanager
def _temporary_sys_path(paths: tuple[Path, ...] | list[Path]) -> Any:
    inserted: list[str] = []
    try:
        for path in paths:
            resolved = str(Path(path).resolve())
            if resolved not in sys.path:
                sys.path.insert(0, resolved)
                inserted.append(resolved)
        yield
    finally:
        for path in inserted:
            try:
                sys.path.remove(path)
            except ValueError:
                pass


def _copy_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _state_dict_equal(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> bool:
    if left.keys() != right.keys():
        return False
    return all(torch.equal(left[key], right[key]) for key in left)


def _ensure_batched_rgb(rgb_batch: torch.Tensor | np.ndarray | list[Any]) -> torch.Tensor:
    tensor = torch.as_tensor(rgb_batch)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4:
        raise ValueError(f"RGB batch must have 3 or 4 dimensions, observed {tuple(tensor.shape)}.")
    if tensor.shape[-1] == 3:
        tensor = tensor.permute(0, 3, 1, 2)
    elif tensor.shape[1] != 3:
        raise ValueError(
            f"RGB batch must be channel-last or channel-first with 3 channels, observed {tuple(tensor.shape)}."
        )
    return tensor.contiguous()


@dataclass(frozen=True)
class FrozenBackbonePreprocessConfig:
    resize_height: int = DEFAULT_FROZEN_VISION_RESIZE[0]
    resize_width: int = DEFAULT_FROZEN_VISION_RESIZE[1]
    mean: tuple[float, float, float] = DEFAULT_FROZEN_VISION_MEAN
    std: tuple[float, float, float] = DEFAULT_FROZEN_VISION_STD
    interpolation: str = "bilinear"
    antialias: bool = True
    scale_to_unit_interval: bool = True

    def to_metadata(self) -> dict[str, Any]:
        return {
            "resize_height": int(self.resize_height),
            "resize_width": int(self.resize_width),
            "target_size": [int(self.resize_height), int(self.resize_width)],
            "mean": [float(value) for value in self.mean],
            "std": [float(value) for value in self.std],
            "interpolation": self.interpolation,
            "antialias": bool(self.antialias),
            "scale_to_unit_interval": bool(self.scale_to_unit_interval),
        }


@dataclass(frozen=True)
class FrozenBackboneLoadConfig:
    dinov3_repo: Path
    dinov3_checkpoint: Path
    dinov3_entrypoint: str
    preprocess_config: FrozenBackbonePreprocessConfig = field(default_factory=FrozenBackbonePreprocessConfig)
    seed: int = 0

    def to_metadata(self) -> dict[str, Any]:
        return {
            "dinov3_repo": str(self.dinov3_repo),
            "dinov3_checkpoint": str(self.dinov3_checkpoint),
            "dinov3_entrypoint": self.dinov3_entrypoint,
            "seed": int(self.seed),
            "preprocess_config": self.preprocess_config.to_metadata(),
        }


def preprocess_rgb_batch(rgb_batch: torch.Tensor | np.ndarray | list[Any], config: FrozenBackbonePreprocessConfig) -> torch.Tensor:
    original = torch.as_tensor(rgb_batch)
    tensor = _ensure_batched_rgb(original)
    original_is_integer = not original.is_floating_point()
    tensor = tensor.to(dtype=torch.float32)
    if original_is_integer:
        tensor = tensor / 255.0
    elif config.scale_to_unit_interval:
        max_value = float(tensor.max().item()) if tensor.numel() else 0.0
        if max_value > 1.5:
            tensor = tensor / 255.0
    if tensor.shape[-2:] != (config.resize_height, config.resize_width):
        tensor = F.interpolate(
            tensor,
            size=(config.resize_height, config.resize_width),
            mode=config.interpolation,
            align_corners=False,
            antialias=config.antialias,
        )
    mean = torch.tensor(config.mean, dtype=torch.float32, device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(config.std, dtype=torch.float32, device=tensor.device).view(1, 3, 1, 1)
    return (tensor - mean) / std


def _coerce_backbone_features(output: object) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        tensor = output
    elif isinstance(output, dict):
        tensor = None
        for key in (
            "x_norm_clstoken",
            "cls_token",
            "features",
            "feature",
            "embedding",
            "embeddings",
            "last_hidden_state",
            "output",
        ):
            value = output.get(key)
            if value is not None:
                tensor = _coerce_backbone_features(value)
                break
        if tensor is None:
            raise ValueError(f"Unsupported backbone output mapping keys: {sorted(output.keys())}.")
        return tensor
    elif isinstance(output, (tuple, list)):
        if not output:
            raise ValueError("Backbone output tuple/list is empty.")
        return _coerce_backbone_features(output[0])
    else:
        raise ValueError(f"Unsupported backbone output type: {type(output)!r}.")

    if tensor.ndim == 4:
        return tensor.mean(dim=(2, 3))
    if tensor.ndim == 3:
        return tensor[:, 0, :]
    if tensor.ndim == 2:
        return tensor
    raise ValueError(f"Unsupported backbone output tensor shape: {tuple(tensor.shape)}.")


class LocalDinov3LoadError(RuntimeError):
    pass


def _parse_local_hubconf_entrypoints(repo_path: Path) -> dict[str, str]:
    hubconf_path = repo_path / "hubconf.py"
    try:
        source = hubconf_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(hubconf_path))
    except Exception as exc:  # pragma: no cover - error path is environment-dependent
        raise LocalDinov3LoadError(
            f"Failed to inspect local DINOv3 hub entrypoints from {hubconf_path}: {exc}"
        ) from exc

    entrypoints: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                public_name = alias.asname or alias.name
                entrypoints[public_name] = node.module
    return dict(sorted(entrypoints.items()))


def _validate_local_dinov3_source(repo_path: Path, checkpoint_path: Path, entrypoint: str) -> dict[str, str]:
    errors: list[str] = []
    if not repo_path.exists():
        errors.append(f"Local DINOv3 repo does not exist: {repo_path}")
    elif not repo_path.is_dir():
        errors.append(f"Local DINOv3 repo is not a directory: {repo_path}")
    hubconf_path = repo_path / "hubconf.py"
    if not hubconf_path.is_file():
        errors.append(f"Local DINOv3 repo is missing hubconf.py: {hubconf_path}")
    if not checkpoint_path.exists():
        errors.append(f"Local DINOv3 checkpoint does not exist: {checkpoint_path}")
    elif not checkpoint_path.is_file():
        errors.append(f"Local DINOv3 checkpoint is not a file: {checkpoint_path}")
    if not entrypoint.strip():
        errors.append("Local DINOv3 entrypoint must be a non-empty string.")
    if errors:
        raise LocalDinov3LoadError("\n".join(errors))

    available_entrypoints = _parse_local_hubconf_entrypoints(repo_path)
    if entrypoint not in available_entrypoints:
        available_text = ", ".join(sorted(available_entrypoints)) or "<none>"
        raise LocalDinov3LoadError(
            f"Local DINOv3 entrypoint {entrypoint!r} was not found in {hubconf_path}.\n"
            f"Available entrypoints: {available_text}"
        )
    return available_entrypoints


def _freeze_backbone_parameters(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()


class SharedFrozenVisionBackbone(nn.Module):
    def __init__(
        self,
        backbone_module: nn.Module,
        *,
        backbone_type: str,
        dinov3_repo: Path,
        dinov3_checkpoint: Path,
        dinov3_entrypoint: str,
        preprocess_config: FrozenBackbonePreprocessConfig,
        model_source_type: str = DEFAULT_MODEL_SOURCE_TYPE,
        load_attempts: tuple[dict[str, Any], ...] = (),
    ) -> None:
        super().__init__()
        self.backbone = backbone_module
        self.backbone_type = str(backbone_type)
        self.dinov3_repo = Path(dinov3_repo)
        self.dinov3_checkpoint = Path(dinov3_checkpoint)
        self.dinov3_entrypoint = str(dinov3_entrypoint)
        self.preprocess_config = preprocess_config
        self.model_source_type = str(model_source_type)
        self.load_attempts = tuple(load_attempts)
        _freeze_backbone_parameters(self.backbone)
        self.eval()

    def preprocess(self, rgb_batch: torch.Tensor | np.ndarray | list[Any]) -> torch.Tensor:
        return preprocess_rgb_batch(rgb_batch, self.preprocess_config)

    def encode(self, rgb_batch: torch.Tensor | np.ndarray | list[Any]) -> torch.Tensor:
        inputs = self.preprocess(rgb_batch)
        with torch.no_grad():
            output = self.backbone(inputs)
        return _coerce_backbone_features(output)

    def forward(self, rgb_batch: torch.Tensor | np.ndarray | list[Any]) -> torch.Tensor:
        return self.encode(rgb_batch)

    def assert_frozen(self) -> None:
        for parameter in self.backbone.parameters():
            if parameter.requires_grad:
                raise AssertionError("Backbone parameters must be frozen.")
        if self.training:
            raise AssertionError("Backbone module must be in eval mode.")

    def to_metadata(self) -> dict[str, Any]:
        return {
            "backbone_type": self.backbone_type,
            "backbone_frozen": True,
            "model_source_type": self.model_source_type,
            "dinov3_repo": str(self.dinov3_repo),
            "dinov3_checkpoint": str(self.dinov3_checkpoint),
            "dinov3_entrypoint": self.dinov3_entrypoint,
            "preprocessing_config": self.preprocess_config.to_metadata(),
            "load_attempts": [_json_ready(attempt) for attempt in self.load_attempts],
        }


class SceneResidualHead(nn.Module):
    def __init__(
        self,
        *,
        scene: str,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = DEFAULT_SCENE_HEAD_HIDDEN_DIM,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.scene = str(scene)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.head_type = DEFAULT_HEAD_TYPE
        self.residual_parameterization = DEFAULT_RESIDUAL_PARAMETERIZATION
        self.net = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )
        self._initialize(seed=seed)
        _freeze_backbone_parameters(self)

    def _initialize(self, *, seed: int) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    fan_in = float(module.in_features)
                    scale = math.sqrt(2.0 / max(fan_in, 1.0))
                    module.weight.copy_(torch.randn(module.weight.shape, generator=generator) * scale)
                    module.bias.zero_()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "scene": self.scene,
            "head_type": self.head_type,
            "residual_parameterization": self.residual_parameterization,
            "input_dim": int(self.input_dim),
            "output_dim": int(self.output_dim),
            "hidden_dim": int(self.hidden_dim),
            "frozen": True,
        }


class FrozenSceneHeadBank(nn.Module):
    def __init__(self, backbone: SharedFrozenVisionBackbone, heads: dict[str, SceneResidualHead]) -> None:
        super().__init__()
        self.backbone = backbone
        self.heads = nn.ModuleDict(heads)
        _freeze_backbone_parameters(self)

    def forward_scene(self, scene: str, rgb_batch: torch.Tensor | np.ndarray | list[Any]) -> tuple[torch.Tensor, torch.Tensor]:
        if scene not in self.heads:
            raise KeyError(f"Unknown scene {scene!r}; available scenes: {sorted(self.heads.keys())}.")
        features = self.backbone.encode(rgb_batch)
        output = self.heads[scene](features)
        return features, output

    def to_metadata(self) -> dict[str, Any]:
        return {
            "backbone": self.backbone.to_metadata(),
            "heads": {scene: head.to_metadata() for scene, head in self.heads.items()},
        }


def _scene_contract(scene: str) -> dict[str, Any]:
    spec = get_scene_spec(scene)
    contract = describe_residual_policy_contract(
        BaseStiffnessSpec.from_matrix(
            np.eye(3, dtype=float),
            active_groups=spec.active_groups,
            active_group_names=spec.active_group_names,
            residual_bound=spec.residual_bound,
        )
    )
    return {
        "scene": spec.name,
        "output_dim": len(spec.active_groups),
        "active_group_names": list(spec.active_group_names),
        "output_space": contract["output_space"],
        "residual_parameterization": contract["residual_parameterization"],
    }


def _load_local_dinov3_model(load_config: FrozenBackboneLoadConfig) -> tuple[SharedFrozenVisionBackbone, dict[str, Any]]:
    repo_path = Path(load_config.dinov3_repo).expanduser().resolve()
    checkpoint_path = Path(load_config.dinov3_checkpoint).expanduser().resolve()
    entrypoint = str(load_config.dinov3_entrypoint).strip()
    available_entrypoints = _validate_local_dinov3_source(repo_path, checkpoint_path, entrypoint)
    entrypoint_module = available_entrypoints[entrypoint]

    load_attempts: list[dict[str, Any]] = [
        {"kind": "repo_validation", "target": str(repo_path), "status": "passed"},
        {"kind": "checkpoint_validation", "target": str(checkpoint_path), "status": "passed"},
        {"kind": "entrypoint_validation", "target": entrypoint, "status": "passed"},
        {
            "kind": "available_entrypoints",
            "target": str(repo_path),
            "status": "passed",
            "entrypoints": available_entrypoints,
        },
    ]

    try:
        with _temporary_sys_path([repo_path]):
            module = importlib.import_module(entrypoint_module)
            entrypoint_fn = getattr(module, entrypoint, None)
            if entrypoint_fn is None or not callable(entrypoint_fn):
                raise LocalDinov3LoadError(
                    f"Local DINOv3 entrypoint {entrypoint!r} is not callable in module {entrypoint_module!r}."
                )
            model = entrypoint_fn(weights=str(checkpoint_path))
        load_attempts.append(
            {
                "kind": "local_hubconf_call",
                "target": f"{repo_path}:{entrypoint}",
                "status": "success",
                "module": entrypoint_module,
                "weights": str(checkpoint_path),
            }
        )
    except Exception as exc:
        raise LocalDinov3LoadError(
            f"Failed to instantiate local DINOv3 entrypoint {entrypoint!r} from repo {repo_path} "
            f"with checkpoint {checkpoint_path}: {exc}"
        ) from exc

    if not isinstance(model, nn.Module):
        raise LocalDinov3LoadError(
            f"Local DINOv3 entrypoint {entrypoint!r} from repo {repo_path} did not return a torch.nn.Module."
        )

    _freeze_backbone_parameters(model)
    backbone = SharedFrozenVisionBackbone(
        backbone_module=model,
        backbone_type=DEFAULT_DINOV3_BACKBONE_TYPE,
        dinov3_repo=repo_path,
        dinov3_checkpoint=checkpoint_path,
        dinov3_entrypoint=entrypoint,
        preprocess_config=load_config.preprocess_config,
        model_source_type=DEFAULT_MODEL_SOURCE_TYPE,
        load_attempts=tuple(load_attempts),
    )
    backbone.assert_frozen()
    return backbone, {
        "backbone_type": DEFAULT_DINOV3_BACKBONE_TYPE,
        "backbone_frozen": True,
        "model_source_type": DEFAULT_MODEL_SOURCE_TYPE,
        "dinov3_repo": str(repo_path),
        "dinov3_checkpoint": str(checkpoint_path),
        "dinov3_entrypoint": entrypoint,
        "load_attempts": load_attempts,
    }


def build_frozen_dinov3_backbone(load_config: FrozenBackboneLoadConfig) -> tuple[SharedFrozenVisionBackbone, dict[str, Any]]:
    return _load_local_dinov3_model(load_config)


def build_scene_head(
    *,
    scene: str,
    feature_dim: int,
    hidden_dim: int = DEFAULT_SCENE_HEAD_HIDDEN_DIM,
    seed: int = 0,
) -> tuple[SceneResidualHead, dict[str, Any]]:
    contract = _scene_contract(scene)
    head = SceneResidualHead(
        scene=scene,
        input_dim=feature_dim,
        output_dim=int(contract["output_dim"]),
        hidden_dim=hidden_dim,
        seed=seed,
    )
    return head, contract


def build_scene_head_bank(
    *,
    backbone: SharedFrozenVisionBackbone,
    feature_dim: int,
    scenes: tuple[str, ...] | None = None,
    hidden_dim: int = DEFAULT_SCENE_HEAD_HIDDEN_DIM,
    seed: int = 0,
) -> FrozenSceneHeadBank:
    scene_list = scenes or scene_names()
    heads: dict[str, SceneResidualHead] = {}
    for index, scene in enumerate(scene_list):
        contract = _scene_contract(scene)
        heads[scene] = SceneResidualHead(
            scene=scene,
            input_dim=feature_dim,
            output_dim=int(contract["output_dim"]),
            hidden_dim=hidden_dim,
            seed=seed + index,
        )
    return FrozenSceneHeadBank(backbone=backbone, heads=heads)


def build_synthetic_rgb_batch(
    *,
    batch_size: int = 4,
    height: int = 128,
    width: int = 128,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(batch_size, height, width, 3), dtype=np.uint8)


def load_rgb_batch_from_dataset(path: Path, *, limit: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        if "rgb_images" not in data.files:
            raise ValueError(f"{path} does not contain rgb_images.")
        rgb_images = np.asarray(data["rgb_images"], dtype=np.uint8)
        metadata = json.loads(str(data["metadata"])) if "metadata" in data.files else {}
    if rgb_images.ndim != 4 or rgb_images.shape[-1] != 3:
        raise ValueError(f"rgb_images must have shape [N, H, W, 3], observed {rgb_images.shape}.")
    if limit is not None:
        rgb_images = rgb_images[: max(int(limit), 1)]
    return rgb_images, metadata


def smoke_scene_head_forward(
    *,
    scene: str,
    backbone: SharedFrozenVisionBackbone,
    rgb_batch: np.ndarray | torch.Tensor | list[Any],
    hidden_dim: int = DEFAULT_SCENE_HEAD_HIDDEN_DIM,
    seed: int = 0,
) -> dict[str, Any]:
    contract = _scene_contract(scene)
    initial_state = _copy_state_dict(backbone.backbone)
    preprocessed = backbone.preprocess(rgb_batch)
    features = backbone.encode(rgb_batch)
    post_state = _copy_state_dict(backbone.backbone)
    if not _state_dict_equal(initial_state, post_state):
        raise RuntimeError("Frozen backbone state changed across a forward pass.")
    head, _ = build_scene_head(scene=scene, feature_dim=int(features.shape[-1]), hidden_dim=hidden_dim, seed=seed)
    output = head(features)
    expected_shape = (int(features.shape[0]), int(contract["output_dim"]))
    if tuple(output.shape) != expected_shape:
        raise RuntimeError(f"{scene} head output shape mismatch: expected {expected_shape}, observed {tuple(output.shape)}.")
    if not torch.isfinite(output).all():
        raise RuntimeError(f"{scene} head produced non-finite output.")
    return {
        "scene": scene,
        "scene_contract": contract,
        "head_type": head.head_type,
        "input_shape": [int(dim) for dim in np.asarray(rgb_batch).shape],
        "preprocessed_shape": [int(dim) for dim in preprocessed.shape],
        "feature_shape": [int(dim) for dim in features.shape],
        "output_shape": [int(dim) for dim in output.shape],
        "feature_dim": int(features.shape[-1]),
        "output_dim": int(contract["output_dim"]),
        "active_group_names": list(contract["active_group_names"]),
        "residual_parameterization": contract["residual_parameterization"],
        "all_finite": bool(torch.isfinite(output).all().item()),
        "backbone_type": backbone.backbone_type,
        "backbone_frozen": True,
        "model_source_type": backbone.model_source_type,
        "dinov3_repo": str(backbone.dinov3_repo),
        "dinov3_checkpoint": str(backbone.dinov3_checkpoint),
        "dinov3_entrypoint": backbone.dinov3_entrypoint,
        "preprocessing_config": backbone.preprocess_config.to_metadata(),
    }


def run_frozen_backbone_head_smoke(
    *,
    dinov3_repo: Path,
    dinov3_checkpoint: Path,
    dinov3_entrypoint: str,
    rgb_batch: np.ndarray | torch.Tensor | list[Any] | None = None,
    dataset_path: Path | None = None,
    batch_size: int = 4,
    hidden_dim: int = DEFAULT_SCENE_HEAD_HIDDEN_DIM,
    seed: int = 0,
    preprocess_config: FrozenBackbonePreprocessConfig | None = None,
    scenes: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    preprocess_config = preprocess_config or FrozenBackbonePreprocessConfig()
    load_config = FrozenBackboneLoadConfig(
        dinov3_repo=dinov3_repo,
        dinov3_checkpoint=dinov3_checkpoint,
        dinov3_entrypoint=dinov3_entrypoint,
        preprocess_config=preprocess_config,
        seed=seed,
    )
    backbone, backbone_metadata = build_frozen_dinov3_backbone(load_config)

    if rgb_batch is None:
        if dataset_path is not None:
            rgb_batch, dataset_metadata = load_rgb_batch_from_dataset(dataset_path, limit=batch_size)
            batch_source = {
                "kind": "dataset",
                "path": str(dataset_path),
                "metadata_scene": dataset_metadata.get("scene"),
                "metadata_setting_id": dataset_metadata.get("setting_id"),
            }
        else:
            rgb_batch = build_synthetic_rgb_batch(batch_size=batch_size, seed=seed)
            batch_source = {"kind": "synthetic", "seed": int(seed)}
    else:
        batch_source = {"kind": "provided"}

    input_shape = [int(dim) for dim in np.asarray(rgb_batch).shape]
    preprocessed = backbone.preprocess(rgb_batch)
    feature_probe = backbone.encode(rgb_batch)
    feature_dim = int(feature_probe.shape[-1])
    scene_list = scenes or scene_names()

    scene_records = []
    output_shapes: dict[str, list[int]] = {}
    for index, scene in enumerate(scene_list):
        record = smoke_scene_head_forward(
            scene=scene,
            backbone=backbone,
            rgb_batch=rgb_batch,
            hidden_dim=hidden_dim,
            seed=seed + index,
        )
        scene_records.append(record)
        output_shapes[scene] = list(record["output_shape"])

    return {
        "status": "passed",
        "backbone_type": backbone_metadata["backbone_type"],
        "backbone_frozen": True,
        "model_source_type": backbone_metadata["model_source_type"],
        "dinov3_repo": backbone_metadata["dinov3_repo"],
        "dinov3_checkpoint": backbone_metadata["dinov3_checkpoint"],
        "dinov3_entrypoint": backbone_metadata["dinov3_entrypoint"],
        "load_attempts": backbone_metadata["load_attempts"],
        "preprocessing_config": backbone.preprocess_config.to_metadata(),
        "head_type": DEFAULT_HEAD_TYPE,
        "head_hidden_dim": int(hidden_dim),
        "scene_count": int(len(scene_list)),
        "scenes": scene_records,
        "scene_records_by_name": {record["scene"]: record for record in scene_records},
        "input_shape": input_shape,
        "preprocessed_shape": [int(dim) for dim in preprocessed.shape],
        "feature_shape": [int(dim) for dim in feature_probe.shape],
        "feature_dim": feature_dim,
        "batch_source": batch_source,
        "output_shape_by_scene": output_shapes,
        "all_finite": bool(all(record["all_finite"] for record in scene_records)),
        "scene_contracts": {scene: _scene_contract(scene) for scene in scene_list},
        "backbone_metadata": backbone.to_metadata(),
    }


__all__ = [
    "DEFAULT_DINOV3_BACKBONE_TYPE",
    "DEFAULT_FROZEN_VISION_MEAN",
    "DEFAULT_FROZEN_VISION_RESIZE",
    "DEFAULT_FROZEN_VISION_STD",
    "DEFAULT_HEAD_TYPE",
    "DEFAULT_MODEL_SOURCE_TYPE",
    "DEFAULT_RESIDUAL_PARAMETERIZATION",
    "DEFAULT_SCENE_HEAD_HIDDEN_DIM",
    "FrozenBackboneLoadConfig",
    "FrozenBackbonePreprocessConfig",
    "FrozenSceneHeadBank",
    "LocalDinov3LoadError",
    "SceneResidualHead",
    "SharedFrozenVisionBackbone",
    "build_frozen_dinov3_backbone",
    "build_scene_head",
    "build_scene_head_bank",
    "build_synthetic_rgb_batch",
    "load_rgb_batch_from_dataset",
    "preprocess_rgb_batch",
    "run_frozen_backbone_head_smoke",
    "scene_names",
    "smoke_scene_head_forward",
]
