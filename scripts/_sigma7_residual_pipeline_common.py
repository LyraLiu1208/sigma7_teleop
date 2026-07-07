from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from _sigma7_runtime import DEFAULT_MUJOCO_ROOT, ROOT, ensure_project_src_on_path

DEFAULT_PIPELINE_ROOT = ROOT / "artifacts" / "sigma7_residual_bc"
COLLECTION_MODE = "collection"
PRACTICE_MODE = "practice"


def ensure_mujoco_src_on_path() -> Path:
    return ensure_project_src_on_path()


def require_safe_segment(value: str, *, name: str) -> str:
    cleaned = str(value).strip()
    if not cleaned:
        raise ValueError(f"{name} must not be empty.")
    if cleaned in {".", ".."} or "/" in cleaned:
        raise ValueError(f"{name} must be a single path segment, observed {value!r}.")
    return cleaned


def scene_participant_mode_root(output_root: Path, scene: str, participant: str, mode: str) -> Path:
    return (
        Path(output_root)
        / require_safe_segment(scene, name="scene")
        / require_safe_segment(participant, name="participant")
        / require_safe_segment(mode, name="mode")
    )


def scene_root(output_root: Path, scene: str) -> Path:
    return Path(output_root) / require_safe_segment(scene, name="scene")


def participant_episodes_root(output_root: Path, scene: str, participant: str, mode: str) -> Path:
    return scene_participant_mode_root(output_root, scene, participant, mode) / "episodes"


def scene_collection_root(output_root: Path, scene: str) -> Path:
    return scene_root(output_root, scene) / COLLECTION_MODE


def scene_dataset_root(output_root: Path, scene: str) -> Path:
    return scene_collection_root(output_root, scene) / "datasets"


def scene_dataset_6d_root(output_root: Path, scene: str) -> Path:
    return scene_collection_root(output_root, scene) / "datasets_6d"


def scene_models_root(output_root: Path, scene: str) -> Path:
    return scene_collection_root(output_root, scene) / "models"


def scene_screening_root(output_root: Path, scene: str) -> Path:
    return scene_collection_root(output_root, scene) / "screening"


def scene_screening_participant_root(output_root: Path, scene: str, participant: str) -> Path:
    return scene_screening_root(output_root, scene) / require_safe_segment(participant, name="participant")


def scene_screening_controller_root(output_root: Path, scene: str, controller: str) -> Path:
    return scene_screening_root(output_root, scene) / require_safe_segment(controller, name="controller")


def scene_screening_participant_controller_root(output_root: Path, scene: str, participant: str, controller: str) -> Path:
    return scene_screening_participant_root(output_root, scene, participant) / require_safe_segment(controller, name="controller")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
