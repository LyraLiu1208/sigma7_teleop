from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from stiffness_copilot_mujoco.learning.residual_label_projection import is_residual_first_projection
from stiffness_copilot_mujoco.learning.residual_dataset import validate_residual_dataset
from stiffness_copilot_mujoco.learning.residual_stiffness import BaseStiffnessSpec
from stiffness_copilot_mujoco.learning.vision_residual_stiffness import (
    describe_residual_policy_contract,
    load_image_only_residual_bc_policy,
)
from stiffness_copilot_mujoco.scenes import get_scene_spec
from stiffness_copilot_mujoco.sim.scene import ROOT


AUDIT_VERSION = "track_a_mainline_artifact_contract_audit_v1"
CURRENT_MAINLINE_SCENE_ROOTS: dict[str, Path] = {
    "circle": ROOT / "artifacts" / "track_a_controller_consistent" / "track_a_c600",
    "polygon": ROOT / "artifacts" / "track_a_controller_consistent" / "polygon" / "track_a_c600",
    "star": ROOT / "artifacts" / "track_a_controller_consistent" / "star" / "track_a_c600",
}
EXCLUDED_ROOT_NAMES = {"archive", "intermediate_results"}


@dataclass(frozen=True)
class TrackAArtifactAuditRecord:
    scene: str
    scene_source: str
    controller_id: str
    scenario_root: str
    dataset_path: str
    policy_path: str
    paired_output_root: str
    paired_summary_path: str
    paired_metadata_path: str
    setting_id: str
    setting_id_source: str
    output_dim: int
    output_dim_source: str
    residual_parameterization: str
    contract_source: str
    active_group_names: tuple[str, ...]
    base_stiffness_spec: dict[str, Any]
    residual_group_target_shape: tuple[int, ...]
    episode_specs_path: str
    episodes_csv_path: str
    controller_policy_consistency_passed: bool
    status: str
    errors: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    discovery_notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["active_group_names"] = list(self.active_group_names)
        payload["base_stiffness_spec"] = _json_ready(payload["base_stiffness_spec"])
        payload["residual_group_target_shape"] = list(self.residual_group_target_shape)
        payload["errors"] = list(self.errors)
        payload["warnings"] = list(self.warnings)
        payload["discovery_notes"] = list(self.discovery_notes)
        return payload


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


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_npz_metadata(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        if "metadata" not in data.files:
            raise ValueError(f"{path} is missing metadata.")
        return json.loads(str(data["metadata"]))


def _resolve_existing_path(value: object, *, base_dirs: Sequence[Path] = ()) -> Path:
    raw = Path(str(value))
    if raw.is_absolute():
        return raw.resolve()
    search_roots = tuple(base_dirs) or (ROOT,)
    for base in search_roots:
        candidate = (base / raw).resolve()
        if candidate.exists():
            return candidate
    return (search_roots[0] / raw).resolve()


def _path_is_primary_candidate(path: Path, *, scenario_root: Path) -> bool:
    try:
        relative = path.resolve().relative_to(scenario_root.resolve())
    except ValueError:
        return False
    return not any(part in EXCLUDED_ROOT_NAMES for part in relative.parts)


def _candidate_search_roots(scenario_root: Path) -> tuple[Path, ...]:
    roots: list[Path] = []
    for child in ("datasets", "models", "evaluations"):
        candidate = scenario_root / child
        if candidate.exists():
            roots.append(candidate)
    return tuple(roots)


def _find_dataset_candidates(scenario_root: Path) -> list[Path]:
    candidates: set[Path] = set()
    for search_root in _candidate_search_roots(scenario_root):
        for path in search_root.rglob("eligible_residual_bc.npz"):
            if path.is_file() and _path_is_primary_candidate(path, scenario_root=scenario_root):
                candidates.add(path.resolve())
    return sorted(candidates)


def _find_policy_candidates(scenario_root: Path) -> list[Path]:
    candidates: set[Path] = set()
    search_root = scenario_root / "models"
    if not search_root.exists():
        return []
    for path in search_root.rglob("*.npz"):
        if path.is_file() and _path_is_primary_candidate(path, scenario_root=scenario_root):
            candidates.add(path.resolve())
    return sorted(candidates)


def _find_paired_candidates(scenario_root: Path) -> list[Path]:
    evaluations_root = scenario_root / "evaluations"
    if not evaluations_root.exists():
        return []
    candidates: set[Path] = set()
    for metadata_path in evaluations_root.rglob("paired_v2_metadata.json"):
        if not metadata_path.is_file() or not _path_is_primary_candidate(metadata_path, scenario_root=scenario_root):
            continue
        parent = metadata_path.parent
        if (parent / "paired_v2_summary.json").exists() and (parent / "paired_v2_episodes.csv").exists():
            candidates.add(parent.resolve())
    return sorted(candidates)


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


def _candidate_score(value: object, expected: object) -> int:
    return 1 if expected is not None and str(value) == str(expected) else 0


def _first_present(*values: object) -> object | None:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _first_present_with_source(*values: tuple[str, object]) -> tuple[object | None, str | None]:
    for source, value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value, source
    return None, None


def _paired_scene_view(metadata: dict[str, Any]) -> dict[str, str | None]:
    scenario = metadata.get("scenario") if isinstance(metadata.get("scenario"), dict) else {}
    working_config = metadata.get("working_config") if isinstance(metadata.get("working_config"), dict) else {}
    scene, scene_source = _first_present_with_source(
        ("paired_v2_metadata.scenario_scene", metadata.get("scenario_scene")),
        ("paired_v2_metadata.scenario_id", metadata.get("scenario_id")),
        ("paired_v2_metadata.scene", metadata.get("scene")),
        ("paired_v2_metadata.scenario.scene", scenario.get("scene")),
        ("paired_v2_metadata.working_config.scene", working_config.get("scene")),
    )
    setting_id, setting_id_source = _first_present_with_source(
        ("paired_v2_metadata.scenario_setting", metadata.get("scenario_setting")),
        ("paired_v2_metadata.setting", metadata.get("setting")),
        ("paired_v2_metadata.scenario.setting", scenario.get("setting")),
        ("paired_v2_metadata.working_config.setting", working_config.get("setting")),
    )
    profile_name, _ = _first_present_with_source(
        ("paired_v2_metadata.scenario_profile_name", metadata.get("scenario_profile_name")),
        ("paired_v2_metadata.profile_name", metadata.get("profile_name")),
        ("paired_v2_metadata.scenario.profile_name", scenario.get("profile_name")),
        ("paired_v2_metadata.working_config.profile_name", working_config.get("profile_name")),
        ("paired_v2_metadata.scenario.contact_profile", scenario.get("contact_profile")),
        ("paired_v2_metadata.working_config.collection_controller_profile", working_config.get("collection_controller_profile")),
    )
    contact_condition_name, _ = _first_present_with_source(
        ("paired_v2_metadata.scenario_contact_condition_name", metadata.get("scenario_contact_condition_name")),
        ("paired_v2_metadata.contact_condition_name", metadata.get("contact_condition_name")),
        ("paired_v2_metadata.scenario.contact_condition_name", scenario.get("contact_condition_name")),
        ("paired_v2_metadata.working_config.contact_condition_name", working_config.get("contact_condition_name")),
    )
    return {
        "scene": None if scene is None else str(scene),
        "scene_source": scene_source,
        "setting_id": None if setting_id is None else str(setting_id),
        "setting_id_source": setting_id_source,
        "profile_name": None if profile_name is None else str(profile_name),
        "contact_condition_name": None if contact_condition_name is None else str(contact_condition_name),
    }


def _score_paired_candidate(*, path: Path, metadata: dict[str, Any], scene: str) -> int:
    view = _paired_scene_view(metadata)
    score = 0
    score += 100 * _candidate_score(view["scene"], scene)
    score += 80 * _candidate_score(metadata.get("controller_policy_consistency_passed"), True)
    if "paired_v2" in path.as_posix():
        score += 5
    if "intermediate_results" in path.as_posix():
        score -= 50
    return score


def _score_policy_candidate(
    *,
    path: Path,
    metadata: dict[str, Any],
    scene: str,
    setting_id: str,
    controller_id: str,
    expected_output_dim: int,
    expected_residual_parameterization: str,
) -> int:
    contract = _scene_contract(scene)
    score = 0
    score += 100 * _candidate_score(metadata.get("scene"), scene)
    score += 80 * _candidate_score(metadata.get("setting_id"), setting_id)
    score += 60 * _candidate_score(metadata.get("reference_controller_id"), controller_id)
    score += 60 * _candidate_score(metadata.get("collection_controller_id"), controller_id)
    score += 40 * _candidate_score(metadata.get("output_dim"), expected_output_dim)
    score += 40 * _candidate_score(metadata.get("residual_parameterization"), expected_residual_parameterization)
    score += 40 * _candidate_score(metadata.get("output_space"), contract["output_space"])
    if metadata.get("method_name") == "image_only_residual_bc":
        score += 20
    if metadata.get("input_mode") == "image_only":
        score += 20
    if "policy" in path.name:
        score += 10
    if "rerun" in path.as_posix():
        score -= 20
    return score


def _score_dataset_candidate(
    *,
    path: Path,
    metadata: dict[str, Any],
    scene: str,
    setting_id: str,
    controller_id: str,
    profile_name: str | None,
    contact_condition_name: str | None,
    expected_output_dim: int,
    expected_residual_parameterization: str,
    expected_active_group_names: tuple[str, ...],
) -> int:
    base_spec = BaseStiffnessSpec.from_metadata(metadata["base_stiffness_spec"])
    contract = describe_residual_policy_contract(base_spec)
    score = 0
    score += 100 * _candidate_score(metadata.get("scene"), scene)
    score += 80 * _candidate_score(metadata.get("setting_id"), setting_id)
    score += 60 * _candidate_score(metadata.get("collection_controller_id"), controller_id)
    score += 20 * _candidate_score(metadata.get("profile_name"), profile_name)
    score += 20 * _candidate_score(metadata.get("contact_condition_name"), contact_condition_name)
    score += 20 * _candidate_score(contract["output_dim"], expected_output_dim)
    score += 20 * _candidate_score(contract["residual_parameterization"], expected_residual_parameterization)
    score += 20 * _candidate_score(tuple(base_spec.active_group_names), expected_active_group_names)
    score += 30 * _candidate_score(is_residual_first_projection(metadata.get("label_projection")), True)
    if "rerun" in path.as_posix():
        score -= 20
    if "observation" in path.as_posix():
        score -= 10
    return score


def _choose_best_candidate(candidates: list[Path], *, score_fn) -> Path:
    if not candidates:
        raise FileNotFoundError("No candidate artifacts were discovered in the scenario root.")
    scored = sorted(((score_fn(path), path) for path in candidates), key=lambda item: (-item[0], item[1].as_posix()))
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        raise ValueError(
            "Discovery is ambiguous; multiple artifacts scored equally well: "
            + ", ".join(str(path) for _, path in scored[:3])
        )
    return scored[0][1]


def _select_paired_result(scene: str, scenario_root: Path) -> tuple[Path, dict[str, Any], tuple[str, ...]]:
    candidates = _find_paired_candidates(scenario_root)
    if not candidates:
        raise FileNotFoundError(f"No paired_v2 artifacts were discovered under {scenario_root}.")
    selected = _choose_best_candidate(
        candidates,
        score_fn=lambda path: _score_paired_candidate(path=path, metadata=_load_json(path / "paired_v2_metadata.json"), scene=scene),
    )
    metadata = _load_json(selected / "paired_v2_metadata.json")
    discovery_notes = (f"paired_candidates={len(candidates)}",)
    return selected, metadata, discovery_notes


def _select_policy(
    *,
    scene: str,
    scenario_root: Path,
    setting_id: str,
    controller_id: str,
    expected_output_dim: int,
    expected_residual_parameterization: str,
    paired_policy_path: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    candidates = _find_policy_candidates(scenario_root)
    if not candidates:
        raise FileNotFoundError(f"No policy artifacts were discovered under {scenario_root}.")
    if paired_policy_path is not None and paired_policy_path.resolve() in candidates:
        selected = paired_policy_path.resolve()
        policy = load_image_only_residual_bc_policy(selected)
        return selected, dict(policy.metadata)
    scored: list[tuple[int, Path, dict[str, Any]]] = []
    for candidate in candidates:
        try:
            policy = load_image_only_residual_bc_policy(candidate)
        except Exception:
            continue
        metadata = dict(policy.metadata)
        scored.append(
            (
                _score_policy_candidate(
                    path=candidate,
                    metadata=metadata,
                    scene=scene,
                    setting_id=setting_id,
                    controller_id=controller_id,
                    expected_output_dim=expected_output_dim,
                    expected_residual_parameterization=expected_residual_parameterization,
                ),
                candidate,
                metadata,
            )
        )
    if not scored:
        raise FileNotFoundError(f"No image-only residual BC policy candidate was discovered under {scenario_root}.")
    scored.sort(key=lambda item: (-item[0], item[1].as_posix()))
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        raise ValueError(
            "Policy discovery is ambiguous; multiple candidates scored equally well: "
            + ", ".join(str(path) for _, path, _ in scored[:3])
        )
    selected_score, selected_path, selected_metadata = scored[0]
    if selected_score < 0:
        raise ValueError(f"Discovered policy candidates do not match the requested scene contract under {scenario_root}.")
    return selected_path, selected_metadata


def _select_dataset(
    *,
    scene: str,
    scenario_root: Path,
    setting_id: str,
    controller_id: str,
    profile_name: str | None,
    contact_condition_name: str | None,
    expected_output_dim: int,
    expected_residual_parameterization: str,
    expected_active_group_names: tuple[str, ...],
) -> tuple[Path, dict[str, Any], tuple[int, int]]:
    candidates = _find_dataset_candidates(scenario_root)
    if not candidates:
        raise FileNotFoundError(f"No eligible_residual_bc.npz datasets were discovered under {scenario_root}.")
    scored: list[tuple[int, Path, dict[str, Any]]] = []
    for candidate in candidates:
        try:
            metadata = _load_npz_metadata(candidate)
            if "base_stiffness_spec" not in metadata:
                continue
            score = _score_dataset_candidate(
                path=candidate,
                metadata=metadata,
                scene=scene,
                setting_id=setting_id,
                controller_id=controller_id,
                profile_name=profile_name,
                contact_condition_name=contact_condition_name,
                expected_output_dim=expected_output_dim,
                expected_residual_parameterization=expected_residual_parameterization,
                expected_active_group_names=expected_active_group_names,
            )
        except Exception:
            continue
        scored.append((score, candidate, metadata))
    if not scored:
        raise FileNotFoundError(f"No eligible_residual_bc.npz candidate with readable metadata was discovered under {scenario_root}.")
    scored.sort(key=lambda item: (-item[0], item[1].as_posix()))
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        raise ValueError(
            "Dataset discovery is ambiguous; multiple candidates scored equally well: "
            + ", ".join(str(path) for _, path, _ in scored[:3])
        )
    selected_score, selected, selected_metadata = scored[0]
    if selected_score < 0:
        raise ValueError(f"Discovered dataset candidates do not match the requested scene contract under {scenario_root}.")
    validate_residual_dataset(selected)
    with np.load(selected, allow_pickle=False) as data:
        if "residual_group_target" not in data.files:
            raise ValueError(f"{selected} is missing residual_group_target.")
        residual_group_target_shape = tuple(int(dim) for dim in data["residual_group_target"].shape)
    return selected, selected_metadata, residual_group_target_shape


def _failure_record(scene: str, *, error: Exception, scenario_root: Path | None = None) -> TrackAArtifactAuditRecord:
    root = scenario_root or CURRENT_MAINLINE_SCENE_ROOTS.get(scene)
    return TrackAArtifactAuditRecord(
        scene=scene,
        scene_source="audit_input.scene",
        controller_id="",
        scenario_root="" if root is None else str(root.resolve()),
        dataset_path="",
        policy_path="",
        paired_output_root="",
        paired_summary_path="",
        paired_metadata_path="",
        setting_id="",
        setting_id_source="unavailable",
        output_dim=0,
        output_dim_source="unavailable",
        residual_parameterization="",
        contract_source="scene_spec / describe_residual_policy_contract",
        active_group_names=(),
        base_stiffness_spec={},
        residual_group_target_shape=(),
        episode_specs_path="",
        episodes_csv_path="",
        controller_policy_consistency_passed=False,
        status="failed",
        errors=(str(error),),
        warnings=(),
        discovery_notes=(f"failure={error.__class__.__name__}",),
    )


def _validate_scene_contract(
    *,
    scene: str,
    output_dim: int,
    residual_parameterization: str,
    active_group_names: tuple[str, ...],
    base_stiffness_spec: BaseStiffnessSpec,
) -> None:
    scene_spec = get_scene_spec(scene)
    expected_output_dim = len(scene_spec.active_groups)
    if output_dim != expected_output_dim:
        raise ValueError(f"Scene {scene!r} expects output_dim={expected_output_dim}, observed {output_dim}.")
    if residual_parameterization != describe_residual_policy_contract(base_stiffness_spec)["residual_parameterization"]:
        raise ValueError(
            f"Scene {scene!r} residual_parameterization mismatch: observed {residual_parameterization!r}."
        )
    if tuple(active_group_names) != tuple(scene_spec.active_group_names):
        raise ValueError(
            f"Scene {scene!r} expects active_group_names={scene_spec.active_group_names!r}, observed {active_group_names!r}."
        )
    if len(base_stiffness_spec.active_groups) != output_dim:
        raise ValueError(
            f"base_stiffness_spec.active_groups length must match output_dim {output_dim}, "
            f"observed {len(base_stiffness_spec.active_groups)}."
        )
    if tuple(base_stiffness_spec.active_group_names) != tuple(active_group_names):
        raise ValueError("base_stiffness_spec.active_group_names must match the normalized active_group_names.")
    if tuple(base_stiffness_spec.residual_bounds.shape) != (output_dim,):
        raise ValueError(
            f"base_stiffness_spec.residual_bounds must have shape ({output_dim},), observed {base_stiffness_spec.residual_bounds.shape}."
        )


def _resolve_consistency(
    *,
    scene: str,
    setting_id: str,
    controller_id: str,
    dataset_metadata: dict[str, Any],
    policy_metadata: dict[str, Any],
    paired_metadata: dict[str, Any],
    dataset_path: Path,
    policy_path: Path,
    paired_path: Path,
) -> bool:
    paired_view = _paired_scene_view(paired_metadata)
    if str(dataset_metadata.get("scene")) != scene:
        raise ValueError(f"Dataset scene mismatch: expected {scene!r}, observed {dataset_metadata.get('scene')!r}.")
    if str(policy_metadata.get("scene")) != scene:
        raise ValueError(f"Policy scene mismatch: expected {scene!r}, observed {policy_metadata.get('scene')!r}.")
    if paired_view["scene"] != scene:
        raise ValueError(f"Paired result scene mismatch: expected {scene!r}, observed {paired_view['scene']!r}.")
    if str(dataset_metadata.get("setting_id")) != setting_id:
        raise ValueError(
            f"Dataset setting_id mismatch: expected {setting_id!r}, observed {dataset_metadata.get('setting_id')!r}."
        )
    if str(policy_metadata.get("setting_id")) != setting_id:
        raise ValueError(
            f"Policy setting_id mismatch: expected {setting_id!r}, observed {policy_metadata.get('setting_id')!r}."
        )
    if paired_view["setting_id"] != setting_id:
        raise ValueError(
            f"Paired result setting mismatch: expected {setting_id!r}, observed {paired_view['setting_id']!r}."
        )
    if str(dataset_metadata.get("collection_controller_id")) != controller_id:
        raise ValueError(
            "Dataset controller mismatch: expected "
            f"{controller_id!r}, observed {dataset_metadata.get('collection_controller_id')!r}."
        )
    if str(policy_metadata.get("reference_controller_id")) != controller_id:
        raise ValueError(
            "Policy reference_controller_id mismatch: expected "
            f"{controller_id!r}, observed {policy_metadata.get('reference_controller_id')!r}."
        )
    if str(paired_metadata.get("reference_controller_id")) != controller_id:
        raise ValueError(
            "Paired result reference_controller_id mismatch: expected "
            f"{controller_id!r}, observed {paired_metadata.get('reference_controller_id')!r}."
        )
    if str(paired_metadata.get("baseline_controller_id")) != controller_id:
        raise ValueError(
            "Paired result baseline_controller_id mismatch: expected "
            f"{controller_id!r}, observed {paired_metadata.get('baseline_controller_id')!r}."
        )
    if str(paired_metadata.get("residual_reference_controller_id")) != controller_id:
        raise ValueError(
            "Paired result residual_reference_controller_id mismatch: expected "
            f"{controller_id!r}, observed {paired_metadata.get('residual_reference_controller_id')!r}."
        )
    paired_flag = paired_metadata.get("controller_policy_consistency_passed")
    if paired_flag is not None and bool(paired_flag) is not True:
        raise ValueError("Paired metadata controller_policy_consistency_passed is false.")
    paired_policy_path = paired_metadata.get("policy_path")
    if paired_policy_path is not None:
        resolved_paired_policy_path = _resolve_existing_path(paired_policy_path, base_dirs=(ROOT, paired_path.parent))
        if resolved_paired_policy_path.resolve() != policy_path.resolve():
            raise ValueError(
                "Paired metadata policy_path disagrees with the selected policy artifact: "
                f"{resolved_paired_policy_path} != {policy_path.resolve()}."
            )
    return True


def audit_track_a_mainline_scene(scene: str, *, scenario_root: Path | None = None) -> TrackAArtifactAuditRecord:
    if scene not in CURRENT_MAINLINE_SCENE_ROOTS:
        raise KeyError(f"Unsupported Track A scene {scene!r}. Expected one of {sorted(CURRENT_MAINLINE_SCENE_ROOTS)}.")
    root = (scenario_root or CURRENT_MAINLINE_SCENE_ROOTS[scene]).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Scenario root does not exist: {root}")

    scene_spec = get_scene_spec(scene)
    expected_contract = _scene_contract(scene)

    paired_root, paired_metadata, paired_notes = _select_paired_result(scene, root)
    paired_view = _paired_scene_view(paired_metadata)
    setting_id = str(paired_view["setting_id"] or "")
    controller_id = str(
        paired_metadata.get("reference_controller_id")
        or paired_metadata.get("baseline_controller_id")
        or paired_metadata.get("residual_reference_controller_id")
        or ""
    )
    if not setting_id:
        raise ValueError(f"Paired metadata under {paired_root} is missing scenario_setting.")
    if not controller_id:
        raise ValueError(f"Paired metadata under {paired_root} is missing controller identifiers.")

    paired_policy_path_value = paired_metadata.get("policy_path")
    paired_policy_path = None
    if paired_policy_path_value is not None:
        candidate = _resolve_existing_path(paired_policy_path_value, base_dirs=(ROOT, paired_root, root))
        if candidate.exists() and candidate.is_relative_to(root):
            paired_policy_path = candidate

    policy_path, policy_metadata = _select_policy(
        scene=scene,
        scenario_root=root,
        setting_id=setting_id,
        controller_id=controller_id,
        expected_output_dim=len(scene_spec.active_groups),
        expected_residual_parameterization=expected_contract["residual_parameterization"],
        paired_policy_path=paired_policy_path,
    )

    base_spec = BaseStiffnessSpec.from_metadata(policy_metadata["base_stiffness_spec"])
    policy_output_dim = int(policy_metadata["output_dim"])
    residual_parameterization = str(policy_metadata["residual_parameterization"])
    active_group_names = tuple(str(name) for name in base_spec.active_group_names)

    profile_name = (
        str(paired_view["profile_name"] or policy_metadata.get("profile_name") or "")
        or None
    )
    contact_condition_name = (
        str(paired_view["contact_condition_name"] or policy_metadata.get("contact_condition_name") or "")
        or None
    )
    dataset_path, dataset_metadata, residual_group_target_shape = _select_dataset(
        scene=scene,
        scenario_root=root,
        setting_id=setting_id,
        controller_id=controller_id,
        profile_name=profile_name,
        contact_condition_name=contact_condition_name,
        expected_output_dim=policy_output_dim,
        expected_residual_parameterization=residual_parameterization,
        expected_active_group_names=active_group_names,
    )

    dataset_base_spec = BaseStiffnessSpec.from_metadata(dataset_metadata["base_stiffness_spec"])
    if dataset_base_spec.base_matrix.shape != base_spec.base_matrix.shape or not np.allclose(
        dataset_base_spec.base_matrix, base_spec.base_matrix, atol=1e-9, rtol=0.0
    ):
        raise ValueError("Dataset base_stiffness_spec.base_matrix does not match the selected policy contract.")
    if tuple(dataset_base_spec.active_group_names) != tuple(active_group_names):
        raise ValueError("Dataset base_stiffness_spec.active_group_names do not match the selected policy contract.")
    if tuple(dataset_base_spec.active_groups) != tuple(base_spec.active_groups):
        raise ValueError("Dataset base_stiffness_spec.active_groups do not match the selected policy contract.")
    if dataset_base_spec.residual_bounds.shape != (policy_output_dim,):
        raise ValueError(
            f"Dataset residual bounds shape {dataset_base_spec.residual_bounds.shape} does not match output_dim {policy_output_dim}."
        )
    if not np.allclose(dataset_base_spec.residual_bounds, base_spec.residual_bounds, atol=1e-12, rtol=0.0):
        raise ValueError("Dataset residual bounds do not match the selected policy contract.")
    if len(residual_group_target_shape) != 2:
        raise ValueError(
            f"Dataset residual_group_target must be 2D, observed shape {residual_group_target_shape}."
        )
    output_dim = int(residual_group_target_shape[1])
    if policy_output_dim != output_dim:
        raise ValueError(
            f"Policy output_dim mismatch: expected {output_dim}, observed {policy_output_dim!r}."
        )
    if int(dataset_metadata.get("residual_dim", output_dim)) != output_dim:
        raise ValueError(
            f"Dataset metadata residual_dim mismatch: expected {output_dim}, observed {dataset_metadata.get('residual_dim')!r}."
        )
    _validate_scene_contract(
        scene=scene,
        output_dim=output_dim,
        residual_parameterization=residual_parameterization,
        active_group_names=active_group_names,
        base_stiffness_spec=base_spec,
    )

    episode_specs_path = _resolve_existing_path(dataset_metadata.get("episode_specs_path"), base_dirs=(ROOT, dataset_path.parent))
    episodes_csv_path = dataset_path.parent / "episodes.csv"
    if not episode_specs_path.exists():
        raise FileNotFoundError(f"Episode specs file not found: {episode_specs_path}")
    if not episodes_csv_path.exists():
        raise FileNotFoundError(f"Episodes CSV file not found: {episodes_csv_path}")

    paired_summary_path = paired_root / "paired_v2_summary.json"
    paired_metadata_path = paired_root / "paired_v2_metadata.json"
    if not paired_summary_path.exists() or not paired_metadata_path.exists():
        raise FileNotFoundError(f"Paired result directory is incomplete: {paired_root}")

    consistency = _resolve_consistency(
        scene=scene,
        setting_id=setting_id,
        controller_id=controller_id,
        dataset_metadata=dataset_metadata,
        policy_metadata=policy_metadata,
        paired_metadata=paired_metadata,
        dataset_path=dataset_path,
        policy_path=policy_path,
        paired_path=paired_root,
    )

    discovery_notes = (
        *paired_notes,
        f"paired_root={paired_root}",
        f"policy_candidates={len(_find_policy_candidates(root))}",
        f"dataset_candidates={len(_find_dataset_candidates(root))}",
    )
    warnings: list[str] = []
    if paired_view["scene_source"] != "paired_v2_metadata.scenario_scene":
        warnings.append(f"scene loaded from fallback source {paired_view['scene_source']}.")
    if paired_view["setting_id_source"] != "paired_v2_metadata.scenario_setting":
        warnings.append(f"setting_id loaded from fallback source {paired_view['setting_id_source']}.")
    if not is_residual_first_projection(dataset_metadata.get("label_projection")):
        warnings.append(
            "selected dataset is not tagged with residual-first baseline-relative label_projection metadata."
        )
    for required_key in ("residual_bound", "baseline_k", "diagnostic_k_min", "diagnostic_k_max", "diagnostic_reconstruction_role"):
        if required_key not in dataset_metadata:
            warnings.append(f"selected dataset metadata is missing {required_key}.")
    return TrackAArtifactAuditRecord(
        scene=scene,
        scene_source=str(paired_view["scene_source"] or "paired_v2_metadata.scenario_scene"),
        controller_id=controller_id,
        scenario_root=str(root),
        dataset_path=str(dataset_path.resolve()),
        policy_path=str(policy_path.resolve()),
        paired_output_root=str(paired_root.resolve()),
        paired_summary_path=str(paired_summary_path.resolve()),
        paired_metadata_path=str(paired_metadata_path.resolve()),
        setting_id=setting_id,
        setting_id_source=str(paired_view["setting_id_source"] or "paired_v2_metadata.scenario_setting"),
        output_dim=output_dim,
        output_dim_source="dataset.residual_group_target.shape[1]",
        residual_parameterization=residual_parameterization,
        contract_source=f"scene_spec[{scene}] / describe_residual_policy_contract",
        active_group_names=active_group_names,
        base_stiffness_spec=dataset_base_spec.to_metadata(),
        residual_group_target_shape=residual_group_target_shape,
        episode_specs_path=str(episode_specs_path.resolve()),
        episodes_csv_path=str(episodes_csv_path.resolve()),
        controller_policy_consistency_passed=consistency,
        status="warning" if warnings else "passed",
        errors=(),
        warnings=tuple(warnings),
        discovery_notes=discovery_notes,
    )


def audit_track_a_mainline_contracts(scenes: Iterable[str] = ("circle", "polygon", "star")) -> dict[str, Any]:
    records: list[TrackAArtifactAuditRecord] = []
    for scene in scenes:
        try:
            records.append(audit_track_a_mainline_scene(scene))
        except Exception as exc:
            records.append(_failure_record(scene, error=exc))
    passed = sum(1 for record in records if record.status == "passed")
    warning = sum(1 for record in records if record.status == "warning")
    failed = sum(1 for record in records if record.status == "failed")
    return {
        "audit_version": AUDIT_VERSION,
        "scene_count": len(records),
        "passed_scene_count": passed,
        "warning_scene_count": warning,
        "failed_scene_count": failed,
        "status": "failed" if failed else "warning" if warning else "passed",
        "records": [record.to_dict() for record in records],
    }


def render_markdown_summary(report: dict[str, Any]) -> str:
    lines = ["# Track A Mainline Artifact Contract Audit", ""]
    lines.append("| scene | status | controller_id | setting_id | output_dim | residual_parameterization | consistency | warnings | errors |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for record in report.get("records", []):
        warnings = "; ".join(record.get("warnings", ())) if record.get("warnings") else ""
        errors = "; ".join(record.get("errors", ())) if record.get("errors") else ""
        lines.append(
            "| {scene} | {status} | {controller_id} | {setting_id} | {output_dim} | {residual_parameterization} | {controller_policy_consistency_passed} | {warnings} | {errors} |".format(
                **record
            )
        )
    return "\n".join(lines) + "\n"


__all__ = [
    "AUDIT_VERSION",
    "CURRENT_MAINLINE_SCENE_ROOTS",
    "TrackAArtifactAuditRecord",
    "audit_track_a_mainline_contracts",
    "audit_track_a_mainline_scene",
    "render_markdown_summary",
]
