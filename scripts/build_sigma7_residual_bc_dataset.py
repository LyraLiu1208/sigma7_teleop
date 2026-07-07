from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from _sigma7_residual_pipeline_common import (
    COLLECTION_MODE,
    DEFAULT_MUJOCO_ROOT,
    DEFAULT_PIPELINE_ROOT,
    ensure_mujoco_src_on_path,
    scene_collection_root,
    scene_dataset_6d_root,
    scene_dataset_root,
    scene_root,
    require_safe_segment,
    write_json,
)


ensure_mujoco_src_on_path()

from stiffness_copilot_mujoco.episodes.episode_spec import (  # noqa: E402
    EpisodeSpec,
    load_episode_specs_jsonl,
    write_episode_specs_jsonl,
)
from stiffness_copilot_mujoco.learning.frozen_train_val_split import FrozenTrainValSplit  # noqa: E402
from stiffness_copilot_mujoco.learning.open_loop_residual_dataset import build_eligible_residual_dataset_from_raw  # noqa: E402
from stiffness_copilot_mujoco.learning.residual_label_projection import (  # noqa: E402
    LABEL_PROJECTION_CONTACT_GATED_CENTERED_V2,
    LABEL_PROJECTION_LEGACY,
    LABEL_PROJECTION_RESIDUAL_FIRST,
)


SAMPLE_KEYS = {
    "state",
    "action",
    "contact_state",
    "task_state",
    "contact_force_world",
    "normal_force",
    "episode_id",
    "sample_step",
    "timestamp",
    "randomization",
    "planned_target_position",
    "planned_target_rotation",
    "trajectory_family_id",
    "trajectory_parameters",
    "phase_id",
    "rgb_images",
}

LABEL_PROJECTION_ALIASES = {
    "residual_first": LABEL_PROJECTION_RESIDUAL_FIRST,
    LABEL_PROJECTION_RESIDUAL_FIRST: LABEL_PROJECTION_RESIDUAL_FIRST,
    "residual_first_contact_gated_centered_v2": LABEL_PROJECTION_CONTACT_GATED_CENTERED_V2,
    LABEL_PROJECTION_CONTACT_GATED_CENTERED_V2: LABEL_PROJECTION_CONTACT_GATED_CENTERED_V2,
    "legacy": LABEL_PROJECTION_LEGACY,
    LABEL_PROJECTION_LEGACY: LABEL_PROJECTION_LEGACY,
}


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


def _load_npz_payload(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files if key != "metadata"}
        metadata = json.loads(str(data["metadata"]))
    return arrays, metadata


def _classify_array_key(key: str) -> str:
    if key in SAMPLE_KEYS:
        return "sample"
    if key == "episode_summary_id" or key.startswith("episode_"):
        return "episode"
    raise KeyError(f"Unsupported raw array key {key!r}.")


def _resolve_label_projection(value: str) -> str:
    try:
        return LABEL_PROJECTION_ALIASES[str(value)]
    except KeyError as exc:
        raise ValueError(f"Unsupported label projection: {value!r}.") from exc


def _assert_same_keys(reference: set[str], current: set[str], *, path: Path) -> None:
    if reference != current:
        missing = sorted(reference - current)
        extra = sorted(current - reference)
        raise ValueError(
            f"Raw dataset keys mismatch for {path}. Missing={missing}, extra={extra}."
        )


def _assert_consistent_metadata(reference: dict[str, Any], current: dict[str, Any], *, path: Path) -> None:
    exact_fields = (
        "scene",
        "scenario_id",
        "schema_version",
        "teleop_mode",
        "teleop_source",
        "sample_stride",
        "rgb_enabled",
        "rgb_camera_name",
        "rgb_image_width",
        "rgb_image_height",
        "rgb_image_stride",
        "renderer_mode",
        "fallback_used",
        "eye_in_hand_camera_pose_version",
        "eye_in_hand_camera_canonical",
        "eye_in_hand_camera_name",
        "eye_in_hand_camera_attachment_parent",
        "eye_in_hand_camera_mount_type",
        "hole_xy_offset_semantics",
        "hole_xy_offset_distribution",
        "trajectory_follows_randomized_hole",
        "contact_generation_parameters_fixed",
        "trajectory_plan",
        "trajectory_source",
        "human_proxy_replaced_by_sigma7_live",
        "target_generation_teleop_noise_disabled",
    )
    for field in exact_fields:
        if reference.get(field) != current.get(field):
            raise ValueError(
                f"Metadata field {field!r} differs between runs; {path} is not merge-compatible."
            )
    if not np.allclose(
        np.asarray(reference.get("collection_stiffness_matrix"), dtype=float),
        np.asarray(current.get("collection_stiffness_matrix"), dtype=float),
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError(f"collection_stiffness_matrix differs between runs; {path} is not merge-compatible.")
    if json.dumps(reference.get("eye_in_hand_camera_pose"), sort_keys=True) != json.dumps(
        current.get("eye_in_hand_camera_pose"), sort_keys=True
    ):
        raise ValueError(f"eye_in_hand_camera_pose differs between runs; {path} is not merge-compatible.")
    if json.dumps(reference.get("legacy_field_mapping"), sort_keys=True) != json.dumps(
        current.get("legacy_field_mapping"), sort_keys=True
    ):
        raise ValueError(f"legacy_field_mapping differs between runs; {path} is not merge-compatible.")


def _rebuild_episode_spec(spec: EpisodeSpec, new_episode_id: int) -> EpisodeSpec:
    return EpisodeSpec.create(
        episode_id=new_episode_id,
        seed=spec.seed,
        scene=spec.scene,
        setting_id=spec.setting_id,
        profile_name=spec.profile_name,
        contact_condition_name=spec.contact_condition_name,
        nominal_hole_position=spec.nominal_hole_position,
        nominal_hole_xy=spec.nominal_hole_xy,
        hole_xy_offset=spec.hole_xy_offset,
        hole_yaw_offset=spec.hole_yaw_offset,
        hole_xy_radius=spec.hole_xy_radius,
        hole_xy_offset_semantics=spec.hole_xy_offset_semantics,
        hole_xy_offset_distribution=spec.hole_xy_offset_distribution,
        trajectory_follows_randomized_hole=spec.trajectory_follows_randomized_hole,
        contact_generation_parameters_fixed=spec.contact_generation_parameters_fixed,
        fixed_contact_condition=spec.fixed_contact_condition,
        trajectory_source=spec.trajectory_source,
        trajectory_family=spec.trajectory_family,
        trajectory_family_id=spec.trajectory_family_id,
        trajectory_parameters=spec.trajectory_parameters,
        target_offsets=spec.target_offsets,
        phase_ids=spec.phase_ids,
        total_steps=spec.total_steps,
        sample_stride=spec.sample_stride,
        image_stride=spec.image_stride,
        native_launcher_required=spec.native_launcher_required,
        teleop_mode=spec.teleop_mode,
        target_rotations=spec.target_rotations,
    )


def _resolve_scene(raw_paths: list[Path], requested_scene: str | None) -> str:
    scenes: dict[str, list[Path]] = {}
    for raw_path in raw_paths:
        _arrays, metadata = _load_npz_payload(raw_path)
        scene = str(metadata.get("scene") or "")
        scenes.setdefault(scene, []).append(raw_path)
    if requested_scene is not None:
        if requested_scene not in scenes:
            raise FileNotFoundError(
                f"Requested scene {requested_scene!r} was not found under the discovered runs. "
                f"Available scenes: {sorted(scenes)}"
            )
        return requested_scene
    if len(scenes) != 1:
        raise ValueError(
            f"Multiple scenes were found under the episode root: {sorted(scenes)}. "
            "Pass --scene to select one."
        )
    return next(iter(scenes))


def _build_trajectory_family_summary(raw_arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    family_ids = np.asarray(raw_arrays["episode_trajectory_family_id"], dtype=np.int64)
    counts: dict[str, int] = {}
    for family_id in family_ids:
        key = str(int(family_id))
        counts[key] = counts.get(key, 0) + 1
    return {
        "trajectory_family_counts": counts,
        "episode_count": int(family_ids.shape[0]),
    }


def _run_external_6d_build(input_root: Path, output_root: Path, *, overwrite: bool) -> None:
    script_path = DEFAULT_MUJOCO_ROOT / "scripts" / "build_residual_bc_6d_dataset.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--input-root",
        str(input_root),
        "--output-root",
        str(output_root),
    ]
    if overwrite:
        cmd.append("--overwrite")
    subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Merge all participants' single-episode Sigma7 collection runs under one scene, then build eligible and 6D datasets."
    )
    parser.add_argument("--scene", type=str, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_PIPELINE_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--label-neighbors", type=int, default=32)
    parser.add_argument("--knn-block-size", type=int, default=1024)
    parser.add_argument(
        "--label-projection",
        choices=tuple(LABEL_PROJECTION_ALIASES.keys()),
        default=LABEL_PROJECTION_CONTACT_GATED_CENTERED_V2,
    )
    parser.add_argument("--label-k-min", type=float, default=300.0)
    parser.add_argument("--label-k-max", type=float, default=600.0)
    parser.add_argument("--baseline-k", type=float, default=600.0)
    parser.add_argument("--diagnostic-k-min", type=float, default=300.0)
    parser.add_argument("--diagnostic-k-max", type=float, default=900.0)
    parser.add_argument("--residual-bound", type=float, default=None)
    parser.add_argument("--l21-coupling-percentile", type=float, default=95.0)
    parser.add_argument("--contact-gate-low", type=float, default=1.0)
    parser.add_argument("--contact-gate-high", type=float, default=10.0)
    parser.add_argument("--neutral-contact-threshold", type=float, default=10.0)
    parser.add_argument("--min-calibration-samples", type=int, default=8)
    args = parser.parse_args(argv)
    label_projection = _resolve_label_projection(args.label_projection)

    scene = require_safe_segment(args.scene, name="scene")
    mode = COLLECTION_MODE
    scene_root_dir = scene_root(args.output_root, scene)
    if not scene_root_dir.exists():
        raise FileNotFoundError(f"Scene root does not exist: {scene_root_dir}")

    all_raw_paths = sorted(scene_root_dir.glob(f"*/{mode}/episodes/**/raw_collection.npz"))
    if not all_raw_paths:
        raise FileNotFoundError(f"No raw_collection.npz files were found under {scene_root_dir}")

    scene = _resolve_scene(all_raw_paths, scene)
    selected_raw_paths = []
    for raw_path in all_raw_paths:
        _arrays, metadata = _load_npz_payload(raw_path)
        if str(metadata.get("scene") or "") == scene:
            selected_raw_paths.append(raw_path)
    if not selected_raw_paths:
        raise FileNotFoundError(f"No runs matched scene {scene!r} under {scene_root_dir}")

    aggregation_root = scene_collection_root(args.output_root, scene)
    dataset_root = scene_dataset_root(args.output_root, scene)
    dataset_6d_root = scene_dataset_6d_root(args.output_root, scene)
    if dataset_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Dataset root already exists: {dataset_root}. Use --overwrite to replace it.")
        shutil.rmtree(dataset_root)
    if dataset_6d_root.exists() and args.overwrite:
        shutil.rmtree(dataset_6d_root)
    dataset_root.mkdir(parents=True, exist_ok=True)

    raw_path = dataset_root / "raw_collection.npz"
    eligible_path = dataset_root / "eligible_residual_bc.npz"
    episode_specs_path = dataset_root / "episode_specs.jsonl"
    frozen_episode_specs_path = dataset_root / "frozen_paired_episode_specs.jsonl"
    frozen_split_path = dataset_root / "frozen_train_val_split.json"
    teleop_trace_path = dataset_root / "sigma7_teleop_trace.npz"
    collection_summary_path = dataset_root / "collection_summary.json"
    collection_metadata_path = dataset_root / "collection_metadata.json"
    trajectory_family_summary_path = dataset_root / "trajectory_family_summary.json"
    source_manifest_path = dataset_root / "source_runs_manifest.json"
    episodes_csv_path = dataset_root / "episodes.csv"

    sample_parts: dict[str, list[np.ndarray]] = {}
    episode_parts: dict[str, list[np.ndarray]] = {}
    teleop_parts: dict[str, list[np.ndarray]] = {}
    merged_specs: list[EpisodeSpec] = []
    merged_episode_rows: list[dict[str, Any]] = []
    source_manifest: list[dict[str, Any]] = []
    source_collection_seeds: list[int] = []
    source_split_seeds: list[int] = []
    source_participants: list[str] = []

    reference_arrays: set[str] | None = None
    reference_metadata: dict[str, Any] | None = None
    total_episode_count = 0
    total_sample_count = 0

    for raw_idx, raw_dataset_path in enumerate(selected_raw_paths):
        arrays, metadata = _load_npz_payload(raw_dataset_path)
        current_keys = set(arrays)
        if reference_arrays is None:
            reference_arrays = current_keys
            reference_metadata = metadata
        else:
            _assert_same_keys(reference_arrays, current_keys, path=raw_dataset_path)
            _assert_consistent_metadata(reference_metadata, metadata, path=raw_dataset_path)
        local_episode_count = int(np.asarray(arrays["episode_summary_id"]).shape[0])
        local_sample_count = int(np.asarray(arrays["episode_id"]).shape[0])
        episode_offset = total_episode_count
        sample_offset = total_sample_count

        for key, value in arrays.items():
            adjusted = np.asarray(value).copy()
            kind = _classify_array_key(key)
            if key == "episode_id":
                adjusted = adjusted.astype(np.int64, copy=False) + episode_offset
            elif key == "episode_summary_id":
                adjusted = adjusted.astype(np.int64, copy=False) + episode_offset
            if kind == "sample":
                sample_parts.setdefault(key, []).append(adjusted)
            else:
                episode_parts.setdefault(key, []).append(adjusted)

        episode_specs_source = Path(str(metadata.get("episode_specs_path") or raw_dataset_path.parent / "episode_specs.jsonl"))
        if episode_specs_source.exists():
            loaded_specs = load_episode_specs_jsonl(episode_specs_source)
            if len(loaded_specs) != local_episode_count:
                raise ValueError(
                    f"Episode spec count mismatch for {episode_specs_source}: "
                    f"{len(loaded_specs)} specs for {local_episode_count} episodes."
                )
            for local_idx, spec in enumerate(loaded_specs):
                rebuilt = _rebuild_episode_spec(spec, episode_offset + local_idx)
                merged_specs.append(rebuilt)

        episode_csv_source = raw_dataset_path.parent / "episodes.csv"
        spec_id_lookup: dict[int, str] = {}
        if merged_specs:
            for spec in merged_specs[-local_episode_count:]:
                spec_id_lookup[int(spec.episode_id)] = spec.episode_spec_id
        if episode_csv_source.exists():
            with episode_csv_source.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    local_episode_id = int(row["episode_id"])
                    global_episode_id = episode_offset + local_episode_id
                    updated = dict(row)
                    updated["episode_id"] = str(global_episode_id)
                    if "sample_start" in updated:
                        updated["sample_start"] = str(sample_offset + int(updated["sample_start"]))
                    if "sample_stop" in updated:
                        updated["sample_stop"] = str(sample_offset + int(updated["sample_stop"]))
                    if "episode_spec_id" in updated and global_episode_id in spec_id_lookup:
                        updated["episode_spec_id"] = spec_id_lookup[global_episode_id]
                    merged_episode_rows.append(updated)

        teleop_trace_source = Path(str(metadata.get("teleop_trace_path") or raw_dataset_path.parent / "sigma7_teleop_trace.npz"))
        if teleop_trace_source.exists():
            trace_arrays, _trace_metadata = _load_npz_payload(teleop_trace_source)
            for key, value in trace_arrays.items():
                adjusted = np.asarray(value).copy()
                if key == "episode_id":
                    adjusted = adjusted.astype(np.int64, copy=False) + episode_offset
                teleop_parts.setdefault(key, []).append(adjusted)

        source_manifest.append(
            {
                "run_index": raw_idx,
                "participant": raw_dataset_path.parents[3].name,
                "run_dir": str(raw_dataset_path.parent),
                "raw_dataset": str(raw_dataset_path),
                "scene": str(metadata.get("scene")),
                "collection_seed": metadata.get("collection_seed"),
                "split_seed": (metadata.get("rng_streams") or {}).get("split_seed"),
                "sample_count": local_sample_count,
                "episode_count": local_episode_count,
            }
        )
        source_participants.append(raw_dataset_path.parents[3].name)
        if metadata.get("collection_seed") is not None:
            source_collection_seeds.append(int(metadata["collection_seed"]))
        split_seed_value = (metadata.get("rng_streams") or {}).get("split_seed")
        if split_seed_value is not None:
            source_split_seeds.append(int(split_seed_value))
        total_episode_count += local_episode_count
        total_sample_count += local_sample_count

    assert reference_metadata is not None

    merged_raw_arrays: dict[str, np.ndarray] = {}
    for key, parts in sample_parts.items():
        merged_raw_arrays[key] = np.concatenate(parts, axis=0)
    for key, parts in episode_parts.items():
        merged_raw_arrays[key] = np.concatenate(parts, axis=0)

    merged_metadata = dict(reference_metadata)
    unique_collection_seed = sorted(set(source_collection_seeds))
    merged_metadata.update(
        {
            "dataset_path": str(raw_path),
            "episode_specs_path": str(episode_specs_path),
            "frozen_paired_episode_specs_path": str(frozen_episode_specs_path),
            "frozen_train_val_split_path": str(frozen_split_path),
            "teleop_trace_path": str(teleop_trace_path) if teleop_parts else None,
            "num_episodes": int(total_episode_count),
            "requested_episodes": int(total_episode_count),
            "num_samples": int(total_sample_count),
            "participant": None,
            "collection_mode": mode,
            "source_run_count": len(source_manifest),
            "source_participants": sorted(set(source_participants)),
            "source_run_dirs": [item["run_dir"] for item in source_manifest],
            "source_raw_datasets": [item["raw_dataset"] for item in source_manifest],
            "source_collection_seeds": unique_collection_seed,
            "source_original_split_seeds": sorted(set(source_split_seeds)),
            "collection_seed": unique_collection_seed[0] if len(unique_collection_seed) == 1 else None,
            "seed": unique_collection_seed[0] if len(unique_collection_seed) == 1 else None,
            "rng_streams": {
                "root_seed": None,
                "perturbation_seed": None,
                "split_seed": int(args.split_seed),
                "aggregation_seed": int(args.split_seed),
                "source_collection_seeds": unique_collection_seed,
            },
            "labels_built": False,
            "training_data_valid": bool(
                merged_metadata.get("rgb_enabled")
                and merged_metadata.get("renderer_mode") == "mujoco_native"
                and not bool(merged_metadata.get("fallback_used", False))
            ),
        }
    )
    if not merged_metadata.get("training_data_valid"):
        merged_metadata["training_data_valid_reason"] = "merged_dataset_not_native_rgb_ready"
    else:
        merged_metadata["training_data_valid_reason"] = None

    np.savez_compressed(raw_path, **merged_raw_arrays, metadata=json.dumps(_json_ready(merged_metadata), sort_keys=True))

    if teleop_parts:
        merged_trace_arrays = {key: np.concatenate(parts, axis=0) for key, parts in teleop_parts.items()}
        np.savez_compressed(
            teleop_trace_path,
            **merged_trace_arrays,
            metadata=json.dumps(_json_ready(merged_metadata), sort_keys=True),
        )

    if merged_specs:
        write_episode_specs_jsonl(episode_specs_path, merged_specs)
        shutil.copy2(episode_specs_path, frozen_episode_specs_path)

    if merged_episode_rows:
        with episodes_csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(merged_episode_rows[0].keys()))
            writer.writeheader()
            writer.writerows(merged_episode_rows)

    write_json(source_manifest_path, {"source_runs": source_manifest})
    write_json(
        trajectory_family_summary_path,
        _build_trajectory_family_summary(merged_raw_arrays),
    )

    build_eligible_residual_dataset_from_raw(
        raw_path,
        output_path=eligible_path,
        label_neighbors=args.label_neighbors,
        knn_block_size=args.knn_block_size,
        label_projection=label_projection,
        label_k_min=args.label_k_min,
        label_k_max=args.label_k_max,
        baseline_k=args.baseline_k,
        diagnostic_k_min=args.diagnostic_k_min,
        diagnostic_k_max=args.diagnostic_k_max,
        residual_bound=args.residual_bound,
        l21_coupling_percentile=args.l21_coupling_percentile,
        contact_gate_low=args.contact_gate_low,
        contact_gate_high=args.contact_gate_high,
        neutral_contact_threshold=args.neutral_contact_threshold,
        min_calibration_samples=args.min_calibration_samples,
    )

    with np.load(eligible_path, allow_pickle=False) as data:
        train_episode_ids = np.asarray(data["train_episode_ids"], dtype=np.int64)
        val_episode_ids = np.asarray(data["val_episode_ids"], dtype=np.int64)
    frozen_split = FrozenTrainValSplit(
        dataset_path=str(eligible_path),
        collection_seed=int(unique_collection_seed[0]) if unique_collection_seed else 0,
        split_seed=int(args.split_seed),
        train_episode_ids=train_episode_ids,
        val_episode_ids=val_episode_ids,
        metadata={
            "scene": scene,
            "participant": None,
            "collection_mode": mode,
            "source_run_count": len(source_manifest),
            "source_participants": sorted(set(source_participants)),
            "collection_controller_id": merged_metadata.get("collection_controller_id"),
        },
    )
    frozen_split.write(frozen_split_path)

    collection_metadata = {
        "schema_version": "sigma7_residual_bc_aggregated_v1",
        "participant": None,
        "collection_mode": mode,
        "scene": scene,
        "scene_collection_root": str(aggregation_root),
        "raw_dataset": str(raw_path),
        "eligible_dataset": str(eligible_path),
        "dataset_6d_root": str(dataset_6d_root),
        "episode_specs_path": str(episode_specs_path) if episode_specs_path.exists() else None,
        "frozen_paired_episode_specs_path": str(frozen_episode_specs_path) if frozen_episode_specs_path.exists() else None,
        "frozen_train_val_split_path": str(frozen_split_path),
        "source_run_manifest": str(source_manifest_path),
        "source_run_count": len(source_manifest),
        "source_participants": sorted(set(source_participants)),
    }
    write_json(collection_metadata_path, collection_metadata)

    collection_summary = {
        "status": "passed",
        "participant": None,
        "collection_mode": mode,
        "scene": scene,
        "scene_collection_root": str(aggregation_root),
        "source_run_count": len(source_manifest),
        "source_participants": sorted(set(source_participants)),
        "episodes": int(total_episode_count),
        "raw_samples": int(total_sample_count),
        "eligible_episodes": int(train_episode_ids.size + val_episode_ids.size),
        "raw_dataset": str(raw_path),
        "eligible_dataset": str(eligible_path),
        "frozen_paired_episode_specs_path": str(frozen_episode_specs_path) if frozen_episode_specs_path.exists() else None,
        "frozen_train_val_split_path": str(frozen_split_path),
        "teleop_trace": str(teleop_trace_path) if teleop_parts else None,
        "source_runs": [item["run_dir"] for item in source_manifest],
    }
    write_json(collection_summary_path, collection_summary)

    _run_external_6d_build(dataset_root, dataset_6d_root, overwrite=args.overwrite)

    print("")
    print("build complete")
    print(f"mode: {mode}")
    print(f"scene: {scene}")
    print(f"scene_root: {scene_root_dir}")
    print(f"scene_collection_root: {aggregation_root}")
    print(f"dataset_root: {dataset_root}")
    print(f"dataset_6d_root: {dataset_6d_root}")
    print(f"source_participants: {sorted(set(source_participants))}")
    print(f"merged_raw_dataset: {raw_path}")
    print(f"eligible_dataset: {eligible_path}")
    print(f"frozen_train_val_split: {frozen_split_path}")
    print(f"source_run_count: {len(source_manifest)}")
    print(f"episodes: {total_episode_count}")
    print(f"samples: {total_sample_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
