from __future__ import annotations

import csv
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from stiffness_copilot_mujoco.contact.state import (
    ContactQuery,
    contact_state_vector,
    extract_contact_state,
    extract_net_peg_hole_contact_force_world,
)
from stiffness_copilot_mujoco.controllers.impedance import task_space_impedance_torque
from stiffness_copilot_mujoco.controllers.track_a_controllers import (
    DEFAULT_TRACK_A_CONTROLLERS_YAML,
    load_track_a_controller_runtime,
)
from stiffness_copilot_mujoco.episodes.episode_spec import (
    EPISODE_SPEC_SCHEMA_VERSION,
    EPISODE_TRAJECTORY_SOURCE_EPISODE_SPEC_REPLAY,
    EpisodeSpec,
    write_episode_specs_jsonl,
)
from stiffness_copilot_mujoco.episodes.teleop_proxy import TELEOP_MODE_POSITION_ONLY
from stiffness_copilot_mujoco.franka_viewer import load_model
from stiffness_copilot_mujoco.learning.dataset_collection import _action_row, _state_row
from stiffness_copilot_mujoco.learning.frozen_train_val_split import FrozenTrainValSplit
from stiffness_copilot_mujoco.learning.open_loop_residual_dataset import classify_episode_admission
from stiffness_copilot_mujoco.learning.residual_dataset import validate_residual_dataset
from stiffness_copilot_mujoco.learning.residual_label_projection import (
    DEFAULT_BASELINE_K,
    DEFAULT_DIAGNOSTIC_K_MAX,
    DEFAULT_DIAGNOSTIC_K_MIN,
    DEFAULT_L21_COUPLING_PERCENTILE,
    project_residual_first_labels,
)
from stiffness_copilot_mujoco.learning.residual_stiffness import BaseStiffnessSpec
from stiffness_copilot_mujoco.learning.stiffness_labels import (
    StiffnessLabelConfig,
    build_stiffness_labels_with_diagnostics,
)
from stiffness_copilot_mujoco.learning.task_state import peg_hole_task_state
from stiffness_copilot_mujoco.metrics.task_metrics import (
    geometry_from_config,
    hole_center_position,
    load_scene_config,
    peg_tip_position,
)
from stiffness_copilot_mujoco.panda_control import arm_qpos, panda_arm_ids, set_arm_torque_ctrl
from stiffness_copilot_mujoco.pose_math import site_rotation
from stiffness_copilot_mujoco.runtime_defaults import default_native_launcher
from stiffness_copilot_mujoco.robustness import (
    ControlledContactProfile,
    get_robustness_preset,
    sample_controlled_contact_perturbations,
    sample_robustness_perturbations,
)
from stiffness_copilot_mujoco.rollout_observation import collect_step, reset_from_config
from stiffness_copilot_mujoco.rollouts.fixed_impedance import (
    DEFAULT_GAIN_CONFIG,
    RolloutConfig,
    cleanup_runtime_scene,
    clip_torque,
    phase_for_step,
    scene_for_rollout,
    target_position_for_phase,
    target_rotation_with_peg_tilt,
)
from stiffness_copilot_mujoco.scenes import get_scene_spec
from stiffness_copilot_mujoco.sim.ids import peg_hole_ids
from stiffness_copilot_mujoco.sim.scene import (
    ROOT,
    apply_eye_in_hand_camera_pose,
    canonical_eye_in_hand_camera_pose,
    eye_in_hand_camera_pose_from_config,
    validate_canonical_eye_in_hand_camera_config,
)
from stiffness_copilot_mujoco.teleop.sigma7_udp import (
    SIGMA7_PACKET_SCHEMA_VERSION,
    Sigma7Packet,
    Sigma7TeleopConfig,
    Sigma7TeleopMapper,
    Sigma7UdpReceiver,
    build_synthetic_sigma7_packet,
)
from stiffness_copilot_mujoco.vision.rendering import MujocoRgbRenderer


DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "datasets" / "residual_bc_residual_first_v1" / "sigma7_live"
DEFAULT_CONTROLLER_ID = "track_a_c600"
TRAJECTORY_FAMILY_SIGMA7_LIVE = "sigma7_live"
TRAJECTORY_FAMILY_SIGMA7_LIVE_ID = 0
RAW_SCHEMA_VERSION = "residual_bc_sigma7_live_raw_v1"
ELIGIBLE_SCHEMA_VERSION = "residual_bc_sigma7_live_residual_first_v1"
TRAJECTORY_PLAN_NAME = "sigma7_live_nominal_phase_plus_device_delta_v1"
RANDOMIZATION_FIELDS = (
    "hole_x",
    "hole_y",
    "hole_yaw",
    "clearance_delta",
    "friction_scale",
    "peg_tilt_x",
    "peg_tilt_y",
    "teleop_noise_xy_amplitude",
    "teleop_noise_cycles",
)
TRAJECTORY_PARAMETER_FIELDS = (
    "approach_hold_steps",
    "descend_steps",
    "insert_steps",
    "final_hold_steps",
    "approach_height",
    "descend_height",
    "insert_depth",
    "initial_target_x",
    "initial_target_y",
    "final_target_x",
    "final_target_y",
    "mean_clamped_delta_norm",
    "max_clamped_delta_norm",
    "num_zero_events",
    "num_pause_events",
    "duration_seconds",
)


@dataclass(frozen=True)
class Sigma7LiveCollectionResult:
    output_dir: Path
    raw_dataset: Path
    eligible_dataset: Path
    collection_summary: Path
    episode_csv: Path
    episode_specs: Path
    frozen_episode_specs: Path
    frozen_split: Path
    teleop_trace: Path


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


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_json_ready(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _available_output_dir(output_root: Path, run_name: str) -> Path:
    candidate = output_root / run_name
    rerun = 0
    while candidate.exists():
        rerun += 1
        candidate = output_root / f"{run_name}_rerun{rerun}"
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def _stack_records(records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {}
    if not records:
        return payload
    keys = sorted({key for record in records for key in record})
    for key in keys:
        values = [record.get(key) for record in records]
        first = next((value for value in values if value is not None), None)
        if first is None:
            payload[key] = np.asarray(values, dtype=float)
            continue
        if isinstance(first, np.ndarray):
            payload[key] = np.stack([np.asarray(value) for value in values], axis=0)
            continue
        if isinstance(first, str):
            payload[key] = np.asarray(["" if value is None else str(value) for value in values], dtype=str)
            continue
        if isinstance(first, (bool, np.bool_)):
            payload[key] = np.asarray([False if value is None else bool(value) for value in values], dtype=bool)
            continue
        if isinstance(first, (int, np.integer)):
            payload[key] = np.asarray([int(value) if value is not None else -1 for value in values], dtype=np.int64)
            continue
        payload[key] = np.asarray([np.nan if value is None else float(value) for value in values], dtype=float)
    return payload


def _stratified_split(
    episode_ids: np.ndarray,
    success: np.ndarray,
    *,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    for outcome in (False, True):
        group = np.asarray(episode_ids[success == outcome], dtype=np.int32).copy()
        rng.shuffle(group)
        if group.size <= 1:
            count = group.size
        else:
            count = min(max(int(round(0.8 * group.size)), 1), group.size - 1)
        train_parts.append(group[:count])
        val_parts.append(group[count:])
    train = np.sort(np.concatenate(train_parts) if train_parts else np.empty(0, dtype=np.int32))
    val = np.sort(np.concatenate(val_parts) if val_parts else np.empty(0, dtype=np.int32))
    if val.size == 0 and train.size > 1:
        val = train[-1:].copy()
        train = train[:-1]
    if np.intersect1d(train, val).size:
        raise RuntimeError("Train and validation episode ids overlap.")
    return train, val


def _randomization_vector(perturbation_dict: dict[str, Any]) -> np.ndarray:
    return np.array(
        [
            float(perturbation_dict["hole_xy_offset"][0]),
            float(perturbation_dict["hole_xy_offset"][1]),
            float(perturbation_dict["hole_yaw_offset"]),
            float(perturbation_dict["clearance_delta"]),
            float(perturbation_dict["friction_scale"]),
            float(perturbation_dict["peg_tilt_x"]),
            float(perturbation_dict["peg_tilt_y"]),
            float(perturbation_dict["teleop_noise_xy_amplitude"]),
            float(perturbation_dict["teleop_noise_cycles"]),
        ],
        dtype=float,
    )


def _teleop_free_perturbation_dict(perturbation: Any) -> dict[str, Any]:
    payload = dict(perturbation.to_dict())
    payload["teleop_noise_xy_amplitude"] = 0.0
    payload["teleop_noise_cycles"] = 0.0
    payload["teleop_noise_phase_x"] = 0.0
    payload["teleop_noise_phase_y"] = 0.0
    return payload


def _phase_counts(phase_ids: np.ndarray) -> tuple[int, int, int, int]:
    phase_ids = np.asarray(phase_ids, dtype=np.int32)
    return tuple(int(np.sum(phase_ids == index)) for index in range(4))


def _trajectory_parameter_vector(
    *,
    phase_ids: np.ndarray,
    rollout_config: RolloutConfig,
    target_offsets: np.ndarray,
    clamped_delta_norm: np.ndarray,
    zero_event_count: int,
    pause_event_count: int,
    duration_seconds: float,
) -> np.ndarray:
    target_offsets = np.asarray(target_offsets, dtype=float)
    clamped_delta_norm = np.asarray(clamped_delta_norm, dtype=float)
    if target_offsets.ndim != 2 or target_offsets.shape[1] != 3:
        raise ValueError("target_offsets must have shape [T, 3].")
    approach_steps, descend_steps, insert_steps, hold_steps = _phase_counts(phase_ids)
    initial_target_xy = target_offsets[0, :2]
    final_target_xy = target_offsets[-1, :2]
    mean_delta_norm = float(np.mean(clamped_delta_norm)) if clamped_delta_norm.size else 0.0
    max_delta_norm = float(np.max(clamped_delta_norm)) if clamped_delta_norm.size else 0.0
    return np.array(
        [
            approach_steps,
            descend_steps,
            insert_steps,
            hold_steps,
            rollout_config.approach_height,
            rollout_config.descend_height,
            rollout_config.insert_depth,
            float(initial_target_xy[0]),
            float(initial_target_xy[1]),
            float(final_target_xy[0]),
            float(final_target_xy[1]),
            mean_delta_norm,
            max_delta_norm,
            int(zero_event_count),
            int(pause_event_count),
            float(duration_seconds),
        ],
        dtype=float,
    )


def _synthetic_packet_for_step(step: int, *, dt_seconds: float) -> Sigma7Packet | None:
    return build_synthetic_sigma7_packet(
        step=step,
        dt_seconds=dt_seconds,
        zero_step=0,
        re_zero_step=110,
        pause_start_step=140,
        pause_end_step=165,
        timeout_start_step=200,
        timeout_end_step=245,
        quit_step=None,
        position_scale_xy=0.05,
        position_scale_z=0.02,
    )


def collect_sigma7_live_residual_dataset(
    *,
    episodes: int = 80,
    seed: int = 2000,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    sample_stride: int = 50,
    save_rgb: bool = False,
    camera_name: str = "eye_in_hand_rgb",
    image_width: int = 128,
    image_height: int = 128,
    image_stride: int = 50,
    max_rgb_episodes: int | None = None,
    renderer_mode: str = "native",
    allow_debug_fallback_renderer: bool = False,
    controller_id: str | None = None,
    controllers_yaml: Path = DEFAULT_TRACK_A_CONTROLLERS_YAML,
    controller_profile: str = DEFAULT_CONTROLLER_ID,
    scenario_id: str | None = None,
    profile_name: str | None = None,
    controlled_contact_profile: ControlledContactProfile | None = None,
    label_neighbors: int = 32,
    knn_block_size: int = 1024,
    label_k_min: float = 300.0,
    label_k_max: float = 600.0,
    baseline_k: float = DEFAULT_BASELINE_K,
    diagnostic_k_min: float = DEFAULT_DIAGNOSTIC_K_MIN,
    diagnostic_k_max: float = DEFAULT_DIAGNOSTIC_K_MAX,
    l21_coupling_percentile: float = DEFAULT_L21_COUPLING_PERCENTILE,
    gain_config: Path = DEFAULT_GAIN_CONFIG,
    synthetic_input: bool = False,
    packet_host: str = "0.0.0.0",
    packet_port: int = 5005,
    operator: str = "unknown",
    realtime: bool = True,
    max_steps: int | None = None,
    teleop_config: Sigma7TeleopConfig | None = None,
) -> Sigma7LiveCollectionResult:
    if episodes <= 0:
        raise ValueError("episodes must be positive.")
    if sample_stride <= 0:
        raise ValueError("sample_stride must be positive.")
    if image_stride <= 0:
        raise ValueError("image_stride must be positive.")
    if save_rgb and sample_stride % image_stride != 0:
        raise ValueError("For synchronized RGB capture, image_stride must evenly divide sample_stride.")
    if renderer_mode not in {"native", "legacy_debug_only"}:
        raise ValueError("renderer_mode must be one of {'native', 'legacy_debug_only'}.")
    if save_rgb and renderer_mode != "native" and not allow_debug_fallback_renderer:
        raise ValueError("Active image dataset collection requires native rendering.")
    if save_rgb and max_rgb_episodes is not None and max_rgb_episodes < episodes:
        raise ValueError("Sigma.7 RGB collection must capture every requested episode for training-compatible datasets.")

    started = time.perf_counter()
    scene = str(scenario_id or "circle")
    scene_spec = get_scene_spec(scene)
    requested_controller_id = controller_id or controller_profile
    controller_entry, profile, gains = load_track_a_controller_runtime(
        requested_controller_id,
        controllers_yaml=controllers_yaml,
        gain_config=gain_config,
    )
    rollout_config = RolloutConfig(
        config_path=scene_spec.config_path,
        gain_config_path=gain_config,
        max_steps=max_steps,
    )
    base_spec = BaseStiffnessSpec.from_matrix(
        controller_entry.position_stiffness_matrix,
        active_groups=scene_spec.active_groups,
        active_group_names=scene_spec.active_group_names,
        residual_bound=scene_spec.residual_bound,
    )
    nominal_scene_config = load_scene_config(scene_spec.config_path)
    nominal_hole_position = np.asarray(nominal_scene_config["hole"]["pos"], dtype=float)
    nominal_hole_xy = np.asarray(nominal_hole_position[:2], dtype=float)
    preset = get_robustness_preset(scene)

    root_seed = np.random.SeedSequence(seed)
    perturbation_seq, split_seq = root_seed.spawn(2)
    perturbation_seed = int(perturbation_seq.generate_state(1, dtype=np.uint32)[0])
    split_seed = int(split_seq.generate_state(1, dtype=np.uint32)[0])
    split_rng = np.random.default_rng(split_seed)

    if controlled_contact_profile is not None:
        perturbations = sample_controlled_contact_perturbations(
            episodes=episodes,
            seed=perturbation_seed,
            profile=controlled_contact_profile,
        )
        profile_name = profile_name or controlled_contact_profile.profile_name
        profile_metadata = controlled_contact_profile.to_metadata()
    else:
        perturbations = sample_robustness_perturbations(
            scene=scene,
            episodes=episodes,
            seed=perturbation_seed,
            preset=preset,
        )
        profile_name = profile_name or f"{preset.setting_id}_legacy_randomized"
        profile_metadata = {
            "hole_xy_offset_semantics": "legacy_contact_randomization",
            "hole_xy_offset_units": "m",
            "hole_xy_offset_distribution": f"legacy_preset_uniform_disk(radius={preset.hole_xy_radius:g})",
            "trajectory_follows_randomized_hole": True,
            "contact_generation_parameters_fixed": False,
            "fixed_hole_yaw_offset": None,
            "fixed_teleop_noise_xy_amplitude": None,
            "fixed_teleop_noise_cycles": None,
            "fixed_teleop_noise_phase_x": None,
            "fixed_teleop_noise_phase_y": None,
            "fixed_clearance_delta": None,
            "fixed_friction_scale": None,
            "fixed_peg_tilt_x": None,
            "fixed_peg_tilt_y": None,
            "legacy_field_mapping": {
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
                }
            },
        }

    run_profile_suffix = (
        f"{profile_name}_{controlled_contact_profile.contact_condition_name}"
        if controlled_contact_profile is not None and controlled_contact_profile.contact_condition_name
        else str(profile_name)
    )
    run_name = f"{preset.setting_id}_sigma7_live_{episodes}ep_seed{seed}_{run_profile_suffix}"
    output_dir = _available_output_dir(output_root, run_name)
    raw_path = output_dir / "raw_collection.npz"
    eligible_path = output_dir / "eligible_residual_bc.npz"
    summary_path = output_dir / "collection_summary.json"
    episode_csv_path = output_dir / "episodes.csv"
    episode_specs_path = output_dir / "episode_specs.jsonl"
    frozen_episode_specs_path = output_dir / "frozen_paired_episode_specs.jsonl"
    frozen_split_path = output_dir / "frozen_train_val_split.json"
    teleop_trace_path = output_dir / "sigma7_teleop_trace.npz"

    sample_rows: dict[str, list[Any]] = {
        "state": [],
        "action": [],
        "contact_state": [],
        "task_state": [],
        "contact_force_world": [],
        "normal_force": [],
        "episode_id": [],
        "sample_step": [],
        "timestamp": [],
        "randomization": [],
        "planned_target_position": [],
        "planned_target_rotation": [],
        "trajectory_family_id": [],
        "trajectory_parameters": [],
        "phase_id": [],
    }
    teleop_rows: list[dict[str, Any]] = []
    rgb_rows: list[np.ndarray] = []
    episode_rows: list[dict[str, Any]] = []
    episode_perturbations: list[np.ndarray] = []
    episode_trajectory_parameters: list[np.ndarray] = []
    episode_specs: list[EpisodeSpec] = []
    collection_renderer_mode: str | None = None
    collection_fallback_used: bool | None = None
    canonical_camera_pose = canonical_eye_in_hand_camera_pose(camera_name)
    teleop_runtime_config = teleop_config or Sigma7TeleopConfig()
    teleop_runtime_config.validate()
    receiver = None if synthetic_input else Sigma7UdpReceiver(packet_host, packet_port, timeout_seconds=0.0)
    stop_reason = "completed"

    try:
        for episode_id, perturbation in enumerate(perturbations):
            mapper = Sigma7TeleopMapper(teleop_runtime_config)
            scene_path, scene_config = scene_for_rollout(scene_spec.config_path, perturbation)
            try:
                model = load_model(scene_path)
            finally:
                cleanup_runtime_scene(scene_path)

            data = mujoco.MjData(model)
            reset_from_config(model, data, scene_config)
            validate_canonical_eye_in_hand_camera_config(scene_config, camera_name=camera_name)
            camera_local_position, camera_rotation, camera_fovy = eye_in_hand_camera_pose_from_config(
                scene_config,
                camera_name=camera_name,
            )
            camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
            if camera_id < 0:
                raise ValueError(f"Unknown camera {camera_name!r}.")
            apply_eye_in_hand_camera_pose(
                model,
                data,
                camera_id=camera_id,
                local_position=camera_local_position,
                rotation=camera_rotation,
                fovy=camera_fovy,
            )
            geometry = geometry_from_config(scene_config)
            simulation_dt_seconds = float(scene_config.get("physics", {}).get("timestep", 0.002))
            task_ids = peg_hole_ids(model, segments=geometry.segments)
            arm_ids = panda_arm_ids(model)
            nullspace_target_qpos = arm_qpos(data, arm_ids)
            hole_center = hole_center_position(data, task_ids)
            target_rotation = target_rotation_with_peg_tilt(
                site_rotation(data, model.site(rollout_config.site_name).id),
                perturbation,
            )
            perturbation_dict = _teleop_free_perturbation_dict(perturbation)
            randomization = _randomization_vector(perturbation_dict)
            episode_perturbations.append(randomization)

            rgb_renderer = None
            if save_rgb:
                rgb_renderer = MujocoRgbRenderer(
                    model,
                    camera_name=camera_name,
                    width=image_width,
                    height=image_height,
                    renderer_mode=renderer_mode,
                )
                if rgb_renderer.fallback_used and not allow_debug_fallback_renderer:
                    raise RuntimeError(
                        "Fallback-rendered RGB capture is legacy debug only and cannot be saved "
                        "for active image-only datasets."
                    )
                if collection_renderer_mode is None:
                    collection_renderer_mode = rgb_renderer.mode
                    collection_fallback_used = bool(rgb_renderer.fallback_used)
                elif collection_renderer_mode != rgb_renderer.mode or collection_fallback_used != bool(rgb_renderer.fallback_used):
                    raise RuntimeError("Renderer provenance changed across Sigma.7 collection episodes.")

            total_steps = rollout_config.total_steps if max_steps is None else min(int(max_steps), rollout_config.total_steps)
            sample_start = len(sample_rows["episode_id"])
            full_max_force = 0.0
            sampled_max_force = 0.0
            contact_count = 0
            contact_onset = -1
            max_torque = 0.0
            torque_saturation_count = 0
            arrays_finite = True
            completed = True
            zero_event_count = 0
            pause_event_count = 0
            timeout_step_count = 0
            actual_target_positions: list[np.ndarray] = []
            actual_phase_ids: list[int] = []
            clamped_delta_norm: list[float] = []

            try:
                for step in range(total_steps + 1):
                    phase_name, phase_step, phase_length = phase_for_step(step, rollout_config)
                    phase_id = {"approach_hold": 0, "descend": 1, "insert": 2, "final_hold": 3, "done": 3}[phase_name]
                    nominal_target_position = target_position_for_phase(
                        phase=phase_name,
                        phase_step=phase_step,
                        phase_length=phase_length,
                        hole_center=hole_center,
                        xy_offset=np.zeros(2, dtype=float),
                        config=rollout_config,
                    )
                    now_seconds = float(step * simulation_dt_seconds) if synthetic_input else time.perf_counter()
                    packet = (
                        _synthetic_packet_for_step(step, dt_seconds=simulation_dt_seconds)
                        if synthetic_input
                        else (receiver.recv_latest() if receiver is not None else None)
                    )
                    teleop_snapshot = mapper.update(
                        packet,
                        step_index=step,
                        now_seconds=now_seconds,
                        nominal_target_position=nominal_target_position,
                        control_dt_seconds=simulation_dt_seconds,
                    )
                    zero_event_count += int(teleop_snapshot.zero_event)
                    pause_event_count += int(teleop_snapshot.pause_event)
                    timeout_step_count += int(teleop_snapshot.timeout_active)
                    target_position = np.asarray(teleop_snapshot.target_position, dtype=float)
                    actual_target_positions.append(target_position.copy())
                    actual_phase_ids.append(int(phase_id))
                    clamped_delta_norm.append(float(np.linalg.norm(teleop_snapshot.clamped_delta)))

                    command = task_space_impedance_torque(
                        model,
                        data,
                        site_name=rollout_config.site_name,
                        target_position=target_position,
                        target_rotation=target_rotation,
                        arm_ids=arm_ids,
                        gains=gains,
                        position_stiffness_matrix=controller_entry.position_stiffness_matrix,
                        nullspace_target_qpos=nullspace_target_qpos,
                        clip_to_ctrlrange=False,
                    )
                    torque, saturated = clip_torque(model, arm_ids, command.torque)
                    contact_query = ContactQuery(model=model, data=data, task_ids=task_ids)
                    contact = extract_contact_state(contact_query)
                    normal_force = float(contact.normal_force)
                    full_max_force = max(full_max_force, normal_force)
                    if contact.in_contact:
                        contact_count += 1
                        if contact_onset < 0:
                            contact_onset = step
                    max_torque = max(max_torque, float(np.max(np.abs(torque))))
                    torque_saturation_count += int(saturated)

                    finite_step = bool(
                        np.all(np.isfinite(data.qpos))
                        and np.all(np.isfinite(data.qvel))
                        and np.all(np.isfinite(torque))
                        and np.isfinite(normal_force)
                    )
                    if not finite_step:
                        arrays_finite = False
                        completed = False
                        stop_reason = "nonfinite_rollout"
                        break

                    teleop_rows.append(
                        {
                            "episode_id": int(episode_id),
                            "step_index": int(step),
                            "time": float(data.time),
                            "received_timestamp": teleop_snapshot.received_timestamp,
                            "packet_timestamp": teleop_snapshot.packet_timestamp,
                            "packet_sequence": teleop_snapshot.packet_sequence,
                            "packet_age_seconds": teleop_snapshot.packet_age_seconds,
                            "raw_position": np.asarray(teleop_snapshot.raw_position, dtype=float),
                            "zero_reference_position": np.asarray(teleop_snapshot.zero_reference_position, dtype=float),
                            "raw_delta": np.asarray(teleop_snapshot.raw_delta, dtype=float),
                            "mapped_delta": np.asarray(teleop_snapshot.mapped_delta, dtype=float),
                            "clamped_delta": np.asarray(teleop_snapshot.clamped_delta, dtype=float),
                            "nominal_target_position": np.asarray(nominal_target_position, dtype=float),
                            "target_position": np.asarray(teleop_snapshot.target_position, dtype=float),
                            "paused": bool(teleop_snapshot.paused),
                            "zeroed": bool(teleop_snapshot.zeroed),
                            "packet_valid": bool(teleop_snapshot.packet_valid),
                            "fresh_packet_received": bool(teleop_snapshot.fresh_packet_received),
                            "timeout_active": bool(teleop_snapshot.timeout_active),
                            "zero_event": bool(teleop_snapshot.zero_event),
                            "pause_event": bool(teleop_snapshot.pause_event),
                            "quit_event": bool(teleop_snapshot.quit_event),
                            "packet_source": teleop_snapshot.packet_source,
                            "packet_json": teleop_snapshot.packet_json,
                            "timeout_mode": teleop_snapshot.timeout_mode,
                        }
                    )

                    if step % sample_stride == 0:
                        obs, u_ref, _ = collect_step(
                            model,
                            data,
                            arm_ids=arm_ids,
                            task_ids=task_ids,
                            target_position=target_position,
                            target_rotation=target_rotation,
                            phase_id=int(phase_id),
                        )
                        task_state = peg_hole_task_state(
                            data,
                            task_ids,
                            hole_clearance_delta=float(perturbation_dict["clearance_delta"]),
                        )
                        force_world = extract_net_peg_hole_contact_force_world(contact_query)
                        sampled_max_force = max(sampled_max_force, normal_force)
                        sample_rows["state"].append(_state_row(obs, data, arm_ids))
                        sample_rows["action"].append(_action_row(u_ref, torque))
                        sample_rows["contact_state"].append(contact_state_vector(contact))
                        sample_rows["task_state"].append(task_state)
                        sample_rows["contact_force_world"].append(force_world)
                        sample_rows["normal_force"].append(normal_force)
                        sample_rows["episode_id"].append(episode_id)
                        sample_rows["sample_step"].append(step)
                        sample_rows["timestamp"].append(float(data.time))
                        sample_rows["randomization"].append(randomization)
                        sample_rows["planned_target_position"].append(target_position)
                        sample_rows["planned_target_rotation"].append(target_rotation.reshape(-1))
                        sample_rows["trajectory_family_id"].append(TRAJECTORY_FAMILY_SIGMA7_LIVE_ID)
                        sample_rows["phase_id"].append(int(phase_id))
                        if rgb_renderer is not None:
                            if step % image_stride != 0:
                                raise RuntimeError("image_stride must align with sampled Sigma.7 collection steps.")
                            rgb_rows.append(rgb_renderer.render(data))

                    if teleop_snapshot.quit_event:
                        completed = False
                        stop_reason = "operator_quit"
                        break

                    if step < total_steps:
                        set_arm_torque_ctrl(model, data, arm_ids, torque)
                        mujoco.mj_step(model, data)
                        if realtime and not synthetic_input:
                            time.sleep(simulation_dt_seconds)
            finally:
                if rgb_renderer is not None:
                    rgb_renderer.close()

            if not actual_target_positions:
                break

            target_offsets = np.stack(actual_target_positions, axis=0) - np.asarray(hole_center, dtype=float)
            phase_ids = np.asarray(actual_phase_ids, dtype=np.int8)
            duration_seconds = float(max(0, len(actual_target_positions) - 1) * simulation_dt_seconds)
            trajectory_parameters = _trajectory_parameter_vector(
                phase_ids=phase_ids,
                rollout_config=rollout_config,
                target_offsets=target_offsets,
                clamped_delta_norm=np.asarray(clamped_delta_norm, dtype=float),
                zero_event_count=zero_event_count,
                pause_event_count=pause_event_count,
                duration_seconds=duration_seconds,
            )
            episode_trajectory_parameters.append(trajectory_parameters)
            for _ in range(sample_start, len(sample_rows["episode_id"])):
                sample_rows["trajectory_parameters"].append(trajectory_parameters)

            fixed_contact_condition = {
                "teleop_noise_xy_amplitude": 0.0,
                "teleop_noise_cycles": 0.0,
                "teleop_noise_phase_x": 0.0,
                "teleop_noise_phase_y": 0.0,
                "clearance_delta": float(perturbation_dict["clearance_delta"]),
                "friction_scale": float(perturbation_dict["friction_scale"]),
                "peg_tilt_x": float(perturbation_dict["peg_tilt_x"]),
                "peg_tilt_y": float(perturbation_dict["peg_tilt_y"]),
                "fixed_hole_yaw_offset": float(perturbation_dict["hole_yaw_offset"]),
                "hole_xy_radius": float(controlled_contact_profile.hole_xy_radius if controlled_contact_profile is not None else preset.hole_xy_radius),
            }
            episode_spec = EpisodeSpec.create(
                episode_id=episode_id,
                seed=seed,
                scene=scene,
                setting_id=preset.setting_id,
                profile_name=str(profile_name),
                contact_condition_name=controlled_contact_profile.contact_condition_name if controlled_contact_profile is not None else None,
                nominal_hole_position=nominal_hole_position,
                nominal_hole_xy=nominal_hole_xy,
                hole_xy_offset=np.asarray(perturbation_dict["hole_xy_offset"], dtype=float),
                hole_yaw_offset=float(perturbation_dict["hole_yaw_offset"]),
                hole_xy_radius=float(fixed_contact_condition["hole_xy_radius"]),
                hole_xy_offset_semantics=str(profile_metadata["hole_xy_offset_semantics"]),
                hole_xy_offset_distribution=str(profile_metadata["hole_xy_offset_distribution"]),
                trajectory_follows_randomized_hole=bool(profile_metadata["trajectory_follows_randomized_hole"]),
                contact_generation_parameters_fixed=bool(profile_metadata["contact_generation_parameters_fixed"]),
                fixed_contact_condition=fixed_contact_condition,
                trajectory_source=EPISODE_TRAJECTORY_SOURCE_EPISODE_SPEC_REPLAY,
                trajectory_family=TRAJECTORY_FAMILY_SIGMA7_LIVE,
                trajectory_family_id=TRAJECTORY_FAMILY_SIGMA7_LIVE_ID,
                trajectory_parameters=trajectory_parameters,
                teleop_mode=TELEOP_MODE_POSITION_ONLY,
                target_rotations=None,
                target_offsets=target_offsets,
                phase_ids=phase_ids,
                total_steps=int(target_offsets.shape[0] - 1),
                sample_stride=sample_stride,
                image_stride=image_stride if save_rgb else None,
            )
            episode_specs.append(episode_spec)

            final_state = peg_hole_task_state(
                data,
                task_ids,
                hole_clearance_delta=float(perturbation_dict["clearance_delta"]),
            )
            final_depth = float(final_state[2])
            final_lateral = float(np.linalg.norm(final_state[:2]))
            final_peg_tip = peg_tip_position(data, task_ids)
            final_hole_center = np.asarray(hole_center, dtype=float).copy()
            final_target_position = np.asarray(actual_target_positions[-1], dtype=float).copy()
            success_depth_threshold = float(0.95 * rollout_config.insert_depth)
            success_lateral_threshold = float(geometry.radial_clearance)
            success = bool(
                completed
                and final_depth >= success_depth_threshold
                and final_lateral <= success_lateral_threshold
            )
            if success:
                failure_reason = ""
            elif not completed and stop_reason == "operator_quit":
                failure_reason = "operator_quit"
            elif not completed:
                failure_reason = "incomplete_rollout"
            elif not arrays_finite:
                failure_reason = "nonfinite_rollout"
            elif final_depth < success_depth_threshold and final_lateral > success_lateral_threshold:
                failure_reason = "insufficient_depth_and_lateral_misalignment"
            elif final_depth < success_depth_threshold:
                failure_reason = "insufficient_depth"
            elif final_lateral > success_lateral_threshold:
                failure_reason = "lateral_misalignment"
            else:
                failure_reason = "unknown"
            label_eligible, spike_suspicious, exclusion_reason = classify_episode_admission(
                episode_complete=completed,
                arrays_finite=arrays_finite,
                full_step_max_force=full_max_force,
                sampled_max_force=sampled_max_force,
            )
            sample_stop = len(sample_rows["episode_id"])
            capture_ratio = sampled_max_force / full_max_force if full_max_force > 0.0 else 1.0
            episode_rows.append(
                {
                    "episode_id": episode_id,
                    "episode_spec_id": episode_spec.episode_spec_id,
                    "episode_spec_schema_version": EPISODE_SPEC_SCHEMA_VERSION,
                    "trajectory_source": EPISODE_TRAJECTORY_SOURCE_EPISODE_SPEC_REPLAY,
                    "trajectory_family": TRAJECTORY_FAMILY_SIGMA7_LIVE,
                    "trajectory_family_id": TRAJECTORY_FAMILY_SIGMA7_LIVE_ID,
                    "teleop_mode": TELEOP_MODE_POSITION_ONLY,
                    "operator": str(operator),
                    "input_device": "sigma7_position_only",
                    "sample_start": sample_start,
                    "sample_stop": sample_stop,
                    "sample_count": sample_stop - sample_start,
                    "total_steps": int(target_offsets.shape[0] - 1),
                    "nominal_hole_xy_x": float(nominal_hole_xy[0]),
                    "nominal_hole_xy_y": float(nominal_hole_xy[1]),
                    "actual_hole_xy_x": float(episode_spec.actual_hole_xy[0]),
                    "actual_hole_xy_y": float(episode_spec.actual_hole_xy[1]),
                    "trajectory_center_xy_x": float(episode_spec.trajectory_center_xy[0]),
                    "trajectory_center_xy_y": float(episode_spec.trajectory_center_xy[1]),
                    "trajectory_minus_hole_xy_x": float(episode_spec.trajectory_minus_hole_xy[0]),
                    "trajectory_minus_hole_xy_y": float(episode_spec.trajectory_minus_hole_xy[1]),
                    "success": success,
                    "final_depth": final_depth,
                    "final_lateral_error": final_lateral,
                    "final_peg_tip_x": float(final_peg_tip[0]),
                    "final_peg_tip_y": float(final_peg_tip[1]),
                    "final_peg_tip_z": float(final_peg_tip[2]),
                    "final_hole_center_x": float(final_hole_center[0]),
                    "final_hole_center_y": float(final_hole_center[1]),
                    "final_hole_center_z": float(final_hole_center[2]),
                    "final_target_position_x": float(final_target_position[0]),
                    "final_target_position_y": float(final_target_position[1]),
                    "final_target_position_z": float(final_target_position[2]),
                    "success_depth_threshold": success_depth_threshold,
                    "success_lateral_threshold": success_lateral_threshold,
                    "hole_yaw": float(scene_config.get("hole", {}).get("rotation", 0.0)) + float(perturbation_dict["hole_yaw_offset"]),
                    "peg_yaw": float(final_state[5]),
                    "insertion_axis_x": 0.0,
                    "insertion_axis_y": 0.0,
                    "insertion_axis_z": -1.0,
                    "failure_reason": failure_reason,
                    "full_step_max_force": full_max_force,
                    "sampled_max_force": sampled_max_force,
                    "capture_ratio": capture_ratio,
                    "contact_count": contact_count,
                    "contact_onset_step": contact_onset,
                    "max_abs_commanded_torque": max_torque,
                    "torque_saturation_count": torque_saturation_count,
                    "episode_complete": completed,
                    "arrays_finite": arrays_finite,
                    "solver_spike_suspicious": spike_suspicious,
                    "label_eligible": label_eligible,
                    "exclusion_reason": exclusion_reason,
                    "zero_event_count": zero_event_count,
                    "pause_event_count": pause_event_count,
                    "timeout_step_count": timeout_step_count,
                    "duration_seconds": duration_seconds,
                    "clearance_delta": float(perturbation_dict["clearance_delta"]),
                    "perturbation": perturbation_dict,
                }
            )
            print(
                f"episode={episode_id:03d} mode=sigma7_live success={success} "
                f"eligible={label_eligible} samples={sample_stop - sample_start} "
                f"depth={final_depth:.6f} lat={final_lateral:.6f} "
                f"full_max={full_max_force:.3f} sampled_max={sampled_max_force:.3f}",
                flush=True,
            )
            if stop_reason == "operator_quit":
                break
    finally:
        if receiver is not None:
            receiver.close()

    actual_episodes = len(episode_rows)
    if actual_episodes == 0:
        raise RuntimeError("Sigma.7 collection did not record any episodes.")

    raw_arrays = {
        "state": np.vstack(sample_rows["state"]),
        "action": np.vstack(sample_rows["action"]),
        "contact_state": np.vstack(sample_rows["contact_state"]),
        "task_state": np.vstack(sample_rows["task_state"]),
        "contact_force_world": np.vstack(sample_rows["contact_force_world"]),
        "normal_force": np.asarray(sample_rows["normal_force"], dtype=float),
        "episode_id": np.asarray(sample_rows["episode_id"], dtype=np.int32),
        "sample_step": np.asarray(sample_rows["sample_step"], dtype=np.int32),
        "timestamp": np.asarray(sample_rows["timestamp"], dtype=float),
        "randomization": np.vstack(sample_rows["randomization"]),
        "planned_target_position": np.vstack(sample_rows["planned_target_position"]),
        "planned_target_rotation": np.vstack(sample_rows["planned_target_rotation"]),
        "trajectory_family_id": np.asarray(sample_rows["trajectory_family_id"], dtype=np.int8),
        "trajectory_parameters": np.vstack(sample_rows["trajectory_parameters"]),
        "phase_id": np.asarray(sample_rows["phase_id"], dtype=np.int8),
        "episode_summary_id": np.arange(actual_episodes, dtype=np.int32),
        "episode_success": np.asarray([row["success"] for row in episode_rows], dtype=bool),
        "episode_final_depth": np.asarray([row["final_depth"] for row in episode_rows], dtype=float),
        "episode_final_lateral_error": np.asarray([row["final_lateral_error"] for row in episode_rows], dtype=float),
        "episode_max_normal_force": np.asarray([row["full_step_max_force"] for row in episode_rows], dtype=float),
        "episode_sampled_max_normal_force": np.asarray([row["sampled_max_force"] for row in episode_rows], dtype=float),
        "episode_force_capture_ratio": np.asarray([row["capture_ratio"] for row in episode_rows], dtype=float),
        "episode_contact_count": np.asarray([row["contact_count"] for row in episode_rows], dtype=np.int32),
        "episode_contact_onset_step": np.asarray([row["contact_onset_step"] for row in episode_rows], dtype=np.int32),
        "episode_perturbation": np.vstack(episode_perturbations),
        "episode_command_xy_offset": np.vstack([parameters[7:9] for parameters in episode_trajectory_parameters]),
        "episode_trajectory_family_id": np.asarray([TRAJECTORY_FAMILY_SIGMA7_LIVE_ID] * actual_episodes, dtype=np.int8),
        "episode_trajectory_parameters": np.vstack(episode_trajectory_parameters),
        "episode_complete": np.asarray([row["episode_complete"] for row in episode_rows], dtype=bool),
        "episode_solver_spike_suspicious": np.asarray([row["solver_spike_suspicious"] for row in episode_rows], dtype=bool),
        "episode_label_eligible": np.asarray([row["label_eligible"] for row in episode_rows], dtype=bool),
    }
    if save_rgb:
        if len(rgb_rows) != int(raw_arrays["task_state"].shape[0]):
            raise RuntimeError("RGB capture count does not match sample count.")
        raw_arrays["rgb_images"] = np.stack(rgb_rows, axis=0).astype(np.uint8, copy=False)

    raw_metadata = {
        "schema_version": RAW_SCHEMA_VERSION,
        "scene": scene,
        "setting_id": preset.setting_id,
        "profile_name": profile_name,
        "teleop_mode": TELEOP_MODE_POSITION_ONLY,
        "teleop_source": "sigma7_live_position_only",
        "sigma7_packet_schema_version": SIGMA7_PACKET_SCHEMA_VERSION,
        "synthetic_input": bool(synthetic_input),
        "operator": str(operator),
        "packet_host": None if synthetic_input else str(packet_host),
        "packet_port": None if synthetic_input else int(packet_port),
        "teleop_config": {
            "deadband": float(teleop_runtime_config.deadband),
            "max_target_velocity": float(teleop_runtime_config.max_target_velocity),
            "timeout_seconds": float(teleop_runtime_config.timeout_seconds),
            "timeout_mode": str(teleop_runtime_config.timeout_mode),
            "workspace_min_delta": list(map(float, teleop_runtime_config.workspace_min_delta)),
            "workspace_max_delta": list(map(float, teleop_runtime_config.workspace_max_delta)),
            "zero_on_first_packet": bool(teleop_runtime_config.zero_on_first_packet),
        },
        "eye_in_hand_camera_pose_version": canonical_camera_pose["pose_version"],
        "eye_in_hand_camera_canonical": bool(canonical_camera_pose["canonical"]),
        "eye_in_hand_camera_name": canonical_camera_pose["camera_name"],
        "eye_in_hand_camera_attachment_parent": canonical_camera_pose["attachment_parent"],
        "eye_in_hand_camera_mount_type": canonical_camera_pose["mount_type"],
        "eye_in_hand_camera_pose": canonical_camera_pose,
        "collection_seed": seed,
        **profile_metadata,
        "robustness_preset": preset.to_metadata(),
        "controller_id": controller_entry.controller_id,
        "scenario_id": scenario_id,
        "collection_controller_id": controller_entry.controller_id,
        "collection_stiffness_matrix": controller_entry.position_stiffness_matrix.tolist(),
        "dataset_path": str(raw_path),
        "controllers_yaml": str(controllers_yaml),
        "base_profile": profile,
        "controller_profile": profile,
        "gain_config": str(gain_config),
        "trajectory_plan": TRAJECTORY_PLAN_NAME,
        "trajectory_source": EPISODE_TRAJECTORY_SOURCE_EPISODE_SPEC_REPLAY,
        "trajectory_families": [TRAJECTORY_FAMILY_SIGMA7_LIVE],
        "trajectory_parameter_fields": list(TRAJECTORY_PARAMETER_FIELDS),
        "num_episodes": actual_episodes,
        "requested_episodes": episodes,
        "num_samples": int(raw_arrays["task_state"].shape[0]),
        "sample_stride": sample_stride,
        "rgb_enabled": bool(save_rgb),
        "rgb_camera_name": camera_name if save_rgb else None,
        "rgb_image_width": int(image_width) if save_rgb else None,
        "rgb_image_height": int(image_height) if save_rgb else None,
        "rgb_image_stride": int(image_stride) if save_rgb else None,
        "rgb_max_episodes": None if max_rgb_episodes is None else int(max_rgb_episodes),
        "renderer_mode": collection_renderer_mode if save_rgb else None,
        "fallback_used": collection_fallback_used if save_rgb else None,
        "native_launcher_required": default_native_launcher(),
        "episode_specs_path": str(episode_specs_path),
        "frozen_paired_episode_specs_path": str(frozen_episode_specs_path),
        "frozen_train_val_split_path": str(frozen_split_path),
        "teleop_trace_path": str(teleop_trace_path),
        "episode_spec_schema_version": EPISODE_SPEC_SCHEMA_VERSION,
        "simulation_timestep": 0.002,
        "sample_rate_hz": 1.0 / (0.002 * sample_stride),
        "seed": seed,
        "rng_streams": {
            "root_seed": seed,
            "perturbation_seed": perturbation_seed,
            "split_seed": split_seed,
        },
        "randomization_fields": list(RANDOMIZATION_FIELDS),
        "legacy_field_mapping": profile_metadata.get("legacy_field_mapping"),
        "admission_rule": {
            "solver_spike_full_step_threshold": 1000.0,
            "solver_spike_capture_ratio_threshold": 0.05,
            "solver_spike_sampled_force_ceiling": 500.0,
            "whole_episode_quarantine": True,
        },
        "labels_built": False,
        "human_proxy_replaced_by_sigma7_live": True,
        "target_generation_teleop_noise_disabled": True,
    }
    np.savez_compressed(raw_path, **raw_arrays, metadata=json.dumps(raw_metadata, sort_keys=True))

    teleop_payload = _stack_records(teleop_rows)
    np.savez_compressed(teleop_trace_path, **teleop_payload, metadata=json.dumps(raw_metadata, sort_keys=True))

    eligible_episode_ids = np.asarray([row["episode_id"] for row in episode_rows if row["label_eligible"]], dtype=np.int32)
    eligible_sample_mask = np.isin(raw_arrays["episode_id"], eligible_episode_ids)
    if not np.any(eligible_sample_mask):
        raise RuntimeError(f"No label-eligible samples were collected. Raw artifact is available at {raw_path}.")

    task_state = raw_arrays["task_state"][eligible_sample_mask]
    contact_force = raw_arrays["contact_force_world"][eligible_sample_mask]
    normalized_matrix, normalized_cholesky, diagnostics = build_stiffness_labels_with_diagnostics(
        task_state,
        contact_force,
        config=StiffnessLabelConfig(neighbors=label_neighbors, knn_block_size=knn_block_size),
    )
    projection = project_residual_first_labels(
        normalized_matrix,
        base_spec=base_spec,
        residual_bound=float(np.max(base_spec.residual_bounds)),
        baseline_k=baseline_k,
        diagnostic_k_min=diagnostic_k_min,
        diagnostic_k_max=diagnostic_k_max,
        l21_coupling_percentile=l21_coupling_percentile,
    )
    residual_group = projection.residual_group_target
    residual_theta = projection.residual_theta_target
    physical_matrix = projection.physical_stiffness_matrix_target
    physical_cholesky = projection.physical_stiffness_cholesky_target

    eligible_rows = [row for row in episode_rows if row["label_eligible"]]
    eligible_success = np.asarray([row["success"] for row in eligible_rows], dtype=bool)
    train_ids, val_ids = _stratified_split(eligible_episode_ids, eligible_success, rng=split_rng)
    eligible_arrays = {
        "state": raw_arrays["state"][eligible_sample_mask],
        "action": raw_arrays["action"][eligible_sample_mask],
        "contact_state": raw_arrays["contact_state"][eligible_sample_mask],
        "task_state": task_state,
        "contact_force_world": contact_force,
        "stiffness_matrix_target": normalized_matrix,
        "stiffness_cholesky_target": normalized_cholesky,
        "physical_stiffness_matrix_target": physical_matrix,
        "physical_stiffness_cholesky_target": physical_cholesky,
        "physical_stiffness_eigenvalues": projection.physical_stiffness_eigenvalues,
        "residual_group_target": residual_group,
        "residual_theta_target": residual_theta,
        "theta_base": base_spec.theta_base,
        "base_stiffness_matrix": base_spec.base_matrix,
        "episode_id": raw_arrays["episode_id"][eligible_sample_mask],
        "sample_step": raw_arrays["sample_step"][eligible_sample_mask],
        "timestamp": raw_arrays["timestamp"][eligible_sample_mask],
        "randomization": raw_arrays["randomization"][eligible_sample_mask],
        "planned_target_position": raw_arrays["planned_target_position"][eligible_sample_mask],
        "planned_target_rotation": raw_arrays["planned_target_rotation"][eligible_sample_mask],
        "trajectory_family_id": raw_arrays["trajectory_family_id"][eligible_sample_mask],
        "trajectory_parameters": raw_arrays["trajectory_parameters"][eligible_sample_mask],
        "phase_id": raw_arrays["phase_id"][eligible_sample_mask],
        "episode_summary_id": eligible_episode_ids,
        "episode_success": eligible_success,
        "episode_final_depth": np.asarray([row["final_depth"] for row in eligible_rows], dtype=float),
        "episode_final_lateral_error": np.asarray([row["final_lateral_error"] for row in eligible_rows], dtype=float),
        "episode_max_normal_force": np.asarray([row["full_step_max_force"] for row in eligible_rows], dtype=float),
        "episode_contact_count": np.asarray([row["contact_count"] for row in eligible_rows], dtype=np.int32),
        "episode_contact_onset_step": np.asarray([row["contact_onset_step"] for row in eligible_rows], dtype=np.int32),
        "episode_perturbation": raw_arrays["episode_perturbation"][eligible_episode_ids],
        "episode_command_xy_offset": raw_arrays["episode_command_xy_offset"][eligible_episode_ids],
        "episode_trajectory_family_id": raw_arrays["episode_trajectory_family_id"][eligible_episode_ids],
        "episode_trajectory_parameters": raw_arrays["episode_trajectory_parameters"][eligible_episode_ids],
        "train_episode_ids": train_ids,
        "val_episode_ids": val_ids,
        **diagnostics,
    }
    if save_rgb:
        eligible_arrays["rgb_images"] = raw_arrays["rgb_images"][eligible_sample_mask]

    eligible_metadata = {
        "schema_version": ELIGIBLE_SCHEMA_VERSION,
        "scene": scene,
        "scene_config": str(scene_spec.config_path),
        "eye_in_hand_camera_pose_version": canonical_camera_pose["pose_version"],
        "eye_in_hand_camera_canonical": bool(canonical_camera_pose["canonical"]),
        "eye_in_hand_camera_name": canonical_camera_pose["camera_name"],
        "eye_in_hand_camera_attachment_parent": canonical_camera_pose["attachment_parent"],
        "eye_in_hand_camera_mount_type": canonical_camera_pose["mount_type"],
        "eye_in_hand_camera_pose": canonical_camera_pose,
        "setting_id": preset.setting_id,
        "profile_name": profile_name,
        "collection_seed": seed,
        **profile_metadata,
        "robustness_preset": preset.to_metadata(),
        "controller_id": controller_entry.controller_id,
        "scenario_id": scenario_id,
        "collection_controller_id": controller_entry.controller_id,
        "collection_stiffness_matrix": controller_entry.position_stiffness_matrix.tolist(),
        "dataset_path": str(eligible_path),
        "controllers_yaml": str(controllers_yaml),
        "base_profile": profile,
        "gain_config": str(gain_config),
        "base_stiffness_spec": base_spec.to_metadata(),
        "trajectory_plan": TRAJECTORY_PLAN_NAME,
        "trajectory_source": EPISODE_TRAJECTORY_SOURCE_EPISODE_SPEC_REPLAY,
        "trajectory_families": [TRAJECTORY_FAMILY_SIGMA7_LIVE],
        "trajectory_parameter_fields": list(TRAJECTORY_PARAMETER_FIELDS),
        "num_episodes": int(eligible_episode_ids.size),
        "raw_episode_count": actual_episodes,
        "requested_episode_count": episodes,
        "num_samples": int(task_state.shape[0]),
        "seed": seed,
        "rng_streams": raw_metadata["rng_streams"],
        "sample_stride": sample_stride,
        "rgb_enabled": bool(save_rgb),
        "rgb_camera_name": camera_name if save_rgb else None,
        "rgb_image_width": int(image_width) if save_rgb else None,
        "rgb_image_height": int(image_height) if save_rgb else None,
        "rgb_image_stride": int(image_stride) if save_rgb else None,
        "rgb_max_episodes": None if max_rgb_episodes is None else int(max_rgb_episodes),
        "renderer_mode": collection_renderer_mode if save_rgb else None,
        "fallback_used": collection_fallback_used if save_rgb else None,
        "native_launcher_required": default_native_launcher(),
        "episode_specs_path": str(episode_specs_path),
        "frozen_paired_episode_specs_path": str(frozen_episode_specs_path),
        "frozen_train_val_split_path": str(frozen_split_path),
        "teleop_trace_path": str(teleop_trace_path),
        "episode_spec_schema_version": EPISODE_SPEC_SCHEMA_VERSION,
        "sample_rate_hz": raw_metadata["sample_rate_hz"],
        "label_neighbors": label_neighbors,
        "knn_block_size": knn_block_size,
        "legacy_label_k_min": label_k_min,
        "legacy_label_k_max": label_k_max,
        "target": "residual_cholesky_group_delta",
        "task_state_dim": int(task_state.shape[1]),
        "residual_dim": int(residual_group.shape[1]),
        "randomization_fields": list(RANDOMIZATION_FIELDS),
        "legacy_field_mapping": profile_metadata.get("legacy_field_mapping"),
        **projection.label_metadata,
        "episode_outcome_fields": [
            "success",
            "final_depth",
            "final_lateral_error",
            "max_normal_force",
            "contact_count",
            "contact_onset_step",
            "perturbation",
            "command_xy_offset",
            "trajectory_family",
            "trajectory_parameters",
        ],
        "episode_split": {
            "strategy": "success_stratified_80_20",
            "split_seed": split_seed,
            "train_episode_count": int(train_ids.size),
            "val_episode_count": int(val_ids.size),
        },
        "raw_source_dataset": str(raw_path),
        "excluded_episode_ids": [row["episode_id"] for row in episode_rows if not row["label_eligible"]],
        "teleop_mode": TELEOP_MODE_POSITION_ONLY,
        "teleop_source": "sigma7_live_position_only",
        "synthetic_input": bool(synthetic_input),
        "operator": str(operator),
        "packet_host": None if synthetic_input else str(packet_host),
        "packet_port": None if synthetic_input else int(packet_port),
        "teleop_config": raw_metadata["teleop_config"],
        "sigma7_packet_schema_version": SIGMA7_PACKET_SCHEMA_VERSION,
        "human_proxy_replaced_by_sigma7_live": True,
        "target_generation_teleop_noise_disabled": True,
    }
    np.savez_compressed(eligible_path, **eligible_arrays, metadata=json.dumps(eligible_metadata, sort_keys=True))
    validate_residual_dataset(eligible_path)

    write_episode_specs_jsonl(episode_specs_path, episode_specs)
    shutil.copy2(episode_specs_path, frozen_episode_specs_path)

    with episode_csv_path.open("w", encoding="utf-8", newline="") as handle:
        csv_fields = [key for key in episode_rows[0] if key != "perturbation"] + [f"perturbation_{name}" for name in RANDOMIZATION_FIELDS]
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row, values in zip(episode_rows, episode_perturbations, strict=True):
            flat = {key: value for key, value in row.items() if key != "perturbation"}
            flat.update({f"perturbation_{name}": float(value) for name, value in zip(RANDOMIZATION_FIELDS, values, strict=True)})
            writer.writerow(flat)

    frozen_split = FrozenTrainValSplit(
        dataset_path=str(eligible_path),
        collection_seed=int(seed),
        split_seed=int(split_seed),
        train_episode_ids=np.asarray(train_ids, dtype=np.int64),
        val_episode_ids=np.asarray(val_ids, dtype=np.int64),
        metadata={
            "scene": scene,
            "setting_id": preset.setting_id,
            "profile_name": profile_name,
            "controllers_yaml": str(controllers_yaml),
            "collection_controller_id": controller_entry.controller_id,
            "teleop_source": "sigma7_live_position_only",
        },
    )
    frozen_split.write(frozen_split_path)

    contact_fraction = float(np.mean(eligible_arrays["contact_state"][:, 0] > 0.5))
    summary = {
        "status": "passed",
        "runtime_seconds": time.perf_counter() - started,
        "raw_dataset": str(raw_path),
        "eligible_dataset": str(eligible_path),
        "teleop_trace": str(teleop_trace_path),
        "collection_seed": int(seed),
        "split_seed": int(split_seed),
        "requested_episodes": int(episodes),
        "episodes": int(actual_episodes),
        "eligible_episodes": int(eligible_episode_ids.size),
        "quarantined_episodes": int(actual_episodes - eligible_episode_ids.size),
        "successes": int(sum(row["success"] for row in episode_rows)),
        "failures": int(sum(not row["success"] for row in episode_rows)),
        "raw_samples": int(raw_arrays["task_state"].shape[0]),
        "eligible_samples": int(eligible_arrays["task_state"].shape[0]),
        "eligible_contact_fraction": contact_fraction,
        "catastrophic_episodes": int(sum(row["full_step_max_force"] >= 1000.0 for row in episode_rows)),
        "solver_spike_suspicious_episodes": int(sum(row["solver_spike_suspicious"] for row in episode_rows)),
        "trajectory_family_counts": {TRAJECTORY_FAMILY_SIGMA7_LIVE: int(actual_episodes)},
        "rng_streams": raw_metadata["rng_streams"],
        "validation": {
            "eligible_residual_dataset": "passed",
            "raw_arrays_finite": bool(
                all(
                    np.all(np.isfinite(value))
                    for value in raw_arrays.values()
                    if np.issubdtype(np.asarray(value).dtype, np.number)
                )
            ),
            "sample_stride": sample_stride,
            "sample_rate_hz": raw_metadata["sample_rate_hz"],
        },
        "frozen_paired_episode_specs_path": str(frozen_episode_specs_path),
        "frozen_train_val_split_path": str(frozen_split_path),
        "synthetic_input": bool(synthetic_input),
        "operator": str(operator),
        "stop_reason": stop_reason,
        "teleop_mode": TELEOP_MODE_POSITION_ONLY,
    }
    _write_json(summary_path, summary)

    return Sigma7LiveCollectionResult(
        output_dir=output_dir,
        raw_dataset=raw_path,
        eligible_dataset=eligible_path,
        collection_summary=summary_path,
        episode_csv=episode_csv_path,
        episode_specs=episode_specs_path,
        frozen_episode_specs=frozen_episode_specs_path,
        frozen_split=frozen_split_path,
        teleop_trace=teleop_trace_path,
    )


__all__ = [
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_CONTROLLER_ID",
    "ELIGIBLE_SCHEMA_VERSION",
    "RAW_SCHEMA_VERSION",
    "Sigma7LiveCollectionResult",
    "TRAJECTORY_FAMILY_SIGMA7_LIVE",
    "TRAJECTORY_PARAMETER_FIELDS",
    "collect_sigma7_live_residual_dataset",
]
