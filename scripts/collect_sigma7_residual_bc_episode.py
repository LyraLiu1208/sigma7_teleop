from __future__ import annotations

import argparse
import csv
import getpass
import json
import os
import secrets
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np

from _sigma7_residual_pipeline_common import (
    DEFAULT_PIPELINE_ROOT,
    ensure_mujoco_src_on_path,
    participant_episodes_root,
    require_safe_segment,
    write_json,
)
from _sigma7_runtime import default_mjpython, default_viewer_python


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MJ_PYTHON = default_mjpython()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ensure_mujoco_src_on_path()

from stiffness_copilot_mujoco.contact.state import (  # noqa: E402
    ContactQuery,
    contact_state_vector,
    extract_contact_state,
    extract_net_peg_hole_contact_force_world,
)
from stiffness_copilot_mujoco.controllers.impedance import task_space_impedance_torque  # noqa: E402
from stiffness_copilot_mujoco.controllers.track_a_controllers import (  # noqa: E402
    DEFAULT_TRACK_A_CONTROLLERS_YAML,
    load_track_a_controller_runtime,
)
from stiffness_copilot_mujoco.episodes.episode_spec import (  # noqa: E402
    EPISODE_SPEC_SCHEMA_VERSION,
    EPISODE_TRAJECTORY_SOURCE_EPISODE_SPEC_REPLAY,
    EpisodeSpec,
    write_episode_specs_jsonl,
)
from stiffness_copilot_mujoco.episodes.teleop_proxy import TELEOP_MODE_POSITION_ORIENTATION  # noqa: E402
from stiffness_copilot_mujoco.franka_viewer import load_model  # noqa: E402
from stiffness_copilot_mujoco.learning.dataset_collection import _action_row, _state_row  # noqa: E402
from stiffness_copilot_mujoco.learning.frozen_train_val_split import FrozenTrainValSplit  # noqa: E402
from stiffness_copilot_mujoco.learning.open_loop_residual_dataset import (  # noqa: E402
    build_eligible_residual_dataset_from_raw,
    classify_episode_admission,
)
from stiffness_copilot_mujoco.learning.task_state import peg_hole_task_state  # noqa: E402
from stiffness_copilot_mujoco.learning.vision_residual_dataset import infer_training_data_valid  # noqa: E402
from stiffness_copilot_mujoco.metrics.task_metrics import (  # noqa: E402
    geometry_from_config,
    hole_center_position,
    insertion_depth,
    lateral_error,
    load_scene_config,
    peg_tip_position,
)
from stiffness_copilot_mujoco.panda_control import arm_qpos, panda_arm_ids, set_arm_torque_ctrl  # noqa: E402
from stiffness_copilot_mujoco.pose_math import site_rotation  # noqa: E402
from stiffness_copilot_mujoco.robustness import (  # noqa: E402
    ControlledContactProfile,
    get_robustness_preset,
    make_controlled_contact_profile,
    sample_controlled_contact_perturbations,
    sample_robustness_perturbations,
)
from stiffness_copilot_mujoco.rollout_observation import collect_step, reset_from_config  # noqa: E402
from stiffness_copilot_mujoco.rollouts.fixed_impedance import (  # noqa: E402
    RolloutConfig,
    RolloutPerturbation,
    cleanup_runtime_scene,
    clip_torque,
    scene_for_rollout,
)
from stiffness_copilot_mujoco.scenes import get_scene_spec  # noqa: E402
from stiffness_copilot_mujoco.sim.ids import peg_hole_ids  # noqa: E402
from stiffness_copilot_mujoco.sim.scene import (  # noqa: E402
    apply_eye_in_hand_camera_pose,
    canonical_eye_in_hand_camera_pose,
    eye_in_hand_camera_pose_from_config,
    validate_canonical_eye_in_hand_camera_config,
)
from stiffness_copilot_mujoco.vision.rendering import MujocoRgbRenderer  # noqa: E402
from tools.run_sigma7_mujoco_live_teleop import (  # noqa: E402
    EyeInHandWindow,
    PoseTeleopConfig,
    PoseTeleopState,
    Sigma7PoseReceiver,
    TerminalKeyMonitor,
    _build_third_person_camera,
    _rotation_matrix_to_row_major,
    _scene_alias,
    _viewer_running,
    map_sigma7_pose,
)


RAW_SCHEMA_VERSION = "residual_bc_sigma7_live_pose_raw_v1"
TRAJECTORY_FAMILY_SIGMA7_LIVE_POSE = "sigma7_live_pose"
TRAJECTORY_FAMILY_SIGMA7_LIVE_POSE_ID = 0
DEFAULT_CONTACT_PROFILE = "sigma7_seeded_pose_v1"
DEFAULT_SEEDED_HOLE_YAW_MAX_DEG = 1.0
DEFAULT_SEEDED_TELEOP_NOISE_CYCLES_MIN = 0.75
DEFAULT_SEEDED_TELEOP_NOISE_CYCLES_MAX = 1.25
DEFAULT_RANDOM_COLLECTION_SEED_MAX = 2**31 - 1
SEED_RANDOMIZED_FIELDS = (
    "hole_xy_offset",
    "hole_yaw_offset",
    "teleop_noise_cycles",
    "teleop_noise_phase_x",
    "teleop_noise_phase_y",
)
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
    "duration_seconds",
    "mean_position_delta_norm",
    "max_position_delta_norm",
    "mean_orientation_error_norm",
    "max_orientation_error_norm",
    "zero_event_count",
    "user_quit_event_count",
    "position_scale",
    "max_orientation_angle_rad",
)


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _unique_output_dir(root: Path, name: str) -> Path:
    candidate = root / name
    suffix = 0
    while candidate.exists():
        suffix += 1
        candidate = root / f"{name}_rerun{suffix}"
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def _randomization_vector(perturbation: RolloutPerturbation) -> np.ndarray:
    payload = perturbation.to_dict()
    hole_xy = np.asarray(payload["hole_xy_offset"], dtype=float)
    return np.array(
        [
            float(hole_xy[0]),
            float(hole_xy[1]),
            float(payload["hole_yaw_offset"]),
            float(payload["clearance_delta"]),
            float(payload["friction_scale"]),
            float(payload["peg_tilt_x"]),
            float(payload["peg_tilt_y"]),
            float(payload["teleop_noise_xy_amplitude"]),
            float(payload["teleop_noise_cycles"]),
        ],
        dtype=float,
    )


def _legacy_randomized_profile_metadata(scene: str) -> tuple[str, dict[str, Any]]:
    preset = get_robustness_preset(scene)
    return (
        f"{preset.setting_id}_legacy_randomized",
        {
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
        },
    )


def _seeded_pose_profile_metadata(
    *,
    scene: str,
    hole_xy_radius: float,
    hole_yaw_max_deg: float,
    teleop_noise_cycles_min: float,
    teleop_noise_cycles_max: float,
    fixed_teleop_noise_xy_amplitude: float,
    fixed_clearance_delta: float,
    fixed_friction_scale: float,
    fixed_peg_tilt_x: float,
    fixed_peg_tilt_y: float,
) -> tuple[str, dict[str, Any]]:
    preset = get_robustness_preset(scene)
    return (
        f"{preset.setting_id}_{DEFAULT_CONTACT_PROFILE}",
        {
            "hole_xy_offset_semantics": "global_object_placement",
            "hole_xy_offset_units": "m",
            "hole_xy_offset_distribution": f"uniform_disk(radius={hole_xy_radius:g})",
            "trajectory_follows_randomized_hole": True,
            "contact_generation_parameters_fixed": False,
            "seed_randomized_fields": list(SEED_RANDOMIZED_FIELDS),
            "seed_randomization_ranges": {
                "hole_xy_radius": float(hole_xy_radius),
                "hole_yaw_max_deg": float(hole_yaw_max_deg),
                "teleop_noise_cycles_min": float(teleop_noise_cycles_min),
                "teleop_noise_cycles_max": float(teleop_noise_cycles_max),
                "teleop_noise_phase_x_min_rad": float(-np.pi),
                "teleop_noise_phase_x_max_rad": float(np.pi),
                "teleop_noise_phase_y_min_rad": float(-np.pi),
                "teleop_noise_phase_y_max_rad": float(np.pi),
            },
            "fixed_hole_yaw_offset": None,
            "fixed_teleop_noise_xy_amplitude": float(fixed_teleop_noise_xy_amplitude),
            "fixed_teleop_noise_cycles": None,
            "fixed_teleop_noise_phase_x": None,
            "fixed_teleop_noise_phase_y": None,
            "fixed_clearance_delta": float(fixed_clearance_delta),
            "fixed_friction_scale": float(fixed_friction_scale),
            "fixed_peg_tilt_x": float(fixed_peg_tilt_x),
            "fixed_peg_tilt_y": float(fixed_peg_tilt_y),
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
        },
    )


def _sample_uniform_disk_offset(*, rng: np.random.Generator, radius: float) -> tuple[float, float]:
    sampled_radius = float(radius) * float(np.sqrt(rng.uniform(0.0, 1.0)))
    angle = float(rng.uniform(-np.pi, np.pi))
    return (sampled_radius * float(np.cos(angle)), sampled_radius * float(np.sin(angle)))


def _sample_sigma7_seeded_pose_perturbation(
    *,
    scene: str,
    seed: int,
    hole_xy_radius: float,
    hole_yaw_max_deg: float,
    teleop_noise_xy_amplitude: float,
    teleop_noise_cycles_min: float,
    teleop_noise_cycles_max: float,
    clearance_delta: float,
    friction_scale: float,
    peg_tilt_x: float,
    peg_tilt_y: float,
) -> tuple[RolloutPerturbation, str, dict[str, Any], dict[str, Any]]:
    if teleop_noise_cycles_min <= 0.0 or teleop_noise_cycles_max <= 0.0:
        raise ValueError("teleop noise cycle bounds must be positive.")
    if teleop_noise_cycles_min > teleop_noise_cycles_max:
        raise ValueError("teleop noise cycle min must not exceed max.")
    rng = np.random.default_rng(seed)
    preset = get_robustness_preset(scene)
    perturbation = RolloutPerturbation(
        hole_xy_offset=_sample_uniform_disk_offset(rng=rng, radius=float(hole_xy_radius)),
        hole_yaw_offset=float(rng.uniform(-np.deg2rad(hole_yaw_max_deg), np.deg2rad(hole_yaw_max_deg))),
        teleop_noise_xy_amplitude=float(teleop_noise_xy_amplitude),
        teleop_noise_cycles=float(rng.uniform(teleop_noise_cycles_min, teleop_noise_cycles_max)),
        teleop_noise_phase_x=float(rng.uniform(-np.pi, np.pi)),
        teleop_noise_phase_y=float(rng.uniform(-np.pi, np.pi)),
        clearance_delta=float(clearance_delta),
        friction_scale=float(friction_scale),
        peg_tilt_x=float(peg_tilt_x),
        peg_tilt_y=float(peg_tilt_y),
    )
    profile_name, profile_metadata = _seeded_pose_profile_metadata(
        scene=scene,
        hole_xy_radius=hole_xy_radius,
        hole_yaw_max_deg=hole_yaw_max_deg,
        teleop_noise_cycles_min=teleop_noise_cycles_min,
        teleop_noise_cycles_max=teleop_noise_cycles_max,
        fixed_teleop_noise_xy_amplitude=teleop_noise_xy_amplitude,
        fixed_clearance_delta=clearance_delta,
        fixed_friction_scale=friction_scale,
        fixed_peg_tilt_x=peg_tilt_x,
        fixed_peg_tilt_y=peg_tilt_y,
    )
    return perturbation, profile_name, profile_metadata, preset.to_metadata()


def _random_collection_seed() -> int:
    return 1 + secrets.randbelow(DEFAULT_RANDOM_COLLECTION_SEED_MAX)


def _sample_perturbation(
    *,
    scene: str,
    seed: int,
    contact_profile: str,
    controlled_contact_profile: ControlledContactProfile | None,
    hole_xy_radius: float,
    fixed_teleop_noise_xy_amplitude: float,
    fixed_clearance_delta: float,
    fixed_friction_scale: float,
    fixed_peg_tilt_x: float,
    fixed_peg_tilt_y: float,
    seeded_hole_yaw_max_deg: float,
    seeded_teleop_noise_cycles_min: float,
    seeded_teleop_noise_cycles_max: float,
) -> tuple[RolloutPerturbation, str, dict[str, Any], dict[str, Any]]:
    if contact_profile == DEFAULT_CONTACT_PROFILE:
        return _sample_sigma7_seeded_pose_perturbation(
            scene=scene,
            seed=seed,
            hole_xy_radius=hole_xy_radius,
            hole_yaw_max_deg=seeded_hole_yaw_max_deg,
            teleop_noise_xy_amplitude=fixed_teleop_noise_xy_amplitude,
            teleop_noise_cycles_min=seeded_teleop_noise_cycles_min,
            teleop_noise_cycles_max=seeded_teleop_noise_cycles_max,
            clearance_delta=fixed_clearance_delta,
            friction_scale=fixed_friction_scale,
            peg_tilt_x=fixed_peg_tilt_x,
            peg_tilt_y=fixed_peg_tilt_y,
        )
    preset = get_robustness_preset(scene)
    if controlled_contact_profile is not None:
        perturbation = sample_controlled_contact_perturbations(
            episodes=1,
            seed=seed,
            profile=controlled_contact_profile,
        )[0]
        profile_name = contact_profile
        profile_metadata = controlled_contact_profile.to_metadata()
    else:
        perturbation = sample_robustness_perturbations(
            scene=scene,
            episodes=1,
            seed=seed,
            preset=preset,
        )[0]
        profile_name, profile_metadata = _legacy_randomized_profile_metadata(scene)
    return perturbation, profile_name, profile_metadata, preset.to_metadata()


def _teleop_config_from_args(args: argparse.Namespace) -> PoseTeleopConfig:
    return PoseTeleopConfig(
        workspace_min_delta=np.asarray(args.workspace_min_delta, dtype=float),
        workspace_max_delta=np.asarray(args.workspace_max_delta, dtype=float),
        position_scale=float(args.position_scale),
        max_orientation_angle_rad=float(args.max_orientation_angle_rad),
        deadband_m=float(args.deadband_m),
    )


def _trajectory_parameters(
    *,
    duration_seconds: float,
    target_positions: list[np.ndarray],
    orientation_error_norms: list[float],
    zero_event_count: int,
    user_quit_event_count: int,
    position_scale: float,
    max_orientation_angle_rad: float,
) -> np.ndarray:
    deltas = np.diff(np.stack(target_positions, axis=0), axis=0) if len(target_positions) >= 2 else np.zeros((0, 3), dtype=float)
    delta_norms = np.linalg.norm(deltas, axis=1) if deltas.size else np.zeros(0, dtype=float)
    orient = np.asarray(orientation_error_norms, dtype=float)
    return np.array(
        [
            float(duration_seconds),
            float(np.mean(delta_norms)) if delta_norms.size else 0.0,
            float(np.max(delta_norms)) if delta_norms.size else 0.0,
            float(np.mean(orient)) if orient.size else 0.0,
            float(np.max(orient)) if orient.size else 0.0,
            float(zero_event_count),
            float(user_quit_event_count),
            float(position_scale),
            float(max_orientation_angle_rad),
        ],
        dtype=float,
    )


def _default_controller_yaml() -> Path:
    return DEFAULT_TRACK_A_CONTROLLERS_YAML


def _maybe_reexec_with_mjpython(args: argparse.Namespace, argv: list[str]) -> int | None:
    if sys.platform != "darwin" or bool(args.disable_third_person):
        return None
    if os.environ.get("SIGMA7_COLLECT_UNDER_MJPYTHON") == "1":
        return None
    mjpython = Path(args.mjpython).expanduser()
    if not mjpython.exists():
        print(
            f"warning: mjpython not found at {mjpython}; continuing with {sys.executable}. "
            "MuJoCo viewer may not open on macOS.",
            flush=True,
        )
        return None
    try:
        if Path(sys.executable).resolve() == mjpython.resolve():
            return None
    except FileNotFoundError:
        pass
    env = os.environ.copy()
    env["SIGMA7_COLLECT_UNDER_MJPYTHON"] = "1"
    cmd = [str(mjpython), str(Path(__file__).resolve()), *argv]
    print("relaunching under mjpython for MuJoCo viewer support", flush=True)
    return subprocess.run(cmd, env=env, check=False).returncode


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="Collect one interactive Sigma7 pose-teleop episode with a MuJoCo viewer and save it under scene/participant/mode/episodes/."
    )
    parser.add_argument("--participant", type=str, required=True)
    parser.add_argument("--mode", choices=("practice", "collection"), required=True)
    parser.add_argument("--scene", type=str, default="circle")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_PIPELINE_ROOT)
    parser.add_argument("--collection-seed", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Legacy alias for --collection-seed.")
    parser.add_argument("--sample-stride", type=int, default=50)
    parser.add_argument("--camera-name", type=str, default="eye_in_hand_rgb")
    parser.add_argument("--image-width", type=int, default=128)
    parser.add_argument("--image-height", type=int, default=128)
    parser.add_argument("--image-stride", type=int, default=50)
    parser.add_argument("--renderer-mode", choices=("native", "legacy_debug_only"), default="native")
    parser.add_argument("--no-save-rgb", action="store_true", help="Disable RGB capture. DINOv3 training will not be possible.")
    parser.add_argument("--show-eye-view", action="store_true")
    parser.add_argument("--disable-third-person", action="store_true")
    parser.add_argument("--eye-render-stride", type=int, default=4)
    parser.add_argument("--third-person-sync-stride", type=int, default=2)
    parser.add_argument("--label-neighbors", type=int, default=32)
    parser.add_argument("--knn-block-size", type=int, default=1024)
    parser.add_argument("--label-k-min", type=float, default=300.0)
    parser.add_argument("--label-k-max", type=float, default=600.0)
    parser.add_argument("--baseline-k", type=float, default=600.0)
    parser.add_argument("--diagnostic-k-min", type=float, default=300.0)
    parser.add_argument("--diagnostic-k-max", type=float, default=900.0)
    parser.add_argument("--l21-coupling-percentile", type=float, default=95.0)
    parser.add_argument("--controller-id", type=str, default="track_a_c600")
    parser.add_argument("--controllers-yaml", type=Path, default=_default_controller_yaml())
    parser.add_argument("--contact-profile", type=str, default=DEFAULT_CONTACT_PROFILE)
    parser.add_argument("--hole-xy-radius", type=float, default=0.02)
    parser.add_argument("--fixed-teleop-noise-xy-amplitude", type=float, default=0.0010)
    parser.add_argument("--fixed-teleop-noise-cycles", type=float, default=1.0)
    parser.add_argument("--fixed-teleop-noise-phase-x", type=float, default=0.0)
    parser.add_argument("--fixed-teleop-noise-phase-y", type=float, default=1.5707963267948966)
    parser.add_argument("--fixed-clearance-delta", type=float, default=-0.0003)
    parser.add_argument("--fixed-friction-scale", type=float, default=1.15)
    parser.add_argument("--fixed-peg-tilt-x", type=float, default=0.0087)
    parser.add_argument("--fixed-peg-tilt-y", type=float, default=-0.0087)
    parser.add_argument("--fixed-hole-yaw-offset", type=float, default=0.0)
    parser.add_argument("--seeded-hole-yaw-max-deg", type=float, default=DEFAULT_SEEDED_HOLE_YAW_MAX_DEG)
    parser.add_argument("--seeded-teleop-noise-cycles-min", type=float, default=DEFAULT_SEEDED_TELEOP_NOISE_CYCLES_MIN)
    parser.add_argument("--seeded-teleop-noise-cycles-max", type=float, default=DEFAULT_SEEDED_TELEOP_NOISE_CYCLES_MAX)
    parser.add_argument("--contact-condition-name", type=str, default=None)
    parser.add_argument("--packet-host", type=str, default="0.0.0.0")
    parser.add_argument("--packet-port", type=int, default=5005)
    parser.add_argument("--operator", type=str, default=getpass.getuser())
    parser.add_argument("--position-scale", type=float, default=3.5)
    parser.add_argument("--max-orientation-angle-rad", type=float, default=np.pi)
    parser.add_argument("--deadband-m", type=float, default=0.0005)
    parser.add_argument("--workspace-min-delta", type=float, nargs=3, default=(-0.08, -0.08, -0.26))
    parser.add_argument("--workspace-max-delta", type=float, nargs=3, default=(0.08, 0.08, 0.05))
    parser.add_argument("--max-steps", type=int, default=45000)
    parser.add_argument(
        "--success-hold-steps",
        type=int,
        default=50,
        help="Consecutive depth-qualified steps required before marking success.",
    )
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--hold-after-finish-seconds", type=float, default=1.0)
    parser.add_argument("--auto-close-after-finish", action="store_true")
    parser.add_argument("--no-realtime", action="store_true")
    parser.add_argument("--viewer-python", type=Path, default=default_viewer_python())
    parser.add_argument("--mjpython", type=Path, default=DEFAULT_MJ_PYTHON)
    args = parser.parse_args(argv)

    reexec_code = _maybe_reexec_with_mjpython(args, argv)
    if reexec_code is not None:
        return int(reexec_code)

    participant = require_safe_segment(args.participant, name="participant")
    mode = require_safe_segment(args.mode, name="mode")
    scene_alias = require_safe_segment(args.scene, name="scene")
    scene_name = _scene_alias(scene_alias)
    explicit_seed = args.collection_seed if args.collection_seed is not None else args.seed
    collection_seed = (
        int(explicit_seed)
        if explicit_seed is not None
        else _random_collection_seed()
    )
    save_rgb = not bool(args.no_save_rgb)
    if save_rgb and args.renderer_mode != "native":
        raise ValueError("RGB training data requires --renderer-mode native.")
    if args.eye_render_stride <= 0 or args.third_person_sync_stride <= 0:
        raise ValueError("Viewer sync strides must be positive.")
    if args.sample_stride <= 0 or args.image_stride <= 0:
        raise ValueError("Sample and image stride must be positive.")
    if save_rgb and args.sample_stride % args.image_stride != 0:
        raise ValueError("image_stride must evenly divide sample_stride.")

    controlled_contact_profile = None
    if args.contact_profile not in {"legacy_randomized", DEFAULT_CONTACT_PROFILE}:
        controlled_contact_profile = make_controlled_contact_profile(
            profile_name=args.contact_profile,
            contact_condition_name=args.contact_condition_name,
            hole_xy_radius=args.hole_xy_radius,
            teleop_noise_xy_amplitude=args.fixed_teleop_noise_xy_amplitude,
            teleop_noise_cycles=args.fixed_teleop_noise_cycles,
            teleop_noise_phase_x=args.fixed_teleop_noise_phase_x,
            teleop_noise_phase_y=args.fixed_teleop_noise_phase_y,
            clearance_delta=args.fixed_clearance_delta,
            friction_scale=args.fixed_friction_scale,
            peg_tilt_x=args.fixed_peg_tilt_x,
            peg_tilt_y=args.fixed_peg_tilt_y,
            hole_yaw_offset=args.fixed_hole_yaw_offset,
        )

    perturbation, profile_name, profile_metadata, preset_metadata = _sample_perturbation(
        scene=scene_name,
        seed=int(collection_seed),
        contact_profile=args.contact_profile,
        controlled_contact_profile=controlled_contact_profile,
        hole_xy_radius=float(args.hole_xy_radius),
        fixed_teleop_noise_xy_amplitude=float(args.fixed_teleop_noise_xy_amplitude),
        fixed_clearance_delta=float(args.fixed_clearance_delta),
        fixed_friction_scale=float(args.fixed_friction_scale),
        fixed_peg_tilt_x=float(args.fixed_peg_tilt_x),
        fixed_peg_tilt_y=float(args.fixed_peg_tilt_y),
        seeded_hole_yaw_max_deg=float(args.seeded_hole_yaw_max_deg),
        seeded_teleop_noise_cycles_min=float(args.seeded_teleop_noise_cycles_min),
        seeded_teleop_noise_cycles_max=float(args.seeded_teleop_noise_cycles_max),
    )

    episodes_root = participant_episodes_root(args.output_root, scene_alias, participant, mode)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{scene_name}_sigma7_pose_episode_seed{collection_seed}_{timestamp}"
    output_dir = _unique_output_dir(episodes_root, run_name)
    raw_path = output_dir / "raw_collection.npz"
    eligible_path = output_dir / "eligible_residual_bc.npz"
    summary_path = output_dir / "collection_summary.json"
    collection_metadata_path = output_dir / "collection_metadata.json"
    episode_csv_path = output_dir / "episodes.csv"
    episode_specs_path = output_dir / "episode_specs.jsonl"
    frozen_episode_specs_path = output_dir / "frozen_paired_episode_specs.jsonl"
    frozen_split_path = output_dir / "frozen_train_val_split.json"
    teleop_trace_path = output_dir / "sigma7_teleop_trace.npz"
    trajectory_family_summary_path = output_dir / "trajectory_family_summary.json"

    scene_spec = get_scene_spec(scene_name)
    rollout_config = RolloutConfig(config_path=scene_spec.config_path, max_steps=int(args.max_steps))
    controller_entry, controller_profile, gains = load_track_a_controller_runtime(
        args.controller_id,
        controllers_yaml=args.controllers_yaml,
    )
    scene_path, scene_config = scene_for_rollout(scene_spec.config_path, perturbation)
    receiver = Sigma7PoseReceiver(args.packet_host, int(args.packet_port))
    rgb_renderer: MujocoRgbRenderer | None = None
    eye_window: EyeInHandWindow | None = None
    passive_viewer = None

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
    trace_rows: dict[str, list[Any]] = {
        "time": [],
        "step": [],
        "phase_id": [],
        "target_position": [],
        "target_rotation": [],
        "actual_position": [],
        "actual_rotation": [],
        "position_error_norm": [],
        "orientation_error_norm": [],
        "sigma_position": [],
        "sigma_rotation": [],
        "sigma_linear_velocity": [],
        "sigma_angular_velocity_rad": [],
        "sigma_gripper_angle_rad": [],
        "sigma_gripper_linear_velocity": [],
        "contact_force_world": [],
        "contact_force_norm": [],
        "contact_active": [],
        "contact_normal_force": [],
        "insertion_depth": [],
        "lateral_error": [],
        "button_mask": [],
        "packet_sequence": [],
        "packet_timestamp": [],
    }
    rgb_rows: list[np.ndarray] = []

    try:
        model = load_model(scene_path)
        data = mujoco.MjData(model)
        reset_from_config(model, data, scene_config)

        validate_canonical_eye_in_hand_camera_config(scene_config, camera_name=args.camera_name)
        camera_local_position, camera_rotation, camera_fovy = eye_in_hand_camera_pose_from_config(
            scene_config,
            camera_name=args.camera_name,
        )
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera_name)
        if camera_id < 0:
            raise ValueError(f"Unknown camera {args.camera_name!r}.")
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
        peg_tip_site_id = model.site(rollout_config.site_name).id
        anchor_position = np.array(data.site_xpos[peg_tip_site_id], dtype=float)
        anchor_rotation = site_rotation(data, peg_tip_site_id)
        teleop_config = _teleop_config_from_args(args)
        teleop_state = PoseTeleopState(
            anchor_position=anchor_position.copy(),
            anchor_rotation=anchor_rotation.copy(),
        )

        if save_rgb:
            rgb_renderer = MujocoRgbRenderer(
                model,
                camera_name=args.camera_name,
                width=int(args.image_width),
                height=int(args.image_height),
                renderer_mode=args.renderer_mode,
            )
            if bool(rgb_renderer.fallback_used):
                raise RuntimeError("Fallback-rendered RGB capture is not accepted for training-compatible collection.")
        if args.show_eye_view and rgb_renderer is not None:
            eye_window = EyeInHandWindow(
                python_path=args.viewer_python,
                script_path=ROOT / "scripts" / "live_image_window.py",
                title=f"sigma7_collect::{scene_name}",
            )
        if not args.disable_third_person:
            passive_viewer = mujoco.viewer.launch_passive(model, data)
            third_person_camera = _build_third_person_camera(hole_center, mujoco)
            passive_viewer.cam.type = third_person_camera.type
            passive_viewer.cam.lookat[:] = third_person_camera.lookat
            passive_viewer.cam.distance = third_person_camera.distance
            passive_viewer.cam.azimuth = third_person_camera.azimuth
            passive_viewer.cam.elevation = third_person_camera.elevation
            passive_viewer.cam.orthographic = third_person_camera.orthographic
            passive_viewer.cam.fixedcamid = third_person_camera.fixedcamid
            passive_viewer.cam.trackbodyid = third_person_camera.trackbodyid
            passive_viewer.sync()

        print(
            json.dumps(
                {
                    "scene": scene_name,
                    "scene_alias": scene_alias,
                    "participant": participant,
                    "collection_mode": mode,
                    "controller_id": args.controller_id,
                    "camera_name": args.camera_name,
                    "output_dir": str(output_dir),
                    "seed": int(collection_seed),
                    "perturbation": perturbation.to_dict(),
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        print("waiting for first Sigma7 pose packet...", flush=True)
        latest_packet = None
        while latest_packet is None:
            latest_packet = receiver.recv_latest()
            if latest_packet is None:
                time.sleep(0.01)
        print("first Sigma7 pose packet received, collection running", flush=True)
        print("press q in this terminal to mark failure, save data, and exit", flush=True)

        success_streak = 0
        max_contact_force = 0.0
        sampled_max_force = 0.0
        contact_count = 0
        contact_onset = -1
        step_count = 0
        zero_event_count = 0
        user_quit_event_count = 0
        arrays_finite = True
        completed = True
        termination_reason = "max_steps_reached"
        final_depth_threshold = float(0.95 * rollout_config.insert_depth)
        final_lateral_threshold = float(geometry.radial_clearance)
        next_tick = time.perf_counter()

        all_target_positions: list[np.ndarray] = []
        all_target_rotations: list[np.ndarray] = []
        orientation_error_norms: list[float] = []

        with TerminalKeyMonitor() as key_monitor:
            for step in range(int(args.max_steps)):
                key = key_monitor.read_key()
                if key is not None and key.lower() == "q":
                    user_quit_event_count += 1
                    completed = False
                    termination_reason = "user_marked_failure"
                    break
                if passive_viewer is not None and not _viewer_running(passive_viewer):
                    passive_viewer = None
                if not args.no_realtime:
                    now = time.perf_counter()
                    if now < next_tick:
                        time.sleep(next_tick - now)
                    next_tick += simulation_dt_seconds

                packet = receiver.recv_latest() or latest_packet
                if packet is None:
                    time.sleep(0.001)
                    continue
                latest_packet = packet

                target_position, target_rotation = map_sigma7_pose(
                    packet,
                    state=teleop_state,
                    config=teleop_config,
                )
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
                torque, _saturated = clip_torque(model, arm_ids, command.torque)
                contact_query = ContactQuery(model=model, data=data, task_ids=task_ids)
                contact = extract_contact_state(contact_query)
                force_world = np.asarray(extract_net_peg_hole_contact_force_world(contact_query), dtype=float)
                force_norm = float(np.linalg.norm(force_world))
                max_contact_force = max(max_contact_force, float(contact.normal_force))

                actual_position = np.array(data.site_xpos[peg_tip_site_id], dtype=float)
                actual_rotation = site_rotation(data, peg_tip_site_id)
                depth = float(insertion_depth(data, task_ids))
                lateral = float(lateral_error(data, task_ids))
                success_now = bool(depth >= final_depth_threshold)
                success_streak = success_streak + 1 if success_now else 0
                if contact.in_contact:
                    contact_count += 1
                    if contact_onset < 0:
                        contact_onset = step

                finite_step = bool(
                    np.all(np.isfinite(data.qpos))
                    and np.all(np.isfinite(data.qvel))
                    and np.all(np.isfinite(torque))
                    and np.isfinite(contact.normal_force)
                    and np.all(np.isfinite(target_position))
                    and np.all(np.isfinite(target_rotation))
                )
                if not finite_step:
                    arrays_finite = False
                    completed = False
                    termination_reason = "nonfinite_rollout"
                    break

                all_target_positions.append(np.asarray(target_position, dtype=float).copy())
                all_target_rotations.append(np.asarray(target_rotation, dtype=float).copy())
                orientation_error_norms.append(float(np.linalg.norm(command.orientation_error)))

                trace_rows["time"].append(float(data.time))
                trace_rows["step"].append(step)
                trace_rows["phase_id"].append(2)
                trace_rows["target_position"].append(np.asarray(target_position, dtype=float).copy())
                trace_rows["target_rotation"].append(_rotation_matrix_to_row_major(target_rotation))
                trace_rows["actual_position"].append(actual_position.copy())
                trace_rows["actual_rotation"].append(_rotation_matrix_to_row_major(actual_rotation))
                trace_rows["position_error_norm"].append(float(np.linalg.norm(command.position_error)))
                trace_rows["orientation_error_norm"].append(float(np.linalg.norm(command.orientation_error)))
                trace_rows["sigma_position"].append(np.asarray(packet.position, dtype=float).copy())
                trace_rows["sigma_rotation"].append(_rotation_matrix_to_row_major(packet.orientation_frame))
                trace_rows["sigma_linear_velocity"].append(np.asarray(packet.linear_velocity, dtype=float).copy())
                trace_rows["sigma_angular_velocity_rad"].append(np.asarray(packet.angular_velocity_rad, dtype=float).copy())
                trace_rows["sigma_gripper_angle_rad"].append(float(packet.gripper_angle_rad))
                trace_rows["sigma_gripper_linear_velocity"].append(float(packet.gripper_linear_velocity))
                trace_rows["contact_force_world"].append(force_world.copy())
                trace_rows["contact_force_norm"].append(force_norm)
                trace_rows["contact_active"].append(bool(contact.in_contact))
                trace_rows["contact_normal_force"].append(float(contact.normal_force))
                trace_rows["insertion_depth"].append(depth)
                trace_rows["lateral_error"].append(lateral)
                trace_rows["button_mask"].append(int(packet.buttons))
                trace_rows["packet_sequence"].append(-1 if packet.sequence is None else int(packet.sequence))
                trace_rows["packet_timestamp"].append(float("nan") if packet.packet_timestamp is None else float(packet.packet_timestamp))

                if step % int(args.sample_stride) == 0:
                    obs, u_ref, _ = collect_step(
                        model,
                        data,
                        arm_ids=arm_ids,
                        task_ids=task_ids,
                        target_position=target_position,
                        target_rotation=target_rotation,
                        phase_id=2,
                    )
                    task_state = peg_hole_task_state(
                        data,
                        task_ids,
                        hole_clearance_delta=float(perturbation.clearance_delta),
                    )
                    sampled_max_force = max(sampled_max_force, float(contact.normal_force))
                    sample_rows["state"].append(_state_row(obs, data, arm_ids))
                    sample_rows["action"].append(_action_row(u_ref, torque))
                    sample_rows["contact_state"].append(contact_state_vector(contact))
                    sample_rows["task_state"].append(task_state)
                    sample_rows["contact_force_world"].append(force_world.copy())
                    sample_rows["normal_force"].append(float(contact.normal_force))
                    sample_rows["episode_id"].append(0)
                    sample_rows["sample_step"].append(step)
                    sample_rows["timestamp"].append(float(data.time))
                    sample_rows["randomization"].append(_randomization_vector(perturbation))
                    sample_rows["planned_target_position"].append(np.asarray(target_position, dtype=float).copy())
                    sample_rows["planned_target_rotation"].append(_rotation_matrix_to_row_major(target_rotation))
                    sample_rows["trajectory_family_id"].append(TRAJECTORY_FAMILY_SIGMA7_LIVE_POSE_ID)
                    sample_rows["phase_id"].append(2)
                    if rgb_renderer is not None:
                        if step % int(args.image_stride) != 0:
                            raise RuntimeError("image_stride must align with sampled steps.")
                        frame = rgb_renderer.render(data)
                        rgb_rows.append(frame.copy())
                        if eye_window is not None and (step == 0 or step % int(args.eye_render_stride) == 0):
                            eye_window.send(frame)
                elif eye_window is not None and rgb_renderer is not None and (step == 0 or step % int(args.eye_render_stride) == 0):
                    frame = rgb_renderer.render(data)
                    eye_window.send(frame)

                if step % max(int(args.print_every), 1) == 0:
                    print(
                        f"step={step:04d} depth={depth:+.4f} lateral={lateral:.4f} "
                        f"force={force_norm:.3f} target=({target_position[0]:+.3f} {target_position[1]:+.3f} {target_position[2]:+.3f})",
                        flush=True,
                    )

                set_arm_torque_ctrl(model, data, arm_ids, torque)
                mujoco.mj_step(model, data)
                if passive_viewer is not None and (step == 0 or step % int(args.third_person_sync_stride) == 0):
                    passive_viewer.sync()
                step_count = step + 1
                if success_streak >= int(args.success_hold_steps):
                    termination_reason = "success"
                    break

        if not sample_rows["state"]:
            raise RuntimeError("No samples were collected. Did the sender produce packets?")

        duration_seconds = float(trace_rows["time"][-1]) if trace_rows["time"] else 0.0
        trajectory_parameters = _trajectory_parameters(
            duration_seconds=duration_seconds,
            target_positions=all_target_positions,
            orientation_error_norms=orientation_error_norms,
            zero_event_count=zero_event_count,
            user_quit_event_count=user_quit_event_count,
            position_scale=float(args.position_scale),
            max_orientation_angle_rad=float(args.max_orientation_angle_rad),
        )
        sample_rows["trajectory_parameters"] = [trajectory_parameters.copy() for _ in sample_rows["state"]]

        final_depth = float(trace_rows["insertion_depth"][-1]) if trace_rows["insertion_depth"] else 0.0
        final_lateral_error = float(trace_rows["lateral_error"][-1]) if trace_rows["lateral_error"] else 0.0
        insertion_success = bool(
            termination_reason == "success"
            and final_depth >= final_depth_threshold
        )
        label_eligible, spike_suspicious, exclusion_reason = classify_episode_admission(
            episode_complete=completed,
            arrays_finite=arrays_finite,
            full_step_max_force=float(max_contact_force),
            sampled_max_force=float(sampled_max_force),
        )

        randomization = _randomization_vector(perturbation)
        target_offsets = np.stack(all_target_positions, axis=0) - np.asarray(hole_center, dtype=float)
        target_rotations = np.stack(all_target_rotations, axis=0)
        phase_ids = np.full(target_offsets.shape[0], 2, dtype=np.int32)
        nominal_scene_config = load_scene_config(scene_spec.config_path)
        nominal_hole_position = np.asarray(nominal_scene_config["hole"]["pos"], dtype=float)
        nominal_hole_xy = np.asarray(nominal_hole_position[:2], dtype=float)
        fixed_contact_condition = {
            "teleop_noise_xy_amplitude": float(perturbation.teleop_noise_xy_amplitude),
            "teleop_noise_cycles": float(perturbation.teleop_noise_cycles),
            "teleop_noise_phase_x": float(perturbation.teleop_noise_phase_x),
            "teleop_noise_phase_y": float(perturbation.teleop_noise_phase_y),
            "clearance_delta": float(perturbation.clearance_delta),
            "friction_scale": float(perturbation.friction_scale),
            "peg_tilt_x": float(perturbation.peg_tilt_x),
            "peg_tilt_y": float(perturbation.peg_tilt_y),
            "fixed_hole_yaw_offset": float(perturbation.hole_yaw_offset),
            "hole_xy_radius": float(args.hole_xy_radius),
        }
        episode_spec = EpisodeSpec.create(
            episode_id=0,
            seed=int(collection_seed),
            scene=scene_name,
            setting_id=str(preset_metadata["setting_id"]),
            profile_name=str(profile_name),
            contact_condition_name=controlled_contact_profile.contact_condition_name if controlled_contact_profile is not None else None,
            nominal_hole_position=nominal_hole_position,
            nominal_hole_xy=nominal_hole_xy,
            hole_xy_offset=np.asarray(perturbation.hole_xy_offset, dtype=float),
            hole_yaw_offset=float(perturbation.hole_yaw_offset),
            hole_xy_radius=float(args.hole_xy_radius),
            hole_xy_offset_semantics=str(profile_metadata["hole_xy_offset_semantics"]),
            hole_xy_offset_distribution=str(profile_metadata["hole_xy_offset_distribution"]),
            trajectory_follows_randomized_hole=bool(profile_metadata["trajectory_follows_randomized_hole"]),
            contact_generation_parameters_fixed=bool(profile_metadata["contact_generation_parameters_fixed"]),
            fixed_contact_condition=fixed_contact_condition,
            trajectory_source=EPISODE_TRAJECTORY_SOURCE_EPISODE_SPEC_REPLAY,
            trajectory_family=TRAJECTORY_FAMILY_SIGMA7_LIVE_POSE,
            trajectory_family_id=TRAJECTORY_FAMILY_SIGMA7_LIVE_POSE_ID,
            trajectory_parameters=trajectory_parameters,
            teleop_mode=TELEOP_MODE_POSITION_ORIENTATION,
            target_rotations=target_rotations,
            target_offsets=target_offsets,
            phase_ids=phase_ids,
            total_steps=int(target_offsets.shape[0] - 1),
            sample_stride=int(args.sample_stride),
            image_stride=int(args.image_stride) if save_rgb else None,
        )

        raw_arrays: dict[str, np.ndarray] = {
            "state": np.vstack(sample_rows["state"]).astype(np.float32),
            "action": np.vstack(sample_rows["action"]).astype(np.float32),
            "contact_state": np.vstack(sample_rows["contact_state"]).astype(np.float32),
            "task_state": np.vstack(sample_rows["task_state"]).astype(np.float32),
            "contact_force_world": np.vstack(sample_rows["contact_force_world"]).astype(np.float32),
            "normal_force": np.asarray(sample_rows["normal_force"], dtype=np.float32),
            "episode_id": np.asarray(sample_rows["episode_id"], dtype=np.int32),
            "sample_step": np.asarray(sample_rows["sample_step"], dtype=np.int32),
            "timestamp": np.asarray(sample_rows["timestamp"], dtype=np.float32),
            "randomization": np.vstack(sample_rows["randomization"]).astype(np.float32),
            "planned_target_position": np.vstack(sample_rows["planned_target_position"]).astype(np.float32),
            "planned_target_rotation": np.vstack(sample_rows["planned_target_rotation"]).astype(np.float32),
            "trajectory_family_id": np.asarray(sample_rows["trajectory_family_id"], dtype=np.int8),
            "trajectory_parameters": np.vstack(sample_rows["trajectory_parameters"]).astype(np.float32),
            "phase_id": np.asarray(sample_rows["phase_id"], dtype=np.int8),
            "episode_summary_id": np.asarray([0], dtype=np.int32),
            "episode_success": np.asarray([insertion_success], dtype=bool),
            "episode_final_depth": np.asarray([final_depth], dtype=np.float32),
            "episode_final_lateral_error": np.asarray([final_lateral_error], dtype=np.float32),
            "episode_max_normal_force": np.asarray([float(max_contact_force)], dtype=np.float32),
            "episode_sampled_max_normal_force": np.asarray([float(sampled_max_force)], dtype=np.float32),
            "episode_force_capture_ratio": np.asarray(
                [float(sampled_max_force / max_contact_force) if max_contact_force > 0.0 else 1.0],
                dtype=np.float32,
            ),
            "episode_contact_count": np.asarray([contact_count], dtype=np.int32),
            "episode_contact_onset_step": np.asarray([contact_onset], dtype=np.int32),
            "episode_perturbation": np.asarray([randomization], dtype=np.float32),
            "episode_command_xy_offset": np.asarray([target_offsets[-1, :2]], dtype=np.float32),
            "episode_trajectory_family_id": np.asarray([TRAJECTORY_FAMILY_SIGMA7_LIVE_POSE_ID], dtype=np.int8),
            "episode_trajectory_parameters": np.asarray([trajectory_parameters], dtype=np.float32),
            "episode_complete": np.asarray([completed], dtype=bool),
            "episode_solver_spike_suspicious": np.asarray([spike_suspicious], dtype=bool),
            "episode_label_eligible": np.asarray([label_eligible], dtype=bool),
        }
        if rgb_rows:
            raw_arrays["rgb_images"] = np.stack(rgb_rows, axis=0).astype(np.uint8)

        canonical_camera_pose = canonical_eye_in_hand_camera_pose(args.camera_name)
        training_data_valid, training_data_valid_reason = infer_training_data_valid(
            {
                "renderer_mode": None if rgb_renderer is None else rgb_renderer.mode,
                "fallback_used": False if rgb_renderer is None else bool(rgb_renderer.fallback_used),
                "rgb_enabled": bool(save_rgb),
            },
            rgb_images_present=bool(rgb_rows),
        )
        raw_metadata = {
            "schema_version": RAW_SCHEMA_VERSION,
            "scene": scene_name,
            "scene_alias": scene_alias,
            "setting_id": preset_metadata["setting_id"],
            "profile_name": profile_name,
            "teleop_mode": TELEOP_MODE_POSITION_ORIENTATION,
            "teleop_source": "sigma7_live_pose_teleop",
            "operator": str(args.operator),
            "packet_host": str(args.packet_host),
            "packet_port": int(args.packet_port),
            "eye_in_hand_camera_pose_version": canonical_camera_pose["pose_version"],
            "eye_in_hand_camera_canonical": bool(canonical_camera_pose["canonical"]),
            "eye_in_hand_camera_name": canonical_camera_pose["camera_name"],
            "eye_in_hand_camera_attachment_parent": canonical_camera_pose["attachment_parent"],
            "eye_in_hand_camera_mount_type": canonical_camera_pose["mount_type"],
            "eye_in_hand_camera_pose": canonical_camera_pose,
            "collection_seed": int(collection_seed),
            **profile_metadata,
            "robustness_preset": preset_metadata,
            "controller_id": controller_entry.controller_id,
            "scenario_id": scene_alias,
            "collection_controller_id": controller_entry.controller_id,
            "collection_stiffness_matrix": controller_entry.position_stiffness_matrix.tolist(),
            "dataset_path": str(raw_path),
            "controllers_yaml": str(args.controllers_yaml),
            "base_profile": controller_profile,
            "controller_profile": controller_profile,
            "gain_config": None,
            "trajectory_plan": "sigma7_live_pose_anchor_delta_v1",
            "trajectory_source": EPISODE_TRAJECTORY_SOURCE_EPISODE_SPEC_REPLAY,
            "trajectory_families": [TRAJECTORY_FAMILY_SIGMA7_LIVE_POSE],
            "trajectory_parameter_fields": list(TRAJECTORY_PARAMETER_FIELDS),
            "num_episodes": 1,
            "requested_episodes": 1,
            "num_samples": int(raw_arrays["task_state"].shape[0]),
            "sample_stride": int(args.sample_stride),
            "rgb_enabled": bool(save_rgb),
            "rgb_camera_name": args.camera_name if save_rgb else None,
            "rgb_image_width": int(args.image_width) if save_rgb else None,
            "rgb_image_height": int(args.image_height) if save_rgb else None,
            "rgb_image_stride": int(args.image_stride) if save_rgb else None,
            "rgb_max_episodes": 1 if save_rgb else None,
            "renderer_mode": None if rgb_renderer is None else rgb_renderer.mode,
            "fallback_used": False if rgb_renderer is None else bool(rgb_renderer.fallback_used),
            "episode_specs_path": str(episode_specs_path),
            "frozen_paired_episode_specs_path": str(frozen_episode_specs_path),
            "frozen_train_val_split_path": str(frozen_split_path),
            "teleop_trace_path": str(teleop_trace_path),
            "episode_spec_schema_version": EPISODE_SPEC_SCHEMA_VERSION,
            "sample_rate_hz": 1.0 / (simulation_dt_seconds * float(args.sample_stride)),
            "seed": int(collection_seed),
            "rng_streams": {
                "root_seed": int(collection_seed),
                "perturbation_seed": int(collection_seed),
                "split_seed": int(collection_seed),
            },
            "randomization_fields": list(RANDOMIZATION_FIELDS),
            "legacy_field_mapping": profile_metadata.get("legacy_field_mapping"),
            "labels_built": False,
            "training_data_valid": bool(training_data_valid),
            "training_data_valid_reason": training_data_valid_reason,
            "participant": participant,
            "collection_mode": mode,
        }
        np.savez_compressed(raw_path, **raw_arrays, metadata=json.dumps(_json_ready(raw_metadata), sort_keys=True))

        trace_arrays = {
            key: np.asarray(value, dtype=bool if key == "contact_active" else np.float32 if key not in {"step", "phase_id", "button_mask", "packet_sequence"} else np.int32)
            for key, value in trace_rows.items()
        }
        np.savez_compressed(teleop_trace_path, **trace_arrays, metadata=json.dumps(_json_ready(raw_metadata), sort_keys=True))

        write_episode_specs_jsonl(episode_specs_path, [episode_spec])
        episode_csv_row = {
            "episode_id": 0,
            "episode_spec_id": episode_spec.episode_spec_id,
            "episode_spec_schema_version": EPISODE_SPEC_SCHEMA_VERSION,
            "trajectory_source": EPISODE_TRAJECTORY_SOURCE_EPISODE_SPEC_REPLAY,
            "trajectory_family": TRAJECTORY_FAMILY_SIGMA7_LIVE_POSE,
            "trajectory_family_id": TRAJECTORY_FAMILY_SIGMA7_LIVE_POSE_ID,
            "teleop_mode": TELEOP_MODE_POSITION_ORIENTATION,
            "operator": str(args.operator),
            "input_device": "sigma7_pose_udp_sender",
            "participant": participant,
            "collection_mode": mode,
            "collection_seed": int(collection_seed),
            "profile_name": profile_name,
            "sample_start": 0,
            "sample_stop": int(raw_arrays["state"].shape[0]),
            "sample_count": int(raw_arrays["state"].shape[0]),
            "total_steps": int(target_offsets.shape[0] - 1),
            "nominal_hole_xy_x": float(nominal_hole_xy[0]),
            "nominal_hole_xy_y": float(nominal_hole_xy[1]),
            "actual_hole_xy_x": float(episode_spec.actual_hole_xy[0]),
            "actual_hole_xy_y": float(episode_spec.actual_hole_xy[1]),
            "trajectory_center_xy_x": float(episode_spec.trajectory_center_xy[0]),
            "trajectory_center_xy_y": float(episode_spec.trajectory_center_xy[1]),
            "trajectory_minus_hole_xy_x": float(episode_spec.trajectory_minus_hole_xy[0]),
            "trajectory_minus_hole_xy_y": float(episode_spec.trajectory_minus_hole_xy[1]),
            "success": insertion_success,
            "final_depth": final_depth,
            "final_lateral_error": final_lateral_error,
            "final_peg_tip_x": float(peg_tip_position(data, task_ids)[0]),
            "final_peg_tip_y": float(peg_tip_position(data, task_ids)[1]),
            "final_peg_tip_z": float(peg_tip_position(data, task_ids)[2]),
            "final_hole_center_x": float(hole_center[0]),
            "final_hole_center_y": float(hole_center[1]),
            "final_hole_center_z": float(hole_center[2]),
            "final_target_position_x": float(all_target_positions[-1][0]),
            "final_target_position_y": float(all_target_positions[-1][1]),
            "final_target_position_z": float(all_target_positions[-1][2]),
            "success_depth_threshold": final_depth_threshold,
            "success_lateral_threshold": final_lateral_threshold,
            "failure_reason": "" if insertion_success else termination_reason,
            "full_step_max_force": float(max_contact_force),
            "sampled_max_force": float(sampled_max_force),
            "capture_ratio": float(sampled_max_force / max_contact_force) if max_contact_force > 0.0 else 1.0,
            "contact_count": contact_count,
            "contact_onset_step": contact_onset,
            "episode_complete": completed,
            "arrays_finite": arrays_finite,
            "solver_spike_suspicious": spike_suspicious,
            "label_eligible": label_eligible,
            "exclusion_reason": exclusion_reason,
            "duration_seconds": duration_seconds,
        }
        with episode_csv_path.open("w", encoding="utf-8", newline="") as handle:
            fieldnames = list(episode_csv_row.keys()) + [f"perturbation_{name}" for name in RANDOMIZATION_FIELDS]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            row = dict(episode_csv_row)
            row.update({f"perturbation_{name}": float(value) for name, value in zip(RANDOMIZATION_FIELDS, randomization, strict=True)})
            writer.writerow(row)

        eligible_built = False
        try:
            build_eligible_residual_dataset_from_raw(
                raw_path,
                output_path=eligible_path,
                label_neighbors=args.label_neighbors,
                knn_block_size=args.knn_block_size,
                label_k_min=args.label_k_min,
                label_k_max=args.label_k_max,
                baseline_k=args.baseline_k,
                diagnostic_k_min=args.diagnostic_k_min,
                diagnostic_k_max=args.diagnostic_k_max,
                l21_coupling_percentile=args.l21_coupling_percentile,
            )
            eligible_built = True
        except Exception as exc:
            print(f"eligible build skipped: {exc}", flush=True)

        if eligible_built:
            with np.load(eligible_path, allow_pickle=False) as data_npz:
                train_episode_ids = np.asarray(data_npz["train_episode_ids"], dtype=np.int64)
                val_episode_ids = np.asarray(data_npz["val_episode_ids"], dtype=np.int64)
            shutil_path = frozen_episode_specs_path
            shutil_path.write_text(episode_specs_path.read_text(encoding="utf-8"), encoding="utf-8")
            FrozenTrainValSplit(
                dataset_path=str(eligible_path),
                collection_seed=int(collection_seed),
                split_seed=int(collection_seed),
                train_episode_ids=train_episode_ids,
                val_episode_ids=val_episode_ids,
                metadata={
                    "scene": scene_name,
                    "setting_id": preset_metadata["setting_id"],
                    "profile_name": profile_name,
                    "collection_controller_id": controller_entry.controller_id,
                    "teleop_source": "sigma7_live_pose_teleop",
                },
            ).write(frozen_split_path)

        collection_metadata = {
            "schema_version": "sigma7_pose_episode_collection_v1",
            "participant": participant,
            "collection_mode": mode,
            "scene": scene_name,
            "scene_alias": scene_alias,
            "raw_dataset": str(raw_path),
            "eligible_dataset": str(eligible_path) if eligible_built else None,
            "teleop_trace": str(teleop_trace_path),
            "episode_specs_path": str(episode_specs_path),
            "frozen_paired_episode_specs_path": str(frozen_episode_specs_path) if eligible_built else None,
            "frozen_train_val_split_path": str(frozen_split_path) if eligible_built else None,
            "run_dir": str(output_dir),
        }
        _write_json(collection_metadata_path, collection_metadata)
        _write_json(
            trajectory_family_summary_path,
            {
                "trajectory_family_counts": {TRAJECTORY_FAMILY_SIGMA7_LIVE_POSE: 1},
                "episode_count": 1,
            },
        )
        summary = {
            "status": "passed",
            "raw_dataset": str(raw_path),
            "eligible_dataset": str(eligible_path) if eligible_built else None,
            "teleop_trace": str(teleop_trace_path),
            "collection_seed": int(collection_seed),
            "episodes": 1,
            "eligible_episodes": 1 if label_eligible else 0,
            "quarantined_episodes": 0 if label_eligible else 1,
            "successes": int(insertion_success),
            "failures": int(not insertion_success),
            "raw_samples": int(raw_arrays["task_state"].shape[0]),
            "eligible_samples": int(raw_arrays["task_state"].shape[0]) if label_eligible else 0,
            "eligible_contact_fraction": float(np.mean(raw_arrays["contact_state"][:, 0] > 0.5)),
            "catastrophic_episodes": int(max_contact_force >= 1000.0),
            "solver_spike_suspicious_episodes": int(spike_suspicious),
            "runtime_seconds": duration_seconds,
            "stop_reason": termination_reason,
            "validation": {
                "raw_arrays_finite": arrays_finite,
                "sample_stride": int(args.sample_stride),
                "sample_rate_hz": raw_metadata["sample_rate_hz"],
            },
        }
        _write_json(summary_path, summary)

        context = {
            "participant": participant,
            "collection_mode": mode,
            "scene": scene_alias,
            "collection_seed": int(collection_seed),
            "save_rgb": bool(save_rgb),
            "run_dir": str(output_dir),
            "raw_dataset": str(raw_path),
            "eligible_dataset": str(eligible_path) if eligible_built else None,
            "teleop_trace": str(teleop_trace_path),
        }
        write_json(output_dir / "run_context.json", context)
        write_json(episodes_root / "latest_run.json", context)

        print("")
        print("collection complete")
        print(f"participant: {participant}")
        print(f"mode: {mode}")
        print(f"scene: {scene_alias}")
        print(f"run_dir: {output_dir}")
        print(f"raw_dataset: {raw_path}")
        print(f"eligible_dataset: {eligible_path if eligible_built else '<not built>'}")
        print(f"teleop_trace: {teleop_trace_path}")
        print(f"episode_specs: {episode_specs_path}")
        print(f"frozen_train_val_split: {frozen_split_path if eligible_built else '<not built>'}")
        print(f"success/failure: {int(insertion_success)}/{int(not insertion_success)}")
        print(f"eligible/quarantined: {int(label_eligible)}/{int(not label_eligible)}")
        print(f"stop_reason: {termination_reason}")

        if passive_viewer is not None and not bool(args.auto_close_after_finish):
            print(
                "teleop finished; data saved; third-person viewer remains open until you close the window or press q/Ctrl-C",
                flush=True,
            )
            try:
                with TerminalKeyMonitor() as key_monitor:
                    while _viewer_running(passive_viewer):
                        key = key_monitor.read_key()
                        if key is not None and key.lower() == "q":
                            break
                        passive_viewer.sync()
                        time.sleep(0.05)
            except KeyboardInterrupt:
                pass
        elif args.hold_after_finish_seconds > 0.0:
            try:
                time.sleep(float(args.hold_after_finish_seconds))
            except KeyboardInterrupt:
                pass
        return 0
    finally:
        receiver.close()
        if rgb_renderer is not None:
            rgb_renderer.close()
        if eye_window is not None:
            eye_window.close()
        if passive_viewer is not None:
            try:
                passive_viewer.close()
            except Exception:
                pass
        cleanup_runtime_scene(scene_path)


if __name__ == "__main__":
    raise SystemExit(main())
