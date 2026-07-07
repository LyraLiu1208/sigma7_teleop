from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

import mujoco
import mujoco.viewer
import numpy as np

from _sigma7_residual_pipeline_common import (
    DEFAULT_PIPELINE_ROOT,
    ensure_mujoco_src_on_path,
    require_safe_segment,
    scene_screening_participant_controller_root,
    write_json,
)
from _sigma7_runtime import default_mjpython, default_viewer_python
from collect_sigma7_residual_bc_episode import (
    DEFAULT_CONTACT_PROFILE,
    DEFAULT_SEEDED_HOLE_YAW_MAX_DEG,
    DEFAULT_SEEDED_TELEOP_NOISE_CYCLES_MAX,
    DEFAULT_SEEDED_TELEOP_NOISE_CYCLES_MIN,
    _sample_perturbation,
    _teleop_config_from_args,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MJ_PYTHON = default_mjpython()
ensure_mujoco_src_on_path()

from stiffness_copilot_mujoco.contact.state import (  # noqa: E402
    ContactQuery,
    extract_contact_state,
    extract_net_peg_hole_contact_force_world,
)
from stiffness_copilot_mujoco.controllers.impedance import task_space_impedance_torque  # noqa: E402
from stiffness_copilot_mujoco.controllers.stiffness_command_smoothing import (  # noqa: E402
    StiffnessCommandSmoother,
    StiffnessCommandSmoothingConfig,
)
from stiffness_copilot_mujoco.controllers.track_a_controllers import (  # noqa: E402
    DEFAULT_TRACK_A_CONTROLLERS_YAML,
    load_track_a_controller_runtime,
)
from stiffness_copilot_mujoco.evaluation.force_metrics import (  # noqa: E402
    ForceMetricThresholds,
    compute_force_metrics,
)
from stiffness_copilot_mujoco.evaluation.track_a_episode_runner import (  # noqa: E402
    summarize_policy_metadata,
    validate_track_a_v2_policy_metadata,
)
from stiffness_copilot_mujoco.franka_viewer import load_model  # noqa: E402
from stiffness_copilot_mujoco.learning.vision_residual_stiffness import load_image_only_residual_bc_policy  # noqa: E402
from stiffness_copilot_mujoco.metrics.task_metrics import (  # noqa: E402
    geometry_from_config,
    hole_center_position,
    insertion_depth,
    lateral_error,
)
from stiffness_copilot_mujoco.panda_control import arm_qpos, panda_arm_ids, set_arm_torque_ctrl  # noqa: E402
from stiffness_copilot_mujoco.pose_math import site_rotation  # noqa: E402
from stiffness_copilot_mujoco.robustness import ControlledContactProfile, make_controlled_contact_profile  # noqa: E402
from stiffness_copilot_mujoco.rollout_observation import reset_from_config  # noqa: E402
from stiffness_copilot_mujoco.rollouts.fixed_impedance import (  # noqa: E402
    RolloutConfig,
    cleanup_runtime_scene,
    clip_torque,
    scene_for_rollout,
)
from stiffness_copilot_mujoco.scenes import get_scene_spec  # noqa: E402
from stiffness_copilot_mujoco.sim.ids import peg_hole_ids  # noqa: E402
from stiffness_copilot_mujoco.sim.scene import (  # noqa: E402
    apply_eye_in_hand_camera_pose,
    eye_in_hand_camera_pose_from_config,
    validate_canonical_eye_in_hand_camera_config,
)
from stiffness_copilot_mujoco.vision.rendering import MujocoRgbRenderer  # noqa: E402
from tools.run_sigma7_mujoco_live_teleop import (  # noqa: E402
    EyeInHandWindow,
    Sigma7PoseReceiver,
    TerminalKeyMonitor,
    _build_third_person_camera,
    _rotation_matrix_to_row_major,
    _scene_alias,
    _viewer_running,
    map_sigma7_pose,
)


RAW_TRACE_SCHEMA_VERSION = "sigma7_single_controller_screening_live_v2"
TRAJECTORY_FAMILY_SIGMA7_LIVE_SCREENING = "sigma7_live_pose_screening"
TRAJECTORY_FAMILY_SIGMA7_LIVE_SCREENING_ID = 0
DEFAULT_CONTROLLER_ID = "track_a_c600"
DEFAULT_STIFFNESS_UPDATE_PERIOD_STEPS = 6
DEFAULT_STIFFNESS_UPDATE_HZ = 90.0
DEFAULT_STIFFNESS_SMOOTHING_METHOD = "log_spd_ema"
DEFAULT_STIFFNESS_SMOOTHING_ALPHA = 0.2


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json_ready(row), sort_keys=True))
            handle.write("\n")


def _write_jsonl_row(handle: TextIO, row: dict[str, Any]) -> None:
    handle.write(json.dumps(_json_ready(row), sort_keys=True))
    handle.write("\n")


def _unique_output_dir(root: Path, name: str) -> Path:
    candidate = root / name
    suffix = 0
    while candidate.exists():
        suffix += 1
        candidate = root / f"{name}_rerun{suffix}"
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def _screening_seed_from_episode_id(episode_id: int) -> int:
    digest = hashlib.sha256(f"sigma7_screening_seed_v1:{int(episode_id)}".encode("utf-8")).digest()
    return 1 + (int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**31 - 1))


def _maybe_reexec_with_mjpython(args: argparse.Namespace, argv: list[str]) -> int | None:
    if sys.platform != "darwin" or bool(args.disable_third_person):
        return None
    if os.environ.get("SIGMA7_SCREENING_UNDER_MJPYTHON") == "1":
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
    env["SIGMA7_SCREENING_UNDER_MJPYTHON"] = "1"
    cmd = [str(mjpython), str(Path(__file__).resolve()), *argv]
    print("relaunching under mjpython for MuJoCo viewer support", flush=True)
    return subprocess.run(cmd, env=env, check=False).returncode


def _build_stiffness_smoothing_config(args: argparse.Namespace) -> StiffnessCommandSmoothingConfig:
    method = args.stiffness_smoothing_method or DEFAULT_STIFFNESS_SMOOTHING_METHOD
    alpha = DEFAULT_STIFFNESS_SMOOTHING_ALPHA if args.stiffness_smoothing_alpha is None else float(args.stiffness_smoothing_alpha)
    if args.stiffness_update_hz is not None and args.stiffness_update_period_steps is not None:
        print(
            "[warn] --stiffness-update-hz overrides --stiffness-update-period-steps for deployment scheduling.",
            file=sys.stderr,
            flush=True,
        )
    if (
        args.stiffness_update_hz is None
        and args.stiffness_update_period_steps is None
        and args.stiffness_smoothing_method is None
        and args.stiffness_smoothing_alpha is None
    ):
        effective_period_steps = DEFAULT_STIFFNESS_UPDATE_PERIOD_STEPS
        effective_update_hz = DEFAULT_STIFFNESS_UPDATE_HZ
        method = DEFAULT_STIFFNESS_SMOOTHING_METHOD
        alpha = DEFAULT_STIFFNESS_SMOOTHING_ALPHA
    else:
        effective_period_steps = (
            DEFAULT_STIFFNESS_UPDATE_PERIOD_STEPS if args.stiffness_update_period_steps is None else int(args.stiffness_update_period_steps)
        )
        effective_update_hz = None if args.stiffness_update_hz is None else float(args.stiffness_update_hz)
    return StiffnessCommandSmoothingConfig(
        enabled=True,
        method=str(method),
        alpha=float(alpha),
        policy_update_period_steps=int(effective_period_steps),
        update_rate_hz=effective_update_hz,
        hold_between_updates=True,
    )


def _trace_force_mask(rows: list[dict[str, Any]]) -> tuple[list[float], list[bool]]:
    forces = [float(row.get("normal_force", 0.0)) for row in rows]
    contact_mask = [bool(row.get("contact_state", row.get("in_contact", False))) for row in rows]
    return forces, contact_mask


def _episode_quick_summary(
    *,
    controller_kind: str,
    summary: dict[str, Any],
    trace_rows: list[dict[str, Any]],
    thresholds: ForceMetricThresholds,
    trace_path: Path,
) -> dict[str, Any]:
    forces, contact_mask = _trace_force_mask(trace_rows)
    force_metrics = compute_force_metrics(forces, contact_mask=contact_mask, thresholds=thresholds)
    raw_max_force = force_metrics["raw_max_force"]
    return {
        "controller_kind": controller_kind,
        "controller_id": summary.get("controller_id"),
        "seed": summary.get("seed"),
        "success": bool(summary.get("depth_reached", False)),
        "low_force_success": bool(summary.get("low_force_success", False)),
        "catastrophic": bool(raw_max_force is not None and float(raw_max_force) >= thresholds.catastrophic_force_threshold),
        "raw_max_force": None if raw_max_force is None else float(raw_max_force),
        "raw_max_force_step": force_metrics["raw_max_force_step"],
        "raw_max_force_contact_state": force_metrics["raw_max_force_contact_state"],
        "depth_reached": bool(summary.get("depth_reached", False)),
        "final_depth": float(summary.get("final_depth", 0.0)),
        "final_lateral_error": float(summary.get("final_lateral_error", 0.0)),
        "termination_reason": summary.get("termination_reason"),
        "trace_path": str(trace_path),
        "trace_row_count": int(len(trace_rows)),
    }


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="Run one live Sigma7 screening episode with either the baseline controller or a trained residual policy."
    )
    parser.add_argument("--participant", type=str, required=True)
    parser.add_argument("--scene", type=str, required=True)
    parser.add_argument("--controller", choices=("baseline", "residual"), required=True)
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--episode-index", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_PIPELINE_ROOT)
    parser.add_argument("--policy", type=Path, default=None, help="Required when --controller residual.")
    parser.add_argument("--controllers-yaml", type=Path, default=DEFAULT_TRACK_A_CONTROLLERS_YAML)
    parser.add_argument("--controller-id", type=str, default=DEFAULT_CONTROLLER_ID)
    parser.add_argument("--screening-seed", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Legacy alias for --screening-seed.")
    parser.add_argument("--renderer-mode", choices=("native", "legacy_debug_only"), default="native")
    parser.add_argument("--camera-name", type=str, default="eye_in_hand_rgb")
    parser.add_argument("--image-width", type=int, default=128)
    parser.add_argument("--image-height", type=int, default=128)
    parser.add_argument("--show-eye-view", action="store_true")
    parser.add_argument("--disable-third-person", action="store_true")
    parser.add_argument("--eye-render-stride", type=int, default=4)
    parser.add_argument("--third-person-sync-stride", type=int, default=2)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument(
        "--stiffness-update-period-steps",
        type=int,
        default=None,
        help="Fixed stiffness refresh period in control steps. The default deployment path is 90 Hz time-accumulator scheduling.",
    )
    parser.add_argument("--stiffness-update-hz", type=float, default=None)
    parser.add_argument("--stiffness-smoothing-method", choices=("diagonal_ema", "log_spd_ema"), default=None)
    parser.add_argument("--stiffness-smoothing-alpha", type=float, default=None)
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
    scene_alias = require_safe_segment(args.scene, name="scene")
    controller_kind = require_safe_segment(args.controller, name="controller")
    if args.episode_index is not None and args.episode_id == 0:
        args.episode_id = int(args.episode_index)
    if controller_kind == "residual" and args.policy is None:
        raise ValueError("--policy is required when --controller residual.")
    if controller_kind == "baseline" and args.policy is not None:
        print("[warn] --policy is ignored for baseline screening runs.", file=sys.stderr, flush=True)
    if controller_kind == "residual" and args.renderer_mode != "native":
        raise ValueError("Residual screening requires --renderer-mode native.")
    if not np.isfinite(args.residual_scale) or args.residual_scale < 0.0:
        raise ValueError("--residual-scale must be a finite, non-negative scalar.")
    if args.eye_render_stride <= 0 or args.third_person_sync_stride <= 0:
        raise ValueError("Viewer sync strides must be positive.")
    scene_name = _scene_alias(scene_alias)
    episode_id = int(args.episode_id)
    explicit_seed = args.screening_seed if args.screening_seed is not None else args.seed
    screening_seed = int(explicit_seed) if explicit_seed is not None else _screening_seed_from_episode_id(episode_id)

    controlled_contact_profile: ControlledContactProfile | None = None
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
        seed=screening_seed,
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

    scene_spec = get_scene_spec(scene_name)
    rollout_config = RolloutConfig(config_path=scene_spec.config_path, max_steps=int(args.max_steps))
    controller_entry, controller_profile, gains = load_track_a_controller_runtime(
        args.controller_id,
        controllers_yaml=args.controllers_yaml,
    )
    reference_stiffness_matrix = np.asarray(controller_entry.position_stiffness_matrix, dtype=float)
    stiffness_smoothing_config = _build_stiffness_smoothing_config(args) if controller_kind == "residual" else None
    policy_path: Path | None = None
    policy = None
    policy_metadata_summary: dict[str, Any] | None = None
    if controller_kind == "residual":
        policy_path = Path(args.policy).expanduser().resolve(strict=False)
        policy = load_image_only_residual_bc_policy(policy_path)
        policy_hard_failures = validate_track_a_v2_policy_metadata(dict(policy.metadata))
        if policy_hard_failures:
            raise ValueError("Policy metadata guard failed: " + "; ".join(policy_hard_failures))
        policy_metadata_summary = summarize_policy_metadata(dict(policy.metadata))
        policy_reference_controller_id = str(policy.metadata["reference_controller_id"])
        if policy_reference_controller_id != args.controller_id:
            raise ValueError(
                f"Policy reference_controller_id {policy_reference_controller_id!r} does not match selected controller_id {args.controller_id!r}."
            )
        policy_base_matrix = np.asarray(policy.base_spec.base_matrix, dtype=float)
        if not np.allclose(policy_base_matrix, reference_stiffness_matrix, atol=1e-9, rtol=0.0):
            raise ValueError("Residual policy base stiffness does not match the selected controller registry entry.")

    controller_root = scene_screening_participant_controller_root(args.output_root, scene_alias, participant, controller_kind)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    policy_suffix = "" if policy_path is None else f"_{policy_path.stem}"
    run_name = f"{participant}_{scene_name}_{controller_kind}_screening_ep{episode_id:04d}_seed{screening_seed}_{timestamp}{policy_suffix}"
    run_root = _unique_output_dir(controller_root, run_name)
    trace_path = run_root / "trace.jsonl"
    summary_path = run_root / "summary.json"
    manifest_path = run_root / "episode_manifest.json"

    print(
        f"screening start participant={participant} controller={controller_kind} scene={scene_alias} "
        f"episode_id={episode_id} screening_seed={screening_seed}",
        flush=True,
    )

    receiver = Sigma7PoseReceiver(args.packet_host, int(args.packet_port))
    rgb_renderer: MujocoRgbRenderer | None = None
    eye_window: EyeInHandWindow | None = None
    passive_viewer = None
    scene_path = None
    trace_rows: list[dict[str, Any]] = []
    trace_handle: TextIO | None = None

    try:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_handle = trace_path.open("w", encoding="utf-8")

        scene_path, scene_config = scene_for_rollout(scene_spec.config_path, perturbation)
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
        from tools.run_sigma7_mujoco_live_teleop import PoseTeleopState  # noqa: E402

        teleop_state = PoseTeleopState(
            anchor_position=anchor_position.copy(),
            anchor_rotation=anchor_rotation.copy(),
        )

        renderer_required = bool(args.show_eye_view) or controller_kind == "residual"
        if renderer_required:
            rgb_renderer = MujocoRgbRenderer(
                model,
                camera_name=args.camera_name,
                width=int(args.image_width),
                height=int(args.image_height),
                renderer_mode=args.renderer_mode,
            )
            if controller_kind == "residual" and bool(rgb_renderer.fallback_used):
                raise RuntimeError("Residual screening does not accept fallback-rendered policy inputs.")
        if args.show_eye_view and rgb_renderer is not None:
            eye_window = EyeInHandWindow(
                python_path=args.viewer_python,
                script_path=ROOT / "scripts" / "live_image_window.py",
                title=f"sigma7_screening::{scene_name}::{controller_kind}",
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
                    "participant": participant,
                    "scene": scene_name,
                    "scene_alias": scene_alias,
                    "controller_kind": controller_kind,
                    "controller_id": args.controller_id,
                    "episode_id": episode_id,
                    "screening_seed": screening_seed,
                    "output_dir": str(run_root),
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
        print("first Sigma7 pose packet received, screening running", flush=True)
        print("press q in this terminal to mark failure, save data, and exit", flush=True)

        smoother = (
            StiffnessCommandSmoother(
                stiffness_smoothing_config,
                simulation_dt_seconds=simulation_dt_seconds,
            )
            if stiffness_smoothing_config is not None
            else None
        )
        policy_update_period_steps = (
            None
            if stiffness_smoothing_config is None
            else stiffness_smoothing_config.resolved_policy_update_period_steps(simulation_dt_seconds=simulation_dt_seconds)
        )
        policy_update_period_seconds = (
            None
            if stiffness_smoothing_config is None or stiffness_smoothing_config.update_rate_hz is None
            else 1.0 / float(stiffness_smoothing_config.update_rate_hz)
        )
        next_policy_refresh_time_seconds = 0.0 if policy_update_period_seconds is not None else None
        last_policy_refresh_step: int | None = None
        current_residual_raw_vector = np.zeros(6, dtype=float)
        current_residual_after_bound_vector = np.zeros(6, dtype=float)
        current_stiffness_after_residual = reference_stiffness_matrix.copy()
        current_stiffness_target = reference_stiffness_matrix.copy()
        current_theta = None
        current_theta_delta = None
        success_streak = 0
        max_normal_force = 0.0
        max_tangential_force = 0.0
        max_penetration_depth = 0.0
        max_abs_torque = 0.0
        torque_saturation_count = 0
        normal_force_sum = 0.0
        normal_force_count = 0
        contact_detected = False
        contact_onset_step = -1
        depth_at_contact = 0.0
        arrays_finite = True
        termination_reason = "max_steps_reached"
        next_tick = time.perf_counter()
        step_count = 0
        final_orientation_error = 0.0
        final_depth_threshold = float(0.95 * rollout_config.insert_depth)
        final_lateral_threshold = float(geometry.radial_clearance)

        with TerminalKeyMonitor() as key_monitor:
            for step in range(int(args.max_steps)):
                key = key_monitor.read_key()
                if key is not None and key.lower() == "q":
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

                frame = None
                residual_raw_vector = np.zeros(6, dtype=float)
                residual_after_bound_vector = np.zeros(6, dtype=float)
                stiffness_matrix_before_residual = reference_stiffness_matrix
                stiffness_matrix_after_residual = current_stiffness_after_residual
                stiffness_matrix_target = current_stiffness_target
                stiffness_matrix_command = reference_stiffness_matrix
                stiffness_smoothing_fields: dict[str, Any] = {
                    "stiffness_matrix_raw_before_smoothing": reference_stiffness_matrix,
                    "stiffness_matrix_after_smoothing": reference_stiffness_matrix,
                    "stiffness_refreshed_this_step": False,
                    "smoothing_update_applied": False,
                    "smoothing_hold_applied": False,
                    "stiffness_update_hz_target": None,
                    "stiffness_update_index": None,
                    "stiffness_update_interval_steps": None,
                    "stiffness_update_interval_seconds": None,
                    "steps_since_last_stiffness_refresh": None,
                    "stiffness_smoothing_scheduler": None,
                    "stiffness_update_scheduler": None,
                    "offdiag_xy_before_smoothing": float(reference_stiffness_matrix[0, 1]),
                        "offdiag_xy_after_smoothing": float(reference_stiffness_matrix[0, 1]),
                    }
                theta = current_theta
                theta_delta = current_theta_delta
                policy_refreshed_this_step = False
                if controller_kind == "residual":
                    assert policy is not None
                    assert rgb_renderer is not None
                    current_time_seconds = float(step) * simulation_dt_seconds
                    should_refresh_policy = bool(last_policy_refresh_step is None)
                    if not should_refresh_policy and policy_update_period_seconds is not None:
                        assert next_policy_refresh_time_seconds is not None
                        should_refresh_policy = bool(current_time_seconds + 1e-12 >= next_policy_refresh_time_seconds)
                    if not should_refresh_policy and policy_update_period_steps is not None and last_policy_refresh_step is not None:
                        should_refresh_policy = bool((step - last_policy_refresh_step) >= policy_update_period_steps)
                    if should_refresh_policy:
                        frame = rgb_renderer.render(data)
                        residual_raw, residual_after_bound, position_stiffness, theta, theta_delta = policy.predict_image_only(
                            frame,
                            residual_scale=float(args.residual_scale),
                        )
                        current_residual_raw_vector = np.asarray(residual_raw, dtype=float).reshape(-1)
                        current_residual_after_bound_vector = np.asarray(residual_after_bound, dtype=float).reshape(-1)
                        current_stiffness_after_residual = np.asarray(position_stiffness, dtype=float).reshape(3, 3)
                        current_stiffness_target = reference_stiffness_matrix + (
                            current_stiffness_after_residual - policy.base_spec.base_matrix
                        )
                        current_theta = None if theta is None else np.asarray(theta, dtype=float).reshape(-1).copy()
                        current_theta_delta = None if theta_delta is None else np.asarray(theta_delta, dtype=float).reshape(-1).copy()
                        last_policy_refresh_step = step
                        if policy_update_period_seconds is not None:
                            if next_policy_refresh_time_seconds is None:
                                next_policy_refresh_time_seconds = policy_update_period_seconds
                            else:
                                next_policy_refresh_time_seconds += policy_update_period_seconds
                        policy_refreshed_this_step = True
                    residual_raw_vector = current_residual_raw_vector
                    residual_after_bound_vector = current_residual_after_bound_vector
                    stiffness_matrix_after_residual = current_stiffness_after_residual
                    stiffness_matrix_target = current_stiffness_target
                    theta = current_theta
                    theta_delta = current_theta_delta
                    smoothing_step = smoother.apply(step=step, target_matrix=stiffness_matrix_target)
                    stiffness_matrix_command = np.asarray(smoothing_step.command_matrix, dtype=float).reshape(3, 3)
                    stiffness_smoothing_fields = {
                        "stiffness_matrix_raw_before_smoothing": np.asarray(
                            smoothing_step.raw_matrix_before_smoothing,
                            dtype=float,
                        ).reshape(3, 3),
                        "stiffness_matrix_after_smoothing": np.asarray(
                            smoothing_step.smoothed_matrix,
                            dtype=float,
                        ).reshape(3, 3),
                        "stiffness_refreshed_this_step": bool(smoothing_step.update_applied),
                        "smoothing_update_applied": bool(smoothing_step.update_applied),
                        "smoothing_hold_applied": bool(smoothing_step.hold_applied),
                        "stiffness_update_hz_target": smoothing_step.stiffness_update_hz_target,
                        "stiffness_update_index": smoothing_step.stiffness_update_index,
                        "stiffness_update_interval_steps": smoothing_step.stiffness_update_interval_steps,
                        "stiffness_update_interval_seconds": smoothing_step.stiffness_update_interval_seconds,
                        "steps_since_last_stiffness_refresh": smoothing_step.steps_since_last_stiffness_refresh,
                        "stiffness_smoothing_scheduler": smoothing_step.scheduler,
                        "stiffness_update_scheduler": smoothing_step.scheduler,
                        "offdiag_xy_before_smoothing": float(
                            np.asarray(smoothing_step.raw_matrix_before_smoothing, dtype=float)[0, 1]
                        ),
                        "offdiag_xy_after_smoothing": float(
                            np.asarray(smoothing_step.smoothed_matrix, dtype=float)[0, 1]
                        ),
                        "policy_refreshed_this_step": bool(policy_refreshed_this_step),
                    }
                elif rgb_renderer is not None and args.show_eye_view and (step == 0 or step % int(args.eye_render_stride) == 0):
                    frame = rgb_renderer.render(data)

                if eye_window is not None and frame is not None and (step == 0 or step % int(args.eye_render_stride) == 0):
                    eye_window.send(frame)

                command = task_space_impedance_torque(
                    model,
                    data,
                    site_name=rollout_config.site_name,
                    target_position=target_position,
                    target_rotation=target_rotation,
                    arm_ids=arm_ids,
                    gains=gains,
                    position_stiffness_matrix=stiffness_matrix_command,
                    nullspace_target_qpos=nullspace_target_qpos,
                    clip_to_ctrlrange=False,
                )
                torque, saturated = clip_torque(model, arm_ids, command.torque)
                max_abs_torque = max(max_abs_torque, float(np.max(np.abs(torque))))
                torque_saturation_count += int(saturated)
                contact_query = ContactQuery(model=model, data=data, task_ids=task_ids)
                contact = extract_contact_state(contact_query)
                force_world = np.asarray(extract_net_peg_hole_contact_force_world(contact_query), dtype=float)
                force_norm = float(np.linalg.norm(force_world))
                max_normal_force = max(max_normal_force, float(contact.normal_force))
                max_tangential_force = max(max_tangential_force, float(contact.tangential_force))
                max_penetration_depth = max(max_penetration_depth, float(contact.penetration_depth))
                if contact.in_contact:
                    if not contact_detected:
                        contact_detected = True
                        contact_onset_step = step
                        depth_at_contact = float(insertion_depth(data, task_ids))
                    normal_force_sum += float(contact.normal_force)
                    normal_force_count += 1

                actual_position = np.array(data.site_xpos[peg_tip_site_id], dtype=float)
                actual_rotation = site_rotation(data, peg_tip_site_id)
                depth = float(insertion_depth(data, task_ids))
                lateral = float(lateral_error(data, task_ids))
                success_now = bool(depth >= final_depth_threshold)
                success_streak = success_streak + 1 if success_now else 0
                final_orientation_error = float(np.linalg.norm(command.orientation_error))

                finite_step = bool(
                    np.all(np.isfinite(data.qpos))
                    and np.all(np.isfinite(data.qvel))
                    and np.all(np.isfinite(torque))
                    and np.isfinite(contact.normal_force)
                    and np.all(np.isfinite(target_position))
                    and np.all(np.isfinite(target_rotation))
                    and np.all(np.isfinite(stiffness_matrix_command))
                )
                if not finite_step:
                    arrays_finite = False
                    termination_reason = "nonfinite_rollout"
                    break

                trace_row = {
                    "participant": participant,
                    "scene": scene_name,
                    "scene_alias": scene_alias,
                    "controller_kind": controller_kind,
                    "controller_id": args.controller_id,
                    "policy_path": None if policy_path is None else str(policy_path),
                    "episode_id": episode_id,
                    "episode_spec_id": f"{scene_name}_screening_ep{episode_id:04d}_seed{screening_seed}",
                    "seed": screening_seed,
                    "screening_seed": screening_seed,
                    "step": int(step),
                    "time": float(data.time),
                    "phase": "teleop",
                    "phase_id": 2,
                    "depth": depth,
                    "insertion_depth": depth,
                    "lateral_error": lateral,
                    "position_error_norm": float(np.linalg.norm(command.position_error)),
                    "orientation_error_norm": final_orientation_error,
                    "normal_force": float(contact.normal_force),
                    "tangential_force": float(contact.tangential_force),
                    "penetration_depth": float(contact.penetration_depth),
                    "contact_state": bool(contact.in_contact),
                    "in_contact": bool(contact.in_contact),
                    "contact_force_world": [float(v) for v in force_world.reshape(-1)],
                    "contact_force_norm": force_norm,
                    "target_position": np.asarray(target_position, dtype=float).copy(),
                    "target_rotation": _rotation_matrix_to_row_major(target_rotation),
                    "actual_position": actual_position.copy(),
                    "actual_rotation": _rotation_matrix_to_row_major(actual_rotation),
                    "sigma_position": np.asarray(packet.position, dtype=float).copy(),
                    "sigma_rotation": _rotation_matrix_to_row_major(packet.orientation_frame),
                    "sigma_linear_velocity": np.asarray(packet.linear_velocity, dtype=float).copy(),
                    "sigma_angular_velocity_rad": np.asarray(packet.angular_velocity_rad, dtype=float).copy(),
                    "sigma_gripper_angle_rad": float(packet.gripper_angle_rad),
                    "sigma_gripper_linear_velocity": float(packet.gripper_linear_velocity),
                    "button_mask": int(packet.buttons),
                    "packet_sequence": -1 if packet.sequence is None else int(packet.sequence),
                    "packet_timestamp": None if packet.packet_timestamp is None else float(packet.packet_timestamp),
                    "residual_scale": float(args.residual_scale) if controller_kind == "residual" else None,
                    "residual_pred_vector": residual_raw_vector.tolist(),
                    "residual_action_vector": residual_raw_vector.tolist(),
                    "residual_after_bound_vector": residual_after_bound_vector.tolist(),
                    "group_delta": residual_after_bound_vector.tolist(),
                    "theta": None if theta is None else [float(v) for v in np.asarray(theta, dtype=float).reshape(-1)],
                    "theta_delta": None if theta_delta is None else [float(v) for v in np.asarray(theta_delta, dtype=float).reshape(-1)],
                    "stiffness_matrix_before_residual": np.asarray(stiffness_matrix_before_residual, dtype=float).reshape(3, 3).tolist(),
                    "stiffness_matrix_after_residual": np.asarray(stiffness_matrix_after_residual, dtype=float).reshape(3, 3).tolist(),
                    "stiffness_matrix_target": np.asarray(stiffness_matrix_target, dtype=float).reshape(3, 3).tolist(),
                    "stiffness_matrix_command": np.asarray(stiffness_matrix_command, dtype=float).reshape(3, 3).tolist(),
                    "stiffness_x": float(stiffness_matrix_after_residual[0, 0]),
                    "stiffness_y": float(stiffness_matrix_after_residual[1, 1]),
                    "stiffness_z": float(stiffness_matrix_after_residual[2, 2]),
                    "reference_stiffness_x": float(reference_stiffness_matrix[0, 0]),
                    "reference_stiffness_y": float(reference_stiffness_matrix[1, 1]),
                    "reference_stiffness_z": float(reference_stiffness_matrix[2, 2]),
                    "stiffness_command_x": float(stiffness_matrix_command[0, 0]),
                    "stiffness_command_y": float(stiffness_matrix_command[1, 1]),
                    "stiffness_command_z": float(stiffness_matrix_command[2, 2]),
                    "renderer_mode": None if rgb_renderer is None else rgb_renderer.mode,
                    "fallback_used": False if rgb_renderer is None else bool(rgb_renderer.fallback_used),
                    **{
                        key: (_json_ready(value) if isinstance(value, np.ndarray) else value)
                        for key, value in stiffness_smoothing_fields.items()
                    },
                }
                trace_rows.append(
                    {
                        "time": float(data.time),
                        "depth": depth,
                        "lateral_error": lateral,
                        "normal_force": float(contact.normal_force),
                        "contact_state": bool(contact.in_contact),
                        "in_contact": bool(contact.in_contact),
                    }
                )
                if trace_handle is None:
                    raise RuntimeError("Trace writer is not available.")
                _write_jsonl_row(trace_handle, trace_row)
                trace_handle.flush()

                if step % max(int(args.print_every), 1) == 0:
                    print(
                        f"step={step:04d} depth={depth:+.4f} lateral={lateral:.4f} force={force_norm:.3f} "
                        f"Kdiag=({stiffness_matrix_command[0,0]:.1f} {stiffness_matrix_command[1,1]:.1f} {stiffness_matrix_command[2,2]:.1f})",
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

        if not trace_rows:
            raise RuntimeError("No screening trace rows were collected. Did the sender produce packets?")

        final_depth = float(trace_rows[-1]["depth"])
        final_lateral_error = float(trace_rows[-1]["lateral_error"])
        depth_reached = bool(final_depth >= final_depth_threshold)
        low_force_success = bool(depth_reached and max_normal_force <= ForceMetricThresholds().low_force_success_threshold)
        completion_like = bool(depth_reached and final_lateral_error <= final_lateral_threshold)
        duration_seconds = float(trace_rows[-1]["time"])
        mean_normal_force_contact = normal_force_sum / normal_force_count if normal_force_count else 0.0
        smoothing_summary = None if smoother is None else smoother.summary_dict()
        summary = {
            "schema_version": RAW_TRACE_SCHEMA_VERSION,
            "participant": participant,
            "scene": scene_name,
            "scene_alias": scene_alias,
            "controller_kind": controller_kind,
            "controller_id": args.controller_id,
            "policy_path": None if policy_path is None else str(policy_path),
            "seed": screening_seed,
            "screening_seed": screening_seed,
            "episode_seed": screening_seed,
            "episode_id": episode_id,
            "episode_spec_id": f"{scene_name}_screening_ep{episode_id:04d}_seed{screening_seed}",
            "trajectory_family": TRAJECTORY_FAMILY_SIGMA7_LIVE_SCREENING,
            "trajectory_family_id": TRAJECTORY_FAMILY_SIGMA7_LIVE_SCREENING_ID,
            "steps": int(step_count),
            "runtime_seconds": duration_seconds,
            "depth_reached": depth_reached,
            "low_force_success": low_force_success,
            "completion_like": completion_like,
            "final_depth": final_depth,
            "final_lateral_error": final_lateral_error,
            "final_orientation_error": final_orientation_error,
            "contact_detected": bool(contact_detected),
            "contact_onset_step": int(contact_onset_step),
            "max_normal_force": float(max_normal_force),
            "mean_normal_force_contact": float(mean_normal_force_contact),
            "max_tangential_force": float(max_tangential_force),
            "max_penetration_depth": float(max_penetration_depth),
            "max_abs_commanded_torque": float(max_abs_torque),
            "torque_saturation_count": int(torque_saturation_count),
            "depth_progress_after_contact": float(final_depth - depth_at_contact) if contact_detected else 0.0,
            "termination_reason": termination_reason,
            "success_depth_threshold": final_depth_threshold,
            "success_lateral_threshold": final_lateral_threshold,
            "arrays_finite": arrays_finite,
            "renderer_mode": None if rgb_renderer is None else rgb_renderer.mode,
            "fallback_used": False if rgb_renderer is None else bool(rgb_renderer.fallback_used),
            "residual_scale": float(args.residual_scale) if controller_kind == "residual" else None,
            "stiffness_smoothing_summary": smoothing_summary,
            "profile_name": profile_name,
            "controller_profile": controller_profile,
            "perturbation": perturbation.to_dict(),
        }

        _write_json(summary_path, summary)

        thresholds = ForceMetricThresholds()
        quick = _episode_quick_summary(
            controller_kind=controller_kind,
            summary=summary,
            trace_rows=trace_rows,
            thresholds=thresholds,
            trace_path=trace_path,
        )
        manifest = {
            "raw_trace_schema_version": RAW_TRACE_SCHEMA_VERSION,
            "participant": participant,
            "scene": scene_name,
            "requested_scene": scene_alias,
            "controller_kind": controller_kind,
            "controller_id": args.controller_id,
            "controllers_yaml": str(args.controllers_yaml),
            "controller_profile": controller_profile,
            "policy_path": None if policy_path is None else str(policy_path),
            "policy_metadata_summary": policy_metadata_summary,
            "residual_scale": float(args.residual_scale) if controller_kind == "residual" else None,
            "screening_seed": screening_seed,
            "episode_seed": screening_seed,
            "episode_id": episode_id,
            "episode_spec_id": summary["episode_spec_id"],
            "trajectory_family": TRAJECTORY_FAMILY_SIGMA7_LIVE_SCREENING,
            "trajectory_family_id": TRAJECTORY_FAMILY_SIGMA7_LIVE_SCREENING_ID,
            "profile_name": profile_name,
            "profile_metadata": profile_metadata,
            "robustness_preset": preset_metadata,
            "perturbation": perturbation.to_dict(),
            "operator": str(args.operator),
            "packet_host": str(args.packet_host),
            "packet_port": int(args.packet_port),
            "teleop_mode": "position_orientation",
            "run_dir": str(run_root),
            "trace_path": str(trace_path),
            "summary_path": str(summary_path),
            "quick_summary": quick,
        }
        _write_json(manifest_path, manifest)

        latest_context = {
            "participant": participant,
            "scene": scene_alias,
            "controller_kind": controller_kind,
            "episode_id": episode_id,
            "episode_seed": screening_seed,
            "screening_seed": screening_seed,
            "run_dir": str(run_root),
            "trace_path": str(trace_path),
            "summary_path": str(summary_path),
            "manifest_path": str(manifest_path),
            "policy_path": None if policy_path is None else str(policy_path),
        }
        write_json(controller_root / "latest_run.json", latest_context)

        print("")
        print("screening complete")
        print(f"participant: {participant}")
        print(f"scene: {scene_alias}")
        print(f"controller: {controller_kind}")
        print(f"episode_id: {episode_id}")
        print(f"episode_seed: {screening_seed}")
        print(f"run_dir: {run_root}")
        print(f"trace: {trace_path}")
        print(f"summary: {summary_path}")
        print(f"manifest: {manifest_path}")
        print(f"success: {depth_reached}")
        print(f"low_force_success: {low_force_success}")
        print(f"raw_max_force: {quick['raw_max_force']}")

        if passive_viewer is not None and not bool(args.auto_close_after_finish):
            print(
                "screening finished; data saved; third-person viewer remains open until you close the window or press q/Ctrl-C",
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
        if trace_handle is not None:
            trace_handle.close()
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
        if scene_path is not None:
            cleanup_runtime_scene(scene_path)


if __name__ == "__main__":
    raise SystemExit(main())
