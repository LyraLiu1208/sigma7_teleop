#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import select
import socket
import struct
import subprocess
import sys
import termios
import time
import tty
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MUJOCO_ROOT = ROOT
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "sigma7_mujoco_live_teleop"
DEFAULT_VIEWER_PYTHON = Path(os.environ.get("SIGMA7_VIEWER_PYTHON", sys.executable)).expanduser()


def _ensure_mujoco_imports(mujoco_root: Path) -> None:
    src_root = mujoco_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))


@dataclass(frozen=True)
class Sigma7PosePacket:
    received_timestamp: float
    sequence: int | None
    packet_timestamp: float | None
    position: np.ndarray
    orientation_frame: np.ndarray
    gripper_angle_rad: float
    linear_velocity: np.ndarray
    angular_velocity_rad: np.ndarray
    gripper_linear_velocity: float
    buttons: int
    raw: dict[str, Any]


@dataclass
class PoseTeleopState:
    zero_position: np.ndarray | None = None
    zero_rotation: np.ndarray | None = None
    anchor_position: np.ndarray | None = None
    anchor_rotation: np.ndarray | None = None


@dataclass(frozen=True)
class PoseTeleopConfig:
    workspace_min_delta: np.ndarray
    workspace_max_delta: np.ndarray
    position_scale: float
    max_orientation_angle_rad: float
    deadband_m: float


class Sigma7PoseReceiver:
    def __init__(self, host: str, port: int) -> None:
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind((host, port))
        self._socket.settimeout(0.0)

    def close(self) -> None:
        self._socket.close()

    def recv_latest(self) -> Sigma7PosePacket | None:
        latest: Sigma7PosePacket | None = None
        while True:
            try:
                payload, _addr = self._socket.recvfrom(65535)
            except (BlockingIOError, socket.timeout):
                break
            latest = parse_sigma7_pose_packet(payload)
        return latest


class TerminalKeyMonitor:
    def __init__(self) -> None:
        self._enabled = False
        self._fd: int | None = None
        self._old_attrs: list[Any] | None = None

    def __enter__(self) -> "TerminalKeyMonitor":
        try:
            if not sys.stdin.isatty():
                return self
            self._fd = sys.stdin.fileno()
            self._old_attrs = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
            self._enabled = True
        except Exception:
            self._enabled = False
            self._fd = None
            self._old_attrs = None
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._enabled and self._fd is not None and self._old_attrs is not None:
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_attrs)
            except Exception:
                pass

    def read_key(self) -> str | None:
        if not self._enabled or self._fd is None:
            return None
        try:
            readable, _, _ = select.select([self._fd], [], [], 0.0)
            if not readable:
                return None
            return os.read(self._fd, 1).decode("utf-8", errors="ignore")
        except Exception:
            return None


class EyeInHandWindow:
    def __init__(self, *, python_path: Path, script_path: Path, title: str) -> None:
        if not python_path.exists():
            raise FileNotFoundError(f"Image viewer Python not found: {python_path}")
        if not script_path.exists():
            raise FileNotFoundError(f"Image viewer script not found: {script_path}")
        self._proc = subprocess.Popen(
            [str(python_path), "-u", str(script_path), "--title", title],
            stdin=subprocess.PIPE,
        )
        if self._proc.stdin is None:
            raise RuntimeError("Failed to open image window stdin.")
        self._stdin = self._proc.stdin
        self._closed = False

    def send(self, image: np.ndarray) -> None:
        import cv2

        if self._closed:
            return
        frame = np.asarray(image, dtype=np.uint8)
        ok, encoded = cv2.imencode(".png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError("Failed to encode image frame.")
        payload = encoded.tobytes()
        self._stdin.write(struct.pack("!I", len(payload)))
        self._stdin.write(payload)
        self._stdin.flush()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._stdin.write(struct.pack("!I", 0))
            self._stdin.flush()
        except Exception:
            pass
        try:
            self._stdin.close()
        except Exception:
            pass
        try:
            self._proc.wait(timeout=1.0)
        except Exception:
            try:
                self._proc.terminate()
            except Exception:
                pass


def _viewer_running(viewer: Any) -> bool:
    try:
        return bool(viewer.is_running())
    except Exception:
        return True


def _build_third_person_camera(lookat: np.ndarray, mujoco_module: Any) -> Any:
    camera = mujoco_module.MjvCamera()
    camera.type = mujoco_module.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = np.asarray(lookat, dtype=float)
    camera.distance = 0.90
    camera.azimuth = 135.0
    camera.elevation = -25.0
    camera.orthographic = 0
    camera.fixedcamid = -1
    camera.trackbodyid = -1
    return camera


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


def _save_rollout_outputs(
    *,
    output_dir: Path,
    metadata: dict[str, Any],
    summary: dict[str, Any],
    packet_log: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
) -> None:
    _write_json(output_dir / "metadata.json", metadata)
    _write_json(output_dir / "episode_summary.json", summary)
    _write_json(output_dir / "packet_log.json", {"packets": packet_log})
    np.savez_compressed(output_dir / "rollout_timeseries.npz", **arrays)


def _rotation_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float).reshape(3)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12 or abs(float(angle)) <= 1e-12:
        return np.eye(3, dtype=float)
    axis = axis / norm
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=float,
    )


def _axis_angle_from_rotation(rotation: np.ndarray) -> tuple[np.ndarray, float]:
    matrix = np.asarray(rotation, dtype=float).reshape(3, 3)
    trace = float(np.trace(matrix))
    cos_angle = max(-1.0, min(1.0, 0.5 * (trace - 1.0)))
    angle = float(math.acos(cos_angle))
    if angle <= 1e-9:
        return np.array([1.0, 0.0, 0.0], dtype=float), 0.0
    axis = np.array(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ],
        dtype=float,
    )
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1e-9:
        eigvals, eigvecs = np.linalg.eigh(matrix)
        axis = np.asarray(eigvecs[:, int(np.argmax(eigvals))], dtype=float)
        axis_norm = float(np.linalg.norm(axis))
    return axis / max(axis_norm, 1e-12), angle


def _rotation_matrix_to_row_major(rotation: np.ndarray) -> np.ndarray:
    return np.asarray(rotation, dtype=float).reshape(9)


def _as_vector3(value: Any, *, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), observed {vector.shape}.")
    return vector


def _as_rotation(value: Any, *, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3), observed {matrix.shape}.")
    return matrix


def parse_sigma7_pose_packet(payload: bytes | str) -> Sigma7PosePacket:
    text = payload.decode("utf-8") if isinstance(payload, (bytes, bytearray)) else str(payload)
    raw = json.loads(text)
    if not isinstance(raw, dict):
        raise ValueError("Sigma7 pose packet must decode to a JSON object.")
    return Sigma7PosePacket(
        received_timestamp=time.perf_counter(),
        sequence=None if raw.get("sequence") is None else int(raw["sequence"]),
        packet_timestamp=None if raw.get("packet_timestamp") is None else float(raw["packet_timestamp"]),
        position=_as_vector3(raw.get("position"), name="position"),
        orientation_frame=_as_rotation(raw.get("orientation_frame"), name="orientation_frame"),
        gripper_angle_rad=float(raw.get("gripper_angle_rad", 0.0)),
        linear_velocity=_as_vector3(raw.get("linear_velocity", [0.0, 0.0, 0.0]), name="linear_velocity"),
        angular_velocity_rad=_as_vector3(raw.get("angular_velocity_rad", [0.0, 0.0, 0.0]), name="angular_velocity_rad"),
        gripper_linear_velocity=float(raw.get("gripper_linear_velocity", 0.0)),
        buttons=int(raw.get("buttons", 0)),
        raw=raw,
    )


def map_sigma7_pose(
    packet: Sigma7PosePacket,
    *,
    state: PoseTeleopState,
    config: PoseTeleopConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if state.zero_position is None:
        state.zero_position = np.asarray(packet.position, dtype=float).copy()
    if state.zero_rotation is None:
        state.zero_rotation = np.asarray(packet.orientation_frame, dtype=float).copy()
    if state.anchor_position is None or state.anchor_rotation is None:
        raise RuntimeError("Pose teleop anchor has not been initialized.")

    raw_delta = (np.asarray(packet.position, dtype=float) - state.zero_position) * float(config.position_scale)
    raw_delta[np.abs(raw_delta) < float(config.deadband_m)] = 0.0
    position_delta = np.clip(raw_delta, config.workspace_min_delta, config.workspace_max_delta)
    target_position = np.asarray(state.anchor_position, dtype=float) + position_delta

    current_rotation = np.asarray(packet.orientation_frame, dtype=float)
    zero_rotation = np.asarray(state.zero_rotation, dtype=float)
    relative_rotation = current_rotation @ zero_rotation.T
    axis, angle = _axis_angle_from_rotation(relative_rotation)
    clipped_angle = min(float(config.max_orientation_angle_rad), abs(angle))
    clipped_angle *= 1.0 if angle >= 0.0 else -1.0
    clipped_relative_rotation = _rotation_from_axis_angle(axis, clipped_angle)
    target_rotation = clipped_relative_rotation @ np.asarray(state.anchor_rotation, dtype=float)
    return target_position, target_rotation


def _scene_alias(user_scene: str) -> str:
    aliases = {
        "circle": "circle",
        "polygon": "polygon_circle_logic_v1",
        "star": "star_circle_logic_v1",
        "polygon_circle_logic_v1": "polygon_circle_logic_v1",
        "star_circle_logic_v1": "star_circle_logic_v1",
    }
    try:
        return aliases[user_scene]
    except KeyError as exc:
        raise ValueError(f"Unsupported scene {user_scene!r}.") from exc


def _unique_output_dir(root: Path, name: str) -> Path:
    candidate = root / name
    suffix = 0
    while candidate.exists():
        suffix += 1
        candidate = root / f"{name}_rerun{suffix}"
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live Sigma7-to-MuJoCo teleop with fixed impedance and camera display.")
    parser.add_argument("--mujoco-root", type=Path, default=DEFAULT_MUJOCO_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--scene", choices=("circle", "polygon", "star"), default="circle")
    parser.add_argument("--seed", type=int, default=9903)
    parser.add_argument("--controller-id", type=str, default="track_a_c600")
    parser.add_argument("--packet-host", type=str, default="0.0.0.0")
    parser.add_argument("--packet-port", type=int, default=5005)
    parser.add_argument("--operator", type=str, default="unknown")
    parser.add_argument("--camera-name", type=str, default="eye_in_hand_rgb")
    parser.add_argument("--image-width", type=int, default=480)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--viewer-python", type=Path, default=DEFAULT_VIEWER_PYTHON)
    parser.add_argument("--disable-eye-view", action="store_true")
    parser.add_argument("--disable-third-person", action="store_true")
    parser.add_argument("--eye-render-stride", type=int, default=4)
    parser.add_argument("--third-person-sync-stride", type=int, default=2)
    parser.add_argument("--position-scale", type=float, default=2.5)
    parser.add_argument("--max-orientation-angle-rad", type=float, default=3.141592653589793)
    parser.add_argument("--deadband-m", type=float, default=0.0005)
    parser.add_argument("--workspace-min-delta", type=float, nargs=3, default=(-0.08, -0.08, -0.26))
    parser.add_argument("--workspace-max-delta", type=float, nargs=3, default=(0.08, 0.08, 0.05))
    parser.add_argument("--max-steps", type=int, default=15000)
    parser.add_argument("--success-hold-steps", type=int, default=50)
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--hold-after-finish-seconds", type=float, default=1.0)
    parser.add_argument("--auto-close-after-finish", action="store_true")
    parser.add_argument("--no-realtime", action="store_true")
    parser.add_argument("--contact-profile", type=str, default="circle_calibrated_v1_global_hole_fixed_contact")
    parser.add_argument("--contact-condition-name", type=str, default="teleop_easy_clearance_p0p0003")
    parser.add_argument("--hole-xy-radius", type=float, default=0.02)
    parser.add_argument("--fixed-teleop-noise-xy-amplitude", type=float, default=0.0010)
    parser.add_argument("--fixed-teleop-noise-cycles", type=float, default=1.0)
    parser.add_argument("--fixed-teleop-noise-phase-x", type=float, default=0.0)
    parser.add_argument("--fixed-teleop-noise-phase-y", type=float, default=1.5707963267948966)
    parser.add_argument("--fixed-clearance-delta", type=float, default=0.0003)
    parser.add_argument("--fixed-friction-scale", type=float, default=1.15)
    parser.add_argument("--fixed-peg-tilt-x", type=float, default=0.0087)
    parser.add_argument("--fixed-peg-tilt-y", type=float, default=-0.0087)
    parser.add_argument("--fixed-hole-yaw-offset", type=float, default=0.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    import mujoco
    import mujoco.viewer

    _ensure_mujoco_imports(args.mujoco_root)

    from stiffness_copilot_mujoco.contact.state import ContactQuery, extract_contact_state, extract_net_peg_hole_contact_force_world
    from stiffness_copilot_mujoco.controllers.impedance import task_space_impedance_torque
    from stiffness_copilot_mujoco.controllers.track_a_controllers import DEFAULT_TRACK_A_CONTROLLERS_YAML, load_track_a_controller_runtime
    from stiffness_copilot_mujoco.franka_viewer import load_model
    from stiffness_copilot_mujoco.metrics.task_metrics import geometry_from_config, hole_center_position, insertion_depth, lateral_error, load_scene_config
    from stiffness_copilot_mujoco.panda_control import arm_qpos, panda_arm_ids, set_arm_torque_ctrl
    from stiffness_copilot_mujoco.pose_math import site_rotation
    from stiffness_copilot_mujoco.robustness import make_controlled_contact_profile, sample_controlled_contact_perturbations
    from stiffness_copilot_mujoco.rollout_observation import reset_from_config
    from stiffness_copilot_mujoco.rollouts.fixed_impedance import RolloutConfig, cleanup_runtime_scene, clip_torque, scene_for_rollout
    from stiffness_copilot_mujoco.scenes import get_scene_spec
    from stiffness_copilot_mujoco.sim.ids import peg_hole_ids
    from stiffness_copilot_mujoco.sim.scene import canonical_eye_in_hand_camera_pose, eye_in_hand_camera_pose_from_config, apply_eye_in_hand_camera_pose, validate_canonical_eye_in_hand_camera_config
    from stiffness_copilot_mujoco.vision.rendering import MujocoRgbRenderer

    scene_name = _scene_alias(args.scene)
    scene_spec = get_scene_spec(scene_name)
    rollout_config = RolloutConfig(config_path=scene_spec.config_path, max_steps=args.max_steps)
    controller_entry, controller_profile, gains = load_track_a_controller_runtime(
        args.controller_id,
        controllers_yaml=DEFAULT_TRACK_A_CONTROLLERS_YAML,
    )
    controlled_profile = make_controlled_contact_profile(
        profile_name=args.contact_profile,
        contact_condition_name=args.contact_condition_name,
        hole_xy_radius=float(args.hole_xy_radius),
        teleop_noise_xy_amplitude=float(args.fixed_teleop_noise_xy_amplitude),
        teleop_noise_cycles=float(args.fixed_teleop_noise_cycles),
        teleop_noise_phase_x=float(args.fixed_teleop_noise_phase_x),
        teleop_noise_phase_y=float(args.fixed_teleop_noise_phase_y),
        clearance_delta=float(args.fixed_clearance_delta),
        friction_scale=float(args.fixed_friction_scale),
        peg_tilt_x=float(args.fixed_peg_tilt_x),
        peg_tilt_y=float(args.fixed_peg_tilt_y),
        hole_yaw_offset=float(args.fixed_hole_yaw_offset),
    )
    perturbation = sample_controlled_contact_perturbations(
        episodes=1,
        seed=int(args.seed),
        profile=controlled_profile,
    )[0]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = _unique_output_dir(args.output_root, f"{scene_name}_sigma7_live_{timestamp}")

    receiver = Sigma7PoseReceiver(args.packet_host, args.packet_port)
    eye_window = None
    passive_viewer = None
    rgb_renderer = None
    model = None
    data = None
    scene_path = None

    packet_log: list[dict[str, Any]] = []
    time_series: dict[str, list[Any]] = {
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

    try:
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
        canonical_camera = canonical_eye_in_hand_camera_pose(args.camera_name)

        mapper_state = PoseTeleopState(
            anchor_position=anchor_position.copy(),
            anchor_rotation=anchor_rotation.copy(),
        )
        mapper_config = PoseTeleopConfig(
            workspace_min_delta=np.asarray(args.workspace_min_delta, dtype=float),
            workspace_max_delta=np.asarray(args.workspace_max_delta, dtype=float),
            position_scale=float(args.position_scale),
            max_orientation_angle_rad=float(args.max_orientation_angle_rad),
            deadband_m=float(args.deadband_m),
        )

        if not args.disable_eye_view:
            rgb_renderer = MujocoRgbRenderer(
                model,
                camera_name=args.camera_name,
                width=int(args.image_width),
                height=int(args.image_height),
                renderer_mode="native",
            )
            eye_window = EyeInHandWindow(
                python_path=args.viewer_python,
                script_path=args.mujoco_root / "scripts" / "live_image_window.py",
                title=f"sigma7_live::{scene_name}",
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
                    "controller_id": args.controller_id,
                    "controller_profile": controller_profile,
                    "camera_name": args.camera_name,
                    "eye_view_enabled": not bool(args.disable_eye_view),
                    "output_dir": str(output_dir),
                    "seed": int(args.seed),
                    "perturbation": perturbation.to_dict(),
                    "contact_profile": controlled_profile.to_metadata(),
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        print("waiting for first Sigma7 packet...", flush=True)

        latest_packet = None
        while latest_packet is None:
            latest_packet = receiver.recv_latest()
            if latest_packet is None:
                time.sleep(0.01)
        print("first Sigma7 packet received, teleop running", flush=True)

        if args.eye_render_stride <= 0:
            raise ValueError("--eye-render-stride must be positive.")
        if args.third_person_sync_stride <= 0:
            raise ValueError("--third-person-sync-stride must be positive.")
        if args.success_hold_steps <= 0:
            raise ValueError("--success-hold-steps must be positive.")

        max_contact_force = 0.0
        step_count = 0
        success_streak = 0
        success_hold_steps = int(args.success_hold_steps)
        final_depth_threshold = float(0.95 * rollout_config.insert_depth)
        final_lateral_threshold = float(geometry.radial_clearance)
        termination_reason = "max_steps_reached"
        next_tick = time.perf_counter()
        try:
            with TerminalKeyMonitor() as key_monitor:
                print("press q in this terminal to mark failure, save data, and exit", flush=True)
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
                        state=mapper_state,
                        config=mapper_config,
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
                    max_contact_force = max(max_contact_force, force_norm)

                    actual_position = np.array(data.site_xpos[peg_tip_site_id], dtype=float)
                    actual_rotation = site_rotation(data, peg_tip_site_id)
                    depth = float(insertion_depth(data, task_ids))
                    lateral = float(lateral_error(data, task_ids))
                    success_now = bool(
                        force_norm > 0.0
                        and depth >= final_depth_threshold
                        and lateral <= final_lateral_threshold
                    )
                    success_streak = success_streak + 1 if success_now else 0

                    time_series["time"].append(float(data.time))
                    time_series["step"].append(step)
                    time_series["phase_id"].append(2)
                    time_series["target_position"].append(target_position.copy())
                    time_series["target_rotation"].append(_rotation_matrix_to_row_major(target_rotation))
                    time_series["actual_position"].append(actual_position.copy())
                    time_series["actual_rotation"].append(_rotation_matrix_to_row_major(actual_rotation))
                    time_series["position_error_norm"].append(float(np.linalg.norm(command.position_error)))
                    time_series["orientation_error_norm"].append(float(np.linalg.norm(command.orientation_error)))
                    time_series["sigma_position"].append(np.asarray(packet.position, dtype=float).copy())
                    time_series["sigma_rotation"].append(_rotation_matrix_to_row_major(packet.orientation_frame))
                    time_series["sigma_linear_velocity"].append(np.asarray(packet.linear_velocity, dtype=float).copy())
                    time_series["sigma_angular_velocity_rad"].append(np.asarray(packet.angular_velocity_rad, dtype=float).copy())
                    time_series["sigma_gripper_angle_rad"].append(float(packet.gripper_angle_rad))
                    time_series["sigma_gripper_linear_velocity"].append(float(packet.gripper_linear_velocity))
                    time_series["contact_force_world"].append(force_world.copy())
                    time_series["contact_force_norm"].append(force_norm)
                    time_series["contact_active"].append(bool(contact.in_contact))
                    time_series["contact_normal_force"].append(float(contact.normal_force))
                    time_series["insertion_depth"].append(depth)
                    time_series["lateral_error"].append(lateral)
                    time_series["button_mask"].append(int(packet.buttons))
                    time_series["packet_sequence"].append(-1 if packet.sequence is None else int(packet.sequence))
                    time_series["packet_timestamp"].append(float("nan") if packet.packet_timestamp is None else float(packet.packet_timestamp))

                    packet_log.append(
                        {
                            "step": step,
                            "sequence": packet.sequence,
                            "packet_timestamp": packet.packet_timestamp,
                            "received_timestamp": packet.received_timestamp,
                            "buttons": packet.buttons,
                        }
                    )

                    if eye_window is not None and rgb_renderer is not None and (step == 0 or step % int(args.eye_render_stride) == 0):
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
                    if success_streak >= success_hold_steps:
                        termination_reason = "success"
                        break
        except KeyboardInterrupt:
            termination_reason = "keyboard_interrupt"

        final_depth = float(time_series["insertion_depth"][-1]) if time_series["insertion_depth"] else 0.0
        final_lateral_error = float(time_series["lateral_error"][-1]) if time_series["lateral_error"] else 0.0
        insertion_success = bool(
            termination_reason == "success"
            and final_depth >= final_depth_threshold
            and final_lateral_error <= final_lateral_threshold
        )
        metadata = {
            "schema_version": "sigma7_mujoco_live_teleop_v1",
            "scenario": scene_name,
            "scene_alias": args.scene,
            "mode": "sigma7_pose_teleop_fixed_impedance",
            "operator": args.operator,
            "policy_path": None,
            "residual_scale": None,
            "controller_id": args.controller_id,
            "controller_profile": controller_profile,
            "collection_controller_id": args.controller_id,
            "collection_stiffness_matrix": controller_entry.position_stiffness_matrix.tolist(),
            "teleop_mode": "position_orientation",
            "input_device": "sigma7_sdk_direct",
            "mujoco_root": str(args.mujoco_root),
            "scene_config_path": str(scene_spec.config_path),
            "camera_name": args.camera_name,
            "eye_view_enabled": not bool(args.disable_eye_view),
            "renderer_mode": None if rgb_renderer is None else rgb_renderer.mode,
            "fallback_used": False if rgb_renderer is None else bool(rgb_renderer.fallback_used),
            "eye_in_hand_camera_pose_version": canonical_camera["pose_version"],
            "eye_in_hand_camera_canonical": bool(canonical_camera["canonical"]),
            "eye_in_hand_camera_name": canonical_camera["camera_name"],
            "eye_in_hand_camera_attachment_parent": canonical_camera["attachment_parent"],
            "eye_in_hand_camera_mount_type": canonical_camera["mount_type"],
            "eye_in_hand_camera_pose": canonical_camera,
            "seed": int(args.seed),
            "contact_profile": controlled_profile.profile_name,
            "contact_condition_name": controlled_profile.contact_condition_name,
            "controlled_contact_profile": controlled_profile.to_metadata(),
            "perturbation": perturbation.to_dict(),
            "workspace_min_delta": mapper_config.workspace_min_delta.tolist(),
            "workspace_max_delta": mapper_config.workspace_max_delta.tolist(),
            "position_scale": float(args.position_scale),
            "max_orientation_angle_rad": float(args.max_orientation_angle_rad),
            "deadband_m": float(args.deadband_m),
            "success_hold_steps": int(args.success_hold_steps),
            "auto_close_after_finish": bool(args.auto_close_after_finish),
            "eye_render_stride": int(args.eye_render_stride),
            "third_person_sync_stride": int(args.third_person_sync_stride),
            "missing_fields": [],
        }
        summary = {
            "insertion_success": insertion_success,
            "termination_reason": termination_reason,
            "max_contact_force": float(max_contact_force),
            "final_depth": final_depth,
            "final_lateral_error": final_lateral_error,
            "contact_fraction": float(np.mean(np.asarray(time_series["contact_active"], dtype=float))) if time_series["contact_active"] else 0.0,
            "duration_seconds": float(time_series["time"][-1]) if time_series["time"] else 0.0,
            "num_steps": int(step_count),
            "success_streak_final": int(success_streak),
            "success_hold_steps": int(args.success_hold_steps),
            "final_depth_threshold": final_depth_threshold,
            "final_lateral_threshold": final_lateral_threshold,
        }

        arrays = {
            "time": np.asarray(time_series["time"], dtype=float),
            "step": np.asarray(time_series["step"], dtype=np.int32),
            "phase_id": np.asarray(time_series["phase_id"], dtype=np.int32),
            "target_position": np.asarray(time_series["target_position"], dtype=float),
            "target_rotation": np.asarray(time_series["target_rotation"], dtype=float),
            "actual_position": np.asarray(time_series["actual_position"], dtype=float),
            "actual_rotation": np.asarray(time_series["actual_rotation"], dtype=float),
            "position_error_norm": np.asarray(time_series["position_error_norm"], dtype=float),
            "orientation_error_norm": np.asarray(time_series["orientation_error_norm"], dtype=float),
            "sigma_position": np.asarray(time_series["sigma_position"], dtype=float),
            "sigma_rotation": np.asarray(time_series["sigma_rotation"], dtype=float),
            "sigma_linear_velocity": np.asarray(time_series["sigma_linear_velocity"], dtype=float),
            "sigma_angular_velocity_rad": np.asarray(time_series["sigma_angular_velocity_rad"], dtype=float),
            "sigma_gripper_angle_rad": np.asarray(time_series["sigma_gripper_angle_rad"], dtype=float),
            "sigma_gripper_linear_velocity": np.asarray(time_series["sigma_gripper_linear_velocity"], dtype=float),
            "contact_force_world": np.asarray(time_series["contact_force_world"], dtype=float),
            "contact_force_norm": np.asarray(time_series["contact_force_norm"], dtype=float),
            "contact_active": np.asarray(time_series["contact_active"], dtype=bool),
            "contact_normal_force": np.asarray(time_series["contact_normal_force"], dtype=float),
            "insertion_depth": np.asarray(time_series["insertion_depth"], dtype=float),
            "lateral_error": np.asarray(time_series["lateral_error"], dtype=float),
            "button_mask": np.asarray(time_series["button_mask"], dtype=np.int32),
            "packet_sequence": np.asarray(time_series["packet_sequence"], dtype=np.int64),
            "packet_timestamp": np.asarray(time_series["packet_timestamp"], dtype=float),
            "metadata": np.asarray(json.dumps(metadata, sort_keys=True)),
        }
        _save_rollout_outputs(
            output_dir=output_dir,
            metadata=metadata,
            summary=summary,
            packet_log=packet_log,
            arrays=arrays,
        )

        print("")
        print(f"output_dir: {output_dir}")
        print(f"metadata: {output_dir / 'metadata.json'}")
        print(f"summary: {output_dir / 'episode_summary.json'}")
        print(f"timeseries: {output_dir / 'rollout_timeseries.npz'}")

        if passive_viewer is not None and not bool(args.auto_close_after_finish):
            print("teleop finished; data saved; third-person viewer remains open until you close the window or press q/Ctrl-C", flush=True)
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
        if scene_path is not None:
            cleanup_runtime_scene(scene_path)


if __name__ == "__main__":
    raise SystemExit(main())
