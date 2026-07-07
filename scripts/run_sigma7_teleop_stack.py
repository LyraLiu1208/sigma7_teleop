from __future__ import annotations

import argparse
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIEWER = ROOT / "scripts" / "run_sigma7_mujoco_live_teleop.py"
DEFAULT_SENDER = ROOT / "scripts" / "run_sigma7_pose_udp_sender.py"


def _terminate(proc: subprocess.Popen[bytes] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=3.0)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Launch the Sigma7 UDP sender and the interactive MuJoCo teleop viewer together."
    )
    parser.add_argument("--viewer-script", type=Path, default=DEFAULT_VIEWER)
    parser.add_argument("--sender-script", type=Path, default=DEFAULT_SENDER)
    parser.add_argument("--scene", choices=("circle", "polygon", "star"), default="polygon")
    parser.add_argument("--operator", type=str, default="lyra")
    parser.add_argument("--packet-host", type=str, default="0.0.0.0")
    parser.add_argument("--packet-port", type=int, default=5005)
    parser.add_argument("--sender-host", type=str, default="127.0.0.1")
    parser.add_argument("--sender-port", type=int, default=5005)
    parser.add_argument("--sender-hz", type=int, default=200)
    parser.add_argument("--controller-id", type=str, default="track_a_c600")
    parser.add_argument("--position-scale", type=float, default=3.5)
    parser.add_argument("--max-steps", type=int, default=45000)
    parser.add_argument("--disable-eye-view", action="store_true", default=True)
    parser.add_argument("--enable-eye-view", action="store_true")
    parser.add_argument("--disable-third-person", action="store_true")
    parser.add_argument("--sender-only", action="store_true")
    parser.add_argument("--viewer-only", action="store_true")
    args = parser.parse_args(argv)

    if args.sender_only and args.viewer_only:
        raise ValueError("sender-only and viewer-only cannot both be set.")

    disable_eye_view = bool(args.disable_eye_view and not args.enable_eye_view)
    sender_cmd = [
        sys.executable,
        str(args.sender_script),
        "--host",
        args.sender_host,
        "--port",
        str(args.sender_port),
        "--hz",
        str(args.sender_hz),
    ]
    viewer_cmd = [
        sys.executable,
        str(args.viewer_script),
        "--scene",
        args.scene,
        "--operator",
        args.operator,
        "--packet-host",
        args.packet_host,
        "--packet-port",
        str(args.packet_port),
        "--controller-id",
        args.controller_id,
        "--position-scale",
        str(args.position_scale),
        "--max-steps",
        str(args.max_steps),
    ]
    if disable_eye_view:
        viewer_cmd.append("--disable-eye-view")
    if args.disable_third_person:
        viewer_cmd.append("--disable-third-person")

    sender_proc: subprocess.Popen[bytes] | None = None
    viewer_proc: subprocess.Popen[bytes] | None = None
    try:
        if not args.viewer_only:
            print("sender:", shlex.join(sender_cmd))
            sender_proc = subprocess.Popen(sender_cmd)
            time.sleep(0.8)
        if not args.sender_only:
            print("viewer:", shlex.join(viewer_cmd))
            viewer_proc = subprocess.Popen(viewer_cmd)
        while True:
            sender_done = sender_proc is None or sender_proc.poll() is not None
            viewer_done = viewer_proc is None or viewer_proc.poll() is not None
            if args.sender_only and sender_done:
                break
            if args.viewer_only and viewer_done:
                break
            if not args.sender_only and not args.viewer_only and viewer_done:
                break
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        _terminate(viewer_proc)
        _terminate(sender_proc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
