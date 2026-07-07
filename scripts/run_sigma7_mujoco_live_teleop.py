from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from _sigma7_runtime import ROOT, default_mjpython

DEFAULT_MJ_PYTHON = default_mjpython()
DEFAULT_TOOL = ROOT / "tools" / "run_sigma7_mujoco_live_teleop.py"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Launcher for the interactive MuJoCo Sigma7 live teleop viewer."
    )
    parser.add_argument("--mjpython", type=Path, default=DEFAULT_MJ_PYTHON)
    parser.add_argument("--tool", type=Path, default=DEFAULT_TOOL)
    parser.add_argument("--scene", choices=("circle", "polygon", "star"), default="polygon")
    parser.add_argument("--operator", type=str, default="lyra")
    parser.add_argument("--packet-host", type=str, default="0.0.0.0")
    parser.add_argument("--packet-port", type=int, default=5005)
    parser.add_argument("--controller-id", type=str, default="track_a_c600")
    parser.add_argument("--position-scale", type=float, default=3.5)
    parser.add_argument("--max-steps", type=int, default=45000)
    parser.add_argument("--disable-eye-view", action="store_true", default=True)
    parser.add_argument("--enable-eye-view", action="store_true")
    parser.add_argument("--disable-third-person", action="store_true")
    parser.add_argument("--hold-after-finish-seconds", type=float, default=1.0)
    parser.add_argument("--auto-close-after-finish", action="store_true")
    parser.add_argument("--no-realtime", action="store_true")
    parser.add_argument("--contact-profile", type=str, default=None)
    parser.add_argument("--contact-condition-name", type=str, default=None)
    args = parser.parse_args(argv)

    disable_eye_view = bool(args.disable_eye_view and not args.enable_eye_view)
    cmd = [
        str(args.mjpython),
        str(args.tool),
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
        "--hold-after-finish-seconds",
        str(args.hold_after_finish_seconds),
    ]
    if disable_eye_view:
        cmd.append("--disable-eye-view")
    if args.disable_third_person:
        cmd.append("--disable-third-person")
    if args.auto_close_after_finish:
        cmd.append("--auto-close-after-finish")
    if args.no_realtime:
        cmd.append("--no-realtime")
    if args.contact_profile is not None:
        cmd.extend(["--contact-profile", args.contact_profile])
    if args.contact_condition_name is not None:
        cmd.extend(["--contact-condition-name", args.contact_condition_name])

    print("launching MuJoCo live teleop viewer")
    print("command:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
