from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from _sigma7_runtime import ROOT

DEFAULT_BINARY = ROOT / "tools" / "build" / "sigma7_pose_udp_sender"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Launcher for the Sigma7 pose UDP sender."
    )
    parser.add_argument("--arch", type=str, default=None, choices=("x86_64", "arm64"))
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--hz", type=int, default=200)
    args = parser.parse_args(argv)

    if not args.binary.exists():
        raise FileNotFoundError(
            f"Sigma7 sender binary not found: {args.binary}. "
            "Build it first with `cmake -S . -B build && cmake --build build` in tools/sigma7_pose_udp_sender."
        )

    cmd = [str(args.binary), "--host", args.host, "--port", str(args.port), "--hz", str(args.hz)]
    if args.arch is not None and sys.platform == "darwin":
        cmd = ["arch", f"-{args.arch}", *cmd]
    print("launching Sigma7 pose UDP sender")
    print("command:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
