#!/usr/bin/env python3
from __future__ import annotations

import importlib
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    warning: bool = False


def check_path(path: Path, *, label: str) -> CheckResult:
    return CheckResult(label, path.exists(), str(path))


def check_import(module_name: str) -> CheckResult:
    try:
        importlib.import_module(module_name)
        return CheckResult(f"import:{module_name}", True, "ok")
    except Exception as exc:
        return CheckResult(f"import:{module_name}", False, str(exc))


def check_git() -> CheckResult:
    git_dir = ROOT / ".git"
    return CheckResult("git", git_dir.exists(), str(git_dir))


def check_virtualenv() -> CheckResult:
    venv_dir = ROOT / ".venv"
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    expected_python = venv_dir / "bin" / "python"
    using_expected_python = Path(sys.executable).resolve() == expected_python.resolve() if expected_python.exists() else False
    detail = f"sys.executable={sys.executable}"
    return CheckResult(
        "python_virtualenv",
        in_venv and using_expected_python,
        detail,
        warning=True,
    )


def check_sender_sdk() -> CheckResult:
    sdk_root = os.environ.get("SIGMA7_SDK_ROOT") or os.environ.get("FORCEDIM_SDK_ROOT")
    if not sdk_root:
        return CheckResult("sigma7_sdk", False, "SIGMA7_SDK_ROOT is not set", warning=True)
    include_dir = Path(sdk_root).expanduser() / "include"
    ok = include_dir.exists()
    return CheckResult("sigma7_sdk", ok, str(include_dir), warning=not ok)


def check_sender_binary() -> CheckResult:
    binary = ROOT / "tools" / "build" / "sigma7_pose_udp_sender"
    return CheckResult("sigma7_sender_binary", binary.exists(), str(binary), warning=not binary.exists())


def check_package_import() -> CheckResult:
    try:
        import stiffness_copilot_mujoco  # noqa: F401
        return CheckResult("import:stiffness_copilot_mujoco", True, "ok")
    except Exception as exc:
        return CheckResult("import:stiffness_copilot_mujoco", False, str(exc))


def main() -> int:
    results = [
        check_git(),
        check_virtualenv(),
        check_path(ROOT / "src" / "stiffness_copilot_mujoco", label="src_package"),
        check_path(ROOT / "configs" / "track_a_controllers.yaml", label="controllers_yaml"),
        check_path(ROOT / "scripts" / "live_image_window.py", label="live_image_window"),
        check_path(ROOT / "third_party" / "mujoco_menagerie" / "franka_emika_panda" / "panda.xml", label="franka_assets"),
        check_path(ROOT / "third_party" / "dinov3" / "hubconf.py", label="dinov3_repo"),
        check_path(ROOT / "checkpoints" / "dinov3_vits16_pretrain_lvd1689m-08c60483.pth", label="dinov3_checkpoint"),
        check_package_import(),
        check_import("numpy"),
        check_import("mujoco"),
        check_import("yaml"),
        check_import("cv2"),
        check_import("torch"),
        check_import("matplotlib"),
        check_sender_sdk(),
        check_sender_binary(),
    ]

    for result in results:
        status = "WARN" if result.warning and not result.ok else ("OK" if result.ok else "FAIL")
        print(f"[{status}] {result.name}: {result.detail}")

    hard_failures = [item for item in results if not item.ok and not item.warning]
    if hard_failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
