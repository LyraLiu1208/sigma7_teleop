#!/usr/bin/env python3
from __future__ import annotations

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
    cmd = [
        sys.executable,
        "-c",
        (
            "import importlib; "
            f"importlib.import_module({module_name!r}); "
            "print('ok', flush=True)"
        ),
    ]
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            env=os.environ.copy(),
        )
    except Exception as exc:
        return CheckResult(f"import:{module_name}", False, str(exc))

    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    if completed.returncode == 0:
        return CheckResult(f"import:{module_name}", True, stdout or "ok")
    if completed.returncode < 0:
        signal_num = -completed.returncode
        detail = f"terminated by signal {signal_num}"
        if stderr:
            detail = f"{detail}; stderr={stderr}"
        return CheckResult(f"import:{module_name}", False, detail)
    detail = stderr or stdout or f"exit code {completed.returncode}"
    return CheckResult(f"import:{module_name}", False, detail)


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


def check_env_profile() -> CheckResult:
    profile_path = ROOT / ".venv" / ".sigma7_env_profile"
    if not profile_path.exists():
        return CheckResult("env_profile", False, f"{profile_path} is missing", warning=True)
    values: dict[str, str] = {}
    for line in profile_path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    profile_name = values.get("ENV_PROFILE_NAME", "<unknown>")
    torch_version = values.get("TORCH_VERSION", "<unknown>")
    index_url = values.get("PYTORCH_INDEX_URL", "<unknown>")
    return CheckResult(
        "env_profile",
        True,
        f"profile={profile_name} torch={torch_version} index={index_url}",
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


def check_torch_runtime() -> CheckResult:
    cmd = [
        sys.executable,
        "-c",
        (
            "import torch; "
            "print(f'version={torch.__version__} cuda_available={torch.cuda.is_available()}', flush=True)"
        ),
    ]
    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        env=os.environ.copy(),
    )
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    if completed.returncode == 0:
        return CheckResult("torch_runtime", True, stdout or "ok")
    if completed.returncode < 0:
        signal_num = -completed.returncode
        detail = f"terminated by signal {signal_num}"
        if stderr:
            detail = f"{detail}; stderr={stderr}"
        return CheckResult("torch_runtime", False, detail)
    detail = stderr or stdout or f"exit code {completed.returncode}"
    return CheckResult("torch_runtime", False, detail)


def main() -> int:
    results = [
        check_git(),
        check_virtualenv(),
        check_env_profile(),
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
        check_torch_runtime(),
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
