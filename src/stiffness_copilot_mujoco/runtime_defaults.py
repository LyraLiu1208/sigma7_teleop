from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DINOV3_REPO = REPO_ROOT / "third_party" / "dinov3"
DEFAULT_DINOV3_CHECKPOINT = REPO_ROOT / "checkpoints" / "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"


def _env_path(*names: str) -> Path | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return Path(value).expanduser()
    return None


def default_mjpython() -> Path:
    explicit = _env_path("SIGMA7_MJPYTHON", "MJPYTHON", "MJ_PYTHON")
    if explicit is not None:
        return explicit
    discovered = shutil.which("mjpython")
    if discovered:
        return Path(discovered)
    return Path(sys.executable)


def default_native_launcher() -> str:
    if sys.platform == "darwin":
        return str(default_mjpython())
    return str(Path(sys.executable))
