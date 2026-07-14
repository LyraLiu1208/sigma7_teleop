from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


class LiveMetricWindow:
    def __init__(
        self,
        *,
        python_path: Path,
        script_path: Path,
        kind: str,
        title: str,
        window_seconds: float,
        draw_stride: int = 1,
    ) -> None:
        if not python_path.exists():
            raise FileNotFoundError(f"Metric viewer Python not found: {python_path}")
        if not script_path.exists():
            raise FileNotFoundError(f"Metric viewer script not found: {script_path}")
        self._proc = subprocess.Popen(
            [
                str(python_path),
                "-u",
                str(script_path),
                "--kind",
                str(kind),
                "--title",
                str(title),
                "--window-seconds",
                str(float(window_seconds)),
                "--draw-stride",
                str(max(1, int(draw_stride))),
            ],
            stdin=subprocess.PIPE,
        )
        if self._proc.stdin is None:
            raise RuntimeError("Failed to open metric window stdin.")
        self._stdin = self._proc.stdin
        self._closed = False

    def send(self, payload: dict[str, Any]) -> None:
        if self._closed:
            return
        if self._proc.poll() is not None:
            self._closed = True
            return
        try:
            self._stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
            self._stdin.flush()
        except Exception:
            self._closed = True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._stdin.write(json.dumps({"close": True}) + "\n")
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
