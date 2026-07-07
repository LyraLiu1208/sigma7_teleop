from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import mujoco
import mujoco.viewer

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from stiffness_copilot_mujoco.paths import PANDA_SCENE
from stiffness_copilot_mujoco.sim.scene import DEFAULT_CONFIG, render_config_file


def load_model(scene_path: Path) -> mujoco.MjModel:
    try:
        return mujoco.MjModel.from_xml_path(str(scene_path))
    except ValueError:
        xml = scene_path.read_text(encoding="utf-8")
        if '<include file="panda.xml"/>' not in xml:
            raise
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".xml",
            dir=PANDA_SCENE.parent,
            encoding="utf-8",
            delete=True,
        ) as handle:
            handle.write(xml)
            handle.flush()
            return mujoco.MjModel.from_xml_path(handle.name)


def launch_viewer(scene_path: Path, *, duration: float | None = None) -> None:
    if not scene_path.exists():
        raise FileNotFoundError(
            f"Franka scene not found: {scene_path}. Fetch MuJoCo Menagerie's Franka folders first."
        )

    model = load_model(scene_path)
    data = mujoco.MjData(model)

    if duration is None:
        mujoco.viewer.launch(model, data)
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        deadline = time.time() + duration
        while viewer.is_running() and time.time() < deadline:
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Open a MuJoCo viewer for a Menagerie Franka scene.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="YAML scene config. Defaults to configs/scenes/panda_peg_in_hole_torque.yaml.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        help="Optional duration in seconds. If omitted, the viewer stays open until closed.",
    )
    args = parser.parse_args(argv)

    scene = render_config_file(args.config)
    print(f"rendered scene: {scene}")
    launch_viewer(scene, duration=args.duration)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
