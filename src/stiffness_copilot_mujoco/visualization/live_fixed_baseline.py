from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from stiffness_copilot_mujoco.rollouts.fixed_impedance import BaselineName, RolloutConfig, run_fixed_stiffness_episode


@dataclass
class LiveViewerCallback:
    print_every: int
    viewer: object | None = None

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def __call__(self, step: int, phase: str, model: mujoco.MjModel, data: mujoco.MjData, metrics: dict) -> None:
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(model, data)
        if not self.viewer.is_running():
            return
        self.viewer.sync()
        if self.print_every and step % self.print_every == 0:
            print(
                f"step={step} phase={phase} "
                f"lat={metrics['lateral_error']:.6f} depth={metrics['depth']:.6f} "
                f"normal={metrics['normal_force']:.3f} max_tau={metrics['max_abs_commanded_torque']:.3f}"
            )
        time.sleep(float(model.opt.timestep))


def view_fixed_stiffness_baseline(
    *,
    baseline: BaselineName,
    seed: int,
    xy_offset: np.ndarray | None = None,
    duration: float | None = None,
    print_every: int = 250,
    config_path: Path | None = None,
    gain_config_path: Path | None = None,
) -> None:
    base = RolloutConfig()
    max_steps = int(duration / 0.002) if duration is not None else None
    config = RolloutConfig(
        config_path=config_path or base.config_path,
        gain_config_path=gain_config_path or base.gain_config_path,
        approach_hold_steps=base.approach_hold_steps,
        descend_steps=base.descend_steps,
        insert_steps=base.insert_steps,
        final_hold_steps=base.final_hold_steps,
        approach_height=base.approach_height,
        descend_height=base.descend_height,
        insert_depth=base.insert_depth,
        site_name=base.site_name,
        print_every=0,
        max_steps=max_steps,
    )
    if xy_offset is None:
        rng = np.random.default_rng(seed)
        radius = rng.uniform(0.0, 0.018)
        angle = rng.uniform(-np.pi, np.pi)
        xy_offset = radius * np.array([np.cos(angle), np.sin(angle)], dtype=float)
    callback = LiveViewerCallback(print_every=print_every)
    try:
        summary = run_fixed_stiffness_episode(
            baseline=baseline,
            seed=seed,
            xy_offset=np.asarray(xy_offset, dtype=float),
            config=config,
            step_callback=callback,
        )
    finally:
        callback.close()
    print(
        f"baseline={summary.baseline} profile={summary.profile} "
        f"final_depth={summary.final_depth:.6f} final_lateral={summary.final_lateral_error:.6f} "
        f"contact={summary.contact_detected} max_normal={summary.max_normal_force:.3f}"
    )

