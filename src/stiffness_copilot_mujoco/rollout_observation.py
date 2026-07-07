from __future__ import annotations

import mujoco
import numpy as np

from stiffness_copilot_mujoco.panda_control import PandaArmIds, arm_qvel, site_jacobians
from stiffness_copilot_mujoco.pose_math import orientation_error, site_rotation
from stiffness_copilot_mujoco.metrics.task_metrics import (
    contact_metrics,
    hole_center_position,
    hole_insertion_axis_world,
    insertion_depth,
    lateral_error,
    peg_axis_alignment,
    peg_tip_position,
)


def reset_from_config(model: mujoco.MjModel, data: mujoco.MjData, config: dict) -> None:
    initial_keyframe = str(config["scene"].get("initial_keyframe", ""))
    if initial_keyframe:
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, initial_keyframe)
        if key_id < 0:
            raise KeyError(f"MuJoCo keyframe {initial_keyframe!r} was not found.")
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)


def _body_rotation(data: mujoco.MjData, body_id: int) -> np.ndarray:
    return np.array(data.xmat[body_id], dtype=float).reshape(3, 3)


def _site_velocity(model: mujoco.MjModel, data: mujoco.MjData, site_name: str, arm_ids: PandaArmIds) -> tuple[np.ndarray, np.ndarray]:
    jacp, jacr = site_jacobians(model, data, site_name, arm_ids)
    qvel = arm_qvel(data, arm_ids)
    return jacp @ qvel, jacr @ qvel


def collect_step(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    arm_ids: PandaArmIds,
    task_ids,
    target_position: np.ndarray,
    target_rotation: np.ndarray,
    phase_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    peg_tip_site_id = model.site("peg_tip").id
    ee_pos = np.array(data.site_xpos[peg_tip_site_id], dtype=float)
    ee_rot = site_rotation(data, peg_tip_site_id)
    ee_lin_vel, ee_ang_vel = _site_velocity(model, data, "peg_tip", arm_ids)
    peg_pos = np.array(data.xpos[task_ids.peg_body], dtype=float)
    peg_rot = _body_rotation(data, task_ids.peg_body)
    hole_pos = hole_center_position(data, task_ids)
    rel = peg_tip_position(data, task_ids) - hole_pos
    contacts = contact_metrics(model, data, task_ids)
    contact_state = 2.0 if contacts.peg_hole_contact_count > 0 else 1.0 if contacts.peg_contact_count > 0 else 0.0
    orient_vec = orientation_error(ee_rot, target_rotation)
    orient_norm = float(np.linalg.norm(orient_vec))
    insert_depth = insertion_depth(data, task_ids)
    lateral = lateral_error(data, task_ids)
    axis = peg_axis_alignment(data, task_ids, hole_insertion_axis_world())
    obs = np.concatenate(
        [
            ee_pos,
            ee_rot.reshape(-1),
            ee_lin_vel,
            ee_ang_vel,
            peg_pos,
            peg_rot.reshape(-1),
            hole_pos,
            rel,
            np.array(
                [
                    insert_depth,
                    lateral,
                    axis,
                    orient_norm,
                    contact_state,
                    contacts.peg_normal_force,
                ],
                dtype=float,
            ),
        ]
    )
    u_ref = np.concatenate(
        [
            target_position,
            target_rotation.reshape(-1),
            target_position - ee_pos,
            orient_vec,
            np.array([float(phase_id)], dtype=float),
        ]
    )
    min_dist = np.nan if contacts.min_peg_contact_dist is None else contacts.min_peg_contact_dist
    info = np.array(
        [
            insert_depth,
            lateral,
            axis,
            orient_norm,
            contacts.peg_contact_count,
            contacts.peg_hole_contact_count,
            contacts.peg_normal_force,
            min_dist,
            contact_state,
        ],
        dtype=float,
    )
    return obs, u_ref, info
