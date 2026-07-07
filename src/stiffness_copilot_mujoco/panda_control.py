from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


ARM_JOINT_NAMES = tuple(f"joint{idx}" for idx in range(1, 8))
ARM_ACTUATOR_NAMES = tuple(f"actuator{idx}" for idx in range(1, 8))


@dataclass(frozen=True)
class PandaArmIds:
    joint_ids: tuple[int, ...]
    actuator_ids: tuple[int, ...]
    qpos_addrs: tuple[int, ...]
    qvel_addrs: tuple[int, ...]


def _required_id(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int:
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id < 0:
        raise KeyError(f"MuJoCo object {name!r} of type {obj_type!r} was not found.")
    return int(obj_id)


def panda_arm_ids(model: mujoco.MjModel) -> PandaArmIds:
    joint_ids = tuple(_required_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ARM_JOINT_NAMES)
    actuator_ids = tuple(_required_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in ARM_ACTUATOR_NAMES)

    for joint_id, actuator_id in zip(joint_ids, actuator_ids, strict=True):
        actuator_joint_id = int(model.actuator_trnid[actuator_id, 0])
        if actuator_joint_id != joint_id:
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
            raise RuntimeError(f"Actuator {actuator_name!r} is not attached to joint {joint_name!r}.")

    return PandaArmIds(
        joint_ids=joint_ids,
        actuator_ids=actuator_ids,
        qpos_addrs=tuple(int(model.jnt_qposadr[joint_id]) for joint_id in joint_ids),
        qvel_addrs=tuple(int(model.jnt_dofadr[joint_id]) for joint_id in joint_ids),
    )


def arm_qpos(data: mujoco.MjData, ids: PandaArmIds) -> np.ndarray:
    return np.array([data.qpos[addr] for addr in ids.qpos_addrs], dtype=float)


def arm_qvel(data: mujoco.MjData, ids: PandaArmIds) -> np.ndarray:
    return np.array([data.qvel[addr] for addr in ids.qvel_addrs], dtype=float)


def set_arm_position_ctrl(model: mujoco.MjModel, data: mujoco.MjData, ids: PandaArmIds, q_target: np.ndarray) -> None:
    q_target = np.asarray(q_target, dtype=float)
    if q_target.shape != (7,):
        raise ValueError(f"Expected q_target shape (7,), got {q_target.shape}.")
    for idx, actuator_id in enumerate(ids.actuator_ids):
        low, high = model.actuator_ctrlrange[actuator_id]
        data.ctrl[actuator_id] = np.clip(q_target[idx], low, high)


def set_arm_torque_ctrl(model: mujoco.MjModel, data: mujoco.MjData, ids: PandaArmIds, torque: np.ndarray) -> None:
    torque = np.asarray(torque, dtype=float)
    if torque.shape != (7,):
        raise ValueError(f"Expected torque shape (7,), got {torque.shape}.")
    for idx, actuator_id in enumerate(ids.actuator_ids):
        low, high = model.actuator_ctrlrange[actuator_id]
        data.ctrl[actuator_id] = np.clip(torque[idx], low, high)


def hold_current_arm_position(model: mujoco.MjModel, data: mujoco.MjData, ids: PandaArmIds) -> None:
    set_arm_position_ctrl(model, data, ids, arm_qpos(data, ids))


def site_jacobians(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_name: str,
    ids: PandaArmIds,
) -> tuple[np.ndarray, np.ndarray]:
    site_id = _required_id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    jacp = np.zeros((3, model.nv), dtype=float)
    jacr = np.zeros((3, model.nv), dtype=float)
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    return jacp[:, ids.qvel_addrs], jacr[:, ids.qvel_addrs]


def site_jacobian(model: mujoco.MjModel, data: mujoco.MjData, site_name: str, ids: PandaArmIds) -> np.ndarray:
    jacp, _ = site_jacobians(model, data, site_name, ids)
    return jacp


def site_rot_jacobian(model: mujoco.MjModel, data: mujoco.MjData, site_name: str, ids: PandaArmIds) -> np.ndarray:
    _, jacr = site_jacobians(model, data, site_name, ids)
    return jacr


def site_velocity(model: mujoco.MjModel, data: mujoco.MjData, site_name: str, ids: PandaArmIds) -> np.ndarray:
    return site_jacobian(model, data, site_name, ids) @ arm_qvel(data, ids)
