from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from stiffness_copilot_mujoco.sim.ids import PegHoleIds


@dataclass(frozen=True)
class ContactState:
    in_contact: bool
    normal_force: float
    tangential_force: float
    contact_point_world: np.ndarray
    contact_normal_world: np.ndarray
    penetration_depth: float
    contact_velocity: float


@dataclass(frozen=True)
class ContactQuery:
    model: mujoco.MjModel
    data: mujoco.MjData
    task_ids: PegHoleIds


def _as_query(sim) -> ContactQuery:
    if isinstance(sim, ContactQuery):
        return sim
    if isinstance(sim, tuple) and len(sim) == 3:
        return ContactQuery(model=sim[0], data=sim[1], task_ids=sim[2])
    if isinstance(sim, dict):
        return ContactQuery(model=sim["model"], data=sim["data"], task_ids=sim["task_ids"])
    return ContactQuery(model=sim.model, data=sim.data, task_ids=sim.task_ids)


def _body_linear_velocity(model: mujoco.MjModel, data: mujoco.MjData, body_id: int) -> np.ndarray:
    velocity = np.zeros(6, dtype=float)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, velocity, 0)
    return velocity[3:6]


def _zero_contact() -> ContactState:
    return ContactState(
        in_contact=False,
        normal_force=0.0,
        tangential_force=0.0,
        contact_point_world=np.zeros(3, dtype=float),
        contact_normal_world=np.zeros(3, dtype=float),
        penetration_depth=0.0,
        contact_velocity=0.0,
    )


def extract_contact_state(sim) -> ContactState:
    query = _as_query(sim)
    model = query.model
    data = query.data
    ids = query.task_ids
    hole_geoms = set(ids.hole_wall_geoms)
    peg_geoms = set(ids.peg_geoms)
    best: ContactState | None = None
    best_force = -1.0

    for contact_idx in range(data.ncon):
        contact = data.contact[contact_idx]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        if geom1 not in peg_geoms and geom2 not in peg_geoms:
            continue
        if geom1 not in hole_geoms and geom2 not in hole_geoms:
            continue

        force_local = np.zeros(6, dtype=float)
        mujoco.mj_contactForce(model, data, contact_idx, force_local)
        frame = np.asarray(contact.frame, dtype=float).reshape(3, 3)
        normal_world = frame[0].copy()
        normal_norm = float(np.linalg.norm(normal_world))
        if normal_norm > 0.0:
            normal_world /= normal_norm

        normal_force = abs(float(force_local[0]))
        tangential_force = float(np.linalg.norm(force_local[1:3]))
        contact_point = np.asarray(contact.pos, dtype=float).copy()
        penetration_depth = max(0.0, -float(contact.dist))

        body1 = int(model.geom_bodyid[geom1])
        body2 = int(model.geom_bodyid[geom2])
        vel1 = _body_linear_velocity(model, data, body1)
        vel2 = _body_linear_velocity(model, data, body2)
        contact_velocity = float(np.dot(vel2 - vel1, normal_world))

        state = ContactState(
            in_contact=True,
            normal_force=normal_force,
            tangential_force=tangential_force,
            contact_point_world=contact_point,
            contact_normal_world=normal_world,
            penetration_depth=penetration_depth,
            contact_velocity=contact_velocity,
        )
        if normal_force > best_force:
            best = state
            best_force = normal_force

    return best if best is not None else _zero_contact()


def extract_net_peg_hole_contact_force_world(sim) -> np.ndarray:
    query = _as_query(sim)
    model = query.model
    data = query.data
    ids = query.task_ids
    hole_geoms = set(ids.hole_wall_geoms)
    peg_geoms = set(ids.peg_geoms)
    net_force = np.zeros(3, dtype=float)

    for contact_idx in range(data.ncon):
        contact = data.contact[contact_idx]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        if geom1 not in peg_geoms and geom2 not in peg_geoms:
            continue
        if geom1 not in hole_geoms and geom2 not in hole_geoms:
            continue

        force_local = np.zeros(6, dtype=float)
        mujoco.mj_contactForce(model, data, contact_idx, force_local)
        frame = np.asarray(contact.frame, dtype=float).reshape(3, 3)
        force_world = force_local[:3] @ frame
        if geom2 in peg_geoms:
            force_world = -force_world
        net_force += force_world

    return net_force


def contact_state_vector(state: ContactState) -> np.ndarray:
    return np.concatenate(
        [
            np.array(
                [
                    float(state.in_contact),
                    state.normal_force,
                    state.tangential_force,
                ],
                dtype=float,
            ),
            state.contact_point_world,
            state.contact_normal_world,
            np.array([state.penetration_depth, state.contact_velocity], dtype=float),
        ]
    )
