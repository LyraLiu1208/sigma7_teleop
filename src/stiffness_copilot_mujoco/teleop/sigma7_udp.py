from __future__ import annotations

import json
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


SIGMA7_PACKET_SCHEMA_VERSION = "sigma7_udp_jsonl_v1"
SIGMA7_TIMEOUT_MODE_HOLD_LAST_TARGET = "hold_last_target"
SIGMA7_TIMEOUT_MODE_PAUSE = "pause"
SIGMA7_TIMEOUT_MODES = (SIGMA7_TIMEOUT_MODE_HOLD_LAST_TARGET, SIGMA7_TIMEOUT_MODE_PAUSE)


def _as_position(value: Any, *, name: str = "position") -> np.ndarray:
    position = np.asarray(value, dtype=float)
    if position.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), observed {position.shape}.")
    return position


def _json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


@dataclass(frozen=True)
class Sigma7Packet:
    received_timestamp: float | None
    sequence: int | None
    position: np.ndarray
    zero: bool = False
    pause: bool = False
    quit: bool = False
    valid: bool = True
    packet_timestamp: float | None = None
    source: str = "udp"
    raw: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": SIGMA7_PACKET_SCHEMA_VERSION,
            "received_timestamp": self.received_timestamp,
            "sequence": self.sequence,
            "packet_timestamp": self.packet_timestamp,
            "position": _as_position(self.position).tolist(),
            "zero": bool(self.zero),
            "pause": bool(self.pause),
            "quit": bool(self.quit),
            "valid": bool(self.valid),
            "source": self.source,
        }
        if self.raw:
            payload["raw"] = _json_ready(self.raw)
        return payload


@dataclass(frozen=True)
class Sigma7TeleopConfig:
    deadband: float = 0.0005
    max_target_velocity: float = 0.05
    timeout_seconds: float = 0.25
    timeout_mode: str = SIGMA7_TIMEOUT_MODE_HOLD_LAST_TARGET
    workspace_min_delta: tuple[float, float, float] = (-0.03, -0.03, -0.03)
    workspace_max_delta: tuple[float, float, float] = (0.03, 0.03, 0.03)
    zero_on_first_packet: bool = True

    def validate(self) -> None:
        if self.deadband < 0.0:
            raise ValueError("deadband must be non-negative.")
        if self.max_target_velocity < 0.0:
            raise ValueError("max_target_velocity must be non-negative.")
        if self.timeout_seconds < 0.0:
            raise ValueError("timeout_seconds must be non-negative.")
        if self.timeout_mode not in SIGMA7_TIMEOUT_MODES:
            raise ValueError(f"timeout_mode must be one of {SIGMA7_TIMEOUT_MODES!r}, observed {self.timeout_mode!r}.")
        if np.asarray(self.workspace_min_delta, dtype=float).shape != (3,):
            raise ValueError("workspace_min_delta must have shape (3,).")
        if np.asarray(self.workspace_max_delta, dtype=float).shape != (3,):
            raise ValueError("workspace_max_delta must have shape (3,).")


@dataclass(frozen=True)
class Sigma7TeleopSnapshot:
    step_index: int
    time_seconds: float
    packet_sequence: int | None
    packet_timestamp: float | None
    received_timestamp: float | None
    packet_age_seconds: float | None
    raw_position: np.ndarray
    zero_reference_position: np.ndarray
    raw_delta: np.ndarray
    mapped_delta: np.ndarray
    clamped_delta: np.ndarray
    target_position: np.ndarray
    packet_valid: bool
    fresh_packet_received: bool
    timeout_active: bool
    paused: bool
    zeroed: bool
    zero_event: bool
    pause_event: bool
    quit_event: bool
    packet_source: str
    packet_json: str
    timeout_mode: str


class Sigma7UdpReceiver:
    def __init__(self, host: str = "0.0.0.0", port: int = 5005, *, timeout_seconds: float = 0.0) -> None:
        self.host = str(host)
        self.port = int(port)
        self.timeout_seconds = float(timeout_seconds)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind((self.host, self.port))
        self._socket.settimeout(self.timeout_seconds)

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None  # type: ignore[assignment]

    def recv_latest(self) -> Sigma7Packet | None:
        latest: Sigma7Packet | None = None
        while True:
            try:
                payload, _addr = self._socket.recvfrom(65535)
            except (socket.timeout, BlockingIOError):
                break
            packet = parse_sigma7_jsonl_packet(payload)
            packet = Sigma7Packet(
                **{**packet.__dict__, "received_timestamp": time.perf_counter(), "source": "udp"},
            )
            latest = packet
        return latest


@dataclass
class Sigma7TeleopState:
    zero_reference_position: np.ndarray | None = None
    last_packet: Sigma7Packet | None = None
    last_packet_received_timestamp: float | None = None
    last_packet_valid: bool = False
    last_clamped_delta: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    paused: bool = False
    emergency_quit: bool = False
    zeroed: bool = False
    packet_source: str = "udp"


def parse_sigma7_jsonl_packet(payload: bytes | str) -> Sigma7Packet:
    text = payload.decode("utf-8") if isinstance(payload, (bytes, bytearray)) else str(payload)
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Sigma7 packet must decode to a JSON object.")

    raw_position = data.get("position")
    if raw_position is None and "position_m" in data:
        raw_position = data["position_m"]
    if raw_position is None and "position_mm" in data:
        raw_position = np.asarray(data["position_mm"], dtype=float) / 1000.0
    position = _as_position(raw_position, name="position")
    packet_timestamp = data.get("packet_timestamp", data.get("timestamp"))
    sequence = data.get("sequence", data.get("seq"))
    return Sigma7Packet(
        received_timestamp=data.get("received_timestamp"),
        sequence=None if sequence is None else int(sequence),
        position=position,
        zero=bool(data.get("zero", data.get("re_zero", False))),
        pause=bool(data.get("pause", False)),
        quit=bool(data.get("quit", data.get("emergency_quit", False))),
        valid=bool(data.get("valid", True)),
        packet_timestamp=None if packet_timestamp is None else float(packet_timestamp),
        source=str(data.get("source", "udp")),
        raw=dict(data),
    )


def build_sigma7_jsonl_packet(
    *,
    sequence: int,
    position: np.ndarray,
    packet_timestamp: float | None = None,
    received_timestamp: float | None = None,
    zero: bool = False,
    pause: bool = False,
    quit: bool = False,
    source: str = "fake_sender",
) -> str:
    packet = Sigma7Packet(
        received_timestamp=received_timestamp,
        sequence=int(sequence),
        position=_as_position(position),
        zero=bool(zero),
        pause=bool(pause),
        quit=bool(quit),
        valid=True,
        packet_timestamp=packet_timestamp,
        source=source,
    )
    return json.dumps(packet.to_json_dict(), sort_keys=True)


def build_synthetic_sigma7_packet(
    *,
    step: int,
    dt_seconds: float,
    sequence_offset: int = 0,
    zero_step: int = 0,
    re_zero_step: int | None = None,
    pause_start_step: int | None = None,
    pause_end_step: int | None = None,
    timeout_start_step: int | None = None,
    timeout_end_step: int | None = None,
    quit_step: int | None = None,
    position_scale_xy: float = 0.05,
    position_scale_z: float = 0.02,
) -> Sigma7Packet | None:
    if timeout_start_step is not None and timeout_end_step is not None:
        if int(timeout_start_step) <= int(step) < int(timeout_end_step):
            return None
    t = float(step) * float(dt_seconds)
    base_position = np.array(
        [
            position_scale_xy * np.sin(0.35 * t) + 0.012 * np.sin(1.2 * t),
            position_scale_xy * np.cos(0.27 * t) + 0.009 * np.cos(0.9 * t),
            position_scale_z * np.sin(0.19 * t),
        ],
        dtype=float,
    )
    zero = int(step) == int(zero_step) or (re_zero_step is not None and int(step) == int(re_zero_step))
    pause = False
    if pause_start_step is not None and pause_end_step is not None:
        pause = int(pause_start_step) <= int(step) < int(pause_end_step)
    quit_flag = quit_step is not None and int(step) == int(quit_step)
    return Sigma7Packet(
        received_timestamp=t,
        sequence=int(sequence_offset + step),
        position=base_position,
        zero=zero,
        pause=pause,
        quit=quit_flag,
        valid=True,
        packet_timestamp=t,
        source="synthetic",
        raw={
            "step": int(step),
            "dt_seconds": float(dt_seconds),
            "position": base_position.tolist(),
            "zero": zero,
            "pause": pause,
            "quit": quit_flag,
        },
    )


class Sigma7TeleopMapper:
    def __init__(self, config: Sigma7TeleopConfig | None = None) -> None:
        self.config = config or Sigma7TeleopConfig()
        self.config.validate()
        self.state = Sigma7TeleopState()

    def reset(self) -> None:
        self.state = Sigma7TeleopState()

    def update(
        self,
        packet: Sigma7Packet | None,
        *,
        step_index: int,
        now_seconds: float,
        nominal_target_position: np.ndarray,
        control_dt_seconds: float,
    ) -> Sigma7TeleopSnapshot:
        nominal_target_position = _as_position(nominal_target_position, name="nominal_target_position")
        control_dt_seconds = float(control_dt_seconds)
        received_timestamp = self.state.last_packet_received_timestamp
        packet_timestamp = self.state.last_packet.packet_timestamp if self.state.last_packet is not None else None
        packet_sequence = self.state.last_packet.sequence if self.state.last_packet is not None else None
        raw_position = self.state.last_packet.position.copy() if self.state.last_packet is not None else np.zeros(3, dtype=float)
        packet_valid = self.state.last_packet_valid
        fresh_packet_received = False
        zero_event = False
        pause_event = False
        quit_event = False

        if packet is not None:
            fresh_packet_received = True
            packet_valid = bool(packet.valid)
            self.state.last_packet = packet
            self.state.last_packet_received_timestamp = float(now_seconds if packet.received_timestamp is None else packet.received_timestamp)
            received_timestamp = self.state.last_packet_received_timestamp
            packet_timestamp = packet.packet_timestamp
            packet_sequence = packet.sequence
            raw_position = packet.position.copy()
            self.state.packet_source = packet.source
            self.state.last_packet_valid = packet_valid
            if packet_valid:
                if self.state.zero_reference_position is None and self.config.zero_on_first_packet:
                    self.state.zero_reference_position = raw_position.copy()
                    self.state.zeroed = True
                    zero_event = True
                    self.state.last_clamped_delta = np.zeros(3, dtype=float)
                if packet.zero:
                    self.state.zero_reference_position = raw_position.copy()
                    self.state.zeroed = True
                    zero_event = True
                    self.state.last_clamped_delta = np.zeros(3, dtype=float)
                self.state.paused = bool(packet.pause)
                pause_event = bool(packet.pause)
                if packet.quit:
                    self.state.emergency_quit = True
                    quit_event = True

        if self.state.zero_reference_position is None:
            self.state.zero_reference_position = raw_position.copy()
            self.state.zeroed = True
            zero_event = True

        raw_delta = raw_position - self.state.zero_reference_position
        mapped_delta = raw_delta.copy()
        mapped_delta[np.abs(mapped_delta) < float(self.config.deadband)] = 0.0
        workspace_min = np.asarray(self.config.workspace_min_delta, dtype=float)
        workspace_max = np.asarray(self.config.workspace_max_delta, dtype=float)
        workspace_clamped_delta = np.clip(mapped_delta, workspace_min, workspace_max)

        timeout_active = False
        packet_age_seconds: float | None = None
        if self.state.last_packet_received_timestamp is not None:
            packet_age_seconds = float(now_seconds - self.state.last_packet_received_timestamp)
            timeout_active = packet_age_seconds > float(self.config.timeout_seconds)

        commanded_delta = workspace_clamped_delta.copy()
        if self.state.paused:
            commanded_delta = self.state.last_clamped_delta.copy()
        if timeout_active:
            if self.config.timeout_mode == SIGMA7_TIMEOUT_MODE_HOLD_LAST_TARGET:
                commanded_delta = self.state.last_clamped_delta.copy()
            elif self.config.timeout_mode == SIGMA7_TIMEOUT_MODE_PAUSE:
                commanded_delta = np.zeros(3, dtype=float)
                self.state.paused = True

        delta_step = commanded_delta - self.state.last_clamped_delta
        max_step = float(self.config.max_target_velocity) * control_dt_seconds
        if max_step > 0.0 and np.linalg.norm(delta_step) > max_step:
            commanded_delta = self.state.last_clamped_delta + delta_step / np.linalg.norm(delta_step) * max_step

        if not timeout_active and not self.state.paused:
            self.state.last_clamped_delta = commanded_delta.copy()
        elif timeout_active and self.config.timeout_mode == SIGMA7_TIMEOUT_MODE_HOLD_LAST_TARGET:
            commanded_delta = self.state.last_clamped_delta.copy()
        elif self.state.paused:
            commanded_delta = self.state.last_clamped_delta.copy()

        target_position = nominal_target_position + commanded_delta
        return Sigma7TeleopSnapshot(
            step_index=int(step_index),
            time_seconds=float(now_seconds),
            packet_sequence=None if packet_sequence is None else int(packet_sequence),
            packet_timestamp=None if packet_timestamp is None else float(packet_timestamp),
            received_timestamp=None if received_timestamp is None else float(received_timestamp),
            packet_age_seconds=packet_age_seconds,
            raw_position=np.asarray(raw_position, dtype=float).copy(),
            zero_reference_position=np.asarray(self.state.zero_reference_position, dtype=float).copy(),
            raw_delta=np.asarray(raw_delta, dtype=float).copy(),
            mapped_delta=np.asarray(mapped_delta, dtype=float).copy(),
            clamped_delta=np.asarray(commanded_delta, dtype=float).copy(),
            target_position=np.asarray(target_position, dtype=float).copy(),
            packet_valid=bool(packet_valid),
            fresh_packet_received=fresh_packet_received,
            timeout_active=bool(timeout_active),
            paused=bool(self.state.paused),
            zeroed=bool(self.state.zeroed),
            zero_event=bool(zero_event),
            pause_event=bool(pause_event),
            quit_event=bool(quit_event or self.state.emergency_quit),
            packet_source=self.state.packet_source,
            packet_json=json.dumps(_json_ready(packet.to_json_dict() if packet is not None else {}), sort_keys=True),
            timeout_mode=self.config.timeout_mode,
        )


__all__ = [
    "SIGMA7_PACKET_SCHEMA_VERSION",
    "SIGMA7_TIMEOUT_MODE_HOLD_LAST_TARGET",
    "SIGMA7_TIMEOUT_MODE_PAUSE",
    "SIGMA7_TIMEOUT_MODES",
    "Sigma7Packet",
    "Sigma7TeleopConfig",
    "Sigma7TeleopMapper",
    "Sigma7TeleopSnapshot",
    "Sigma7UdpReceiver",
    "build_sigma7_jsonl_packet",
    "build_synthetic_sigma7_packet",
    "parse_sigma7_jsonl_packet",
]
