from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


StiffnessSmoothingMethod = Literal["diagonal_ema", "log_spd_ema"]


def _as_spd_matrix(matrix: np.ndarray, *, name: str) -> np.ndarray:
    value = np.asarray(matrix, dtype=float)
    if value.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3), got {value.shape}.")
    value = 0.5 * (value + value.T)
    eigvals = np.linalg.eigvalsh(value)
    if np.any(eigvals <= 0.0):
        raise ValueError(f"{name} must be SPD, got eigenvalues {eigvals}.")
    return value


def _diag_summary(diag_rows: list[np.ndarray]) -> dict[str, object]:
    if not diag_rows:
        zeros = [0.0, 0.0, 0.0]
        return {
            "count": 0,
            "diag_mean": zeros,
            "diag_min": zeros,
            "diag_max": zeros,
        }
    stacked = np.vstack(diag_rows)
    return {
        "count": int(stacked.shape[0]),
        "diag_mean": [float(v) for v in np.mean(stacked, axis=0)],
        "diag_min": [float(v) for v in np.min(stacked, axis=0)],
        "diag_max": [float(v) for v in np.max(stacked, axis=0)],
    }


@dataclass(frozen=True)
class StiffnessCommandSmoothingConfig:
    enabled: bool = False
    method: StiffnessSmoothingMethod = "diagonal_ema"
    alpha: float = 0.2
    policy_update_period_steps: int | None = 1
    update_rate_hz: float | None = None
    hold_between_updates: bool = True
    target_kind: str = "stiffness_command"
    eps: float = 1e-8

    def validate(self) -> None:
        if not 0.0 <= float(self.alpha) <= 1.0:
            raise ValueError(f"alpha must be within [0, 1], got {self.alpha!r}.")
        if self.policy_update_period_steps is not None and int(self.policy_update_period_steps) <= 0:
            raise ValueError("policy_update_period_steps must be positive when provided.")
        if self.update_rate_hz is not None and float(self.update_rate_hz) <= 0.0:
            raise ValueError("update_rate_hz must be positive when provided.")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive.")
        if self.method not in {"diagonal_ema", "log_spd_ema"}:
            raise ValueError(f"Unsupported stiffness smoothing method {self.method!r}.")

    def resolved_policy_update_period_steps(self, *, simulation_dt_seconds: float | None = None) -> int:
        self.validate()
        if self.policy_update_period_steps is not None:
            return max(1, int(self.policy_update_period_steps))
        if self.update_rate_hz is None:
            return 1
        if simulation_dt_seconds is None or simulation_dt_seconds <= 0.0:
            raise ValueError("simulation_dt_seconds must be positive when resolving update_rate_hz.")
        return max(1, int(round(1.0 / (simulation_dt_seconds * float(self.update_rate_hz)))))

    def to_dict(self, *, simulation_dt_seconds: float | None = None) -> dict[str, object]:
        resolved_steps = self.resolved_policy_update_period_steps(simulation_dt_seconds=simulation_dt_seconds)
        effective_update_rate_hz = None if self.update_rate_hz is None else float(self.update_rate_hz)
        if self.update_rate_hz is not None:
            if simulation_dt_seconds is not None and simulation_dt_seconds > 0.0:
                effective_steps = max(1, int(round(1.0 / (simulation_dt_seconds * float(self.update_rate_hz)))))
                effective_seconds = float(1.0 / float(self.update_rate_hz))
            else:
                effective_steps = resolved_steps
                effective_seconds = float(1.0 / float(self.update_rate_hz))
        else:
            effective_steps = resolved_steps
            effective_seconds = None if simulation_dt_seconds is None or simulation_dt_seconds <= 0.0 else float(simulation_dt_seconds * resolved_steps)
        payload: dict[str, object] = {
            "enabled": bool(self.enabled),
            "method": self.method,
            "alpha": float(self.alpha),
            "policy_update_period_steps": resolved_steps,
            "policy_update_period_steps_requested": None if self.policy_update_period_steps is None else int(self.policy_update_period_steps),
            "effective_policy_update_period_steps": int(effective_steps),
            "effective_policy_update_period_seconds": None if effective_seconds is None else float(effective_seconds),
            "update_rate_hz": None if self.update_rate_hz is None else float(self.update_rate_hz),
            "stiffness_update_hz_target": effective_update_rate_hz,
            "scheduler": "time_accumulator" if self.update_rate_hz is not None else "fixed_period",
            "hold_between_updates": bool(self.hold_between_updates),
            "target_kind": self.target_kind,
        }
        if simulation_dt_seconds is not None and simulation_dt_seconds > 0.0:
            payload["simulation_dt_seconds"] = float(simulation_dt_seconds)
            payload["control_rate_hz"] = float(1.0 / simulation_dt_seconds)
        return payload


def deployment_stiffness_smoothing_config(
    *,
    alpha: float = 0.2,
    policy_update_period_steps: int = 6,
    update_rate_hz: float | None = 90.0,
    hold_between_updates: bool = True,
    target_kind: str = "stiffness_command",
    eps: float = 1e-8,
) -> StiffnessCommandSmoothingConfig:
    return StiffnessCommandSmoothingConfig(
        enabled=True,
        method="log_spd_ema",
        alpha=alpha,
        policy_update_period_steps=policy_update_period_steps,
        update_rate_hz=update_rate_hz,
        hold_between_updates=hold_between_updates,
        target_kind=target_kind,
        eps=eps,
    )


def resolve_deployment_stiffness_smoothing_config(
    config: StiffnessCommandSmoothingConfig | None = None,
    *,
    alpha: float = 0.2,
    policy_update_period_steps: int = 6,
    update_rate_hz: float | None = 90.0,
    hold_between_updates: bool = True,
    target_kind: str = "stiffness_command",
    eps: float = 1e-8,
) -> StiffnessCommandSmoothingConfig:
    if config is None or not config.enabled:
        return deployment_stiffness_smoothing_config(
            alpha=alpha,
            policy_update_period_steps=policy_update_period_steps,
            update_rate_hz=update_rate_hz,
            hold_between_updates=hold_between_updates,
            target_kind=target_kind,
            eps=eps,
        )
    return config


@dataclass(frozen=True)
class StiffnessCommandStepResult:
    command_matrix: np.ndarray
    update_applied: bool
    hold_applied: bool
    raw_matrix_before_smoothing: np.ndarray | None = None
    smoothed_matrix: np.ndarray | None = None
    scheduler: str | None = None
    stiffness_update_hz_target: float | None = None
    stiffness_update_index: int | None = None
    stiffness_update_interval_steps: int | None = None
    stiffness_update_interval_seconds: float | None = None
    steps_since_last_stiffness_refresh: int | None = None


class StiffnessCommandSmoother:
    def __init__(
        self,
        config: StiffnessCommandSmoothingConfig | None = None,
        *,
        simulation_dt_seconds: float | None = None,
    ) -> None:
        self.config = config or StiffnessCommandSmoothingConfig()
        self.config.validate()
        self.simulation_dt_seconds = simulation_dt_seconds
        self.policy_update_period_steps = self.config.resolved_policy_update_period_steps(
            simulation_dt_seconds=simulation_dt_seconds
        )
        if self.config.update_rate_hz is not None and (simulation_dt_seconds is None or simulation_dt_seconds <= 0.0):
            raise ValueError("simulation_dt_seconds must be positive when update_rate_hz is provided.")
        self.scheduler = "time_accumulator" if self.config.update_rate_hz is not None else "fixed_period"
        self.stiffness_update_hz_target = None if self.config.update_rate_hz is None else float(self.config.update_rate_hz)
        self._initialized = False
        self._command_matrix: np.ndarray | None = None
        self._last_refresh_step: int | None = None
        self._last_refresh_time_seconds: float | None = None
        self._next_refresh_time_seconds: float | None = 0.0 if self.config.update_rate_hz is not None else None
        self._last_refresh_interval_steps: int | None = None
        self._last_refresh_interval_seconds: float | None = None
        self._before_diag_rows: list[np.ndarray] = []
        self._after_diag_rows: list[np.ndarray] = []
        self._diag_delta_rows: list[np.ndarray] = []
        self._step_count = 0
        self._update_count = 0
        self._hold_count = 0

    def apply(self, *, step: int, target_matrix: np.ndarray) -> StiffnessCommandStepResult:
        target_spd = _as_spd_matrix(target_matrix, name="target_matrix")
        target_diag = np.diag(target_spd).astype(float, copy=False)
        self._before_diag_rows.append(target_diag.copy())
        self._step_count += 1
        current_time_seconds = None if self.simulation_dt_seconds is None else float(step) * float(self.simulation_dt_seconds)
        steps_since_last_refresh = 0 if self._last_refresh_step is None else int(step - self._last_refresh_step)
        seconds_since_last_refresh = (
            None
            if current_time_seconds is None or self._last_refresh_time_seconds is None
            else float(current_time_seconds - self._last_refresh_time_seconds)
        )

        if not self.config.enabled:
            command_spd = target_spd.copy()
            result = StiffnessCommandStepResult(
                command_matrix=command_spd,
                update_applied=False,
                hold_applied=False,
                raw_matrix_before_smoothing=target_spd.copy(),
                smoothed_matrix=command_spd.copy(),
                scheduler="disabled",
                stiffness_update_hz_target=self.stiffness_update_hz_target,
                stiffness_update_index=0,
                stiffness_update_interval_steps=0,
                stiffness_update_interval_seconds=0.0 if current_time_seconds is not None else None,
                steps_since_last_stiffness_refresh=0,
            )
            self._record_after(command_spd)
            self._command_matrix = command_spd
            self._initialized = True
            self._last_refresh_step = step
            self._last_refresh_time_seconds = current_time_seconds
            return result

        if not self._initialized or self._command_matrix is None:
            should_refresh = True
        elif self.config.update_rate_hz is not None:
            assert self._next_refresh_time_seconds is not None
            should_refresh = bool(
                (not self.config.hold_between_updates)
                or (current_time_seconds is not None and current_time_seconds + 1e-12 >= self._next_refresh_time_seconds)
            )
        else:
            should_refresh = (
                self._last_refresh_step is None
                or (step - self._last_refresh_step) >= self.policy_update_period_steps
                or not self.config.hold_between_updates
            )
        if should_refresh:
            if not self._initialized or self._command_matrix is None:
                command_spd = target_spd.copy()
            elif self.config.method == "diagonal_ema":
                previous_diag = np.diag(self._command_matrix).astype(float, copy=False)
                smoothed_diag = (1.0 - float(self.config.alpha)) * previous_diag + float(self.config.alpha) * target_diag
                command_spd = np.diag(smoothed_diag)
            else:
                from stiffness_copilot_mujoco.learning.supervised_policy import log_euclidean_ema_spd

                command_spd = log_euclidean_ema_spd(
                    self._command_matrix,
                    target_spd,
                    alpha=float(self.config.alpha),
                    eps=float(self.config.eps),
                )
            self._command_matrix = _as_spd_matrix(command_spd, name="command_matrix")
            self._initialized = True
            previous_refresh_step = self._last_refresh_step
            previous_refresh_time = self._last_refresh_time_seconds
            self._last_refresh_step = step
            self._last_refresh_time_seconds = current_time_seconds
            if self.config.update_rate_hz is not None:
                if self._next_refresh_time_seconds is None:
                    self._next_refresh_time_seconds = 1.0 / float(self.config.update_rate_hz)
                else:
                    self._next_refresh_time_seconds += 1.0 / float(self.config.update_rate_hz)
            self._last_refresh_interval_steps = None if previous_refresh_step is None else int(step - previous_refresh_step)
            self._last_refresh_interval_seconds = None if previous_refresh_time is None or current_time_seconds is None else float(current_time_seconds - previous_refresh_time)
            self._update_count += 1
            result = StiffnessCommandStepResult(
                command_matrix=self._command_matrix.copy(),
                update_applied=True,
                hold_applied=False,
                raw_matrix_before_smoothing=target_spd.copy(),
                smoothed_matrix=self._command_matrix.copy(),
                scheduler=self.scheduler,
                stiffness_update_hz_target=self.stiffness_update_hz_target,
                stiffness_update_index=int(self._update_count),
                stiffness_update_interval_steps=self._last_refresh_interval_steps,
                stiffness_update_interval_seconds=self._last_refresh_interval_seconds,
                steps_since_last_stiffness_refresh=steps_since_last_refresh,
            )
            self._record_after(self._command_matrix)
            return result

        self._hold_count += 1
        assert self._command_matrix is not None  # established by initialized path above
        result = StiffnessCommandStepResult(
            command_matrix=self._command_matrix.copy(),
            update_applied=False,
            hold_applied=True,
            raw_matrix_before_smoothing=target_spd.copy(),
            smoothed_matrix=self._command_matrix.copy(),
            scheduler=self.scheduler,
            stiffness_update_hz_target=self.stiffness_update_hz_target,
            stiffness_update_index=int(self._update_count),
            stiffness_update_interval_steps=self._last_refresh_interval_steps,
            stiffness_update_interval_seconds=self._last_refresh_interval_seconds,
            steps_since_last_stiffness_refresh=steps_since_last_refresh,
        )
        self._record_after(self._command_matrix)
        return result

    def _record_after(self, command_matrix: np.ndarray) -> None:
        after_diag = np.diag(_as_spd_matrix(command_matrix, name="command_matrix")).astype(float, copy=False)
        self._after_diag_rows.append(after_diag.copy())
        self._diag_delta_rows.append(np.abs(after_diag - self._before_diag_rows[-1]))

    def summary_dict(self) -> dict[str, object]:
        if self._diag_delta_rows:
            delta = np.vstack(self._diag_delta_rows)
            delta_summary = {
                "mean_abs_diag_delta": [float(v) for v in np.mean(delta, axis=0)],
                "max_abs_diag_delta": [float(v) for v in np.max(delta, axis=0)],
            }
        else:
            zeros = [0.0, 0.0, 0.0]
            delta_summary = {
                "mean_abs_diag_delta": zeros,
                "max_abs_diag_delta": zeros,
            }
        return {
            "config": self.config.to_dict(simulation_dt_seconds=self.simulation_dt_seconds),
            "step_count": int(self._step_count),
            "update_count": int(self._update_count),
            "hold_count": int(self._hold_count),
            "scheduler": self.scheduler,
            "stiffness_update_hz_target": self.stiffness_update_hz_target,
            "last_refresh_step": None if self._last_refresh_step is None else int(self._last_refresh_step),
            "last_refresh_time_seconds": self._last_refresh_time_seconds,
            "last_refresh_interval_steps": self._last_refresh_interval_steps,
            "last_refresh_interval_seconds": self._last_refresh_interval_seconds,
            "stiffness_before_smoothing_summary": _diag_summary(self._before_diag_rows),
            "stiffness_after_smoothing_summary": _diag_summary(self._after_diag_rows),
            **delta_summary,
        }


def summary_fields_from_smoothing_summary(summary: dict[str, object]) -> dict[str, object]:
    config = dict(summary["config"])
    return {
        "smoothing_enabled": bool(config["enabled"]),
        "smoothing_method": str(config["method"]),
        "smoothing_alpha": float(config["alpha"]),
        "policy_update_period_steps": int(config["policy_update_period_steps"]),
        "policy_update_period_steps_requested": None
        if config.get("policy_update_period_steps_requested") is None
        else int(config["policy_update_period_steps_requested"]),
        "effective_policy_update_period_steps": int(config["effective_policy_update_period_steps"]),
        "effective_policy_update_period_seconds": None
        if config.get("effective_policy_update_period_seconds") is None
        else float(config["effective_policy_update_period_seconds"]),
        "update_rate_hz": None if config.get("update_rate_hz") is None else float(config["update_rate_hz"]),
        "stiffness_update_hz_target": None
        if config.get("stiffness_update_hz_target") is None
        else float(config["stiffness_update_hz_target"]),
        "scheduler": str(config.get("scheduler", "fixed_period")),
        "hold_between_updates": bool(config["hold_between_updates"]),
        "smoothing_target_kind": str(config["target_kind"]),
        "stiffness_before_smoothing_summary": dict(summary["stiffness_before_smoothing_summary"]),
        "stiffness_after_smoothing_summary": dict(summary["stiffness_after_smoothing_summary"]),
        "step_count": int(summary.get("step_count", 0)),
        "update_count": int(summary.get("update_count", 0)),
        "hold_count": int(summary.get("hold_count", 0)),
        "last_refresh_step": summary.get("last_refresh_step"),
        "last_refresh_time_seconds": summary.get("last_refresh_time_seconds"),
        "last_refresh_interval_steps": summary.get("last_refresh_interval_steps"),
        "last_refresh_interval_seconds": summary.get("last_refresh_interval_seconds"),
    }


def disabled_smoothing_summary_fields() -> dict[str, object]:
    smoother = StiffnessCommandSmoother(StiffnessCommandSmoothingConfig(enabled=False))
    return summary_fields_from_smoothing_summary(smoother.summary_dict())


__all__ = [
    "disabled_smoothing_summary_fields",
    "deployment_stiffness_smoothing_config",
    "summary_fields_from_smoothing_summary",
    "resolve_deployment_stiffness_smoothing_config",
    "StiffnessCommandSmoother",
    "StiffnessCommandSmoothingConfig",
    "StiffnessCommandStepResult",
]
