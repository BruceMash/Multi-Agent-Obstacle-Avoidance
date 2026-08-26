"""Evaluation-only timing hooks for the frozen online planning/control chain.

The hooks neither alter inputs nor outputs.  They synchronize CUDA around actor
and GAT calls when CUDA is available, distinguish FP-SHEP preview inference from
real execution inference, and time only ``compute_dmp_transition`` for the DMP
component (not environment stepping, sensing, collision checks, or I/O).
"""

from __future__ import annotations

import contextlib
import time
from typing import Any, Iterator, Mapping

import torch


def synchronize_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class OnlineRuntimeRecorder:
    def __init__(self) -> None:
        self.actor_rows: list[dict[str, Any]] = []
        self.dmp_rows: list[dict[str, Any]] = []
        self.upper_rows: list[dict[str, Any]] = []
        self._actor_mode: str | None = None
        self._dmp_mode: str | None = None
        self._context: dict[str, Any] = {}
        self._original_compute_dmp_transition: Any | None = None
        self._original_historical_dmp_transition: Any | None = None

    @property
    def context(self) -> dict[str, Any]:
        return dict(self._context)

    @contextlib.contextmanager
    def scoped_context(self, **values: Any) -> Iterator[None]:
        previous = self._context
        self._context = {**previous, **values}
        try:
            yield
        finally:
            self._context = previous

    @contextlib.contextmanager
    def actor_mode(self, mode: str) -> Iterator[None]:
        previous = self._actor_mode
        self._actor_mode = str(mode)
        try:
            yield
        finally:
            self._actor_mode = previous

    @contextlib.contextmanager
    def dmp_mode(self, mode: str) -> Iterator[None]:
        previous = self._dmp_mode
        self._dmp_mode = str(mode)
        try:
            yield
        finally:
            self._dmp_mode = previous

    def record_actor(self, runtime_ms: float, input_shape: Any) -> None:
        if self._actor_mode is None:
            return
        self.actor_rows.append(
            {
                **self._context,
                "actor_mode": self._actor_mode,
                "call_index": len(self.actor_rows),
                "runtime_ms": float(runtime_ms),
                "input_shape": list(input_shape),
                "cuda_synchronized": bool(torch.cuda.is_available()),
            }
        )

    def record_upper_event(self, row: Mapping[str, Any]) -> None:
        self.upper_rows.append({**self._context, **dict(row)})

    @contextlib.contextmanager
    def instrument_dmp(self) -> Iterator[None]:
        import Environment.frozen_sac_dmp_execution as execution
        import planning.historical_forcing_gate as historical

        if self._original_compute_dmp_transition is not None:
            raise RuntimeError("DMP timing instrumentation is already active")
        original = execution.compute_dmp_transition
        original_historical = historical.compute_historical_checkpoint_dmp_transition
        self._original_compute_dmp_transition = original
        self._original_historical_dmp_transition = original_historical

        def timed_call(function: Any, *args: Any, **kwargs: Any) -> Any:
            if self._dmp_mode is None:
                return function(*args, **kwargs)
            started = time.perf_counter_ns()
            result = function(*args, **kwargs)
            runtime_ms = (time.perf_counter_ns() - started) / 1.0e6
            self.dmp_rows.append(
                {
                    **self._context,
                    "dmp_mode": self._dmp_mode,
                    "call_index": len(self.dmp_rows),
                    "runtime_ms": float(runtime_ms),
                }
            )
            return result

        def timed_compute_dmp_transition(*args: Any, **kwargs: Any) -> Any:
            return timed_call(original, *args, **kwargs)

        def timed_historical_dmp_transition(*args: Any, **kwargs: Any) -> Any:
            return timed_call(original_historical, *args, **kwargs)

        execution.compute_dmp_transition = timed_compute_dmp_transition
        historical.compute_historical_checkpoint_dmp_transition = (
            timed_historical_dmp_transition
        )
        try:
            yield
        finally:
            execution.compute_dmp_transition = original
            historical.compute_historical_checkpoint_dmp_transition = original_historical
            self._original_compute_dmp_transition = None
            self._original_historical_dmp_transition = None


class TimedPolicyProxy:
    """Transparent policy proxy whose ``predict`` calls are timed by mode."""

    def __init__(self, policy: Any, recorder: OnlineRuntimeRecorder) -> None:
        self._policy = policy
        self._recorder = recorder

    def __getattr__(self, name: str) -> Any:
        return getattr(self._policy, name)

    def timing_mode(self, mode: str) -> contextlib.AbstractContextManager[None]:
        return self._recorder.actor_mode(mode)

    def predict(self, observation: Any, *args: Any, **kwargs: Any) -> Any:
        synchronize_cuda()
        started = time.perf_counter_ns()
        result = self._policy.predict(observation, *args, **kwargs)
        synchronize_cuda()
        runtime_ms = (time.perf_counter_ns() - started) / 1.0e6
        self._recorder.record_actor(runtime_ms, getattr(observation, "shape", ()))
        return result
