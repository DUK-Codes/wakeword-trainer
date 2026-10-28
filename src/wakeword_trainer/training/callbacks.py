"""Training events and callbacks.

The GUI and library users consume the same :class:`TrainingEvent` stream,
so progress bars, logs and subprocess communication all share one format.
"""

from __future__ import annotations

import contextlib
import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ..exceptions import TrainingCancelled

EventKind = Literal[
    "start",
    "step",
    "metrics",
    "checkpoint",
    "log",
    "done",
    "error",
    "cancelled",
    "prepare",
]


@dataclass
class TrainingEvent:
    kind: EventKind
    step: int = 0
    total_steps: int = 0
    message: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "step": self.step,
            "total_steps": self.total_steps,
            "message": self.message,
            "metrics": _jsonable(self.metrics),
            "payload": _jsonable(self.payload),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingEvent:
        return cls(
            kind=data.get("kind", "log"),
            step=int(data.get("step", 0)),
            total_steps=int(data.get("total_steps", 0)),
            message=str(data.get("message", "")),
            metrics=dict(data.get("metrics") or {}),
            payload=dict(data.get("payload") or {}),
            timestamp=float(data.get("timestamp", time.time())),
        )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # noqa: BLE001
            return str(value)
    return value


EventHandler = Callable[[TrainingEvent], None]


class Callback:
    """Base class; override only the hooks you need."""

    def on_event(self, event: TrainingEvent) -> None:
        return None

    def on_train_start(self, trainer, total_steps: int) -> None:
        return None

    def on_step(self, trainer, step: int, loss: float, lr: float) -> None:
        return None

    def on_eval(self, trainer, step: int, metrics: dict[str, Any]) -> None:
        return None

    def on_checkpoint(self, trainer, step: int, path: Path, metrics: dict[str, Any]) -> None:
        return None

    def on_train_end(self, trainer, result) -> None:
        return None

    def on_log(self, message: str) -> None:
        return None


class CallbackList(Callback):
    def __init__(self, callbacks: list[Callback] | None = None) -> None:
        self.callbacks = list(callbacks or [])

    def add(self, callback: Callback) -> CallbackList:
        self.callbacks.append(callback)
        return self

    def _forward(self, name: str, *args, **kwargs) -> None:
        for callback in self.callbacks:
            getattr(callback, name)(*args, **kwargs)

    def on_event(self, event: TrainingEvent) -> None:
        self._forward("on_event", event)

    def on_train_start(self, trainer, total_steps: int) -> None:
        self._forward("on_train_start", trainer, total_steps)

    def on_step(self, trainer, step: int, loss: float, lr: float) -> None:
        self._forward("on_step", trainer, step, loss, lr)

    def on_eval(self, trainer, step: int, metrics: dict[str, Any]) -> None:
        self._forward("on_eval", trainer, step, metrics)

    def on_checkpoint(self, trainer, step: int, path: Path, metrics: dict[str, Any]) -> None:
        self._forward("on_checkpoint", trainer, step, path, metrics)

    def on_train_end(self, trainer, result) -> None:
        self._forward("on_train_end", trainer, result)

    def on_log(self, message: str) -> None:
        self._forward("on_log", message)


class FunctionCallback(Callback):
    """Adapt a plain ``event -> None`` function to the callback protocol."""

    def __init__(self, handler: EventHandler) -> None:
        self.handler = handler

    def _emit(self, event: TrainingEvent) -> None:
        self.handler(event)

    def on_event(self, event: TrainingEvent) -> None:
        self._emit(event)

    def on_train_start(self, trainer, total_steps: int) -> None:
        self._emit(TrainingEvent("start", step=0, total_steps=total_steps, message="Training started"))

    def on_step(self, trainer, step: int, loss: float, lr: float) -> None:
        total = getattr(trainer, "total_steps", 0)
        every = max(1, getattr(trainer, "log_every", 1))
        if step % every != 0 and step not in {1, total}:
            return
        self._emit(
            TrainingEvent(
                "step",
                step=step,
                total_steps=total,
                metrics={"loss": float(loss), "lr": float(lr)},
            )
        )

    def on_eval(self, trainer, step: int, metrics: dict[str, Any]) -> None:
        self._emit(
            TrainingEvent(
                "metrics",
                step=step,
                total_steps=getattr(trainer, "total_steps", 0),
                metrics=metrics,
            )
        )

    def on_checkpoint(self, trainer, step: int, path: Path, metrics: dict[str, Any]) -> None:
        self._emit(
            TrainingEvent(
                "checkpoint",
                step=step,
                total_steps=getattr(trainer, "total_steps", 0),
                metrics=metrics,
                payload={"path": str(path)},
            )
        )

    def on_train_end(self, trainer, result) -> None:
        self._emit(
            TrainingEvent(
                "done",
                step=getattr(result, "best_step", 0),
                total_steps=getattr(result, "total_steps", 0),
                message="Training finished",
                payload=result.summary() if hasattr(result, "summary") else {},
            )
        )

    def on_log(self, message: str) -> None:
        self._emit(TrainingEvent("log", message=message))


class ConsoleCallback(Callback):
    def __init__(self, prefix: str = "") -> None:
        self.prefix = prefix

    def on_step(self, trainer, step: int, loss: float, lr: float) -> None:
        if step % max(1, getattr(trainer, "log_every", 20)) == 0:
            total = getattr(trainer, "total_steps", 0)
            print(f"{self.prefix}step {step}/{total}  loss={loss:.4f}  lr={lr:.2e}", flush=True)

    def on_eval(self, trainer, step: int, metrics: dict[str, Any]) -> None:
        recall = metrics.get("recall", 0.0)
        fph = metrics.get("fp_per_hour", 0.0)
        print(f"{self.prefix}eval step {step}: recall={recall:.3f} fp/hour={fph:.2f}", flush=True)

    def on_log(self, message: str) -> None:
        print(f"{self.prefix}{message}", flush=True)


class CancelToken:
    """Thread/process-safe cancellation flag."""

    def __init__(self, event: threading.Event | None = None) -> None:
        self._event = event or threading.Event()

    def set(self) -> None:
        self._event.set()

    def clear(self) -> None:
        self._event.clear()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()


class CancellationCallback(Callback):
    def __init__(self, token: CancelToken) -> None:
        self.token = token

    def on_step(self, trainer, step: int, loss: float, lr: float) -> None:
        if self.token.cancelled:
            raise TrainingCancelled("Training cancelled by user.")

    def on_train_start(self, trainer, total_steps: int) -> None:
        if self.token.cancelled:
            raise TrainingCancelled("Training cancelled by user.")


class JSONLWriter(Callback):
    """Append every event as one JSON line (used by the GUI worker)."""

    def __init__(self, path: str | Path, *, include_steps: bool = True) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.include_steps = include_steps
        self._handle = self.path.open("a", encoding="utf-8")

    def _write(self, event: TrainingEvent) -> None:
        self._handle.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
        self._handle.flush()

    def on_event(self, event: TrainingEvent) -> None:
        self._write(event)

    def on_train_start(self, trainer, total_steps: int) -> None:
        self._write(TrainingEvent("start", total_steps=total_steps))

    def on_step(self, trainer, step: int, loss: float, lr: float) -> None:
        if not self.include_steps:
            return
        every = max(1, getattr(trainer, "log_every", 20))
        if step % every == 0 or step == getattr(trainer, "total_steps", 0):
            self._write(
                TrainingEvent(
                    "step",
                    step=step,
                    total_steps=getattr(trainer, "total_steps", 0),
                    metrics={"loss": float(loss), "lr": float(lr)},
                )
            )

    def on_eval(self, trainer, step: int, metrics: dict[str, Any]) -> None:
        self._write(
            TrainingEvent("metrics", step=step, total_steps=getattr(trainer, "total_steps", 0), metrics=metrics)
        )

    def on_checkpoint(self, trainer, step: int, path: Path, metrics: dict[str, Any]) -> None:
        self._write(
            TrainingEvent(
                "checkpoint",
                step=step,
                total_steps=getattr(trainer, "total_steps", 0),
                metrics=metrics,
                payload={"path": str(path)},
            )
        )

    def on_train_end(self, trainer, result) -> None:
        self._write(
            TrainingEvent(
                "done",
                step=getattr(result, "best_step", 0),
                total_steps=getattr(result, "total_steps", 0),
                payload=result.summary() if hasattr(result, "summary") else {},
            )
        )

    def on_log(self, message: str) -> None:
        self._write(TrainingEvent("log", message=message))

    def close(self) -> None:
        self._handle.close()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self._handle.close()
