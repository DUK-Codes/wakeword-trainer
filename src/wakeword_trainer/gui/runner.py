"""GUI-side controller for one training subprocess."""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from ..config import TrainerConfig
from ..training.callbacks import TrainingEvent


class TrainingRun:
    """Owns the worker process and tails its event log."""

    def __init__(self, config: TrainerConfig, *, tflite: bool = False, resume: bool = False) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.output_dir / "gui"
        run_dir.mkdir(parents=True, exist_ok=True)
        self.config_path = run_dir / f"run_{stamp}.json"
        self.events_path = run_dir / f"events_{stamp}.jsonl"
        self.cancel_path = run_dir / f"cancel_{stamp}.flag"
        self.log_path = run_dir / f"worker_{stamp}.log"
        config.save(self.config_path)
        self.events_path.write_text("", encoding="utf-8")
        self.tflite = bool(tflite)
        self.resume = bool(resume)

        self._process: subprocess.Popen | None = None
        self._handle = None
        self._started = time.time()
        self._finished: float | None = None
        self._cancel_requested = False
        self._error: str | None = None

    # ------------------------------------------------------------------
    def start(self) -> None:
        command = [
            sys.executable,
            "-m",
            "wakeword_trainer.gui.worker",
            "--config",
            str(self.config_path),
            "--events",
            str(self.events_path),
            "--cancel",
            str(self.cancel_path),
        ]
        if self.tflite:
            command.append("--tflite")
        if self.resume:
            command.append("--resume")

        env = dict(os.environ)
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("PYTHONUNBUFFERED", "1")

        creationflags = 0
        if sys.platform.startswith("win"):
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        log_handle = self.log_path.open("a", encoding="utf-8", errors="replace")
        try:
            self._process = subprocess.Popen(
                command,
                cwd=str(self.output_dir),
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                creationflags=creationflags,
            )
        finally:
            log_handle.close()

    # ------------------------------------------------------------------
    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def returncode(self) -> int | None:
        return None if self._process is None else self._process.poll()

    @property
    def elapsed(self) -> float:
        end = self._finished if self._finished is not None else time.time()
        return max(0.0, end - self._started)

    @property
    def cancel_requested(self) -> bool:
        return self._cancel_requested

    @property
    def pid(self) -> int | None:
        return self._process.pid if self._process else None

    # ------------------------------------------------------------------
    def read_events(self) -> list[TrainingEvent]:
        """Return events appended since the previous call."""
        if self._handle is None:
            if not self.events_path.is_file():
                return []
            self._handle = self.events_path.open("r", encoding="utf-8", errors="replace")

        events: list[TrainingEvent] = []
        while True:
            position = self._handle.tell()
            line = self._handle.readline()
            if not line:
                break
            if not line.endswith("\n"):
                # A partially written line; retry after the next flush.
                self._handle.seek(position)
                break
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            event = TrainingEvent.from_dict(data)
            if event.kind == "error":
                self._error = event.message
            events.append(event)
        return events

    def drain(self) -> list[TrainingEvent]:
        """Read remaining events after the process exited."""
        events = self.read_events()
        if self._finished is None and not self.running:
            self._finished = time.time()
            events.extend(self.read_events())
        return events

    # ------------------------------------------------------------------
    def cancel(self) -> None:
        self._cancel_requested = True
        with contextlib.suppress(OSError):
            self.cancel_path.write_text("cancel\n", encoding="utf-8")

    def terminate(self) -> None:
        if self._process is not None and self._process.poll() is None:
            with contextlib.suppress(Exception):
                self._process.terminate()

    def kill(self) -> None:
        if self._process is not None and self._process.poll() is None:
            with contextlib.suppress(Exception):
                self._process.kill()

    def close(self) -> None:
        if self._handle is not None:
            try:
                self._handle.close()
            finally:
                self._handle = None
        if self.running:
            self.terminate()

    # ------------------------------------------------------------------
    def tail_log(self, max_chars: int = 4000) -> str:
        if not self.log_path.is_file():
            return ""
        try:
            text = self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return text[-max_chars:]

    def failure_message(self) -> str:
        if self._error:
            return self._error
        tail = self.tail_log(1200).strip()
        if tail:
            return tail.splitlines()[-1]
        code = self.returncode
        return f"Worker exited with code {code}" if code not in (None, 0) else ""
