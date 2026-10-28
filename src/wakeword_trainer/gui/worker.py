"""Subprocess entry point for one training run.

The GUI keeps Tk responsive by running training in a separate process. This
module is intentionally small and dependency-free beyond the package itself:

    python -m wakeword_trainer.gui.worker \
        --config run.json --events events.jsonl [--tflite] [--resume] [--cancel cancel.flag]

All progress is appended to ``events.jsonl`` as one :class:`TrainingEvent`
JSON object per line. The parent process tails that file, which avoids
stdout/stderr buffering issues and third-party library noise.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import traceback
from pathlib import Path

from ..config import TrainerConfig
from ..exceptions import TrainingCancelled, WakeWordTrainerError
from ..training.callbacks import Callback, CancelToken, JSONLWriter, TrainingEvent
from ..training.trainer import Trainer


class FileCancellationCallback(Callback):
    """Stops training when the parent process creates a cancel marker file."""

    def __init__(self, token: CancelToken, marker: Path) -> None:
        self.token = token
        self.marker = marker

    def on_step(self, trainer, step: int, loss: float, lr: float) -> None:
        if self.marker.exists():
            self.token.set()
            raise TrainingCancelled("Training cancelled by the user.")

    def on_train_start(self, trainer, total_steps: int) -> None:
        if self.marker.exists():
            self.token.set()
            raise TrainingCancelled("Training cancelled by the user.")


def _write_error(events_path: Path, message: str, details: str = "") -> None:
    event = TrainingEvent(
        "error",
        message=message,
        payload={"details": details[-8000:]},
    )
    with events_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
        handle.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="wakeword-trainer background worker")
    parser.add_argument("--config", required=True, help="TrainerConfig JSON file")
    parser.add_argument("--events", required=True, help="JSONL event output file")
    parser.add_argument("--cancel", default="", help="Cancel marker file")
    parser.add_argument("--tflite", action="store_true", help="Also export TFLite")
    parser.add_argument("--resume", action="store_true", help="Resume from last.pt")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    events_path = Path(args.events)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    cancel_marker = Path(args.cancel) if args.cancel else events_path.with_suffix(".cancel")

    try:
        config = TrainerConfig.load(config_path)
        config.validate()
    except Exception as exc:  # noqa: BLE001
        _write_error(events_path, f"Invalid configuration: {exc}", traceback.format_exc())
        return 2

    writer = JSONLWriter(events_path)
    token = CancelToken()
    trainer = Trainer(
        config,
        callbacks=[writer, FileCancellationCallback(token, cancel_marker)],
        stop_token=token,
    )

    exit_code = 0
    try:
        result = trainer.train(
            export=True,
            export_tflite_model=bool(args.tflite),
            resume=bool(args.resume),
        )
        print(json.dumps(result.summary(), ensure_ascii=False), flush=True)
    except TrainingCancelled:
        _write_error(events_path, "Training cancelled")
        exit_code = 130
    except WakeWordTrainerError as exc:
        _write_error(events_path, str(exc), traceback.format_exc())
        exit_code = 1
    except KeyboardInterrupt:
        _write_error(events_path, "Training interrupted")
        exit_code = 130
    except Exception as exc:  # noqa: BLE001 - report anything to the GUI
        _write_error(events_path, f"{type(exc).__name__}: {exc}", traceback.format_exc())
        exit_code = 1
    finally:
        with contextlib.suppress(Exception):
            writer.close()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
