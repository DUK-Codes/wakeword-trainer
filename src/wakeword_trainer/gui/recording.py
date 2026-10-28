"""Microphone recording helpers (optional, requires ``sounddevice``)."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..audio.io import save_wav, to_int16
from ..exceptions import WakeWordTrainerError

SAMPLE_RATE = 16_000


class MicrophoneUnavailable(WakeWordTrainerError):
    """Raised when ``sounddevice`` or an input device is missing."""


@dataclass(frozen=True)
class InputDevice:
    index: int | None
    name: str
    is_default: bool = False


def _sounddevice():
    try:
        import sounddevice as sd  # noqa: PLC0415 - optional dependency
    except Exception as exc:  # noqa: BLE001
        raise MicrophoneUnavailable(
            'sounddevice is not installed. Install it with `pip install "wakeword-trainer[gui]"`.'
        ) from exc
    return sd


def input_devices() -> list[InputDevice]:
    """List available input devices (system default first)."""
    sd = _sounddevice()
    devices: list[InputDevice] = [InputDevice(None, "System default", True)]
    try:
        default_in = sd.default.device[0]
    except Exception:  # noqa: BLE001
        default_in = None
    try:
        raw = sd.query_devices()
    except Exception as exc:  # noqa: BLE001
        raise MicrophoneUnavailable(f"Could not enumerate audio devices: {exc}") from exc
    for index, info in enumerate(raw):
        if int(info.get("max_input_channels", 0)) < 1:
            continue
        name = str(info.get("name", f"Device {index}"))
        devices.append(InputDevice(index, name, index == default_in))
    return devices


def _level_from_block(block: np.ndarray) -> float:
    """Map RMS in dBFS to 0..1 for a level meter."""
    arr = np.asarray(block, dtype=np.float32)
    if arr.size == 0:
        return 0.0
    rms = float(np.sqrt(np.mean(arr * arr)))
    if rms <= 1e-7:
        return 0.0
    db = 20.0 * np.log10(max(rms, 1e-7))
    return float(np.clip((db + 60.0) / 60.0, 0.0, 1.0))


class Recorder:
    """Collects 16 kHz mono int16 audio from the default input device."""

    def __init__(self, *, sample_rate: int = SAMPLE_RATE) -> None:
        self.sample_rate = int(sample_rate)
        self._stream = None
        self._chunks: list[np.ndarray] = []
        self._recording = False
        self._started = 0.0
        self._level_callback: Callable[[float], None] | None = None
        self._last_level_time = 0.0

    # ------------------------------------------------------------------
    @property
    def recording(self) -> bool:
        return self._recording

    @property
    def duration(self) -> float:
        if not self._recording:
            return sum(chunk.size for chunk in self._chunks) / self.sample_rate
        return time.time() - self._started

    def start(
        self,
        device: int | None = None,
        *,
        level_callback: Callable[[float], None] | None = None,
    ) -> None:
        if self._recording:
            return
        sd = _sounddevice()
        self._chunks = []
        self._level_callback = level_callback
        self._last_level_time = 0.0

        def callback(indata, frames, time_info, status) -> None:  # noqa: ANN001, ARG001
            del frames, time_info, status
            block = np.asarray(indata[:, 0], dtype=np.float32).copy()
            self._chunks.append(block)
            if level_callback is not None:
                now = time.time()
                if now - self._last_level_time >= 0.04:
                    self._last_level_time = now
                    level_callback(_level_from_block(block))

        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                device=device,
                callback=callback,
            )
            self._stream.start()
        except Exception as exc:  # noqa: BLE001
            self._stream = None
            raise MicrophoneUnavailable(f"Could not open the microphone: {exc}") from exc
        self._recording = True
        self._started = time.time()

    def stop(self) -> np.ndarray:
        """Stop recording and return the captured int16 samples."""
        if not self._recording:
            return np.zeros((0,), dtype=np.int16)
        self._recording = False
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:  # noqa: BLE001
                pass
        if not self._chunks:
            return np.zeros((0,), dtype=np.int16)
        audio = np.concatenate(self._chunks).astype(np.float32)
        self._chunks = []
        return to_int16(np.clip(audio, -1.0, 1.0))

    def abort(self) -> None:
        self._recording = False
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:  # noqa: BLE001
                pass
        self._chunks = []


def unique_path(folder: str | Path, prefix: str = "rec", suffix: str = ".wav") -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    index = 0
    while True:
        candidate = folder / f"{prefix}_{index:04d}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def save_recording(
    samples: np.ndarray,
    folder: str | Path,
    *,
    prefix: str = "rec",
    sample_rate: int = SAMPLE_RATE,
) -> Path:
    if samples.size == 0:
        raise WakeWordTrainerError("The recording is empty; check the microphone.")
    target = unique_path(folder, prefix)
    return save_wav(target, samples, sample_rate)
