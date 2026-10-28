"""Audio loading, resampling and writing.

The canonical in-memory representation is **mono, 16 kHz, int16 PCM**,
because that is what the openWakeWord feature models expect. Everything
else (float WAVs, stereo, 44.1 kHz, MP3, ...) is normalized here.

``soundfile`` handles WAV/FLAC/OGG natively. Files it cannot decode
(typically MP3/M4A/AAC) are passed to ``ffmpeg`` when it is available.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf

from ..exceptions import AudioFormatError

TARGET_SAMPLE_RATE = 16_000

AUDIO_EXTENSIONS = frozenset(
    {".wav", ".flac", ".ogg", ".oga", ".opus", ".mp3", ".m4a", ".aac", ".wma", ".aiff", ".aif", ".au"}
)

# Formats that libsndfile historically fails to decode and that need ffmpeg.
_FFMPEG_ONLY = frozenset({".m4a", ".aac", ".wma", ".mp4", ".webm", ".mkv"})

_FFMPEG_ENV = "WAKEWORD_TRAINER_FFMPEG"


@dataclass(frozen=True)
class AudioInfo:
    path: Path
    sample_rate: int
    channels: int
    duration: float

    @property
    def frames(self) -> int:
        return int(round(self.duration * self.sample_rate))


@lru_cache(maxsize=1)
def ffmpeg_path() -> str | None:
    """Return the ffmpeg executable path, or ``None`` when unavailable."""
    override = os.environ.get(_FFMPEG_ENV)
    if override:
        return override if Path(override).is_file() else None
    return shutil.which("ffmpeg")


def ffmpeg_available() -> bool:
    return ffmpeg_path() is not None


def _run_ffmpeg(args: list[str], *, binary: bool) -> bytes:
    exe = ffmpeg_path()
    if exe is None:
        raise AudioFormatError(
            "This audio format requires ffmpeg, but ffmpeg was not found. "
            "Install ffmpeg and make sure it is on PATH (or set "
            f"{_FFMPEG_ENV} to the executable)."
        )
    with tempfile.TemporaryFile() as err:
        proc = subprocess.run([exe, *args], stdout=subprocess.PIPE, stderr=err, check=False)
        if proc.returncode != 0:
            err.seek(0)
            message = err.read().decode("utf-8", "replace").strip()
            raise AudioFormatError(f"ffmpeg failed: {message[:500]}")
        return proc.stdout if binary else b""


def probe_audio(path: str | Path) -> AudioInfo:
    """Read sample rate/channels/duration without decoding the whole file."""
    path = Path(path)
    try:
        info = sf.info(str(path))
        return AudioInfo(path, int(info.samplerate), int(info.channels), float(info.duration))
    except Exception:
        pass

    if ffmpeg_path() is None:
        raise AudioFormatError(f"Cannot read {path.name!r}. Unsupported or corrupt file, and ffmpeg is not installed.")

    # ffmpeg's null muxer does not report duration on stdout; parse the
    # `ffmpeg -i <file>` banner instead.
    exe = ffmpeg_path()
    proc = subprocess.run(
        [exe, "-hide_banner", "-i", str(path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )
    text = proc.stderr.decode("utf-8", "replace")
    duration = 0.0
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("Duration:"):
            stamp = line.split("Duration:", 1)[1].split(",", 1)[0].strip()
            h, m, s = stamp.split(":")
            duration = int(h) * 3600 + int(m) * 60 + float(s)
        if "Audio:" in line and " Hz" in line:
            sr = int(line.split(" Hz", 1)[0].rsplit(",", 1)[-1].strip())
            channels = 2 if "stereo" in line else 1
            return AudioInfo(path, sr, channels, duration)
    raise AudioFormatError(f"Cannot determine audio properties for {path}.")


def _read_with_ffmpeg(path: Path, target_sr: int) -> np.ndarray:
    raw = _run_ffmpeg(
        [
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ac",
            "1",
            "-ar",
            str(target_sr),
            "-",
        ],
        binary=True,
    )
    return np.frombuffer(raw, dtype="<f4").astype(np.float32)


def read_audio(
    path: str | Path,
    *,
    sample_rate: int = TARGET_SAMPLE_RATE,
    mono: bool = True,
) -> np.ndarray:
    """Decode ``path`` to float32 in ``[-1, 1]`` at ``sample_rate``."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix not in AUDIO_EXTENSIONS:
        raise AudioFormatError(f"Unsupported audio extension {suffix!r} for {path.name!r}.")

    use_ffmpeg = suffix in _FFMPEG_ONLY
    if not use_ffmpeg:
        try:
            data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        except Exception:
            if ffmpeg_path() is None:
                raise AudioFormatError(
                    f"Cannot decode {path.name!r} with libsndfile and ffmpeg is not installed."
                ) from None
            data, sr = None, None  # type: ignore[assignment]
        if data is not None:
            data = data.mean(axis=1) if mono else data[:, 0]
            if sr != sample_rate:
                data = resample(data, sr, sample_rate)
            return np.ascontiguousarray(data, dtype=np.float32)

    if ffmpeg_path() is None:
        raise AudioFormatError(
            f"{path.name!r} needs ffmpeg to decode. Install ffmpeg or convert the file to WAV/FLAC/OGG."
        )
    return _read_with_ffmpeg(path, sample_rate)


def resample(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Rational resampling with a polyphase filter (no external audio deps)."""
    if orig_sr == target_sr or samples.size == 0:
        return samples.astype(np.float32, copy=False)
    from math import gcd

    from scipy.signal import resample_poly

    factor = gcd(int(orig_sr), int(target_sr))
    up = int(target_sr) // factor
    down = int(orig_sr) // factor
    out = resample_poly(samples.astype(np.float32, copy=False), up, down)
    return np.ascontiguousarray(out, dtype=np.float32)


def to_int16(samples: np.ndarray) -> np.ndarray:
    """Convert float samples in ``[-1, 1]`` to int16 PCM with clipping."""
    arr = np.asarray(samples, dtype=np.float32)
    if arr.size and np.max(np.abs(arr)) > 1.5:
        # Already int16-scaled data stored as float.
        arr = arr / 32768.0
    arr = np.clip(arr, -1.0, 1.0)
    return np.round(arr * 32767.0).astype(np.int16)


def to_float(samples: np.ndarray) -> np.ndarray:
    arr = np.asarray(samples)
    if arr.dtype == np.int16:
        return arr.astype(np.float32) / 32768.0
    return arr.astype(np.float32, copy=False)


def normalize_peak(samples: np.ndarray, peak_dbfs: float = -3.0) -> np.ndarray:
    """Scale the signal so its peak sits at ``peak_dbfs``."""
    arr = np.asarray(samples, dtype=np.float32)
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    if peak <= 0:
        return arr
    target = 10.0 ** (peak_dbfs / 20.0)
    return (arr * (target / peak)).astype(np.float32, copy=False)


def load_audio(
    path: str | Path,
    *,
    sample_rate: int = TARGET_SAMPLE_RATE,
    mono: bool = True,
    normalize: bool = False,
    peak_dbfs: float = -3.0,
) -> np.ndarray:
    """Load an audio file as mono int16 PCM at ``sample_rate``."""
    data = read_audio(path, sample_rate=sample_rate, mono=mono)
    if normalize:
        data = normalize_peak(data, peak_dbfs)
    return to_int16(data)


def trim_silence(samples: np.ndarray, top_db: float = 40.0) -> np.ndarray:
    """Trim leading/trailing near-silence from int16 or float samples."""
    arr = np.asarray(samples)
    if arr.size == 0:
        return arr
    peak = float(np.max(np.abs(arr.astype(np.float32))))
    if peak <= 0:
        return arr[:0]
    threshold = peak * (10.0 ** (-abs(top_db) / 20.0))
    above = np.flatnonzero(np.abs(arr.astype(np.float32)) > threshold)
    if above.size == 0:
        return arr[:0]
    pad = int(0.05 * (16000 if arr.dtype == np.int16 else 1))
    start = max(0, int(above[0]) - pad)
    end = min(arr.size, int(above[-1]) + pad + 1)
    return arr[start:end]


def save_wav(
    path: str | Path,
    samples: np.ndarray,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> Path:
    """Write int16 (or float) samples to a PCM WAV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(samples)
    if arr.dtype != np.int16:
        arr = to_int16(arr)
    sf.write(str(path), arr, sample_rate, subtype="PCM_16")
    return path


def wav_duration(path: str | Path) -> float:
    with wave.open(str(path), "rb") as handle:
        frames = handle.getnframes()
        rate = handle.getframerate()
    return frames / float(rate) if rate else 0.0


def silence(seconds: float, sample_rate: int = TARGET_SAMPLE_RATE, rms: float = 0.0) -> np.ndarray:
    """Return int16 noise at a given RMS (0.0 gives digital silence)."""
    n = max(1, int(round(seconds * sample_rate)))
    if rms <= 0:
        return np.zeros(n, dtype=np.int16)
    noise = np.random.default_rng().normal(0.0, rms, n).astype(np.float32)
    return to_int16(noise)


def write_metadata(path: str | Path, data: dict) -> Path:
    path = Path(path)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def human_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)} min {int(sec)} s"
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours} h {minutes} min"


def estimated_hours(num_windows: int, hop_seconds: float) -> float:
    return max(1e-9, num_windows * hop_seconds / 3600.0)


def db_to_amplitude(db: float) -> float:
    return 10.0 ** (db / 20.0)


def amplitude_to_db(amplitude: float) -> float:
    return 20.0 * math.log10(max(amplitude, 1e-12))
