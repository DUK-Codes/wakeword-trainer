"""On-the-fly audio augmentation and fixed-length clip rendering."""

from __future__ import annotations

import random
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve, resample_poly

from ..audio.io import (
    AUDIO_EXTENSIONS,
    db_to_amplitude,
    load_audio,
    normalize_peak,
    to_float,
    to_int16,
    trim_silence,
)
from ..config import AudioConfig, AugmentConfig


def _list_audio(directory: str | Path | None) -> list[Path]:
    if directory is None:
        return []
    root = Path(directory)
    if not root.is_dir():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS)


@lru_cache(maxsize=128)
def _cached_audio(path: str) -> np.ndarray:
    return load_audio(path)


def _rms(samples: np.ndarray) -> float:
    arr = np.asarray(samples, dtype=np.float32)
    return float(np.sqrt(np.mean(arr * arr))) if arr.size else 0.0


def _fit_length(
    samples: np.ndarray,
    target_samples: int,
    *,
    pad_mode: str,
    rng: random.Random,
    background: np.ndarray | None = None,
    fade_samples: int = 0,
) -> np.ndarray:
    """Place ``samples`` inside a ``target_samples`` window."""
    arr = np.asarray(samples, dtype=np.float32)
    if arr.size == 0:
        return np.zeros(target_samples, dtype=np.float32)
    if arr.size > target_samples:
        if pad_mode == "start":
            start = 0
        elif pad_mode == "end":
            start = arr.size - target_samples
        else:
            start = rng.randint(0, arr.size - target_samples)
        arr = arr[start : start + target_samples]

    if background is not None and background.size >= target_samples:
        start = rng.randint(0, background.size - target_samples)
        canvas = background[start : start + target_samples].astype(np.float32)
    else:
        canvas = np.zeros(target_samples, dtype=np.float32)

    offset = {
        "start": 0,
        "end": target_samples - arr.size,
        "center": (target_samples - arr.size) // 2,
    }.get(pad_mode)
    if offset is None:
        offset = rng.randint(0, target_samples - arr.size)

    canvas[offset : offset + arr.size] += arr

    if fade_samples > 0 and arr.size > 2 * fade_samples:
        ramp = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
        canvas[offset : offset + fade_samples] *= ramp
        canvas[offset + arr.size - fade_samples : offset + arr.size] *= ramp[::-1]
    return canvas


class Augmenter:
    """Random augmentation chain for positive wake word clips."""

    def __init__(
        self,
        config: AugmentConfig | None = None,
        audio_config: AudioConfig | None = None,
        *,
        seed: int | None = None,
    ) -> None:
        self.config = config or AugmentConfig()
        self.audio_config = audio_config or AudioConfig()
        self._rng = random.Random(seed)
        self._backgrounds = _list_audio(self.config.background_dir)
        self._rirs = _list_audio(self.config.rir_dir)

    # ------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def child_rng(self) -> random.Random:
        return random.Random(self._rng.getrandbits(64))

    def _pick_background(self, rng: random.Random, target_samples: int) -> np.ndarray | None:
        if not self._backgrounds or rng.random() >= self.config.background_prob:
            return None
        path = self._backgrounds[rng.randrange(len(self._backgrounds))]
        audio = _cached_audio(str(path))
        if audio.size < target_samples:
            repeats = int(np.ceil(target_samples / max(1, audio.size)))
            audio = np.tile(audio, repeats)
        return audio

    def _pick_rir(self, rng: random.Random) -> np.ndarray | None:
        if not self._rirs or rng.random() >= self.config.reverb_prob:
            return None
        path = self._rirs[rng.randrange(len(self._rirs))]
        rir = to_float(_cached_audio(str(path)))
        if rir.size == 0:
            return None
        peak = float(np.max(np.abs(rir)))
        return rir / peak if peak > 0 else rir

    # ------------------------------------------------------------------
    def augment(self, samples: np.ndarray, rng: random.Random | None = None) -> np.ndarray:
        """Apply the random augmentation chain to int16 mono audio."""
        if not self.config.enabled:
            return np.asarray(samples, dtype=np.int16)
        rng = rng or self.child_rng()
        cfg = self.config
        audio = to_float(samples)

        if rng.random() < cfg.gain_prob:
            gain_db = rng.uniform(*cfg.gain_db)
            audio = audio * db_to_amplitude(gain_db)

        if rng.random() < cfg.speed_prob:
            rate = rng.uniform(*cfg.speed_rate)
            if abs(rate - 1.0) > 1e-3:
                audio = self._resample_factor(audio, 1.0 / rate)

        if rng.random() < cfg.pitch_prob:
            semitones = rng.uniform(*cfg.pitch_semitones)
            if abs(semitones) > 1e-3:
                audio = self._pitch_shift(audio, semitones)

        rir = self._pick_rir(rng)
        if rir is not None and rir.size > 1:
            wet = fftconvolve(audio, rir)[: audio.size]
            peak = float(np.max(np.abs(wet))) if wet.size else 0.0
            if peak > 0:
                wet = wet / peak * max(1e-4, float(np.max(np.abs(audio))))
            audio = wet.astype(np.float32)

        if rng.random() < cfg.noise_prob:
            snr_db = rng.uniform(*cfg.noise_snr_db)
            audio = self._add_white_noise(audio, snr_db, rng)

        return to_int16(np.clip(audio, -1.0, 1.0))

    # ------------------------------------------------------------------
    @staticmethod
    def _resample_factor(samples: np.ndarray, factor: float) -> np.ndarray:
        if factor <= 0:
            return samples
        # Resample by a rational approximation of the factor.
        denominator = 1000
        numerator = max(1, int(round(factor * denominator)))
        return resample_poly(samples, numerator, denominator).astype(np.float32)

    @staticmethod
    def _pitch_shift(samples: np.ndarray, semitones: float) -> np.ndarray:
        """Fast pitch augmentation via resampling.

        A true phase-vocoder pitch shift keeps the duration constant but is
        orders of magnitude slower on CPU (torchaudio's implementation takes
        seconds per clip). Resampling changes pitch and duration together,
        which is a reasonable augmentation for keyword spotting; the clip is
        padded/cropped back to the target length afterwards.
        """
        if abs(semitones) < 1e-3:
            return samples.astype(np.float32, copy=False)
        factor = 2.0 ** (semitones / 12.0)
        denominator = 1000
        numerator = max(1, int(round(denominator / factor)))
        return resample_poly(samples, numerator, denominator).astype(np.float32)

    @staticmethod
    def _add_white_noise(samples: np.ndarray, snr_db: float, rng: random.Random) -> np.ndarray:
        signal_rms = _rms(samples)
        if signal_rms <= 0:
            return samples
        noise = np.random.default_rng(rng.getrandbits(64)).normal(0.0, 1.0, samples.size).astype(np.float32)
        noise_rms = _rms(noise)
        if noise_rms <= 0:
            return samples
        target_rms = signal_rms / (10.0 ** (snr_db / 20.0))
        return samples + noise * (target_rms / noise_rms)

    # ------------------------------------------------------------------
    def render(
        self,
        samples: np.ndarray,
        target_samples: int,
        *,
        augment: bool = True,
        rng: random.Random | None = None,
        pad_mode: str | None = None,
    ) -> np.ndarray:
        """Trim, augment and pad a clip into a fixed-length int16 window."""
        rng = rng or self.child_rng()
        audio = np.asarray(samples, dtype=np.int16)
        if self.audio_config.trim_silence and audio.size:
            trimmed = trim_silence(audio, self.audio_config.trim_top_db)
            if trimmed.size >= int(0.15 * self.audio_config.sample_rate):
                audio = trimmed
        if augment:
            audio = self.augment(audio, rng)

        if audio.size > target_samples:
            # Keep at most the configured maximum, then crop in fit_length.
            max_samples = int(self.audio_config.max_clip_seconds * self.audio_config.sample_rate)
            audio = audio[:max_samples]

        background = self._pick_background(rng, target_samples) if augment else None
        fade = int(self.audio_config.fade_ms / 1000.0 * self.audio_config.sample_rate)
        speech = to_float(audio)
        rendered = _fit_length(
            speech,
            target_samples,
            pad_mode=pad_mode or self.audio_config.pad_mode,
            rng=rng,
            fade_samples=fade,
        )
        if background is not None:
            bg = to_float(background)
            start = rng.randint(0, bg.size - target_samples)
            bg = bg[start : start + target_samples]
            speech_rms = _rms(speech)
            bg_rms = _rms(bg)
            if speech_rms > 0 and bg_rms > 0:
                snr_db = rng.uniform(*self.config.background_snr_db)
                rendered = rendered + bg * (speech_rms / bg_rms) * db_to_amplitude(-snr_db)
        if self.audio_config.normalize_peak:
            rendered = normalize_peak(rendered, -3.0)
        return to_int16(np.clip(rendered, -1.0, 1.0))


class ClipBank:
    """In-memory cache of decoded clips for fast repeated sampling."""

    def __init__(self, paths: Iterable[str | Path], *, max_bytes: int = 2 * 1024**3) -> None:
        self._audio: dict[Path, np.ndarray] = {}
        total = 0
        for path in paths:
            key = Path(path)
            audio = load_audio(key)
            if total + audio.nbytes > max_bytes:
                continue
            self._audio[key] = audio
            total += audio.nbytes

    def __len__(self) -> int:
        return len(self._audio)

    def get(self, path: str | Path) -> np.ndarray:
        key = Path(path)
        audio = self._audio.get(key)
        return audio if audio is not None else load_audio(key)
