"""Shared fixtures: tiny synthetic audio, no external downloads.

Everything in the test suite is generated on the fly. No personal dataset,
no multi-gigabyte feature files, no network access.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

SAMPLE_RATE = 16_000


# ---------------------------------------------------------------------------
# synthetic signals
# ---------------------------------------------------------------------------


def word_signal(seed: int, *, seconds: float = 1.4, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """A deterministic three-syllable 'word' with a distinct pitch per seed."""
    rng = np.random.default_rng(seed)
    total = int(seconds * sample_rate)
    base = 170.0 + 25.0 * (seed % 9)
    signal = np.zeros(total, dtype=np.float32)
    for index in range(3):
        start = int((0.12 + index * 0.36) * sample_rate)
        length = min(int(0.24 * sample_rate), total - start)
        if length <= 0:
            continue
        segment = np.arange(length, dtype=np.float32) / sample_rate
        window = np.sin(np.pi * np.arange(length, dtype=np.float32) / length) ** 2
        frequency = base * (1.0 + 0.15 * index)
        signal[start : start + length] += 0.35 * np.sin(2 * np.pi * frequency * segment) * window
    signal += 0.008 * rng.standard_normal(total).astype(np.float32)
    peak = float(np.max(np.abs(signal)))
    if peak > 0:
        signal *= 0.85 / peak
    return signal.astype(np.float32)


def noise_signal(seed: int, *, seconds: float = 2.0, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Low-level coloured-ish noise used as a negative example."""
    rng = np.random.default_rng(seed + 10_000)
    total = int(seconds * sample_rate)
    white = rng.standard_normal(total).astype(np.float32)
    # Cheap low-pass: cumulative average of a short window.
    kernel = np.ones(16, dtype=np.float32) / 16.0
    smoothed = np.convolve(white, kernel, mode="same")
    smoothed *= 0.05 / max(1e-6, float(np.max(np.abs(smoothed))))
    return smoothed.astype(np.float32)


def write_audio(path: str | Path, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(samples, dtype=np.float32), sample_rate, subtype="PCM_16")
    return path


# ---------------------------------------------------------------------------
# session-wide datasets
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def positives_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """12 independent recordings (one group each, named like real sessions)."""
    root = tmp_path_factory.mktemp("positives")
    for index in range(12):
        write_audio(root / f"speaker_{index // 4:02d}_session_{index:02d}.wav", word_signal(index))
    return root


@pytest.fixture(scope="session")
def negative_audio_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Four short noise recordings used as a negative audio folder."""
    root = tmp_path_factory.mktemp("negatives")
    for index in range(4):
        write_audio(root / f"noise_{index:02d}.wav", noise_signal(index))
    return root


@pytest.fixture(scope="session")
def clear_isolated_audio(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Positive folder for leakage tests: source words with augmented copies."""
    root = tmp_path_factory.mktemp("groups")
    sources = {
        "hello": word_signal(99),
        "other": word_signal(100),
        "third": word_signal(101),
        "fourth": word_signal(102),
    }
    write_audio(root / "hello.wav", sources["hello"])
    write_audio(root / "hello_aug1.wav", sources["hello"] * 0.8)
    write_audio(root / "hello_v2.wav", sources["hello"] * 0.9)
    write_audio(root / "other.wav", sources["other"])
    write_audio(root / "other_noise.wav", sources["other"] * 0.7)
    write_audio(root / "third.wav", sources["third"])
    write_audio(root / "third_aug1.wav", sources["third"] * 0.85)
    write_audio(root / "fourth.wav", sources["fourth"])
    write_audio(root / "fourth_v3.wav", sources["fourth"] * 0.75)
    return root


# ---------------------------------------------------------------------------
# isolation
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolated_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Keep caches, plugin folders and GUI state inside the test tmp dir."""
    monkeypatch.setenv("WAKEWORD_TRAINER_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("WAKEWORD_TRAINER_PLUGIN_DIR", str(tmp_path / "plugins"))
    yield


@pytest.fixture
def copied_audio(tmp_path: Path, positives_dir: Path) -> Path:
    """A folder with two identical files (to exercise duplicate detection)."""
    root = tmp_path / "duplicates"
    root.mkdir()
    shutil.copyfile(positives_dir / "speaker_00_session_00.wav", root / "a.wav")
    shutil.copyfile(positives_dir / "speaker_00_session_00.wav", root / "b.wav")
    return root
