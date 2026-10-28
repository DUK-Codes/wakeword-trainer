"""Audio I/O normalization to mono 16 kHz int16."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from wakeword_trainer.audio.io import (
    amplitude_to_db,
    db_to_amplitude,
    estimated_hours,
    human_duration,
    load_audio,
    probe_audio,
    read_audio,
    resample,
    save_wav,
    silence,
    to_float,
    to_int16,
    trim_silence,
)


def test_to_int16_and_back():
    samples = np.array([-1.0, 0.0, 0.5, 1.0], dtype=np.float32)
    ints = to_int16(samples)
    assert ints.dtype == np.int16
    assert ints[0] == -32767
    np.testing.assert_allclose(to_float(ints), samples, atol=1e-4)


def test_save_wav_roundtrip(tmp_path: Path, positives_dir: Path):
    audio = load_audio(positives_dir / "speaker_00_session_00.wav")
    assert audio.dtype == np.int16
    target = save_wav(tmp_path / "copy.wav", audio, 16_000)
    loaded = load_audio(target)
    assert loaded.dtype == np.int16
    assert loaded.shape == audio.shape
    # int16 -> float -> int16 uses a 1/32767 scale, so allow a single LSB.
    np.testing.assert_allclose(loaded, audio, atol=1)


def test_load_audio_downmixes_and_resamples(tmp_path: Path):
    sample_rate = 32_000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    stereo = np.stack([np.sin(2 * np.pi * 440 * time), np.zeros_like(time)], axis=1)
    source = tmp_path / "stereo.wav"
    sf.write(str(source), stereo, sample_rate, subtype="PCM_16")

    audio = load_audio(source, sample_rate=16_000, mono=True)
    assert audio.dtype == np.int16
    # One second at 16 kHz, allowing for resampler edges.
    assert abs(audio.size - 16_000) < 200


def test_probe_audio_reports_format(positives_dir: Path):
    path = positives_dir / "speaker_00_session_00.wav"
    info = probe_audio(path)
    assert info.sample_rate == 16_000
    assert info.channels == 1
    assert info.duration == pytest.approx(1.4, abs=0.01)
    assert info.frames > 0


def test_read_audio_returns_float_in_range(positives_dir: Path):
    audio = read_audio(positives_dir / "speaker_00_session_01.wav", sample_rate=16_000)
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert np.max(np.abs(audio)) <= 1.0


def test_resample_preserves_duration():
    samples = np.sin(2 * np.pi * 200 * np.arange(16_000) / 16_000).astype(np.float32)
    out = resample(samples, 16_000, 8_000)
    assert abs(out.size - 8_000) < 5


def test_trim_silence_removes_padding():
    samples = np.zeros(32_000, dtype=np.int16)
    samples[8_000:24_000] = (np.sin(np.linspace(0, 2 * np.pi * 20, 16_000)) * 10_000).astype(np.int16)
    trimmed = trim_silence(samples, top_db=40.0)
    assert trimmed.size < samples.size
    assert trimmed.size > 10_000


def test_silence_helper():
    assert silence(0.5).size == 8_000
    assert np.all(silence(0.1) == 0)
    noisy = silence(0.1, rms=0.01)
    assert noisy.dtype == np.int16
    assert np.any(noisy != 0)


def test_db_helpers():
    assert amplitude_to_db(db_to_amplitude(-6.0)) == pytest.approx(-6.0, abs=1e-4)
    assert amplitude_to_db(0.0) == -240.0


def test_estimated_hours_and_human_duration():
    assert estimated_hours(360_000, 0.08) == pytest.approx(8.0)
    assert human_duration(5) == "5.0 s"
    assert human_duration(65) == "1 min 5 s"
    assert human_duration(3_665) == "1 h 1 min"
