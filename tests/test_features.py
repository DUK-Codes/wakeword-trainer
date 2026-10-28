"""Feature extraction: known shapes, determinism and the mel transform."""

from __future__ import annotations

import numpy as np
import pytest

from wakeword_trainer.audio.features import (
    EMBEDDING_DIM,
    FeatureExtractor,
    mel_frames_to_embedding_frames,
    num_embedding_frames,
    num_mel_frames,
)
from wakeword_trainer.audio.io import load_audio


@pytest.fixture(scope="module")
def extractor() -> FeatureExtractor:
    return FeatureExtractor(num_threads=1, providers=["CPUExecutionProvider"])


def test_known_frame_counts():
    # Parity with openWakeWord: 2 s -> 197 mel frames -> 16 embedding frames.
    assert num_mel_frames(32_000) == 197
    assert num_embedding_frames(32_000) == 16
    assert num_mel_frames(0) == 0
    assert num_embedding_frames(1_000) == 0


def test_mel_to_embedding_formula():
    assert mel_frames_to_embedding_frames(76) == 1
    assert mel_frames_to_embedding_frames(84) == 2
    assert mel_frames_to_embedding_frames(75) == 0


def test_output_shape(extractor: FeatureExtractor):
    assert extractor.output_shape(32_000) == (16, EMBEDDING_DIM)
    assert extractor.output_shape(16_000) == (3, EMBEDDING_DIM)


def test_mel_transform_is_window_scaled(extractor: FeatureExtractor):
    samples = np.zeros(16_000, dtype=np.int16)
    samples[4_000:12_000] = np.sin(np.linspace(0, 2 * np.pi * 30, 8_000)).astype(np.int16) * 8_000

    raw = extractor.melspectrogram(samples, transform=False)
    transformed = extractor.melspectrogram(samples, transform=True)
    assert raw.shape == transformed.shape
    np.testing.assert_allclose(transformed, raw / 10.0 + 2.0, rtol=0, atol=1e-6)


def test_embeddings_shape_and_determinism(extractor: FeatureExtractor):
    samples = np.zeros(32_000, dtype=np.int16)
    samples[2_000:30_000] = np.tile((np.sin(np.linspace(0, 2 * np.pi * 50, 1_400)) * 10_000).astype(np.int16), 20)

    first = extractor.embeddings(samples)
    second = extractor.embeddings(samples)
    assert first.shape == (16, EMBEDDING_DIM)
    assert first.dtype == np.float32
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)


def test_embeddings_batch_matches_single(extractor: FeatureExtractor, positives_dir):
    audio_a = load_audio(positives_dir / "speaker_00_session_00.wav")
    audio_b = load_audio(positives_dir / "speaker_00_session_01.wav")
    batch = extractor.embeddings(np.stack([audio_a, audio_b]))
    single = extractor.embeddings(audio_a)
    assert batch.shape[0] == 2
    np.testing.assert_allclose(batch[0], single, rtol=0, atol=1e-5)


def test_short_audio_produces_no_frames(extractor: FeatureExtractor):
    samples = np.zeros(320, dtype=np.int16)
    assert extractor.melspectrogram(samples).shape[0] == 0
    assert extractor.embeddings(samples).shape == (0, EMBEDDING_DIM)
