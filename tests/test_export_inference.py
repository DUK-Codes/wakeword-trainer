"""ONNX export, metadata and inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from conftest import word_signal

from wakeword_trainer.api import detect, export_model, predict
from wakeword_trainer.audio.io import to_int16
from wakeword_trainer.config import ModelConfig
from wakeword_trainer.export.onnx import export_onnx, verify_onnx
from wakeword_trainer.models.dnn import build_model
from wakeword_trainer.models.inference import (
    StreamingDetector,
    WakeWordDetector,
    load_model_metadata,
)

FRAMES = 8
CLIP_SECONDS = 1.28


@pytest.fixture(scope="module")
def exported(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("onnx")
    model = build_model(FRAMES, ModelConfig(layer_dim=16, n_blocks=1, dropout=0.0))
    return export_onnx(
        model,
        root / "hey-test.onnx",
        frames=FRAMES,
        word="hey test",
        threshold=0.6,
        clip_seconds=CLIP_SECONDS,
        sample_rate=16_000,
        verify=True,
    )


def test_export_writes_metadata(exported: Path):
    metadata = load_model_metadata(exported)
    assert metadata.word == "hey test"
    assert metadata.frames == FRAMES
    assert metadata.threshold == pytest.approx(0.6)
    assert metadata.clip_seconds == pytest.approx(CLIP_SECONDS)
    assert metadata.sample_rate == 16_000
    assert metadata.version
    assert metadata.extra["input_layout"] == "batch,frames,96"


def test_verify_onnx_matches_torch(tmp_path: Path):
    model = build_model(FRAMES, ModelConfig(layer_dim=16, n_blocks=1, dropout=0.0))
    path = export_onnx(
        model,
        tmp_path / "verified.onnx",
        frames=FRAMES,
        word="verified",
        clip_seconds=CLIP_SECONDS,
        verify=False,
    )
    report = verify_onnx(model, path, frames=FRAMES)
    assert report["input"] == "input"
    assert report["output_shape"] == [2, 1]
    assert report["max_abs_diff"] is not None
    assert report["max_abs_diff"] <= 1e-4


def test_predict_accepts_arrays_and_files(exported: Path, tmp_path: Path):
    audio = to_int16(word_signal(3, seconds=CLIP_SECONDS))

    score = predict(exported, audio)
    assert 0.0 <= score <= 1.0

    from wakeword_trainer.audio.io import save_wav

    path = save_wav(tmp_path / "probe.wav", audio, 16_000)
    from_file = predict(exported, path)
    assert from_file == pytest.approx(score, abs=1e-4)


def test_detect_returns_structured_result(exported: Path):
    audio = to_int16(word_signal(4, seconds=CLIP_SECONDS))
    result = detect(exported, audio, threshold=0.42)
    payload = result.as_dict()
    assert set(payload) >= {"score", "detected", "threshold"}
    assert payload["threshold"] == pytest.approx(0.42)
    assert 0.0 <= payload["score"] <= 1.0


def test_score_features_batch(exported: Path):
    detector = WakeWordDetector(exported, num_threads=1)
    features = np.zeros((3, FRAMES, 96), dtype=np.float32)
    scores = detector.score_features(features)
    assert scores.shape == (3,)
    single = detector.score_features(features[0])
    assert np.ndim(single) == 0
    assert float(single) == pytest.approx(float(scores[0]), abs=1e-6)


def test_streaming_detector_emits_scores(exported: Path):
    detector = WakeWordDetector(exported, num_threads=1)
    streaming = StreamingDetector(detector, hop_seconds=0.08)
    audio = np.tile(to_int16(word_signal(5, seconds=CLIP_SECONDS)), 3)
    block = 1_280  # 80 ms
    scores = []
    for start in range(0, audio.size, block):
        value = streaming.push(audio[start : start + block])
        if value is not None:
            scores.append(value)
    assert scores
    assert all(0.0 <= value <= 1.0 for value in scores)


def test_export_model_copies_onnx(exported: Path, tmp_path: Path):
    target = export_model(exported, tmp_path / "copy")
    assert target.suffix == ".onnx"
    assert target.is_file()
    assert load_model_metadata(target).word == "hey test"
