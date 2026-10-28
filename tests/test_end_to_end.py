"""End-to-end: train a tiny model, evaluate it, export it and run inference.

This is the only test that exercises the whole pipeline on disk. It stays
small on purpose (20 steps, 1 s clips) so it runs in seconds on CPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import wakeword_trainer as wwt
from wakeword_trainer.audio.io import load_audio
from wakeword_trainer.models.inference import load_model_metadata

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory, positives_dir: Path, negative_audio_dir: Path):
    output = tmp_path_factory.mktemp("e2e") / "hey-test"
    result = wwt.train(
        "hey test",
        positives_dir,
        output,
        preset="quick",
        negatives=negative_audio_dir,
        steps=20,
        batch_size=8,
        clip_seconds=1.0,
        layer_dim=16,
        n_blocks=1,
        val_every=10,
        val_windows=64,
        device="cpu",
        num_threads=1,
        seed=7,
        download_negatives=False,
    )
    return result


def test_training_writes_artifacts(trained):
    output = Path(trained.output_dir)
    assert trained.onnx_path is not None
    assert Path(trained.onnx_path).is_file()
    assert trained.total_steps == 20
    assert trained.best_step <= 20
    assert trained.parameters > 0

    for name in (
        "training_result.json",
        "history.json",
        "trainer_config.json",
        "prepared.json",
        "training_curves.png",
    ):
        assert (output / name).is_file(), name
    assert (output / "split" / "train.json").is_file()
    assert (output / "evaluation" / "evaluation.md").is_file()
    assert (output / "evaluation" / "evaluation.json").is_file()

    payload = json.loads((output / "training_result.json").read_text(encoding="utf-8"))
    assert payload["word"] == "hey test"
    assert payload["onnx_path"]


def test_exported_model_metadata(trained):
    metadata = load_model_metadata(trained.onnx_path)
    assert metadata.word == "hey test"
    assert metadata.sample_rate == 16_000
    assert metadata.clip_seconds == pytest.approx(1.0)
    assert metadata.frames > 0
    assert 0.0 <= metadata.threshold <= 1.0
    assert metadata.version


def test_predict_and_detect_on_held_out_audio(trained, positives_dir: Path):
    audio = load_audio(positives_dir / "speaker_00_session_00.wav")
    score = wwt.predict(trained, audio)
    assert 0.0 <= score <= 1.0

    from_file = wwt.predict(trained, positives_dir / "speaker_00_session_00.wav")
    assert 0.0 <= from_file <= 1.0

    result = wwt.detect(trained, audio, threshold=0.5)
    payload = result.as_dict()
    assert payload["threshold"] == pytest.approx(0.5)
    assert 0.0 <= payload["score"] <= 1.0


def test_evaluate_returns_honest_metrics(trained, positives_dir: Path, negative_audio_dir: Path):
    result = wwt.evaluate(
        trained,
        positives_dir,
        negatives=negative_audio_dir,
        target_fp_per_hour=10.0,
        max_negative_windows=200,
    )
    metrics = result.metrics
    assert metrics["positives"] > 0
    assert metrics["negatives"] > 0
    assert 0.0 <= metrics["auc"] <= 1.0
    assert 0.0 <= metrics["operating_point"]["recall"] <= 1.0
    assert metrics["operating_point"]["fp_per_hour"] >= 0.0
    assert result.report_dir is not None
    assert (Path(result.report_dir) / "evaluation.json").is_file()


def test_export_model_copies_onnx_and_detector_loads(trained, tmp_path: Path):
    target = wwt.export_model(trained, tmp_path / "deploy")
    assert target.is_file()
    assert target.suffix == ".onnx"

    detector = wwt.WakeWordDetector(target, threshold=0.5)
    samples = np.zeros(16_000, dtype=np.int16)
    assert 0.0 <= detector.score_audio(samples) <= 1.0
