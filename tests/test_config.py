"""Configuration presets, overrides, serialization and validation."""

from __future__ import annotations

import pytest

from wakeword_trainer.config import PRESETS, AudioConfig, TrainerConfig
from wakeword_trainer.exceptions import ConfigurationError


def test_presets_are_documented_and_ordered():
    assert list(PRESETS) == ["quick", "balanced", "best"]
    steps = [PRESETS[name]["steps"] for name in PRESETS]
    assert steps == sorted(steps)
    assert len(set(steps)) == 3


def test_from_preset_applies_model_and_train_fields():
    config = TrainerConfig.from_preset("quick", word="hey test")
    assert config.train.preset == "quick"
    assert config.train.steps == PRESETS["quick"]["steps"]
    assert config.model.layer_dim == PRESETS["quick"]["layer_dim"]
    assert config.word == "hey test"


def test_short_and_dotted_overrides():
    config = TrainerConfig.from_preset(
        "balanced",
        **{
            "steps": 123,
            "layer_dim": 24,
            "clip_seconds": 1.5,
            "train.batch_size": 6,
            "evaluation.target_fp_per_hour": 0.25,
        },
    )
    assert config.train.steps == 123
    assert config.model.layer_dim == 24
    assert config.audio.clip_seconds == 1.5
    assert config.train.batch_size == 6
    assert config.evaluation.target_fp_per_hour == 0.25


def test_unknown_override_is_rejected():
    with pytest.raises(ConfigurationError):
        TrainerConfig.from_preset("quick", not_a_field=1)


def test_unknown_preset_is_rejected():
    with pytest.raises(ConfigurationError):
        TrainerConfig.from_preset("nonexistent")


def test_roundtrip_through_dict_and_json(tmp_path):
    config = TrainerConfig.from_preset(
        "balanced",
        word="hey computer",
        dataset=tmp_path,
        output_dir=tmp_path / "out",
        steps=321,
        augment={"enabled": False},
    )
    restored = TrainerConfig.from_dict(config.to_dict())
    assert restored.word == "hey computer"
    assert restored.train.steps == 321
    assert restored.augment.enabled is False

    path = config.save(tmp_path / "config.json")
    loaded = TrainerConfig.load(path)
    assert loaded.to_dict() == config.to_dict()


def test_validation_rejects_invalid_audio_settings():
    config = TrainerConfig.from_preset("quick", word="x")
    config.audio.clip_seconds = 0.1
    with pytest.raises(ConfigurationError):
        config.validate()

    config = TrainerConfig.from_preset("quick", word="x")
    config.audio.sample_rate = 8_000
    with pytest.raises(ConfigurationError):
        config.validate()


def test_validation_rejects_missing_dataset_and_negatives(tmp_path):
    config = TrainerConfig.from_preset("quick", word="x")
    config.dataset = tmp_path / "does-not-exist"
    with pytest.raises(ConfigurationError):
        config.validate()

    config = TrainerConfig.from_preset("quick", word="x")
    config.negatives.source = str(tmp_path / "missing.npy")
    with pytest.raises(ConfigurationError):
        config.validate()


def test_empty_word_is_rejected():
    config = TrainerConfig.from_preset("quick", word="   ")
    with pytest.raises(ConfigurationError):
        config.validate()


def test_clip_samples_helper():
    audio = AudioConfig(clip_seconds=1.5)
    assert audio.clip_samples == 24_000


def test_device_validation():
    config = TrainerConfig.from_preset("quick", word="x", device="cpu")
    config.validate()
    config.train.device = "tpu"
    with pytest.raises(ConfigurationError):
        config.validate()
