"""Persisted GUI state and the TrainerConfig it builds."""

from __future__ import annotations

from pathlib import Path

import pytest

from wakeword_trainer.exceptions import ConfigurationError
from wakeword_trainer.gui import state as state_module
from wakeword_trainer.gui.state import GuiState


@pytest.fixture
def gui_state_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(state_module, "config_dir", lambda: tmp_path)
    return tmp_path


def test_defaults(gui_state_dir: Path):
    state = GuiState()
    assert state.language == "ru"
    assert state.preset == "balanced"
    assert state.negatives_mode == "auto"
    assert state.negatives_source() == "auto"


def test_save_and_load_roundtrip(gui_state_dir: Path):
    state = GuiState(
        language="en",
        dataset="D:/data/positives",
        output="D:/models/hey",
        word="hey computer",
        preset="best",
        negatives_mode="npy",
        negatives_path="D:/negatives.npy",
        download_negatives=True,
        export_tflite=True,
        resume=True,
        last_model="D:/models/hey.onnx",
        advanced={"steps": 999, "layer_dim": 128},
    )
    state.save()
    assert GuiState.path().is_file()

    loaded = GuiState.load()
    assert loaded.language == "en"
    assert loaded.dataset == "D:/data/positives"
    assert loaded.output == "D:/models/hey"
    assert loaded.word == "hey computer"
    assert loaded.preset == "best"
    assert loaded.negatives_mode == "npy"
    assert loaded.negatives_path == "D:/negatives.npy"
    assert loaded.download_negatives is True
    assert loaded.export_tflite is True
    assert loaded.resume is True
    assert loaded.advanced == {"steps": 999, "layer_dim": 128}


def test_load_ignores_corrupt_state(gui_state_dir: Path):
    GuiState.path().write_text("{not json", encoding="utf-8")
    state = GuiState.load()
    assert state.language == "ru"
    assert state.preset == "balanced"


def test_load_normalizes_unknown_values(gui_state_dir: Path):
    GuiState.path().write_text(
        '{"language": "de", "preset": "impossible", "negatives_mode": "weird"}',
        encoding="utf-8",
    )
    state = GuiState.load()
    assert state.language == "ru"
    assert state.preset == "balanced"


def test_negatives_source_modes(gui_state_dir: Path):
    state = GuiState(negatives_mode="none")
    assert state.negatives_source() == "none"

    state = GuiState(negatives_mode="audio", negatives_path="/tmp/neg")
    assert state.negatives_source() == "/tmp/neg"

    state = GuiState(negatives_mode="npy", negatives_path="")
    assert state.negatives_source() == "auto"


def test_build_config_applies_advanced_and_paths(gui_state_dir: Path, tmp_path: Path):
    state = GuiState(
        preset="quick",
        word="hey test",
        dataset=str(tmp_path),
        output=str(tmp_path / "out"),
        negatives_mode="none",
        advanced={
            "steps": 55,
            "layer_dim": 32,
            "background_dir": str(tmp_path),
            "_no_augment": True,
        },
    )
    config = state.build_config()
    assert config.word == "hey test"
    assert config.train.steps == 55
    assert config.model.layer_dim == 32
    assert config.augment.enabled is False
    assert config.augment.background_dir == tmp_path
    assert config.negatives.source == "none"


def test_build_config_rejects_invalid_values(gui_state_dir: Path, tmp_path: Path):
    state = GuiState(
        preset="quick",
        word="hey",
        dataset=str(tmp_path),
        output=str(tmp_path / "out"),
        advanced={"steps": 3},
    )
    with pytest.raises(ConfigurationError):
        state.build_config()
