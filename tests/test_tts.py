"""TTS registry: built-in metadata, user plugins, template and synthesis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from wakeword_trainer.audio.io import load_audio
from wakeword_trainer.exceptions import TTSError
from wakeword_trainer.tts import registry


@pytest.fixture
def clean_registry():
    registry._instances.clear()
    registry._plugins.clear()
    registry._plugin_errors.clear()
    registry._plugins_loaded = False
    yield
    registry._instances.clear()
    registry._plugins.clear()
    registry._plugin_errors.clear()
    registry._plugins_loaded = False


def test_builtin_backends_are_listed(clean_registry):
    infos = registry.list_backends()
    assert {"piper", "coqui"} <= set(infos)
    for name in ("piper", "coqui"):
        info = infos[name]
        assert info["builtin"] is True
        assert isinstance(info["available"], bool)
        assert info["description"]


def test_plugin_template_roundtrip(clean_registry, tmp_path: Path):
    path = registry.install_plugin_template(name="demo_backend")
    assert path.is_file()

    errors = registry.refresh_plugins()
    assert errors == {}
    infos = registry.list_backends()
    assert "my_backend" in infos
    assert infos["my_backend"]["builtin"] is False
    assert infos["my_backend"]["available"] is True
    assert registry.available_backends().count("my_backend") == 1

    voices = registry.list_voices("my_backend")
    assert len(voices) == 1
    assert voices[0].id == "default"

    output = tmp_path / "hello.wav"
    result = registry.synthesize("hello world", output, backend="my_backend")
    assert result.backend == "my_backend"
    assert result.path.is_file()
    audio = load_audio(output)
    assert audio.size > 0

    results = registry.synthesize_many(["one", "two", "three"], tmp_path / "batch", backend="my_backend", prefix="line")
    assert len(results) == 3
    assert len(sorted((tmp_path / "batch").glob("line_*.wav"))) == 3


def test_install_template_refuses_to_overwrite(clean_registry):
    registry.install_plugin_template(name="demo_backend")
    with pytest.raises(TTSError):
        registry.install_plugin_template(name="demo_backend")
    replacement = registry.install_plugin_template(name="demo_backend", overwrite=True)
    assert replacement.is_file()


def test_broken_plugin_is_reported_without_breaking_others(clean_registry):
    directory = registry.plugin_directory()
    (directory / "broken.py").write_text("raise RuntimeError('boom')\n", encoding="utf-8")
    errors = registry.refresh_plugins()
    assert "broken.py" in errors
    assert "boom" in errors["broken.py"]
    # Built-ins are still listed.
    assert {"piper", "coqui"} <= set(registry.list_backends())


def test_unknown_backend_raises(clean_registry):
    with pytest.raises(TTSError):
        registry.get_backend("does-not-exist")
    with pytest.raises(TTSError):
        registry.synthesize("hi", "out.wav", backend="does-not-exist")


def test_voice_catalog_is_json_friendly(clean_registry):
    registry.install_plugin_template(name="demo_backend")
    registry.refresh_plugins()
    catalog = registry.voice_catalog()
    assert isinstance(catalog, list)
    assert all(isinstance(item, dict) for item in catalog)
    assert any(item["backend"] == "my_backend" for item in catalog)
    import json

    json.dumps(catalog)  # must not raise


def test_template_silence_is_well_formed(clean_registry, tmp_path: Path):
    registry.install_plugin_template(name="demo_backend")
    registry.refresh_plugins()
    backend = registry.get_backend("my_backend")
    result = backend.synthesize("a phrase with several words", tmp_path / "long.wav")
    audio = load_audio(result.path)
    assert audio.dtype == np.int16
    assert audio.size >= int(0.5 * 16_000)
