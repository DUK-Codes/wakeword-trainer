"""Filesystem locations used by the package.

All writable data (downloads, caches, user TTS plugins) lives outside the
installed package so that a wheel installation stays read-only.
"""

from __future__ import annotations

import os
from pathlib import Path

from platformdirs import user_cache_dir, user_config_dir

from .exceptions import FeatureExtractionError

APP_NAME = "wakeword-trainer"

_CACHE_ENV = "WAKEWORD_TRAINER_CACHE_DIR"
_PLUGIN_ENV = "WAKEWORD_TRAINER_PLUGIN_DIR"


def package_dir() -> Path:
    """Directory of the installed :mod:`wakeword_trainer` package."""
    return Path(__file__).resolve().parent


def resources_dir() -> Path:
    return package_dir() / "resources"


def bundled_model_paths() -> tuple[Path, Path]:
    """Return ``(melspectrogram.onnx, embedding_model.onnx)`` paths."""
    models = resources_dir() / "models"
    mel = models / "melspectrogram.onnx"
    emb = models / "embedding_model.onnx"
    for path in (mel, emb):
        if not path.is_file():
            raise FeatureExtractionError(
                f"Bundled feature model is missing: {path}. Reinstall the package or pass an explicit model path."
            )
    return mel, emb


def cache_dir() -> Path:
    """Root cache directory, overridable with ``WAKEWORD_TRAINER_CACHE_DIR``."""
    override = os.environ.get(_CACHE_ENV)
    path = Path(override).expanduser() if override else Path(user_cache_dir(APP_NAME, appauthor=False))
    path.mkdir(parents=True, exist_ok=True)
    return path


def downloads_dir() -> Path:
    path = cache_dir() / "downloads"
    path.mkdir(parents=True, exist_ok=True)
    return path


def features_cache_dir() -> Path:
    path = cache_dir() / "features"
    path.mkdir(parents=True, exist_ok=True)
    return path


def tts_models_dir() -> Path:
    path = cache_dir() / "tts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def tts_plugins_dir() -> Path:
    """Folder scanned for user-supplied TTS backends (``*.py``)."""
    override = os.environ.get(_PLUGIN_ENV)
    path = Path(override).expanduser() if override else Path(user_config_dir(APP_NAME, appauthor=False)) / "tts_plugins"
    path.mkdir(parents=True, exist_ok=True)
    return path


def config_dir() -> Path:
    path = Path(user_config_dir(APP_NAME, appauthor=False))
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    """Create ``path`` (including parents) and return it as a :class:`Path`."""
    resolved = Path(path).expanduser()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved
