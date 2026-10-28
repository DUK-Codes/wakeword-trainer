"""Backend discovery, user plugins and a small convenience API.

Built-in backends are imported lazily, so installing the base package never
pulls in Piper/Coqui/Silero. User backends are plain ``.py`` files inside the
plugin directory; each file may expose a ``TTSBackend`` subclass through a
module-level ``BACKEND`` object or class.

Run :func:`install_plugin_template` to drop a commented template into the
plugin folder and start from a working example.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
import traceback
from collections.abc import Iterable
from pathlib import Path
from types import ModuleType
from typing import Any

from ..exceptions import TTSError
from ..paths import tts_plugins_dir
from .base import SynthesisResult, TTSBackend, VoiceInfo

_BUILTIN = {
    "piper": ("wakeword_trainer.tts.piper", "PiperBackend"),
    "coqui": ("wakeword_trainer.tts.coqui", "CoquiBackend"),
    "silero": ("wakeword_trainer.tts.silero", "SileroBackend"),
}

_TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "custom_backend.py"

_instances: dict[str, TTSBackend] = {}
_plugins: dict[str, TTSBackend] = {}
_plugin_errors: dict[str, str] = {}
_plugins_loaded = False


# ---------------------------------------------------------------------------
# plugin loading
# ---------------------------------------------------------------------------


def plugin_directory() -> Path:
    """Folder scanned for user-supplied TTS backends."""
    return tts_plugins_dir()


def _load_module(path: Path) -> ModuleType:
    module_name = f"wakeword_trainer_tts_plugin_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise TTSError(f"Cannot load plugin module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _instantiate(module: ModuleType) -> TTSBackend:
    candidate = getattr(module, "BACKEND", None)
    if candidate is None and hasattr(module, "get_backend"):
        candidate = module.get_backend()
    if candidate is None:
        for value in vars(module).values():
            if isinstance(value, type) and issubclass(value, TTSBackend) and value is not TTSBackend:
                candidate = value
                break
    if candidate is None:
        raise TTSError("Plugin does not define BACKEND, get_backend(), or a TTSBackend subclass.")

    backend = candidate() if isinstance(candidate, type) else candidate
    if not isinstance(backend, TTSBackend):
        raise TTSError("Plugin BACKEND is not a TTSBackend instance.")
    if not backend.name:
        backend.name = module.__name__.rsplit("_", 1)[-1]
    return backend


def refresh_plugins() -> dict[str, str]:
    """Rescan the plugin folder. Returns a mapping of failed plugins to errors."""
    global _plugins_loaded  # noqa: PLW0603 - module-level cache
    _plugins.clear()
    _plugin_errors.clear()
    directory = plugin_directory()
    for path in sorted(directory.glob("*.py")):
        if path.name.startswith("_"):
            continue
        try:
            module = _load_module(path)
            backend = _instantiate(module)
            if backend.name in _BUILTIN:
                raise TTSError(f"Plugin {path.name!r} reuses the built-in backend name {backend.name!r}.")
            _plugins[backend.name] = backend
        except Exception as exc:  # noqa: BLE001 - a broken plugin must not break the app
            _plugin_errors[path.name] = f"{exc}\n{traceback.format_exc(limit=2)}"
    _plugins_loaded = True
    return dict(_plugin_errors)


def plugin_errors() -> dict[str, str]:
    if not _plugins_loaded:
        refresh_plugins()
    return dict(_plugin_errors)


def install_plugin_template(*, name: str = "my_tts_backend", overwrite: bool = False) -> Path:
    """Copy the backend template into the plugin folder and return its path."""
    target = plugin_directory() / f"{name}.py"
    if target.exists() and not overwrite:
        raise TTSError(f"Plugin already exists: {target}. Pass overwrite=True to replace it.")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(_TEMPLATE_PATH, target)
    return target


# ---------------------------------------------------------------------------
# backend registry
# ---------------------------------------------------------------------------


def _ensure_plugins() -> None:
    if not _plugins_loaded:
        refresh_plugins()


def available_backends() -> list[str]:
    """Names of backends whose dependencies are importable right now."""
    return [name for name, info in list_backends().items() if info["available"]]


def list_backends(*, include_unavailable: bool = True) -> dict[str, dict[str, Any]]:
    """Return ``{name: info}`` for built-in and user backends."""
    _ensure_plugins()
    result: dict[str, dict[str, Any]] = {}
    for name, (_module_name, _class_name) in _BUILTIN.items():
        info: dict[str, Any] = {
            "name": name,
            "builtin": True,
            "available": False,
            "description": "",
            "license": "",
            "requires": [],
            "error": None,
        }
        try:
            instance = get_backend(name)
            info["available"] = bool(instance.available())
            info["description"] = instance.description
            info["license"] = instance.license
            info["requires"] = list(instance.requires)
            info["error"] = instance.availability_error()
        except Exception as exc:  # noqa: BLE001
            info["error"] = str(exc)
        result[name] = info

    for name, instance in _plugins.items():
        try:
            available = bool(instance.available())
            error = instance.availability_error()
        except Exception as exc:  # noqa: BLE001
            available, error = False, str(exc)
        result[name] = {
            "name": name,
            "builtin": False,
            "available": available,
            "description": instance.description,
            "license": instance.license,
            "requires": list(instance.requires),
            "error": error,
        }

    if not include_unavailable:
        result = {name: info for name, info in result.items() if info["available"]}
    return result


def get_backend(name: str | TTSBackend) -> TTSBackend:
    """Return a backend instance by name (built-in or plugin)."""
    if isinstance(name, TTSBackend):
        return name
    _ensure_plugins()
    key = str(name).lower()
    if key in _instances:
        return _instances[key]
    if key in _plugins:
        _instances[key] = _plugins[key]
        return _instances[key]
    if key not in _BUILTIN:
        known = sorted({*_BUILTIN, *_plugins})
        raise TTSError(f"Unknown TTS backend {name!r}. Available: {', '.join(known) or 'none'}.")
    module_name, class_name = _BUILTIN[key]
    import importlib

    module = importlib.import_module(module_name)
    backend = getattr(module, class_name)()
    _instances[key] = backend
    return backend


def list_voices(
    backend: str | None = None,
    *,
    language: str | None = None,
    available_only: bool = False,
) -> list[VoiceInfo]:
    """List downloadable/installed voices, optionally filtered."""
    names = [backend] if backend else list(list_backends())
    voices: list[VoiceInfo] = []
    for name in names:
        try:
            instance = get_backend(name)
        except Exception:  # noqa: BLE001
            continue
        if available_only and not instance.available():
            continue
        try:
            items = instance.voices()
        except Exception:  # noqa: BLE001 - a backend must not break the list
            items = []
        for voice in items:
            if language and not voice.language.lower().startswith(language.lower()):
                continue
            voices.append(voice)
    return voices


def voice_catalog(language: str | None = None) -> list[dict[str, Any]]:
    """JSON-friendly voice list for GUIs."""
    return [voice.as_dict() for voice in list_voices(language=language)]


def download_voice(backend: str, voice: str, *, progress=None) -> Path:
    return get_backend(backend).download_voice(voice, progress=progress)


def synthesize(
    text: str,
    output_path: str | Path,
    *,
    backend: str = "piper",
    voice: str | None = None,
    language: str | None = None,
    sample_rate: int = 16_000,
) -> SynthesisResult:
    """Synthesize one utterance with the selected backend."""
    instance = get_backend(backend)
    error = instance.availability_error()
    if error:
        raise TTSError(error)
    return instance.synthesize(
        text,
        output_path,
        voice=voice,
        language=language,
        sample_rate=sample_rate,
    )


def synthesize_many(
    texts: Iterable[str],
    output_dir: str | Path,
    *,
    backend: str = "piper",
    voice: str | None = None,
    language: str | None = None,
    prefix: str = "tts",
    sample_rate: int = 16_000,
    progress=None,
) -> list[SynthesisResult]:
    """Synthesize several phrases into ``output_dir`` (used to build a dataset)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    texts = [text.strip() for text in texts if text and text.strip()]
    results: list[SynthesisResult] = []
    for index, text in enumerate(texts, start=1):
        target = output_dir / f"{prefix}_{index:04d}.wav"
        results.append(
            synthesize(
                text,
                target,
                backend=backend,
                voice=voice,
                language=language,
                sample_rate=sample_rate,
            )
        )
        if progress is not None:
            progress(index, len(texts), target.name)
    return results
