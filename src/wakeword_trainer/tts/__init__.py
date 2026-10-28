"""Optional TTS plugin system.

No TTS engine is bundled or installed by default. Backends are discovered
lazily and only import their heavy dependencies when used.
"""

from __future__ import annotations

from .base import SynthesisResult, TTSBackend, VoiceInfo
from .registry import (
    available_backends,
    get_backend,
    install_plugin_template,
    list_backends,
    list_voices,
    plugin_directory,
    refresh_plugins,
    synthesize,
)

__all__ = [
    "SynthesisResult",
    "TTSBackend",
    "VoiceInfo",
    "available_backends",
    "get_backend",
    "install_plugin_template",
    "list_backends",
    "list_voices",
    "plugin_directory",
    "refresh_plugins",
    "synthesize",
]
