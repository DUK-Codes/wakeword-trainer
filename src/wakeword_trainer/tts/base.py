"""Base types for optional speech synthesis backends."""

from __future__ import annotations

import importlib.util
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from ..exceptions import TTSError


@dataclass(frozen=True)
class VoiceInfo:
    id: str
    name: str
    language: str
    backend: str
    description: str = ""
    download_url: str | None = None
    config_url: str | None = None
    size_mb: float | None = None
    license: str = ""

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "language": self.language,
            "backend": self.backend,
            "description": self.description,
            "download_url": self.download_url,
            "size_mb": self.size_mb,
            "license": self.license,
        }


@dataclass
class SynthesisResult:
    path: Path
    backend: str
    voice: str | None
    language: str | None
    text: str
    extra: dict = field(default_factory=dict)


class TTSBackend(ABC):
    """Interface implemented by built-in and user-supplied backends.

    Subclasses must set :attr:`name` and implement :meth:`synthesize`.
    ``available()`` should report whether the required package is installed.
    """

    name: str = ""
    description: str = ""
    requires: tuple[str, ...] = ()
    license: str = ""

    def available(self) -> bool:
        if not self.requires:
            return True
        return all(importlib.util.find_spec(module) is not None for module in self.requires)

    def availability_error(self) -> str | None:
        if self.available():
            return None
        missing = [module for module in self.requires if importlib.util.find_spec(module) is None]
        return f"Backend {self.name!r} needs: {', '.join(missing)}. Install the matching optional extra."

    def languages(self) -> list[str]:
        return sorted({voice.language for voice in self.voices()})

    def voices(self) -> list[VoiceInfo]:
        return []

    def download_voice(self, voice: str, *, progress=None) -> Path:
        raise TTSError(f"Backend {self.name!r} does not support downloading voice {voice!r}.")

    @abstractmethod
    def synthesize(
        self,
        text: str,
        output_path: str | Path,
        *,
        voice: str | None = None,
        language: str | None = None,
        sample_rate: int = 16_000,
    ) -> SynthesisResult:
        """Synthesize ``text`` into a WAV file and return the result."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r} available={self.available()}>"
