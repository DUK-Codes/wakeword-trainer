"""Silero TTS backend (optional).

Silero models are small, fast and run on CPU. Two loading paths are supported:

1. the ``silero`` pip package (``pip install silero``), and
2. ``torch.hub`` from the ``snakers4/silero-models`` repository (downloads the
   model on first use, then caches it).

The ``wakeword-trainer[tts-silero]`` extra installs ``omegaconf``, which the
Silero models require.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..audio.io import save_wav
from ..exceptions import TTSError
from .base import SynthesisResult, TTSBackend, VoiceInfo

_RU_SPEAKERS = ["aidar", "baya", "kseniya", "xenia", "eugene"]
_EN_SPEAKERS = [f"en_{index}" for index in range(0, 118)]

_MODEL_IDS = {
    "ru": "v4_ru",
    "en": "v3_en",
    "de": "v3_de",
    "es": "v3_es",
    "fr": "v3_fr",
}


class SileroBackend(TTSBackend):
    name = "silero"
    description = "Silero TTS: compact CPU-friendly voices for Russian, English and more."
    requires = ("torch", "omegaconf")
    license = "Silero models are distributed under their own license; see the upstream repository."

    def __init__(self) -> None:
        self._model = None
        self._model_language: str | None = None
        self._sample_rate = 48_000

    # ------------------------------------------------------------------
    def availability_error(self) -> str | None:
        if self.available():
            return None
        return (
            "Silero TTS is not available. Run `pip install wakeword-trainer[tts-silero]` "
            "and, on first use, allow torch.hub to download the model."
        )

    def voices(self) -> list[VoiceInfo]:
        voices = [
            VoiceInfo(
                id=f"ru:{speaker}",
                name=f"{speaker} (Russian)",
                language="ru",
                backend=self.name,
                description=f"Silero Russian speaker {speaker}.",
                license="See the snakers4/silero-models license.",
            )
            for speaker in _RU_SPEAKERS
        ]
        voices.extend(
            VoiceInfo(
                id=f"en:{speaker}",
                name=f"{speaker} (English)",
                language="en",
                backend=self.name,
                description=f"Silero English speaker {speaker}.",
                license="See the snakers4/silero-models license.",
            )
            for speaker in _EN_SPEAKERS
        )
        return voices

    # ------------------------------------------------------------------
    def _load(self, language: str):
        language = language.split("-")[0].lower()
        if language not in _MODEL_IDS:
            language = "en"
        if self._model is not None and self._model_language == language:
            return self._model

        model_id = _MODEL_IDS[language]
        try:
            self._model, _example_text, _speakers = self._load_via_package(language, model_id)
        except Exception:
            try:
                self._model, _example_text, _speakers = self._load_via_torch_hub(language, model_id)
            except Exception as exc:  # noqa: BLE001
                raise TTSError(f"Could not load the Silero model {model_id!r}: {exc}") from exc

        self._model_language = language
        return self._model

    @staticmethod
    def _load_via_package(language: str, model_id: str):
        from silero import silero_tts  # type: ignore

        model, example_text = silero_tts(language=language, speaker=model_id)
        return model, example_text, []

    @staticmethod
    def _load_via_torch_hub(language: str, model_id: str):
        import torch

        model, example_text = torch.hub.load(
            repo_or_dir="snakers4/silero-models",
            model="silero_tts",
            language=language,
            speaker=model_id,
            trust_repo=True,
        )
        return model, example_text, []

    # ------------------------------------------------------------------
    def _default_speaker(self, language: str) -> str:
        if language.startswith("ru"):
            return _RU_SPEAKERS[0]
        return "en_0"

    def synthesize(
        self,
        text: str,
        output_path: str | Path,
        *,
        voice: str | None = None,
        language: str | None = None,
        sample_rate: int = 16_000,
    ) -> SynthesisResult:
        if not text.strip():
            raise TTSError("Cannot synthesize empty text.")

        if voice and ":" in voice:
            prefix, _, speaker = voice.partition(":")
            language = language or prefix
        else:
            speaker = voice or self._default_speaker(language or "ru")
        language = (language or "ru").split("-")[0].lower()

        model = self._load(language)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            audio = model.apply_tts(
                text=text,
                speaker=speaker,
                sample_rate=self._sample_rate,
            )
        except TypeError:
            audio = model.apply_tts(
                text=text,
                speaker=speaker,
            )
        except Exception as exc:  # noqa: BLE001
            raise TTSError(f"Silero synthesis failed: {exc}") from exc

        samples = audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio)
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            raise TTSError("Silero produced no audio.")

        source_rate = int(getattr(model, "sample_rate", self._sample_rate) or self._sample_rate)
        if source_rate != sample_rate:
            from ..audio.io import resample

            samples = resample(samples, source_rate, sample_rate)

        save_wav(output_path, samples, sample_rate)
        return SynthesisResult(
            path=output_path,
            backend=self.name,
            voice=speaker,
            language=language,
            text=text,
            extra={"model": _MODEL_IDS.get(language, "v3_en")},
        )
