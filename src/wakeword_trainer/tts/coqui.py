"""Coqui XTTS backend (optional).

Install with ``pip install wakeword-trainer[tts-coqui]``.

Coqui XTTS v2 supports zero-shot voice cloning from a short reference clip and
many languages. It is a large download and is only loaded when used.

Licensing note: Coqui XTTS model weights are released under the Coqui Public
Model License (CPML), which is **non-commercial**. You are responsible for
checking that your use is permitted.
"""

from __future__ import annotations

from pathlib import Path

from ..audio.io import load_audio, save_wav
from ..exceptions import TTSError
from ..paths import tts_models_dir
from .base import SynthesisResult, TTSBackend, VoiceInfo

_SPEAKERS = [
    ("Ana Florence", "en", "Soft female English voice."),
    ("Claribel Dervla", "en", "Clear female English voice."),
    ("Daisy Studious", "en", "Bright female English voice."),
    ("Gracie Wise", "en", "Warm female English voice."),
    ("Alison Dietlinde", "en", "Mature female English voice."),
    ("Torcull Diarmuid", "en", "Male English voice."),
    ("Ludvig Milivoj", "en", "Male English voice."),
    ("Baldur Sanjin", "en", "Deep male English voice."),
    ("Andrew Chipper", "en", "Male English voice."),
    ("Badr Odhiambo", "en", "Male English voice."),
]

_LANGUAGES = [
    "en",
    "es",
    "fr",
    "de",
    "it",
    "pt",
    "pl",
    "tr",
    "ru",
    "nl",
    "cs",
    "ar",
    "zh-cn",
    "ja",
    "hu",
    "ko",
    "hi",
]


class CoquiBackend(TTSBackend):
    name = "coqui"
    description = "Coqui XTTS v2: multilingual TTS with zero-shot voice cloning."
    requires = ("TTS",)
    license = "Coqui XTTS weights use the non-commercial Coqui Public Model License (CPML)."

    def __init__(self) -> None:
        self._model = None
        self._model_name: str | None = None

    # ------------------------------------------------------------------
    def availability_error(self) -> str | None:
        if self.available():
            return None
        return (
            "Coqui TTS is not installed. Run `pip install wakeword-trainer[tts-coqui]`.\n"
            "Note: XTTS weights are non-commercial; check the CPML before publishing generated audio."
        )

    def voices(self) -> list[VoiceInfo]:
        voices = [
            VoiceInfo(
                id=name,
                name=name,
                language=language,
                backend=self.name,
                description=description,
                license="Coqui Public Model License (non-commercial)",
            )
            for name, language, description in _SPEAKERS
        ]
        voices.append(
            VoiceInfo(
                id="__clone__",
                name="Custom clone (reference WAV)",
                language="multi",
                backend=self.name,
                description=(
                    "Pass a 6+ second reference recording as `voice=/path/to/reference.wav` to clone that speaker."
                ),
                license="Reference audio must be yours or properly licensed.",
            )
        )
        return voices

    # ------------------------------------------------------------------
    def _load(self, model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"):
        try:
            from TTS.api import TTS
        except Exception as exc:  # noqa: BLE001
            raise TTSError(f"Could not import Coqui TTS: {exc}") from exc

        if self._model is not None and self._model_name == model_name:
            return self._model
        cache = tts_models_dir() / "coqui"
        cache.mkdir(parents=True, exist_ok=True)
        try:
            self._model = TTS(model_name=model_name, progress_bar=False)
        except Exception as exc:  # noqa: BLE001
            raise TTSError(f"Could not load Coqui model {model_name!r}: {exc}") from exc
        self._model_name = model_name
        return self._model

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
        model = self._load()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        speaker = None
        speaker_wav = None
        if voice and Path(voice).is_file():
            speaker_wav = str(Path(voice).expanduser())
        elif voice:
            speaker = voice
        else:
            speaker = _SPEAKERS[0][0]

        language = (language or "en").lower()
        if language not in _LANGUAGES:
            # XTTS accepts the language code it knows; fall back to English.
            language = "en" if language not in {"ru"} else "ru"

        kwargs = {
            "text": text,
            "language": language,
            "file_path": str(output_path),
        }
        if speaker_wav:
            kwargs["speaker_wav"] = speaker_wav
        else:
            kwargs["speaker"] = speaker
        try:
            model.tts_to_file(**kwargs)
        except TypeError:
            # Some releases expect `split_sentences`/`speed` keywords only.
            model.tts_to_file(
                text=text,
                language=language,
                speaker=speaker,
                file_path=str(output_path),
            )
        except Exception as exc:  # noqa: BLE001
            raise TTSError(f"Coqui synthesis failed: {exc}") from exc

        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise TTSError("Coqui produced no audio.")
        audio = load_audio(output_path, sample_rate=sample_rate)
        save_wav(output_path, audio, sample_rate)
        return SynthesisResult(
            path=output_path,
            backend=self.name,
            voice=speaker_wav or speaker,
            language=language,
            text=text,
            extra={"model": self._model_name},
        )
