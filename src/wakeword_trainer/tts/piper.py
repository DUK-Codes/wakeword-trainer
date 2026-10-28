"""Piper TTS backend (optional).

Install with ``pip install wakeword-trainer[tts-piper]``. Voice models are
downloaded on demand from the official Piper voice repository; they are not
bundled with this package.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from ..audio.io import load_audio, save_wav
from ..exceptions import TTSError
from ..paths import tts_models_dir
from .base import SynthesisResult, TTSBackend, VoiceInfo

CATALOG_PATH = Path(__file__).resolve().parent / "catalog" / "piper_voices.json"


def _catalog() -> dict:
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


class PiperBackend(TTSBackend):
    name = "piper"
    description = "Fast local neural TTS from the Piper project (many languages, downloadable voices)."
    license = "Piper is MIT-licensed; individual voice models carry their own dataset licenses."

    def __init__(self) -> None:
        self._cache: dict[str, dict] = {}

    # ------------------------------------------------------------------
    def available(self) -> bool:
        if shutil.which("piper") is not None:
            return True
        try:
            import importlib.util

            return importlib.util.find_spec("piper") is not None
        except Exception:  # noqa: BLE001
            return False

    def availability_error(self) -> str | None:
        if self.available():
            return None
        return "Piper is not installed. Run `pip install wakeword-trainer[tts-piper]` or install `piper-tts` manually."

    def voices(self) -> list[VoiceInfo]:
        catalog = _catalog()
        base = catalog["base_url"]
        voices: list[VoiceInfo] = []
        for item in catalog["voices"]:
            voices.append(
                VoiceInfo(
                    id=item["id"],
                    name=item["name"],
                    language=item["language"],
                    backend=self.name,
                    download_url=f"{base}{item['path']}.onnx",
                    config_url=f"{base}{item['path']}.onnx.json",
                    size_mb=item.get("size_mb"),
                    license=item.get("license", ""),
                )
            )
        return voices

    # ------------------------------------------------------------------
    def _voice_paths(self, voice_id: str) -> tuple[Path, Path]:
        root = tts_models_dir() / "piper"
        root.mkdir(parents=True, exist_ok=True)
        return root / f"{voice_id}.onnx", root / f"{voice_id}.onnx.json"

    def is_downloaded(self, voice_id: str) -> bool:
        model, config = self._voice_paths(voice_id)
        return model.is_file() and config.is_file()

    def download_voice(self, voice: str, *, progress=None) -> Path:
        from ..data.negatives import download_file

        info = next((item for item in self.voices() if item.id == voice), None)
        if info is None:
            raise TTSError(f"Unknown Piper voice: {voice!r}. Use list_voices(backend='piper').")
        model_path, config_path = self._voice_paths(voice)
        if info.download_url:
            download_file(info.download_url, model_path, progress=progress)
        if info.config_url:
            download_file(info.config_url, config_path, progress=progress)
        return model_path

    # ------------------------------------------------------------------
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
        model_path, config_path, voice_id = self._resolve_voice(voice, language)
        if not self.is_downloaded(voice_id) and not (model_path and model_path.is_file()):
            self.download_voice(voice_id)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "piper.wav"
            self._run_piper(text, model_path, config_path, raw)
            if not raw.is_file() or raw.stat().st_size == 0:
                raise TTSError("Piper produced no audio.")
            audio = load_audio(raw, sample_rate=sample_rate)
            save_wav(output_path, audio, sample_rate)

        return SynthesisResult(
            path=output_path,
            backend=self.name,
            voice=voice_id,
            language=language,
            text=text,
        )

    def _resolve_voice(self, voice: str | None, language: str | None) -> tuple[Path, Path, str]:
        if voice and Path(voice).exists():
            model_path = Path(voice)
            config_path = Path(str(model_path) + ".json")
            return model_path, config_path, model_path.stem

        voices = self.voices()
        if not voices:
            raise TTSError("Piper voice catalog is empty.")
        if voice is None:
            language = language or "ru"
            prefix = language.split("_")[0].lower()
            candidates = [item for item in voices if item.language.lower().startswith(prefix)]
            chosen = candidates[0] if candidates else voices[0]
        else:
            chosen = next((item for item in voices if item.id == voice), None)
            if chosen is None:
                raise TTSError(f"Unknown Piper voice: {voice!r}.")
        model_path, config_path = self._voice_paths(chosen.id)
        return model_path, config_path, chosen.id

    @staticmethod
    def _run_piper(text: str, model_path: Path, config_path: Path, output: Path) -> None:
        executable = shutil.which("piper")
        if executable:
            command = [
                executable,
                "--model",
                str(model_path),
                "--output_file",
                str(output),
            ]
            if config_path.is_file():
                command.extend(["--config", str(config_path)])
            proc = subprocess.run(
                command,
                input=text.encode("utf-8"),
                capture_output=True,
                check=False,
            )
            if proc.returncode != 0:
                raise TTSError(proc.stderr.decode("utf-8", "replace")[-800:] or "Piper failed.")
            return

        try:
            from piper import PiperVoice
        except Exception as exc:  # noqa: BLE001
            raise TTSError(
                f"Piper is installed but neither the `piper` executable nor the Python API could be used: {exc}"
            ) from exc

        try:
            import wave

            voice = PiperVoice.load(
                str(model_path),
                config_path=str(config_path) if config_path.is_file() else None,
            )
            with wave.open(str(output), "wb") as handle:
                voice.synthesize(text, handle)
        except TypeError:
            # Older piper releases return raw PCM instead of writing to a wave handle.
            voice = PiperVoice.load(str(model_path))  # type: ignore[call-arg]
            chunks = list(voice.synthesize_stream_raw(text))  # type: ignore[attr-defined]
            data = b"".join(chunks)
            import wave

            with wave.open(str(output), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(22050)
                handle.writeframes(data)
        except Exception as exc:  # noqa: BLE001
            raise TTSError(f"Piper synthesis failed: {exc}") from exc
