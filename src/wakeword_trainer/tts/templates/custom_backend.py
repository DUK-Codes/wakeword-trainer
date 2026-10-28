"""Template for a custom TTS backend.

How to use it
-------------
1. Copy this file into the plugin folder::

       import wakeword_trainer as wwt
       print(wwt.tts.install_plugin_template())   # or use the GUI button

   The folder is printed by ``wakeword_trainer.tts.plugin_directory()``.

2. Edit the class below: set ``name``, implement ``synthesize`` and optionally
   override ``voices`` / ``available``.

3. Restart the app or call ``wakeword_trainer.tts.refresh_plugins()``.

The file only has to define a :class:`~wakeword_trainer.tts.base.TTSBackend`
subclass and expose it as ``BACKEND`` (a class or an instance).

Requirements
------------
``synthesize`` must write a mono WAV file (any sample rate; the package
resamples it) and return a :class:`SynthesisResult`. Raise ``TTSError`` for
user-facing failures.

Example: wrapping a command-line TTS engine
-------------------------------------------
    subprocess.run([exe, "--voice", voice, "--text", text, "--out", str(path)], check=True)
"""

from __future__ import annotations

from pathlib import Path

from wakeword_trainer.audio.io import load_audio, save_wav
from wakeword_trainer.exceptions import TTSError
from wakeword_trainer.tts.base import SynthesisResult, TTSBackend, VoiceInfo


class MyBackend(TTSBackend):
    # Unique name shown in the GUI and used by ``synthesize(backend="my_backend")``.
    name = "my_backend"
    description = "Example custom backend. Replace this with a real implementation."
    # Optional: pip modules that must be importable for this backend to work.
    requires = ()
    license = ""

    def available(self) -> bool:
        # Return False (and explain in availability_error) when the engine or a
        # voice model is missing.
        return True

    def availability_error(self) -> str | None:
        return None

    def voices(self) -> list[VoiceInfo]:
        return [
            VoiceInfo(
                id="default",
                name="My default voice",
                language="en",
                backend=self.name,
                description="Replace with your own voice list.",
            )
        ]

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

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------
        # Replace this block with a call to your TTS engine, writing a WAV to
        # ``output_path``.
        #
        # A silent placeholder is written so the template runs out of the box:
        import numpy as np

        seconds = max(0.5, len(text.split()) * 0.35)
        samples = np.zeros(int(seconds * 16_000), dtype=np.float32)
        save_wav(output_path, samples, 16_000)
        # ------------------------------------------------------------------

        audio = load_audio(output_path, sample_rate=sample_rate)
        save_wav(output_path, audio, sample_rate)
        return SynthesisResult(
            path=output_path,
            backend=self.name,
            voice=voice or "default",
            language=language,
            text=text,
        )


# The registry looks for BACKEND, then get_backend(), then any TTSBackend subclass.
BACKEND = MyBackend
