# TTS backends and plugins

Text-to-speech is **optional and not bundled**. It is useful for bootstrapping a dataset, covering phrases you cannot record easily, or adding voice variety — but synthetic samples are not a replacement for real recordings from the target microphone.

No TTS dependency is imported unless you use the corresponding backend.

---

## Built-in backends

| Backend | Extra | Notes |
| --- | --- | --- |
| `piper` | `tts-piper` | Fast, local, many languages; uses the `piper` CLI if installed, otherwise the Python API |
| `coqui` | `tts-coqui` | Coqui TTS / XTTS; heavier, supports voice cloning |
| `silero` | `tts-silero` | Silero models; downloads model files on first use |

Install what you need:

```bash
pip install "wakeword-trainer[gui,tts-piper]"
```

Check what is available:

```python
from wakeword_trainer.tts import registry

for name, info in registry.list_backends().items():
    print(name, info["available"], info["description"])

print(registry.voice_catalog("ru"))      # JSON-friendly list
```

Each `VoiceInfo` includes `id`, `name`, `language`, `backend`, `description`, `size_mb`, `license` and download URLs where applicable.

---

## Synthesizing samples

```python
from wakeword_trainer.tts import registry

# One utterance
result = registry.synthesize(
    "привет, компьютер",
    "samples/hello.wav",
    backend="piper",
    voice="ru_RU-irina-medium",
)
print(result.path, result.backend, result.voice)

# A batch into a dataset folder
results = registry.synthesize_many(
    [
        "привет, компьютер",
        "эй, компьютер",
        "компьютер, привет",
    ],
    "data/hey-computer",
    backend="piper",
    voice="ru_RU-irina-medium",
    prefix="tts",
)
```

`prefix` controls file names: `tts_0001.wav`, `tts_0002.wav`, ... — easy to delete in bulk if the experiment does not help.

Output is always a mono WAV at 16 kHz (the package resamples the engine output). Empty text raises `TTSError`.

### Downloading voices

```python
registry.download_voice("piper", "ru_RU-irina-medium")
```

Voices are stored in `<cache>/tts` and reused. If a voice needs a manual download, `list_backends()` / `list_voices()` shows the URLs and license. Always check the license of a voice before shipping a model trained on its output.

### Piper voices shipped in the catalog

The bundled catalog (`tts/catalog/piper_voices.json`) lists 16 medium-quality voices, including:

| Language | Voice IDs |
| --- | --- |
| Russian | `ru_RU-dmitri-medium`, `ru_RU-irina-medium` |
| English | `en_US-lessac-medium`, `en_US-amy-medium` |
| German | `de_DE-thorsten-medium` |
| French | `fr_FR-siwis-medium` |
| Spanish | `es_ES-davefx-medium` |
| Italian | `it_IT-paola-medium` |
| Polish | `pl_PL-darkman-medium` |
| Portuguese | `pt_BR-faber-medium` |
| Turkish | `tr_TR-dfki-medium` |
| Ukrainian | `uk_UA-ukrainian_tts-medium` |
| Chinese | `zh_CN-huayan-medium` |
| Dutch | `nl_NL-mls-medium` |
| Swedish | `sv_SE-nst-medium` |
| Czech | `cs_CZ-jirka-medium` |

---

## Custom backends (plugins)

Any `.py` file in the plugin folder is loaded at startup:

```python
from wakeword_trainer.tts import registry

print(registry.plugin_directory())          # where to put files
path = registry.install_plugin_template()   # create an editable template
registry.refresh_plugins()                  # reload after editing
print(registry.plugin_errors())             # broken plugins do not break the rest
```

A plugin defines a `TTSBackend` subclass and exposes it as `BACKEND`:

```python
from pathlib import Path

from wakeword_trainer.tts.base import SynthesisResult, TTSBackend, VoiceInfo


class MyEngine(TTSBackend):
    name = "my_engine"                 # unique; used as backend="my_engine"
    description = "Wraps my local TTS CLI"
    requires = ()                      # optional pip modules
    license = ""

    def available(self) -> bool:
        return shutil.which("my-tts") is not None

    def availability_error(self) -> str | None:
        return None if self.available() else "Install my-tts and put it on PATH."

    def voices(self) -> list[VoiceInfo]:
        return [VoiceInfo(id="default", name="Default", language="en", backend=self.name)]

    def synthesize(self, text, output_path, *, voice=None, language=None, sample_rate=16_000):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["my-tts", "--text", text, "--out", str(output_path)],
            check=True,
        )
        audio = load_audio(output_path, sample_rate=sample_rate)
        save_wav(output_path, audio, sample_rate)
        return SynthesisResult(
            path=output_path,
            backend=self.name,
            voice=voice or "default",
            language=language,
            text=text,
        )


BACKEND = MyEngine
```

Rules:

- write a **mono WAV** to `output_path`; the registry resamples it and writes it back normalized;
- raise `TTSError` for user-facing failures (missing model, bad voice);
- keep the module import fast — heavy dependencies should be imported inside `synthesize`;
- the registry looks for `BACKEND`, then `get_backend()`, then any `TTSBackend` subclass;
- a broken plugin is reported in `registry.plugin_errors()` and ignored.

The GUI creates the template and opens the folder for you.

---

## Recommendations

- Use TTS to **augment** real data, not to replace it. A model trained only on one synthetic voice usually fails on a real person.
- Generate several phrasings and speaking rates, then treat them like normal recordings — they will be split by group, so keep the `tts_` prefix and add a per-voice suffix if you generate many variants.
- Check the voice license before redistributing a trained model.
- Synthetic samples are ideal for testing whether a phrase is acoustically distinguishable before you record hundreds of takes.
