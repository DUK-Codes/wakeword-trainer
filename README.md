# wakeword-trainer

Train **openWakeWord-compatible** wake word models on your own recordings — from a Python script or a desktop app. Standalone: no `openwakeword` dependency, no upstream notebooks.

[Русская версия](https://github.com/DUK-Codes/wakeword-trainer/blob/main/README.ru.md)

Telegram contact: @DUK_Creativity
Have a question? Message me.

[![PyPI](https://img.shields.io/badge/PyPI-wakeword--trainer-blue)](https://pypi.org/project/wakeword-trainer/)
[![Release](https://img.shields.io/github/v/release/DUK-Codes/wakeword-trainer)](https://github.com/DUK-Codes/wakeword-trainer/releases/latest)

## Install

```bash
pip install wakeword-trainer          # library
pip install "wakeword-trainer[gui]"   # + desktop app and microphone support
```

Python 3.10+. The feature models (mel + embedding) are bundled with the package.

Prefer no Python at all? Download the Windows **installer**
`wakeword-trainer-<version>-windows-x64-setup.exe` (or the portable
`...-portable.zip`) from [Releases](https://github.com/DUK-Codes/wakeword-trainer/releases/latest).
The installer lets you tick the optional speech engine (Piper); it is not
installed by default and can be added later on the **Components** page.

If PyPI is unreachable, install from source:

```bash
pip install "wakeword-trainer[gui] @ git+https://github.com/DUK-Codes/wakeword-trainer.git"
```

## Quick start

```python
import wakeword_trainer as wwt

result = wwt.train(
    word="hey computer",
    dataset="data/hey-computer",      # folder with your recordings
    output="models/hey-computer",
    preset="balanced",                # quick | balanced | best
    negatives="auto",                 # or a .npy file / folder of negative audio
    download_negatives=True,          # one-time ~17 GB download (optional)
)

print(result.best_metrics)
print(wwt.predict(result.onnx_path, "test.wav"))
```

Use your own negatives (no download):

```python
wwt.train(
    "hey computer",
    "data/hey-computer",
    "models/hey-computer",
    negatives="data/negative_audio",
)
```

Inspect data before training:

```python
report = wwt.inspect_dataset("data/hey-computer")
print(report.summary())
```

## Desktop app

```bash
wakeword-trainer-gui
```

Dataset validation, microphone recording, training progress and charts, held-out metrics, live mic test, TTS samples, ONNX/TFLite export — all in one window, RU/EN.

Everything runs on your machine: the feature models ship with the package, PyTorch does the training, and datasets, models and voices stay in the folders you choose. No account or API key is required.

A prebuilt Windows installer and a portable archive (no Python required) are attached to [Releases](https://github.com/DUK-Codes/wakeword-trainer/releases). The library itself is distributed separately via pip.

### Optional components

The base application is deliberately small. Extras are installed later:

- **Speech engine** (Piper) — tick it in the installer or add it on the
  **Components** page; Coqui XTTS remains available through the Python package only.
- **Training negatives** — downloaded automatically during training or prepared in advance
  on the same page (validation set ~185 MB, ACAV100M ~17 GB).
- **Voices** — downloaded from the Speech page when you need them.

See [docs/components.md](https://github.com/DUK-Codes/wakeword-trainer/blob/main/docs/components.md).

## What you get

- **Dataset tools** — recursive scan, validation, duplicate detection, group-aware train/val/test split (augmented copies never leak).
- **Honest evaluation** — recall, FRR, false positives per hour, AUC, threshold sweep, JSON/Markdown/PNG reports on held-out clips.
- **Training** — presets, augmentation, negative-ratio schedule, checkpoint averaging, resume, live JSONL events.
- **Export** — ONNX with metadata and runtime verification; TFLite optional.
- **Inference** — `predict`, `detect`, `WakeWordDetector`, rolling `StreamingDetector`.

## Data in short

50–300 short recordings of the exact phrase, from the microphones and rooms where it will be used; 1.5–2.5 s each; WAV/FLAC/OGG (MP3/M4A with `ffmpeg`). Negatives are required for a usable model.

## Documentation

| | |
| --- | --- |
| [Installation](https://github.com/DUK-Codes/wakeword-trainer/blob/main/docs/installation.md) | extras, installer, ffmpeg, caches |
| [Python API](https://github.com/DUK-Codes/wakeword-trainer/blob/main/docs/python-api.md) | every function, config, callbacks |
| [GUI](https://github.com/DUK-Codes/wakeword-trainer/blob/main/docs/gui.md) | page-by-page walkthrough |
| [Components](https://github.com/DUK-Codes/wakeword-trainer/blob/main/docs/components.md) | optional engines, voices, negatives |
| [Dataset guide](https://github.com/DUK-Codes/wakeword-trainer/blob/main/docs/dataset.md) | recording, grouping, splits |
| [Negatives](https://github.com/DUK-Codes/wakeword-trainer/blob/main/docs/negatives.md) | auto / `.npy` / audio folder |
| [TTS](https://github.com/DUK-Codes/wakeword-trainer/blob/main/docs/tts.md) | Piper, Coqui, plugins |
| [Export & deployment](https://github.com/DUK-Codes/wakeword-trainer/blob/main/docs/export.md) | ONNX contract, openWakeWord integration |
| [Troubleshooting](https://github.com/DUK-Codes/wakeword-trainer/blob/main/docs/troubleshooting.md) | common errors |
| [Architecture](https://github.com/DUK-Codes/wakeword-trainer/blob/main/docs/architecture.md) | modules and pipeline |

## Development

```bash
git clone https://github.com/DUK-Codes/wakeword-trainer.git
cd wakeword-trainer
pip install -e ".[gui,dev]"
ruff check src tests && pytest -q
```

## License

Apache-2.0 — see [LICENSE](https://github.com/DUK-Codes/wakeword-trainer/blob/main/LICENSE) and [NOTICE](https://github.com/DUK-Codes/wakeword-trainer/blob/main/NOTICE).
