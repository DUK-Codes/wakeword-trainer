# wakeword-trainer

Train **openWakeWord-compatible** wake word models on your own recordings — from a Python script or a desktop app. Standalone: no `openwakeword` dependency, no upstream notebooks.

[Русская версия](README.ru.md)

## Install

```bash
pip install wakeword-trainer          # library
pip install "wakeword-trainer[gui]"   # + desktop app and microphone support
```

Python 3.10+. The feature models (mel + embedding) are bundled.

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

Prebuilt Windows `.exe` (no Python required) is attached to [Releases](https://github.com/DUK-Codes/wakeword-trainer/releases). The library itself is distributed separately via pip.

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
| [Installation](docs/installation.md) | extras, ffmpeg, caches |
| [Python API](docs/python-api.md) | every function, config, callbacks |
| [GUI](docs/gui.md) | page-by-page walkthrough |
| [Dataset guide](docs/dataset.md) | recording, grouping, splits |
| [Negatives](docs/negatives.md) | auto / `.npy` / audio folder |
| [TTS](docs/tts.md) | Piper, Coqui, Silero, plugins |
| [Export & deployment](docs/export.md) | ONNX contract, openWakeWord integration |
| [Troubleshooting](docs/troubleshooting.md) | common errors |
| [Architecture](docs/architecture.md) | modules and pipeline |

## Development

```bash
git clone https://github.com/DUK-Codes/wakeword-trainer.git
cd wakeword-trainer
pip install -e ".[gui,dev]"
ruff check src tests && pytest -q
```

## License

Apache-2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
