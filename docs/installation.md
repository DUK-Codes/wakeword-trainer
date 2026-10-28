# Installation

## Requirements

- Python **3.10, 3.11 or 3.12**
- ~2 GB free disk for dependencies, plus space for your dataset
- Optional: `ffmpeg` for MP3/M4A/AAC input
- Optional: NVIDIA GPU with CUDA for faster training (`torch` decides automatically; CPU works fine for `quick`/`balanced`)

## Install from PyPI

```bash
# Library + desktop app
pip install "wakeword-trainer[gui]"

# Library only
pip install wakeword-trainer
```

To pin a version or pre-release:

```bash
pip install "wakeword-trainer==0.1.0"
pip install --pre "wakeword-trainer[gui]"
```

## Install from GitHub (development builds)

```bash
pip install "wakeword-trainer[gui] @ git+https://github.com/DUK-Codes/wakeword-trainer.git"
```

## Install from a checkout

```bash
git clone https://github.com/DUK-Codes/wakeword-trainer.git
cd wakeword-trainer
python -m venv venv

# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

pip install -e ".[gui,dev]"
```

`-e` installs the package in editable mode and makes the bundled ONNX feature models available in place.

## Extras

| Extra | Packages | Needed for |
| --- | --- | --- |
| `gui` | `customtkinter`, `sounddevice` | Desktop app, microphone recording and live test |
| `tflite` | `ai-edge-torch` | TFLite export |
| `tts-piper` | `piper-tts` | Piper backend |
| `tts-coqui` | `TTS` | Coqui XTTS backend |
| `tts-silero` | `omegaconf` (+ Silero model download at runtime) | Silero backend |
| `dev` | `pytest`, `pytest-cov`, `ruff`, `build`, `twine`, `pyinstaller` | Development and packaging |

You can install extras later without reinstalling the base package:

```bash
pip install "wakeword-trainer[gui]"
```

## ffmpeg

`soundfile` (libsndfile) handles WAV, FLAC, OGG/Opus, AIFF and AU natively. MP3, M4A, AAC, WMA and video containers need `ffmpeg`:

- **Windows:** `winget install Gyan.FFmpeg` (or download from ffmpeg.org and add `bin` to `PATH`)
- **macOS:** `brew install ffmpeg`
- **Linux:** `sudo apt install ffmpeg`

The package looks for `ffmpeg` on `PATH`; you can also point at a specific binary:

```bash
# Windows PowerShell
$env:WAKEWORD_TRAINER_FFMPEG = "C:\tools\ffmpeg\bin\ffmpeg.exe"
```

If a format cannot be decoded and ffmpeg is missing, you get a clear `AudioFormatError` instead of a silent failure.

## Verify the install

```bash
python -c "import wakeword_trainer as wwt; print(wwt.__version__)"
python -c "from wakeword_trainer.audio.features import FeatureExtractor; print(FeatureExtractor().output_shape(32000))"
```

Expected feature shape for 2 seconds of audio: `(16, 96)`.

Launch the graphical app:

```bash
wakeword-trainer-gui
# or
python -m wakeword_trainer
```

## Where files are stored

| Data | Location (override) |
| --- | --- |
| Feature cache (negative windows) | `user_cache_dir("wakeword-trainer")` → `WAKEWORD_TRAINER_CACHE_DIR` |
| Downloaded negatives | `<cache>/downloads` |
| TTS voice models | `<cache>/tts` |
| TTS plugins | `user_config_dir("wakeword-trainer")/tts_plugins` → `WAKEWORD_TRAINER_PLUGIN_DIR` |
| GUI state | `user_config_dir("wakeword-trainer")/gui_state.json` |

On Windows the cache defaults to `%LOCALAPPDATA%\wakeword-trainer\Cache`, on Linux to `~/.cache/wakeword-trainer`, on macOS to `~/Library/Caches/wakeword-trainer`.

Training outputs go wherever you point `output` / `output_dir`; nothing is written into the installed package.

## Offline installation

The bundled mel and embedding models mean feature extraction works offline. Training with `negatives="auto"` needs a one-time download of the openWakeWord negative features; for fully offline training use a `.npy` file or a folder of negative audio.

## Windows executable (desktop app)

The library and the app are distributed separately:

- **library** — `pip install wakeword-trainer` (or `[gui]` to run the app from Python);
- **app** — a prebuilt `.exe` in [GitHub Releases](https://github.com/DUK-Codes/wakeword-trainer/releases).

The executable is self-contained (no Python installation needed), uses a CPU build of PyTorch, and runs the same GUI. Datasets, models and caches are still read from/written to the paths you choose. Windows SmartScreen may warn about unsigned binaries; build it yourself with `packaging/build_exe.ps1` if you prefer.

The executable is built for Windows x64 only. On other platforms run the app from source (`pip install "wakeword-trainer[gui]"`, then `wakeword-trainer-gui`) or use the library directly.
