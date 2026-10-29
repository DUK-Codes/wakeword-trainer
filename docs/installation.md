# Installation

## Requirements

- Python **3.10, 3.11 or 3.12** (for the pip package)
- ~2 GB free disk for dependencies, plus space for your dataset
- Optional: `ffmpeg` for MP3/M4A/AAC input
- Optional: NVIDIA GPU with CUDA for faster training (`torch` decides automatically; CPU works fine for `quick`/`balanced`)

The Windows build does not need Python at all.

## Install from PyPI

```bash
# Library + desktop app
pip install "wakeword-trainer[gui]"

# Library only
pip install wakeword-trainer
```

To pin a version or pre-release:

```bash
pip install "wakeword-trainer==0.1.2"
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
| `dev` | `pytest`, `pytest-cov`, `ruff`, `build`, `twine`, `pyinstaller` | Development and packaging |

Extras can be added later without reinstalling the base package:

```bash
pip install "wakeword-trainer[tts-piper]"
```

The desktop application uses a different mechanism — see
[components.md](components.md) — but the Python extras work the same way when the
app is started from a Python environment.

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

## Windows installer and portable archive

The desktop application is distributed in two forms, both on
[GitHub Releases](https://github.com/DUK-Codes/wakeword-trainer/releases/latest):

| Asset | Use |
| --- | --- |
| `wakeword-trainer-<version>-windows-x64-setup.exe` | normal installation: Start-menu shortcut, uninstall entry, optional engines |
| `wakeword-trainer-<version>-windows-x64-portable.zip` | run without installing; unzip and start `WakeWordTrainer.exe` |

The setup installs **per user** by default (`%LOCALAPPDATA%\Programs\WakeWord Trainer`)
and does not require administrator rights; the elevation dialog is offered if you
want a machine-wide installation. Windows 10 or newer, x64.

### Choosing what to install

The wizard offers:

- **Application only** — default; smallest download;
- **Application and speech engines** — adds the Piper engine;
- **Custom** — tick `piper` to add the engine to a minimal install.

Engines can be added or removed later on the **Components** page — see
[components.md](components.md). The base application never contains speech
engines.

For unattended deployment:

```powershell
# Application only
.\wakeword-trainer-0.1.2-windows-x64-setup.exe /VERYSILENT /TYPE=core

# With both engines
.\wakeword-trainer-0.1.2-windows-x64-setup.exe /VERYSILENT /TYPE=withtts

# Explicit component list
.\wakeword-trainer-0.1.2-windows-x64-setup.exe /VERYSILENT /COMPONENTS="piper"
```

The installer is unsigned, so Windows SmartScreen may show a warning. The
binaries are built from this repository by the release workflow; you can also
build them yourself (see [RELEASING.md](../RELEASING.md)).

### Uninstalling

**Settings → Apps → WakeWord Trainer → Uninstall** removes the application and
the engines that came with it. Downloaded voices, negatives and training results
are kept, because they can be large and are often stored outside the application
folder. Delete `%LOCALAPPDATA%\wakeword-trainer\Cache` and
`%APPDATA%\wakeword-trainer` to remove them as well.

### Starting on a specific page

For shortcuts and support screenshots both the installer build and the Python
entry point accept:

```powershell
WakeWordTrainer.exe --page components
wakeword-trainer-gui --lang en --page tts
```

Valid pages: `dataset`, `settings`, `train`, `results`, `test`, `tts`, `components`.

## Where files are stored

| Data | Location (override) |
| --- | --- |
| Feature cache (negative windows) | `user_cache_dir("wakeword-trainer")` → `WAKEWORD_TRAINER_CACHE_DIR` |
| Downloaded negatives | `<cache>/downloads` |
| TTS voice models | `<cache>/tts` |
| TTS plugins | `user_config_dir("wakeword-trainer")/tts_plugins` → `WAKEWORD_TRAINER_PLUGIN_DIR` |
| Downloaded components | `user_config_dir("wakeword-trainer")/components` → `WAKEWORD_TRAINER_COMPONENTS_DIR` (extra root) |
| GUI state | `user_config_dir("wakeword-trainer")/gui_state.json` |

On Windows the cache defaults to `%LOCALAPPDATA%\wakeword-trainer\Cache` and the
config to `%APPDATA%\wakeword-trainer`; on Linux to `~/.cache/wakeword-trainer` and
`~/.config/wakeword-trainer`; on macOS to `~/Library/Caches/wakeword-trainer` and
`~/Library/Application Support/wakeword-trainer`.

Training outputs go wherever you point `output` / `output_dir`; nothing is written into the installed package.

## Offline installation

The bundled mel and embedding models mean feature extraction works offline. Training with `negatives="auto"` needs a one-time download of the openWakeWord negative features; for fully offline training use a `.npy` file or a folder of negative audio.

The same applies to the desktop build: download voices and negatives while you
have a connection, or copy an existing cache folder to the target machine.

## Other platforms

Only Windows x64 has a prebuilt application. On Linux and macOS run the app from
source (`pip install "wakeword-trainer[gui]"`, then `wakeword-trainer-gui`) or use
the library directly; TTS engines are installed as pip extras there.
