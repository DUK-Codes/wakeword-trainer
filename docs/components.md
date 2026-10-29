# Optional components

The desktop application ships with everything needed to train a wake word model:
the GUI, PyTorch, onnxruntime and the mel/embedding feature models. Speech
engines are **not** included, because most people never use them and they would
add tens of megabytes to every download.

Everything optional is a *component*: a small folder that can be added at install
time or downloaded later. Training negatives and voices are plain downloads and
are managed by the same page.

> Python users do not need components at all — install the matching extra
> (`wakeword-trainer[tts-piper]`) and the backend is imported from
> `site-packages` as usual.

---

## What the installer offers

The Windows setup has three choices:

| Type | Contents |
| --- | --- |
| **Application only** (default) | GUI, library, feature models, CPU PyTorch |
| **Application and speech engines** | the above plus the Piper engine |
| **Custom** | tick `piper` individually |

The choice is not final: the **Components** page can add or remove engines later.

## The Components page

Open **Components** in the sidebar. It shows:

- **Speech engines** — installed version and size, or the download size with an
  **Install** button. Engines deployed by the installer cannot be deleted from
  the app (Windows removes them with the application) and show a disabled button.
- **Training negatives** — whether the validation set (~185 MB) and ACAV100M
  (~17 GB) are already downloaded, with a download button and confirmation for
  the large archive.
- **Feature models** — confirms `melspectrogram.onnx` and `embedding_model.onnx`
  are present.
- **Data folders** — cache/downloads, components/plugins, with "open" buttons.

Downloads report progress in the bar above the cards and never block the window.
A failed download leaves a `.part` file that is resumed on the next attempt.

## Where components live

Two roots are scanned, in this order:

| Root | Written by | Removed by |
| --- | --- | --- |
| `<application folder>\components` | the installer | the uninstaller |
| `%APPDATA%\wakeword-trainer\components` | the Components page | the **Remove** button |

A component is a folder with a manifest and a `site-packages` tree:

```
components/
  tts-piper/
    component.json
    site-packages/
      piper/
      pathvalidate/
```

At import time the application appends every `site-packages` folder to
`sys.path` (and registers it as a DLL directory on Windows). Folders are
appended, not prepended, so the packages bundled with the application — numpy,
onnxruntime, torch — always win over a component copy.

`WAKEWORD_TRAINER_COMPONENTS_DIR` points at an extra root; useful for a portable
unzip on a read-only drive or for testing a component before installing it.

## Available engines

| Engine | Component | Size | Notes |
| --- | --- | --- | --- |
| Piper | `component-tts-piper.zip` | ~33 MB | Engine only; voices are downloaded from the Speech page (~63 MB each) |
| Coqui XTTS | — | — | Not part of the desktop build; `pip install wakeword-trainer[tts-coqui]` |

The engine list on the Speech page reflects what is actually importable. If an
engine is missing, the page shows an **Install engine…** button that opens the
Components page.

## Negatives and voices

These are data, not code, and are stored in the cache directory
(`%LOCALAPPDATA%\wakeword-trainer\Cache` by default):

| Data | Path | Size |
| --- | --- | --- |
| Validation features | `<cache>\downloads\validation_set_features.npy` | ~185 MB |
| ACAV100M features | `<cache>\downloads\openwakeword_features_ACAV100M_2000_hrs_16bit.npy` | ~17 GB |
| Piper voices | `<cache>\tts\piper\<voice>.onnx` | ~63 MB each |

They are kept when the application is uninstalled, so a reinstall does not
download them again. Delete the folders manually to reclaim the space.

## Installing a component by hand

```python
from wakeword_trainer import components

components.install_archive("component-tts-piper.zip")   # validates component.json
components.find_component("tts-piper")                  # (path, version) or None
components.remove_component("tts-piper")                # user-level components only
components.activate()                                   # re-scan without restarting
```

Archives are checked for unsafe paths and for a matching `id`. The manifest looks
like this:

```json
{
  "id": "tts-piper",
  "version": "0.1.2",
  "description": "Piper speech engine (voices are downloaded separately).",
  "python": "3.12",
  "packages": ["piper-tts==1.8.0", "pathvalidate>=3,<4"]
}
```

The `python` field records the interpreter used to build the archive; compiled
wheels must match the application's Python version (3.12 for the official build).

## Building components (maintainers)

`packaging/build_components.ps1` creates both the folder tree and the archives
from the same Python environment used for the application build:

```powershell
packaging\build_components.ps1 -Python ".\.venv-build\Scripts\python.exe" -Version 0.1.2
```

Dependencies already present in the application bundle are installed with
`--no-deps` on purpose: without that, pip would pull in a second copy of
onnxruntime, numpy and sympy and the Piper archive would grow from ~33 MB to
~80 MB. `packaging/build_win.ps1` calls this script automatically, so the
installer and the downloadable archives always match.

The archive names are versionless (`component-tts-piper.zip`) and are resolved
through `releases/latest/download/...`, so the Components page always fetches the
latest compatible build.
