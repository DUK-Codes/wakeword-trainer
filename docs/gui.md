# GUI walkthrough

The desktop app wraps the library in a full cycle: validate data, record samples, configure, train, inspect results, test on a live microphone, export.

Launch it with:

```bash
wakeword-trainer-gui
```

or from a checkout:

```bash
python -m wakeword_trainer
```

The interface switches between **Русский** and **English** from the top bar; the choice is saved. Everything you enter is persisted in `gui_state.json` and restored next time.

---

## 1. Dataset

- Pick the folder with your wake word recordings. The app scans it recursively and shows:
  - number of valid/invalid files, total duration, sample rates, channels;
  - groups and duplicate pairs;
  - warnings (short clips, silence, too few files, inconsistent formats) — the
    **Checks** card appears after the first validation and stays hidden while the
    folder has not been scanned.
- **Record from the microphone** directly into the dataset folder:
  1. choose the input device (or use the system default);
  2. press **Start recording**, say the phrase, press **Stop and save**;
  3. the clip is normalized to 16 kHz mono and written as a new WAV file;
  4. **Discard** cancels the take.
- The level meter helps you avoid clipping and too-quiet recordings.

If the microphone is unavailable, the page stays usable for validation and explains how to install `sounddevice`.

Re-validate after adding files — the training page requires a successful validation.

## 2. Settings

- **Wake word** — the exact phrase; used in the model metadata and file name.
- **Output folder** — where the run writes its artifacts.
- **Preset** — `quick`, `balanced`, `best`. Changing the preset refreshes preset-controlled fields.
- **Negatives** — one of:
  - `auto` (openWakeWord feature sets; enable **Download negatives** for the first run);
  - a `.npy` feature file;
  - a folder of negative audio;
  - `none` (not recommended).
- **Advanced** (collapsible):
  - steps, batch size, layer dimension, number of blocks;
  - validation interval, checkpoint interval, early-stop patience;
  - seed, learning rate, clip length, target FP/hour;
  - device (`auto`/`cpu`/`cuda`) and augmentation toggle;
  - background-noise folder and RIR folder;
- **TFLite** — also export `.tflite` (requires the `tflite` extra).
- **Resume** — continue from the existing checkpoint in the output folder.

Invalid values are rejected with a readable message before training starts.

## 3. Training

- Starts a **separate worker process**, so a heavy run never freezes the interface.
- Shows progress (step/total), current loss, recall, FP/hour and a live chart.
- Streams the training log; warnings are highlighted.
- **Cancel** asks the worker to stop at a safe point; if it does not respond, the process is terminated.
- Closing the window during training asks for confirmation and stops the worker cleanly.
- The run writes `wakeword.pt` checkpoints, so a cancelled run can be resumed.

Events travel from the worker as JSONL and are tailed by the GUI — the same `TrainingEvent` objects the library API exposes.

## 4. Results

- Loads `training_result.json` and the evaluation report from the output folder.
- Shows best metrics, operating point, threshold table and per-clip scores.
- Buttons:
  - **Open output folder**;
  - **Open report** (Markdown/figures);
  - **Save ONNX as…** / **Save TFLite as…** — copy the exported model anywhere.
- If the run has no held-out evaluation (too few independent groups), the page says so instead of showing optimistic numbers.

## 5. Test

- Choose an ONNX model (the latest run is preselected).
- Set a threshold.
- **Start microphone** — rolling 2-second windows are scored every 80 ms; the page shows the current confidence, a bar and a **DETECTED** badge when the threshold is crossed.
- **Test a file** — score a recording and see the maximum window score.

This uses the same `StreamingDetector` and `WakeWordDetector` as the Python API, so the numbers match `wwt.predict`.

## 6. Speech (TTS)

Optional page for generating extra positive samples. The desktop build ships
**without** engines: the page lists them and, when one is missing, shows an
**Install engine…** button that jumps to the **Components** page.

- lists installed backends and whether they are available;
- shows the voice catalog (language, size, license);
- **Download voice** fetches the model into the cache;
- enter phrases (one per line), choose a voice and **Generate into dataset**;
- generated files go to the dataset folder with a `tts_` prefix, so they are easy to find and remove;
- **Open plugin folder** / **Create plugin template** manage custom backends.

None of this is required for training — it is a convenience for bootstrapping a dataset.

## 7. Components

The page for everything that is not part of the base build:

- **Speech engine** — Piper with size, status and an
  **Install**/**Remove** button. Engines installed with the setup are managed by
  Windows and cannot be removed here.
- **Training negatives** — download the validation set (~185 MB) or ACAV100M
  (~17 GB, asks for confirmation) before starting a long run.
- **Feature models** — confirms the bundled `melspectrogram.onnx` and
  `embedding_model.onnx` are present.
- **Data folders** — open the cache/downloads folder, the components/plugins
  folder, or create a TTS plugin template.

Downloads show a progress bar and keep the interface responsive. See
[components.md](components.md) for the folder layout and manual installation.

---

## State and files

| File | Contents |
| --- | --- |
| `gui_state.json` | last paths, phrase, preset, advanced values, language, last model |
| `<output>/training_result.json` | result summary shown on the Results page |
| `<output>/evaluation/*` | report the Results page opens |

The state file lives in the per-user config directory (`%APPDATA%\wakeword-trainer` on Windows). Delete it to reset the app.

## Keyboard and accessibility

The app uses standard Tk focus traversal (`Tab`/`Shift+Tab`). Text fields accept pasted paths. There is no dark/light theme switch — the interface is dark by design for long training sessions.

## Troubleshooting the GUI

See [troubleshooting.md](troubleshooting.md), especially:

- `ModuleNotFoundError: customtkinter` → install the `gui` extra;
- no microphone listed → check Windows microphone privacy settings and `sounddevice`;
- training page stuck at "starting" → read the log; the worker writes errors into the event stream.
