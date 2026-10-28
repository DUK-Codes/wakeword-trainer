# Troubleshooting

## Installation

**`ModuleNotFoundError: No module named 'customtkinter'` / `sounddevice`**

The GUI dependencies are an extra:

```bash
pip install "wakeword-trainer[gui]"
```

**`ModuleNotFoundError: wakeword_trainer` after installing**

Make sure you installed the package into the same interpreter you run:

```bash
python -m pip install -e ".[gui,dev]"
python -c "import wakeword_trainer; print(wakeword_trainer.__file__)"
```

With a virtualenv, activate it first (`venv\Scripts\activate`).

**`pip` tries to build `torch` from source**

Update pip (`python -m pip install -U pip`). On Python versions without prebuilt wheels, install a supported Python (3.10–3.12).

---

## Audio and datasets

**`DatasetError: No audio files found in ... Supported extensions: ...`**

The folder is empty, files have other extensions, or they are nested deeper than expected. Supported natively: `.wav`, `.flac`, `.ogg`, `.oga`, `.opus`, `.aiff`, `.aif`, `.au`. MP3/M4A/AAC/WMA need `ffmpeg`.

**`AudioFormatError: This audio format requires ffmpeg ...`**

Install ffmpeg and put it on `PATH`, or set:

```powershell
$env:WAKEWORD_TRAINER_FFMPEG = "C:\tools\ffmpeg\bin\ffmpeg.exe"
```

```bash
export WAKEWORD_TRAINER_FFMPEG=/usr/local/bin/ffmpeg
```

**All clips were rejected as too short**

The default minimum is 0.25 s. Check the warnings from `inspect_dataset` — files shorter than that are rejected. Re-record or raise `clip_seconds`/`min_clip_seconds` only if that matches your use case.

**"Too few independent groups for a held-out split"**

Fewer than 3 groups or 6 clips. Everything goes to training and metrics are optimistic. Add recordings from more sessions/speakers, or pass a custom `group_by`.

**Augmented copies ended up in different splits**

Name copies with the recognized suffixes (`take_aug1.wav`, `take_noise.wav`, `take_v2.wav`) or provide `group_by`. See [dataset.md](dataset.md#grouping-and-leakage).

**Training wants 17 GB**

`negatives="auto"` needs the openWakeWord ACAV100M features. Either set `download_negatives=True` once, or use a folder of negative audio / a `.npy` file. See [negatives.md](negatives.md).

**`DownloadError: Downloaded ... bytes but expected ...`**

The download was interrupted. The `.part` file is kept for resume; run again. If it keeps failing, download the file manually into `<cache>/downloads` with the exact name, or use a local negative source.

---

## Training

**`ConfigurationError: train.steps must be at least 10`**

The pipeline needs a few steps to warm up. Use at least 10 (the `quick` preset uses 4 000).

**Training is very slow on CPU**

- use the `quick` preset first;
- reduce `batch_size` only if memory is the issue (it does not make it faster);
- set `device="cuda"` if you have a supported GPU (check `torch.cuda.is_available()`);
- close other memory-hungry applications;
- negative feature embedding for a large audio folder happens once and is cached.

**CUDA out of memory**

```python
wwt.train(..., device="cuda", batch_size=32)
```

or fall back to `device="cpu"` for small models.

**`TrainingCancelled`**

Expected after `token.cancel()` or the GUI cancel button. The checkpoint is preserved; re-run with `resume=True`.

**Resume starts from scratch**

`resume=True` looks for `wakeword.pt` in the output folder. If the previous run never wrote a checkpoint (very short run) or the folder changed, there is nothing to resume.

**Loss looks fine but the model triggers on everything**

Almost always missing or unrepresentative negatives. Use `negatives="auto"` with the download, or a folder with speech/music/TV. See [negatives.md](negatives.md).

**Training metrics are perfect, real-world recall is bad**

- too few recordings or one speaker/room;
- split too small to be meaningful (watch the warnings);
- overfitting: use more data, more augmentation, fewer steps or a smaller `layer_dim`;
- the evaluation used the same kind of audio as training — test on new takes.

---

## Export

**ONNX export fails with a torch/ONNX error**

Make sure `onnx` and `torch` are installed in compatible versions (`pip install -U "onnx>=1.14,<2" "torch>=2.0,<3"`). The exporter falls back if the `dynamo` argument is unsupported by your torch version.

**`ExportError: Training result has no exported ONNX model`**

Run training with `export=True` (the default), or export from the checkpoint: `wwt.export_model(".../wakeword.pt", "out.onnx")`.

**TFLite export is unavailable**

Install the extra (`pip install "wakeword-trainer[tflite]"`). Conversion depends on `ai-edge-torch`; if it fails, use ONNX.

**Model loads in Python but not in openWakeWord**

Check the metadata: `clip_seconds`/`frames` must match your runtime's feature window. Use `wwt.load_model_metadata(path)` to inspect. The input expected by the classifier is `(batch, frames, 96)` embeddings, not raw audio.

---

## GUI

**The app does not start / window is blank**

Run it from a terminal to see the traceback:

```bash
wakeword-trainer-gui
```

Most often a missing `gui` extra or a broken Tk installation. On Linux, install `python3-tk`.

**No microphone in the device list**

- Windows: Settings → Privacy → Microphone → allow desktop apps;
- check that another application is not holding the device in exclusive mode;
- `sounddevice` must be installed (`pip install "wakeword-trainer[gui]"`);
- try the system default device.

**Training stays at "starting" or "preparing"**

The worker writes its errors into the event stream — open the training log. Exit code 1 with a traceback in the log is the usual cause. For the first `auto` negative run, expect a long download.

**Cancel does not stop the run immediately**

Cancellation is cooperative: the worker checks between steps. If it does not respond, the GUI terminates the process after a timeout.

**Where is my state/config?**

`gui_state.json` in the per-user config folder (`%APPDATA%\wakeword-trainer` on Windows). Delete it to reset the app.

---

## Cache and disk

**How do I free space?**

Delete the cache folder:

- Windows: `%LOCALAPPDATA%\wakeword-trainer\Cache`
- Linux: `~/.cache/wakeword-trainer`
- macOS: `~/Library/Caches/wakeword-trainer`

or set `WAKEWORD_TRAINER_CACHE_DIR` to another disk before training.

**`PermissionError` / mmap errors on Windows**

Something still holds the feature cache. Close the app and any Python processes, then retry. The package loads cached features fully before deleting them, so this usually indicates an external process is reading the cache.

---

## Getting help

When reporting a problem, include:

- `python -c "import wakeword_trainer as wwt; print(wwt.__version__)"`;
- OS and Python version;
- the full traceback;
- the relevant part of `trainer_config.json` (remove personal paths if needed);
- whether `negatives` was `auto`, a `.npy` file or an audio folder.

Never upload personal recordings or trained personal models to a public issue tracker. Use a tiny synthetic reproduction instead.
