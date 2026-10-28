# Architecture

The project is deliberately layered: a small public API on top of pipeline modules, a thin GUI on top of the same library, and no hidden global state except caches and the TTS registry.

```
                 ┌────────────────────────────┐
                 │        api.py              │  train / evaluate / predict / export
                 └────────────┬───────────────┘
                              │
     ┌────────────────────────┼─────────────────────────┐
     │                        │                         │
┌────▼─────┐          ┌───────▼────────┐        ┌───────▼──────┐
│training/ │          │  evaluation/   │        │  export/     │
│trainer   │          │ evaluator      │        │ onnx, tflite │
└────┬─────┘          └───────┬────────┘        └───────┬──────┘
     │                        │                         │
┌────▼────────────────────────▼─────────────────────────▼──────┐
│ data/ (dataset, split, augment, negatives, sampler)          │
│ audio/ (io, features)   models/ (dnn, inference)             │
└───────────────────────────────────────────────────────────────┘
```

## Modules

| Path | Responsibility |
| --- | --- |
| `api.py` | Public functions and the `WakeWordTrainer` wrapper; argument handling and validation |
| `config.py` | Dataclass configuration, presets, override resolution, JSON serialization |
| `paths.py` | Cache/download/TTS/plugin locations, bundled model paths |
| `audio/io.py` | Decoding, resampling, normalization to mono 16 kHz int16, writing, silence handling |
| `audio/features.py` | ONNX mel + embedding front end; frame-count math |
| `data/dataset.py` | Folder scan, clip validation, duplicate detection, grouping, manifests |
| `data/split.py` | Group-aware train/val/test assignment |
| `data/augment.py` | Gain/noise/pitch/speed/reverb/background augmentation, fixed-window rendering |
| `data/negatives.py` | ACAV/validation downloads, `.npy` pools, audio-folder embedding cache |
| `data/sampler.py` | Positive/negative batch construction |
| `models/dnn.py` | Fully-connected classifier (same topology as upstream) |
| `models/inference.py` | ONNX runtime wrapper, file scoring, rolling detector, metadata |
| `training/schedule.py` | Learning-rate warmup, negative-ratio schedule |
| `training/callbacks.py` | `Callback` protocol, `TrainingEvent`, console and cancellation callbacks |
| `training/trainer.py` | Prepare → train → validate → select → evaluate → export |
| `evaluation/metrics.py` | Recall/FRR, FP/hour, threshold sweeps, AUC, operating-point choice |
| `evaluation/evaluator.py` | Held-out scoring and report payload |
| `evaluation/report.py` | JSON, Markdown and PNG reports |
| `export/onnx.py` | ONNX export, metadata, runtime verification |
| `export/tflite.py` | Optional best-effort TFLite conversion |
| `tts/*` | Optional backends, registry, plugin loader, voice catalog |
| `gui/*` | CustomTkinter app, pages, worker subprocess, JSONL event tailing |

## Feature pipeline and parity

openWakeWord models do not consume raw audio. The front end is two ONNX models:

1. **`melspectrogram.onnx`** — input `(batch, samples)` float32 (int16-scaled values), output `(batch, 1, time, 32)`; transformed as `x / 10 + 2`;
2. **`embedding_model.onnx`** — sliding 76-frame windows with step 8, output 96-dimensional embeddings per window.

For 2 s at 16 kHz: `197` mel frames → `16` embedding windows. Both models are bundled in `resources/models` and loaded with ONNX Runtime (CPU by default). The exact same tensors are produced as by the upstream implementation — the test suite asserts frame counts and shape contracts, and the parity check used during development compared mel and embedding outputs on identical audio (`max abs diff = 0.0`).

Because the classifiers are trained on embeddings, models trained here are architecture-compatible with openWakeWord's own classifiers.

## Dataset and split

- `scan_dataset` probes every file, computes durations, groups by stem with augmentation suffixes stripped, and detects duplicates by SHA-1 of file content.
- `split_dataset` assigns whole groups greedily to train/val/test (12% / 18% by default), never splitting a group.
- If there are fewer than 3 groups or 6 clips, everything becomes training data and a warning explains that evaluation will be optimistic.

## Training loop

```
prepare():
  scan -> split -> resolve negatives -> cache audio negatives -> write artifacts
train():
  for step in 1..N:
    batch = positives (augmented) + negatives (scheduled ratio)
    forward/backward, LR schedule
    every val_every: score validation positives + negative sample, select by
                     (recall, -fp_per_hour, -val_loss)
    every checkpoint_every: save model+optimizer+step
  average top-k checkpoints -> final model
  final held-out evaluation -> report
  export ONNX (+ optional TFLite), verify
```

All randomness is seeded (`train.seed`). Checkpoints are saved on CPU tensors so they can be resumed on a different device.

## Events and the GUI process model

`TrainingEvent` is a small dataclass (`kind`, `step`, `total_steps`, `message`, `metrics`, `payload`, `timestamp`) that serializes to JSON. The library passes events to callbacks; the GUI starts

```
python -m wakeword_trainer.gui.worker --config config.json --events events.jsonl ...
```

as a **separate process**, writes events as JSONL, and tails the file. This keeps Tk responsive during heavy training and lets the GUI cancel through a marker file, with `terminate()` as a last resort. The worker writes `error`/`cancelled` events and a meaningful exit code.

The GUI itself is a `CTk` window with lazily built pages (`dataset`, `settings`, `train`, `results`, `test`, `tts`). Each page implements `build`, `retranslate`, `on_show`/`on_hide` where needed. `GuiState` persists paths and options to `gui_state.json`.

## Caches

| Cache | Key | Notes |
| --- | --- | --- |
| Downloaded negatives | file name + expected size | resumable, size-verified |
| Audio-folder features | SHA-1 of paths, sizes, mtimes, clip length, frames | memory-mapped `.npy` |
| TTS voices | backend + voice id | managed by each backend |

Caches live outside the package so wheels stay read-only and upgrades do not invalidate data.

## Testing strategy

- Synthetic audio generated at runtime (three-syllable “words”, noise) — no personal dataset, no network.
- Unit tests for I/O, frame math, splitting/grouping, augmentation, metrics, export/inference, TTS registry, i18n, GUI state.
- One `slow` end-to-end test trains a tiny model (20 steps, 1 s clips), checks artifacts and metadata, then runs `predict`, `detect`, `evaluate` and `export_model`.
- Feature parity with upstream is part of the design contract; frame counts (`197` / `16` for 2 s) are asserted explicitly.

## Design constraints

- **Standalone:** no import of `openwakeword`, no upstream training code.
- **Library-first:** the CLI surface is only the GUI entry point.
- **Honest evaluation:** group-aware splits, held-out metrics, explicit warnings.
- **No personal data in the repository:** `.gitignore`, tests and CI never touch real recordings.
