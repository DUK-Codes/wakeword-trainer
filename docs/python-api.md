# Python API

Everything importable from the top level:

```python
import wakeword_trainer as wwt
```

| Function | Purpose |
| --- | --- |
| `wwt.inspect_dataset(...)` | Validate a recordings folder, return a report |
| `wwt.prepare_dataset(...)` | Scan, split, resolve negatives, write preparation artifacts |
| `wwt.train(...)` | Full run: prepare → train → evaluate → export |
| `wwt.evaluate(...)` | Score an exported model on held-out clips |
| `wwt.predict(...)` | One score for a file or waveform |
| `wwt.detect(...)` | Thresholded detection result |
| `wwt.export_model(...)` | Copy/rebuild a deployable ONNX (or TFLite) |
| `wwt.WakeWordTrainer` | Stateful wrapper for step-by-step runs |

Exceptions are all subclasses of `wwt.WakeWordTrainerError`: `ConfigurationError`, `DatasetError`, `AudioFormatError`, `FeatureExtractionError`, `NegativeDataError`, `DownloadError`, `ExportError`, `TTSError`, `TrainingCancelled`.

---

## Dataset stages

### `inspect_dataset`

```python
report = wwt.inspect_dataset(
    "data/hey-computer",
    clip_seconds=2.0,
    group_by="stem",          # "stem" | "file" | callable(Path) -> str
    recursive=True,
    detect_duplicates=True,
)

print(report.ok)              # at least one valid clip
print(report.summary())       # JSON-friendly dict
report.save("dataset_report.json")
```

`summary()` includes file counts, total duration, sample rates, channels, group count, duplicates, rejected files and warnings.

Each valid clip is a `wwt.Clip` with `path`, `duration`, `sample_rate`, `channels` and `group`.

### `prepare_dataset`

```python
summary = wwt.prepare_dataset(
    "data/hey-computer",
    "models/hey-computer",
    word="hey computer",
    negatives="auto",
    download_negatives=True,
)
```

Writes `dataset_report.json`, `prepared.json`, `trainer_config.json` and `split/{train,val,test}.json` into the output folder. Useful when you want to inspect the split before spending compute.

---

## Training

```python
result = wwt.train(
    word="hey computer",
    dataset="data/hey-computer",
    output="models/hey-computer",
    preset="balanced",
    negatives="auto",
    download_negatives=False,
    export=True,
    export_tflite=False,
    resume=False,
    progress=True,
    # any config override:
    steps=15_000,
    batch_size=64,
    layer_dim=64,
    clip_seconds=2.0,
    seed=42,
)
```

### Parameters

| Name | Default | Meaning |
| --- | --- | --- |
| `word` | required | Phrase; used in metadata and file names |
| `dataset` | required | Folder with positive recordings |
| `output` | `wakeword_models/<slug>` | Output folder |
| `preset` | `"balanced"` | `quick` / `balanced` / `best` |
| `negatives` | `"auto"` | `auto`, `none`, `.npy` path, audio folder |
| `download_negatives` | `False` | Allow the one-time ~17 GB ACAV100M download |
| `callbacks` | `None` | List of `Callback` instances |
| `event_handler` | `None` | Function receiving `TrainingEvent` objects |
| `progress` | `False` | Print progress when no callback is provided |
| `export` | `True` | Export ONNX at the end |
| `export_tflite` | `False` | Also export TFLite (needs the `tflite` extra) |
| `resume` | `False` | Continue from `wakeword.pt` in the output folder |

### Result

`TrainingResult` fields:

```python
result.word
result.output_dir
result.best_step
result.total_steps
result.parameters
result.duration_seconds
result.best_metrics        # recall, fp_per_hour, auc, val_loss, threshold, ...
result.history
result.onnx_path
result.tflite_path
result.evaluation_payload
result.warnings
result.config
result.summary()           # JSON-friendly dict
```

### Live events

```python
from wakeword_trainer import TrainingEvent

def on_event(event: TrainingEvent) -> None:
    if event.kind == "step":
        print(f"{event.step}/{event.total_steps}")
    elif event.kind == "metrics":
        print(event.metrics)       # recall, fp_per_hour, auc, val_loss
    elif event.kind == "log":
        print(event.message)

wwt.train("hey computer", "data/hey-computer", "models/hey-computer", event_handler=on_event)
```

Event kinds: `prepare`, `start`, `step`, `metrics`, `checkpoint`, `log`, `done`, `error`, `cancelled`. Every event is JSON-serializable via `event.to_dict()`.

### Cancellation

```python
from wakeword_trainer import CancelToken, Trainer, TrainerConfig

cfg = TrainerConfig.from_preset(
    "balanced",
    word="hey computer",
    dataset="data/hey-computer",
    output_dir="models/hey-computer",
    negatives="data/negative_audio",
)

token = CancelToken()
trainer = Trainer(cfg, stop_token=token)

# From another thread / signal handler:
# token.cancel()

trainer.train(export=True)   # raises TrainingCancelled at the next safe point
```

The partial checkpoint (`wakeword.pt`) is kept, so the run can be resumed. The GUI uses the same mechanism through a marker file and a subprocess.

### Resume

```python
wwt.train(..., resume=True)
```

Restores model, optimizer and step from `wakeword.pt` and continues to `train.steps`.

---

## Configuration

`wwt.TrainerConfig` groups seven dataclasses:

```python
from wakeword_trainer import TrainerConfig

cfg = TrainerConfig.from_preset(
    "balanced",
    word="hey computer",
    dataset="data/hey-computer",
    output_dir="models/hey-computer",
    negatives="data/negative_audio",
    augment={"enabled": True, "noise_prob": 0.7},
    "train.batch_size": 96,
    "evaluation.target_fp_per_hour": 0.5,
)

cfg.validate()          # raises ConfigurationError with a readable message
cfg.save("config.json")
cfg2 = TrainerConfig.load("config.json")
```

Common fields:

| Section | Field | Default |
| --- | --- | --- |
| `audio` | `clip_seconds` | `2.0` |
| `audio` | `trim_silence`, `trim_top_db` | `True`, `40.0` |
| `audio` | `pad_mode` | `"random"` (`start`/`center`/`end`/`random`) |
| `features` | `window_size`, `step_size` | `76`, `8` |
| `augment` | `enabled`, `noise_prob`, `gain_prob`, `pitch_prob`, `speed_prob`, `reverb_prob`, `background_prob` | on, 0.5, 0.5, 0.3, 0.2, 0.2, 0.3 |
| `model` | `layer_dim`, `n_blocks`, `dropout` | preset-dependent |
| `train` | `steps`, `batch_size`, `learning_rate` | preset / 64 / `3e-4` |
| `train` | `negative_ratio_start`, `negative_ratio_end` | `4.0`, preset |
| `train` | `val_every`, `average_top_k` | preset |
| `train` | `device` (`"auto"`/`"cpu"`/`"cuda"`), `num_threads` | `"auto"`, `4` |
| `evaluation` | `target_fp_per_hour` | `1.0` |
| `evaluation` | `max_negative_windows` | `100_000` |
| `negatives` | `source`, `max_audio_hours` | `"auto"`, `5.0` |

Short override names work for the most common fields: `steps`, `batch_size`, `layer_dim`, `n_blocks`, `dropout`, `clip_seconds`, `sample_rate`, `trim_silence`, `pad_mode`, `negatives`, `download`, `augment`, `target_fp_per_hour`, `threshold_min/max/steps`, `max_negative_windows`, `augmented_test_variants`.

---

## Evaluation

```python
evaluation = wwt.evaluate(
    result,                         # or an .onnx path
    "data/hey-computer",
    split="test",                   # "test" | "all"
    negatives="auto",
    download_negatives=False,
    target_fp_per_hour=1.0,
    max_negative_windows=100_000,
    output_report=True,
)

print(evaluation.metrics["auc"])
print(evaluation.operating_point["recall"])
print(evaluation.operating_point["fp_per_hour"])
print(evaluation.threshold)
print(evaluation.report_dir)
```

`split="test"` re-creates the group-aware split with the same seed, so the clips are held out from training (provided the dataset has not changed). `split="all"` evaluates every clip — useful for a final sanity check, but optimistic.

Report files: `evaluation.json`, `evaluation.md`, `threshold_sweep.png`, `score_distribution.png`, plus per-clip rows inside the JSON.

### Reading the report

- **recall** — fraction of held-out positives above the chosen threshold;
- **FRR** — `1 - recall`;
- **fp_per_hour** — false positives on the negative pool, normalized by hours;
- **operating_point** — the highest-recall threshold within `target_fp_per_hour`;
- **at_threshold_0_5** — the same metrics at a fixed 0.5 threshold;
- **warnings** — statistically weak estimates, unattainable targets, perfect separation.

Negative audio is often measured in minutes, not hours, so `fp_per_hour` can be very noisy. Treat it as an ordering signal, not a guarantee.

---

## Inference

```python
score = wwt.predict("models/hey-computer/hey-computer.onnx", "test.wav")
score = wwt.predict(model, waveform_int16)          # numpy array

result = wwt.detect(model, "test.wav", threshold=0.6)
result.score, result.detected, result.threshold
result.as_dict()
```

Lower-level classes:

```python
from wakeword_trainer import WakeWordDetector, StreamingDetector

detector = WakeWordDetector("model.onnx", threshold=0.5, num_threads=2)
detector.score_features(features)     # (batch, frames, 96) -> scores
detector.score_audio(samples)         # int16 or float waveform -> float
detector.score_file("clip.wav")       # best window score
detector.detect_file("clip.wav")      # DetectionResult

stream = StreamingDetector(detector, hop_seconds=0.08)
for score in stream.push(block_1280_samples):   # 80 ms int16 blocks
    ...
```

`src/sample_rate` handling: pass 16 kHz audio. Files are resampled automatically; raw arrays are assumed to be at the requested rate.

---

## Export

```python
path = wwt.export_model(result, "deploy/hey-computer.onnx")
path = wwt.export_model("runs/hey-computer/wakeword.pt", "deploy/hey-computer")   # checkpoint
path = wwt.export_model("old.onnx", "deploy/copy.onnx")                            # copy
path = wwt.export_model(result, "deploy/hey-computer", format="tflite")            # needs extra
```

`export_model` accepts a `TrainingResult`, a `.pt` checkpoint, or an existing `.onnx` file.

---

## Object wrapper

```python
trainer = wwt.WakeWordTrainer(
    "hey computer",
    "data/hey-computer",
    "models/hey-computer",
    preset="balanced",
    negatives="auto",
    download_negatives=True,
)

trainer.prepare(force=False)   # -> summary dict
trainer.train(export=True, export_tflite=False, resume=False)
trainer.evaluate(use_exported=True)
trainer.predict("test.wav")
trainer.export("deploy/hey-computer.onnx", tflite=False)
trainer.exported_model()       # Path
trainer.model_metadata         # word, threshold, frames, ...
trainer.result                 # TrainingResult | None
trainer.prepared               # PreparedData | None
```

---

## Error handling

```python
import wakeword_trainer as wwt

try:
    wwt.train("hey computer", "missing-folder", "models/hey-computer")
except wwt.DatasetError as exc:
    print("dataset problem:", exc)
except wwt.ConfigurationError as exc:
    print("bad setting:", exc)
except wwt.DownloadError as exc:
    print("download failed:", exc)
except wwt.WakeWordTrainerError as exc:
    print("other trainer error:", exc)
```

Validation happens before any heavy work: invalid clip lengths, unknown overrides, missing datasets and missing negative sources fail fast with a specific message.
