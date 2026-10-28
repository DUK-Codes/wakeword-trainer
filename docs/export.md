# Export and deployment

Training already exports the model unless you pass `export=False`. This page explains what is written, how to move it, and how to consume it.

---

## Output artifacts

```
models/hey-computer/
  hey-computer.onnx                  # classifier (torch -> ONNX)
  hey-computer.tflite                # optional
  wakeword.pt                        # training checkpoint (model + optimizer + step + config)
  trainer_config.json                # the exact config, reloadable with TrainerConfig.load()
  prepared.json                      # dataset report + split + negative source summary
  history.json                       # per-validation metrics
  training_result.json               # final summary (best step, metrics, paths, warnings)
  training_curves.png                # loss / recall / FP-hour curves
  split/
    train.json  val.json  test.json
    split.json
  evaluation/
    evaluation.json  evaluation.md
    threshold_sweep.png  score_distribution.png
```

| File | Purpose |
| --- | --- |
| `.onnx` | deploy this |
| `.pt` | resume training or re-export with a different threshold |
| `trainer_config.json` | exact reproduction |
| `training_result.json` | machine-readable summary for dashboards |
| `evaluation/` | honest held-out metrics |

---

## ONNX model contract

| Property | Value |
| --- | --- |
| Input name | `input` |
| Input shape | `(batch, frames, 96)` — `frames` is in metadata (16 for the default 2 s window) |
| Input dtype | `float32` |
| Output name | `output` |
| Output shape | `(batch, 1)` |
| Output | sigmoid probability in `[0, 1]` |
| Batch axis | dynamic |

The input is **speech embedding features**, not raw audio. The bundled `melspectrogram.onnx` + `embedding_model.onnx` (or openWakeWord's own copies) produce them.

### Metadata

The export stores metadata as ONNX `metadata_props`:

| Key | Example |
| --- | --- |
| `word` | `hey computer` |
| `threshold` | `0.5` (operating point chosen during training) |
| `clip_seconds` | `2.000000` |
| `frames` | `16` |
| `sample_rate` | `16000` |
| `version` | package version |
| `input_layout` | `batch,frames,96` |
| `output_layout` | `batch,1` |
| `extra` | JSON string with additional values passed at export |

Read them with:

```python
from wakeword_trainer import load_model_metadata

meta = load_model_metadata("hey-computer.onnx")
print(meta.word, meta.threshold, meta.frames, meta.clip_seconds)
```

The export verifies the ONNX graph against the PyTorch model (`max_abs_diff` on random inputs) and fails loudly if the runtime disagrees.

---

## Re-exporting

```python
import wakeword_trainer as wwt

# Copy an existing model
wwt.export_model("models/hey-computer/hey-computer.onnx", "deploy/hey-computer.onnx")

# Rebuild from a checkpoint with a different threshold
wwt.export_model(
    "models/hey-computer/wakeword.pt",
    "deploy/hey-computer.onnx",
    threshold=0.35,
)

# From a TrainingResult
wwt.export_model(result, "deploy/hey-computer.onnx")
```

The threshold is only metadata — it does not change the network. Your runtime decides when a score is a detection.

---

## Using the model with openWakeWord

The exported model is a drop-in classifier for openWakeWord-based assistants:

```python
from openwakeword.model import Model

model = Model(
    wakeword_models=["deploy/hey-computer.onnx"],
    inference_framework="onnx",
)

# audio_frame: 16 kHz, mono, int16 numpy array (80 ms blocks are typical)
scores = model.predict(audio_frame)
print(scores["hey-computer"])
```

openWakeWord supplies the mel/embedding front end, so you only need the classifier file. The input/output names and shapes match the upstream convention.

For a custom runtime, score the embeddings directly:

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("hey-computer.onnx", providers=["CPUExecutionProvider"])
scores = session.run(["output"], {"input": features})[0]   # features: (1, 16, 96) float32
```

`features` must come from the same feature pipeline — use `WakeWordDetector` if you do not want to reimplement it.

---

## Choosing a threshold

The training report contains a threshold table. Pick a point that satisfies your tolerance for false activations:

- **assistant in a quiet room** → lower threshold (higher recall, more false positives);
- **always-on device in a noisy home** → higher threshold (fewer false positives, more missed phrases);
- start from the report's `operating_point.threshold` for `target_fp_per_hour`, then verify on the real device.

The GUI **Test** page and `wwt.detect(..., threshold=...)` let you try thresholds without retraining.

---

## TFLite

```bash
pip install "wakeword-trainer[tflite]"
```

```python
wwt.train(..., export_tflite=True)
# or
wwt.export_model(result, "deploy/hey-computer", format="tflite")
```

TFLite export converts the ONNX classifier through `ai-edge-torch`. It is **best-effort**:

- conversion support depends on the installed versions;
- the resulting model is verified for basic load/shape compatibility, not bit-exactness;
- if conversion is unavailable, ONNX remains the fully supported path.

Check `result.warnings` and the log after a TFLite run.

---

## Reproducing a run

```python
from wakeword_trainer import TrainerConfig

config = TrainerConfig.load("models/hey-computer/trainer_config.json")
print(config.to_dict())
```

Keep `trainer_config.json`, `prepared.json` and `split/` together with the model if you need to explain or recreate a result. They contain paths, not audio.

---

## Shipping checklist

- [ ] evaluation on held-out clips looks reasonable (recall and FP/hour)
- [ ] tested on the target device with a **new** recording
- [ ] threshold chosen for the real environment
- [ ] model metadata (`word`, `clip_seconds`) matches your runtime
- [ ] license of the dataset and any TTS voices checked
- [ ] `.onnx` (and optional `.tflite`) copied out of the training folder
- [ ] you did not commit personal recordings or models to a public repository
