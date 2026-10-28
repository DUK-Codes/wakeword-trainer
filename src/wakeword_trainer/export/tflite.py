"""Optional TFLite export.

Two back-ends are attempted in order:

1. ``ai-edge-torch`` (installed with ``pip install wakeword-trainer[tflite]``)
2. ``onnx2tf`` (TensorFlow based, much heavier)

TFLite conversion is best-effort: the exported ONNX model is always the
primary artifact.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch

from ..exceptions import ExportError


def tflite_available() -> bool:
    return importlib.util.find_spec("ai_edge_torch") is not None or importlib.util.find_spec("onnx2tf") is not None


def export_tflite(
    model: torch.nn.Module | None,
    path: str | Path,
    *,
    frames: int,
    onnx_path: str | Path | None = None,
    quantize: bool = False,
) -> Path:
    """Convert a trained model to ``.tflite``.

    Provide either ``model`` (preferred with ai-edge-torch) or an existing
    ``onnx_path`` (required for the onnx2tf fallback).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if model is not None and importlib.util.find_spec("ai_edge_torch") is not None:
        try:
            import ai_edge_torch

            model = model.eval().to("cpu")
            sample = (torch.randn(1, int(frames), 96),)
            edge = ai_edge_torch.convert(model, sample)
            edge.export(str(path))
            return _verify_tflite(path)
        except Exception as exc:  # pragma: no cover - depends on converter
            if onnx_path is None:
                raise ExportError(
                    f"ai-edge-torch conversion failed. Install onnx2tf or export ONNX only.\nUnderlying error: {exc}"
                ) from exc

    if importlib.util.find_spec("onnx2tf") is None:
        raise ExportError(
            "TFLite export requires the optional extra:\n"
            "    pip install wakeword-trainer[tflite]\n"
            "or convert the ONNX model with onnx2tf manually."
        )
    if onnx_path is None:
        raise ExportError("onnx2tf conversion requires an existing ONNX model (onnx_path=...).")

    # onnx2tf keeps its outputs next to the ONNX file by default.
    try:
        import onnx2tf

        onnx2tf.convert(input_onnx_file_path=str(onnx_path), output_folder_path=str(path.parent))
    except Exception as exc:  # pragma: no cover
        raise ExportError(f"onnx2tf conversion failed: {exc}") from exc

    candidates = sorted(path.parent.glob("*float32.tflite")) + sorted(path.parent.glob("*.tflite"))
    if not candidates:
        raise ExportError("onnx2tf finished but no .tflite file was produced.")
    produced = candidates[0]
    if produced != path:
        produced.replace(path)
    return _verify_tflite(path)


def _verify_tflite(path: Path) -> Path:
    """Sanity-check the TFLite file if a runtime is available."""
    try:
        from ai_edge_litert.interpreter import Interpreter
    except Exception:
        try:
            from tflite_runtime.interpreter import Interpreter  # type: ignore
        except Exception:
            return path
    interpreter = Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    shape = input_details["shape"]
    sample = np.zeros(shape, dtype=np.float32)
    interpreter.set_tensor(input_details["index"], sample)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])
    if output.size == 0:
        raise ExportError("TFLite model produced an empty output.")
    return path
