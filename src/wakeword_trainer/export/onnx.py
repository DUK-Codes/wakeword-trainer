"""ONNX export with openWakeWord-compatible input/output names and metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..exceptions import ExportError
from ..version import __version__


def _metadata_dict(
    *,
    word: str,
    threshold: float,
    frames: int,
    clip_seconds: float,
    sample_rate: int,
    extra: dict[str, Any] | None,
) -> dict[str, str]:
    values = {
        "word": str(word),
        "threshold": f"{float(threshold):.6f}",
        "frames": str(int(frames)),
        "clip_seconds": f"{float(clip_seconds):.6f}",
        "sample_rate": str(int(sample_rate)),
        "version": __version__,
        "input_layout": "batch,frames,96",
        "output_layout": "batch,1",
    }
    if extra:
        for key, value in extra.items():
            values[str(key)] = json.dumps(value) if isinstance(value, (dict, list)) else str(value)
    return values


def export_onnx(
    model: torch.nn.Module,
    path: str | Path,
    *,
    frames: int,
    word: str = "",
    threshold: float = 0.5,
    clip_seconds: float = 2.0,
    sample_rate: int = 16_000,
    opset: int = 17,
    extra_metadata: dict[str, Any] | None = None,
    verify: bool = True,
) -> Path:
    """Export a trained classifier to ONNX.

    The exported graph takes ``input`` of shape ``(batch, frames, 96)`` and
    returns ``output`` of shape ``(batch, 1)`` with sigmoid probabilities.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = model.eval().to("cpu")
    dummy = torch.zeros((1, int(frames), 96), dtype=torch.float32)

    try:
        torch.onnx.export(
            model,
            dummy,
            str(path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )
    except TypeError:
        # Older/newer torch versions may not accept `dynamo`.
        torch.onnx.export(
            model,
            dummy,
            str(path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=opset,
            do_constant_folding=True,
        )
    except Exception as exc:  # pragma: no cover - torch/onnx environment dependent
        raise ExportError(f"ONNX export failed: {exc}") from exc

    try:
        import onnx

        onnx_model = onnx.load(str(path))
        onnx_model.doc_string = f"Wake word detector for {word!r} (wakeword-trainer {__version__})"
        del onnx_model.metadata_props[:]
        for key, value in _metadata_dict(
            word=word,
            threshold=threshold,
            frames=frames,
            clip_seconds=clip_seconds,
            sample_rate=sample_rate,
            extra=extra_metadata,
        ).items():
            entry = onnx_model.metadata_props.add()
            entry.key = key
            entry.value = value
        onnx.save(onnx_model, str(path))
    except Exception as exc:  # pragma: no cover
        raise ExportError(f"Failed to write ONNX metadata: {exc}") from exc

    if verify:
        verify_onnx(model, path, frames=frames)
    return path


def verify_onnx(
    model: torch.nn.Module | None,
    path: str | Path,
    *,
    frames: int,
    tolerance: float = 1e-4,
) -> dict[str, Any]:
    """Run the ONNX graph and optionally compare it with the PyTorch model."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    sample = np.random.default_rng(0).normal(0, 1, (2, int(frames), 96)).astype(np.float32)
    output = session.run([output_name], {input_name: sample})[0]
    output = np.asarray(output, dtype=np.float32).reshape(sample.shape[0], -1)
    result = {
        "input": session.get_inputs()[0].name,
        "output": session.get_outputs()[0].name,
        "input_shape": list(session.get_inputs()[0].shape),
        "output_shape": list(output.shape),
        "max_abs_diff": None,
    }
    if model is not None:
        model = model.eval().to("cpu")
        with torch.no_grad():
            expected = model(torch.from_numpy(sample)).numpy().reshape(-1, 1)
        diff = float(np.max(np.abs(expected - output)))
        result["max_abs_diff"] = diff
        if diff > tolerance:
            raise ExportError(f"ONNX output does not match PyTorch (max abs diff {diff:.3e} > {tolerance:.1e}).")
    return result
