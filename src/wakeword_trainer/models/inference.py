"""Inference helpers for exported ONNX models."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnxruntime as ort

from ..audio.features import FeatureExtractor
from ..audio.io import load_audio
from ..exceptions import ExportError, FeatureExtractionError

DEFAULT_CLIP_SECONDS = 2.0


@dataclass
class DetectionResult:
    score: float
    threshold: float
    detected: bool

    def as_dict(self) -> dict:
        return {"score": round(self.score, 6), "threshold": self.threshold, "detected": self.detected}


@dataclass
class ModelMetadata:
    word: str = ""
    threshold: float = 0.5
    sample_rate: int = 16_000
    clip_seconds: float = DEFAULT_CLIP_SECONDS
    frames: int = 16
    version: str = ""
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "word": self.word,
            "threshold": self.threshold,
            "sample_rate": self.sample_rate,
            "clip_seconds": self.clip_seconds,
            "frames": self.frames,
            "version": self.version,
            **self.extra,
        }


def _pad_or_trim(samples: np.ndarray, target: int) -> np.ndarray:
    arr = np.asarray(samples)
    if arr.size >= target:
        return arr[:target]
    if arr.size == 0:
        return np.zeros(target, dtype=np.int16)
    pad = target - arr.size
    left = pad // 2
    fill = np.zeros(target, dtype=arr.dtype)
    fill[left : left + arr.size] = arr
    return fill


class WakeWordDetector:
    """Scores audio with an exported ONNX wake word model."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        feature_extractor: FeatureExtractor | None = None,
        threshold: float | None = None,
        clip_seconds: float | None = None,
        providers: Iterable[str] | None = None,
        num_threads: int = 2,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.is_file():
            raise ExportError(f"Model file not found: {model_path}")
        self.model_path = model_path
        self.metadata = load_model_metadata(model_path)
        self.clip_seconds = float(clip_seconds or self.metadata.clip_seconds or DEFAULT_CLIP_SECONDS)
        self.threshold = float(threshold if threshold is not None else (self.metadata.threshold or 0.5))
        self.extractor = feature_extractor or FeatureExtractor(num_threads=num_threads)
        self.clip_samples = int(round(self.clip_seconds * 16_000))

        available = set(ort.get_available_providers())
        resolved = [p for p in (providers or ["CPUExecutionProvider"]) if p in available]
        if "CPUExecutionProvider" not in resolved:
            resolved.append("CPUExecutionProvider")
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, num_threads)
        self._session = ort.InferenceSession(str(model_path), options, providers=resolved)
        self._input = self._session.get_inputs()[0].name
        self._output = self._session.get_outputs()[0].name
        self._input_frames = self.metadata.frames or None

    # ------------------------------------------------------------------
    def score_features(self, features: np.ndarray) -> np.ndarray:
        arr = np.asarray(features, dtype=np.float32)
        single = arr.ndim == 2
        if single:
            arr = arr[None, ...]
        if arr.ndim != 3:
            raise FeatureExtractionError(f"Expected (frames, 96) features, got shape {arr.shape}.")
        arr = self._adapt_frames(arr)
        out = self._session.run([self._output], {self._input: np.ascontiguousarray(arr, dtype=np.float32)})[0]
        scores = np.asarray(out, dtype=np.float32).reshape(arr.shape[0], -1)[:, 0]
        return scores[0] if single else scores

    def _adapt_frames(self, features: np.ndarray) -> np.ndarray:
        frames = features.shape[1]
        expected = None
        shape = self._session.get_inputs()[0].shape
        if isinstance(shape, list) and len(shape) == 3 and isinstance(shape[1], int) and shape[1] > 0:
            expected = shape[1]
        if expected is None:
            return features
        if frames == expected:
            return features
        if frames > expected:
            return features[:, :expected, :]
        pad = np.repeat(features[:, -1:, :], expected - frames, axis=1)
        return np.concatenate([features, pad], axis=1)

    # ------------------------------------------------------------------
    def score_audio(self, samples: np.ndarray, *, sample_rate: int = 16_000) -> float:
        if sample_rate != 16_000:
            from ..audio.io import resample

            samples = resample(samples.astype(np.float32), sample_rate, 16_000)
        audio = _pad_or_trim(samples, self.clip_samples)
        features = self.extractor.embeddings(audio)
        if features.shape[0] == 0:
            return 0.0
        return float(self.score_features(features))

    def score_file(self, path: str | Path, *, sample_rate: int = 16_000) -> float:
        audio = load_audio(path, sample_rate=sample_rate)
        return self.score_audio(audio, sample_rate=sample_rate)

    def score_file_windows(self, path: str | Path) -> tuple[float, np.ndarray]:
        """Return ``(max_score, per_window_scores)`` for a long file."""
        audio = load_audio(path)
        if audio.size == 0:
            return 0.0, np.zeros((0,), dtype=np.float32)
        features = self.extractor.embeddings(audio)
        if features.shape[0] == 0:
            return 0.0, np.zeros((0,), dtype=np.float32)
        scores = np.atleast_1d(np.asarray(self.score_features(features), dtype=np.float32))
        return float(np.max(scores)), scores

    def detect(self, samples: np.ndarray, *, sample_rate: int = 16_000) -> DetectionResult:
        score = self.score_audio(samples, sample_rate=sample_rate)
        return DetectionResult(score, self.threshold, score >= self.threshold)

    def detect_file(self, path: str | Path) -> DetectionResult:
        score, _ = self.score_file_windows(path)
        return DetectionResult(score, self.threshold, score >= self.threshold)

    # yaml-style convenience alias
    predict = detect


class StreamingDetector:
    """Rolling 2-second detector for microphone input."""

    def __init__(
        self,
        detector: WakeWordDetector,
        *,
        hop_seconds: float = 0.08,
    ) -> None:
        self.detector = detector
        self.hop = max(1, int(round(hop_seconds * 16_000)))
        self._buffer = np.zeros((0,), dtype=np.int16)
        self._since_last = 0

    def reset(self) -> None:
        self._buffer = np.zeros((0,), dtype=np.int16)
        self._since_last = 0

    def push(self, samples: np.ndarray) -> float | None:
        block = np.asarray(samples, dtype=np.int16)
        self._buffer = np.concatenate([self._buffer, block])
        self._since_last += block.size
        target = self.detector.clip_samples
        if self._buffer.size > target:
            self._buffer = self._buffer[-target:]
        if self._buffer.size < target or self._since_last < self.hop:
            return None
        self._since_last = 0
        return self.detector.score_audio(self._buffer)


def load_model_metadata(path: str | Path) -> ModelMetadata:
    """Read metadata written by :func:`wakeword_trainer.export.export_onnx`."""
    path = Path(path)
    try:
        import onnx

        model = onnx.load(str(path), load_external_data=False)
    except Exception:
        return ModelMetadata()
    values: dict = {}
    for prop in model.metadata_props:
        values[prop.key] = prop.value
    if not values:
        return ModelMetadata()
    extra = {
        k: v
        for k, v in values.items()
        if k not in {"word", "threshold", "sample_rate", "clip_seconds", "frames", "version"}
    }
    try:
        frames = int(values.get("frames", 0)) or 0
    except ValueError:
        frames = 0
    try:
        clip_seconds = float(values.get("clip_seconds", DEFAULT_CLIP_SECONDS))
    except ValueError:
        clip_seconds = DEFAULT_CLIP_SECONDS
    try:
        threshold = float(values.get("threshold", 0.5))
    except ValueError:
        threshold = 0.5
    try:
        sample_rate = int(values.get("sample_rate", 16_000))
    except ValueError:
        sample_rate = 16_000
    return ModelMetadata(
        word=values.get("word", ""),
        threshold=threshold,
        sample_rate=sample_rate,
        clip_seconds=clip_seconds,
        frames=frames,
        version=values.get("version", ""),
        extra=extra,
    )


def metadata_to_json(metadata: ModelMetadata) -> str:
    return json.dumps(metadata.as_dict(), indent=2, ensure_ascii=False)


def expected_frames(clip_seconds: float) -> int:
    samples = int(round(clip_seconds * 16_000))
    mel = max(0, int(math.floor(samples / 160.0)) - 3)
    return 1 + (mel - 76) // 8 if mel >= 76 else 0
