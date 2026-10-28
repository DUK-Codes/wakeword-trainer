"""openWakeWord-compatible feature extraction.

Two small ONNX models are used:

``melspectrogram.onnx``
    input  ``(batch, samples)`` float32 (int16-scaled values)
    output ``(batch, 1, time, 32)``

``embedding_model.onnx``
    input  ``(batch, 76, 32, 1)`` float32
    output ``(batch, 1, 1, 96)``

The mel output is transformed with ``x / 10 + 2`` to match the native
TensorFlow implementation used by Google's ``speech_embedding`` model.

Two seconds of 16 kHz audio (32000 samples) produce 197 mel frames and
16 embedding frames with a 76-frame window and an 8-frame hop.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import onnxruntime as ort

from ..exceptions import FeatureExtractionError
from ..paths import bundled_model_paths

MEL_BINS = 32
EMBEDDING_DIM = 96
DEFAULT_WINDOW = 76
DEFAULT_STEP = 8


def num_mel_frames(num_samples: int) -> int:
    """Number of mel frames produced for ``num_samples`` at 16 kHz."""
    if num_samples <= 0:
        return 0
    return max(0, int(math.floor(num_samples / 160.0)) - 3)


def num_embedding_frames(
    num_samples: int,
    window_size: int = DEFAULT_WINDOW,
    step_size: int = DEFAULT_STEP,
) -> int:
    """Number of embedding frames produced for ``num_samples`` at 16 kHz."""
    mel = num_mel_frames(num_samples)
    if mel < window_size:
        return 0
    return 1 + (mel - window_size) // step_size


def mel_frames_to_embedding_frames(
    num_frames: int,
    window_size: int = DEFAULT_WINDOW,
    step_size: int = DEFAULT_STEP,
) -> int:
    if num_frames < window_size:
        return 0
    return 1 + (num_frames - window_size) // step_size


class FeatureExtractor:
    """Batched mel + speech-embedding feature extractor."""

    def __init__(
        self,
        mel_model_path: str | Path | None = None,
        embedding_model_path: str | Path | None = None,
        *,
        window_size: int = DEFAULT_WINDOW,
        step_size: int = DEFAULT_STEP,
        providers: Iterable[str] | None = None,
        num_threads: int = 4,
        max_embedding_batch: int = 1024,
    ) -> None:
        bundled_mel, bundled_emb = bundled_model_paths()
        self.mel_model_path = Path(mel_model_path) if mel_model_path else bundled_mel
        self.embedding_model_path = Path(embedding_model_path) if embedding_model_path else bundled_emb
        self.window_size = int(window_size)
        self.step_size = int(step_size)
        self.max_embedding_batch = int(max_embedding_batch)

        for path in (self.mel_model_path, self.embedding_model_path):
            if not path.is_file():
                raise FeatureExtractionError(f"Feature model not found: {path}")

        self.providers = self._resolve_providers(providers)
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, int(num_threads))
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        try:
            self._mel = ort.InferenceSession(str(self.mel_model_path), options, providers=self.providers)
            self._emb = ort.InferenceSession(str(self.embedding_model_path), options, providers=self.providers)
        except Exception as exc:  # pragma: no cover - depends on runtime setup
            raise FeatureExtractionError(f"Failed to load ONNX feature models: {exc}") from exc

        self._mel_input = self._mel.get_inputs()[0].name
        self._emb_input = self._emb.get_inputs()[0].name
        self._mel_output = self._mel.get_outputs()[0].name
        self._emb_output = self._emb.get_outputs()[0].name

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _resolve_providers(providers: Iterable[str] | None) -> list[str]:
        available = set(ort.get_available_providers())
        if providers is None:
            return ["CPUExecutionProvider"]
        resolved = [p for p in providers if p in available]
        if not resolved:
            resolved = ["CPUExecutionProvider"]
        if "CPUExecutionProvider" not in resolved:
            resolved.append("CPUExecutionProvider")
        return resolved

    def output_shape(self, num_samples: int) -> tuple[int, int]:
        return num_embedding_frames(num_samples, self.window_size, self.step_size), EMBEDDING_DIM

    def num_windows(self, num_mel_frames_value: int) -> int:
        return mel_frames_to_embedding_frames(num_mel_frames_value, self.window_size, self.step_size)

    # --------------------------------------------------------- extraction
    def melspectrogram(self, samples: np.ndarray, *, transform: bool = True) -> np.ndarray:
        """Compute mel features.

        ``samples`` may be ``(samples,)`` or ``(batch, samples)`` and should
        contain int16-scaled values. Returns ``(frames, 32)`` or
        ``(batch, frames, 32)``.
        """
        arr = np.asarray(samples)
        single = arr.ndim == 1
        if single:
            arr = arr[None, :]
        if arr.ndim != 2:
            raise FeatureExtractionError(f"Expected 1-D or 2-D audio, got shape {arr.shape}.")
        audio = arr.astype(np.float32, copy=False)
        if audio.shape[1] == 0 or num_mel_frames(audio.shape[1]) < 1:
            # The mel model needs at least 640 samples (4 STFT frames).
            empty = np.zeros((audio.shape[0], 0, MEL_BINS), dtype=np.float32)
            return empty[0] if single else empty

        try:
            outputs = self._mel.run([self._mel_output], {self._mel_input: audio})
        except Exception as exc:  # pragma: no cover - runtime dependent
            raise FeatureExtractionError(f"Melspectrogram inference failed: {exc}") from exc

        spec = np.asarray(outputs[0], dtype=np.float32)
        # (batch, 1, time, 32) -> (batch, time, 32)
        spec = np.squeeze(spec, axis=1) if spec.ndim == 4 else spec
        if transform:
            spec = spec / 10.0 + 2.0
        return spec[0] if single else spec

    def embeddings_from_melspec(self, melspec: np.ndarray) -> np.ndarray:
        """Embed a mel spectrogram or a batch of mel spectrograms.

        Returns ``(n_windows, 96)`` for a single spectrogram and
        ``(batch, n_windows, 96)`` for a batch.
        """
        mel = np.asarray(melspec, dtype=np.float32)
        single = mel.ndim == 2
        if single:
            mel = mel[None, ...]
        if mel.ndim != 3 or mel.shape[-1] != MEL_BINS:
            raise FeatureExtractionError(
                f"Expected mel shape (frames, {MEL_BINS}) or (batch, frames, {MEL_BINS}), got {mel.shape}."
            )

        batch_windows: list[np.ndarray] = []
        counts: list[int] = []
        for item in mel:
            windows = self._windows(item)
            counts.append(windows.shape[0])
            batch_windows.append(windows)
        if not counts or counts[0] == 0:
            empty = np.zeros((len(counts), 0, EMBEDDING_DIM), dtype=np.float32)
            return empty[0] if single else empty

        flat = np.concatenate(batch_windows, axis=0) if batch_windows else np.zeros((0, 0, 0))
        embedded = self._embed_windows(flat)
        if len(set(counts)) != 1:
            chunks = np.split(embedded, np.cumsum(counts)[:-1])
            return chunks[0] if single else chunks
        reshaped = embedded.reshape(len(counts), counts[0], EMBEDDING_DIM)
        return reshaped[0] if single else reshaped

    def _windows(self, mel: np.ndarray) -> np.ndarray:
        n = 1 + (mel.shape[0] - self.window_size) // self.step_size if mel.shape[0] >= self.window_size else 0
        if n <= 0:
            return np.zeros((0, self.window_size, MEL_BINS), dtype=np.float32)
        windows = np.stack(
            [mel[i * self.step_size : i * self.step_size + self.window_size] for i in range(n)],
            axis=0,
        )
        return windows.astype(np.float32, copy=False)

    def _embed_windows(self, windows: np.ndarray) -> np.ndarray:
        if windows.size == 0:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        batch = windows[..., None]
        results: list[np.ndarray] = []
        for start in range(0, batch.shape[0], self.max_embedding_batch):
            chunk = np.ascontiguousarray(batch[start : start + self.max_embedding_batch], dtype=np.float32)
            try:
                out = self._emb.run([self._emb_output], {self._emb_input: chunk})[0]
            except Exception as exc:  # pragma: no cover - runtime dependent
                raise FeatureExtractionError(f"Embedding inference failed: {exc}") from exc
            out = np.asarray(out, dtype=np.float32).reshape(chunk.shape[0], EMBEDDING_DIM)
            results.append(out)
        return np.concatenate(results, axis=0)

    def embeddings(self, samples: np.ndarray) -> np.ndarray:
        """Compute embedding features for one clip or a batch of clips."""
        mel = self.melspectrogram(samples)
        return self.embeddings_from_melspec(mel)

    # Convenience aliases used by the rest of the package.
    extract = embeddings
    process_batch = embeddings


def features_from_file(
    path: str | Path,
    extractor: FeatureExtractor | None = None,
    *,
    sample_rate: int = 16_000,
) -> np.ndarray:
    from .io import load_audio

    audio = load_audio(path, sample_rate=sample_rate)
    return (extractor or FeatureExtractor()).embeddings(audio)
