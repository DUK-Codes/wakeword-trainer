"""High-level, importable API.

The package is meant to be used from a Python script, not from a flag-heavy
command line. A complete run looks like this::

    import wakeword_trainer as wwt

    wwt.train(
        word="hey computer",
        dataset="data/positives",
        output="models/hey-computer",
        negatives="auto",
        download_negatives=True,
        preset="balanced",
    )

Everything is also available as a small object-oriented wrapper::

    trainer = wwt.WakeWordTrainer("hey computer", "data/positives", "models/hey-computer")
    trainer.prepare()
    trainer.train()
    trainer.evaluate()
    trainer.export()

Individual stages can be called separately: :func:`inspect_dataset`,
:func:`prepare_dataset`, :func:`train`, :func:`evaluate`, :func:`predict`,
:func:`detect` and :func:`export_model`.
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from .audio.features import FeatureExtractor
from .config import PRESETS, Preset, TrainerConfig
from .data.augment import Augmenter
from .data.dataset import DatasetReport, scan_dataset
from .data.negatives import resolve_negatives
from .data.sampler import PositiveSampler
from .data.split import split_dataset
from .evaluation.evaluator import EvaluationResult, Evaluator
from .exceptions import ConfigurationError, DatasetError, ExportError
from .export.onnx import export_onnx
from .export.tflite import export_tflite
from .models.dnn import build_model
from .models.inference import DetectionResult, ModelMetadata, WakeWordDetector, load_model_metadata
from .training.callbacks import Callback, ConsoleCallback
from .training.trainer import Trainer, TrainingResult

__all__ = [
    "WakeWordTrainer",
    "PRESETS",
    "detect",
    "evaluate",
    "export_model",
    "inspect_dataset",
    "predict",
    "prepare_dataset",
    "train",
]


def _slug(text: str) -> str:
    import re

    value = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return value or "wakeword"


def _default_output(word: str) -> Path:
    return Path("wakeword_models") / _slug(word)


def _build_config(
    word: str,
    dataset: str | Path | None,
    output: str | Path | None,
    preset: Preset,
    negatives,
    download_negatives: bool,
    overrides: dict[str, Any],
) -> TrainerConfig:
    config = TrainerConfig.from_preset(
        preset,
        word=word,
        dataset=dataset,
        output_dir=output or _default_output(word),
        **overrides,
    )
    if negatives is not None:
        config.negatives.source = negatives if isinstance(negatives, Path) else str(negatives)
    if download_negatives:
        config.negatives.download = True
    config.validate()
    return config


# ---------------------------------------------------------------------------
# dataset stages
# ---------------------------------------------------------------------------


def inspect_dataset(
    dataset: str | Path,
    *,
    clip_seconds: float = 2.0,
    group_by: str = "stem",
    recursive: bool = True,
    detect_duplicates: bool = True,
    progress=None,
) -> DatasetReport:
    """Scan a folder of wake word recordings and return a validation report."""
    return scan_dataset(
        dataset,
        group_by=group_by,
        recursive=recursive,
        detect_duplicates=detect_duplicates,
        clip_seconds=clip_seconds,
        progress=progress,
    )


def prepare_dataset(
    dataset: str | Path,
    output_dir: str | Path,
    *,
    word: str = "wakeword",
    negatives="none",
    download_negatives: bool = False,
    preset: Preset = "balanced",
    **overrides: Any,
) -> dict[str, Any]:
    """Validate, split and cache everything needed before training.

    Returns a JSON-friendly summary and writes ``dataset_report.json``,
    ``split/`` and ``prepared.json`` into ``output_dir``.
    """
    config = _build_config(
        word,
        dataset,
        output_dir,
        preset,
        negatives,
        download_negatives,
        overrides,
    )
    trainer = Trainer(config)
    prepared = trainer.prepare()
    summary = prepared.summary()
    summary["output_dir"] = str(Path(output_dir).resolve())
    return summary


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def train(
    word: str,
    dataset: str | Path,
    output: str | Path | None = None,
    *,
    preset: Preset = "balanced",
    negatives="auto",
    download_negatives: bool = False,
    callbacks: Sequence[Callback] | None = None,
    event_handler=None,
    progress: bool = False,
    export: bool = True,
    export_tflite: bool = False,
    resume: bool = False,
    **overrides: Any,
) -> TrainingResult:
    """Train a wake word model end to end.

    Parameters
    ----------
    word:
        The phrase to detect. Used for metadata and the output file name.
    dataset:
        Folder with wake word recordings (WAV/FLAC/OGG, MP3/M4A with ffmpeg).
    output:
        Output folder. Defaults to ``wakeword_models/<slug>``.
    preset:
        ``"quick"``, ``"balanced"`` or ``"best"``. Individual fields can be
        overridden with extra keyword arguments (``steps=...``,
        ``layer_dim=...``, ``batch_size=...``, ``clip_seconds=...``, ...).
    negatives:
        ``"auto"``, ``"none"``, a ``.npy`` feature file, or a folder of audio.
    download_negatives:
        Allow downloading the openWakeWord negative feature sets (~17 GB).
    progress:
        Print progress to stdout when no other callback is supplied.
    """
    config = _build_config(word, dataset, output, preset, negatives, download_negatives, overrides)
    callback_list = list(callbacks or [])
    if progress and not callback_list and event_handler is None:
        callback_list.append(ConsoleCallback())
    trainer = Trainer(config, callbacks=callback_list, event_handler=event_handler)
    return trainer.train(export=export, export_tflite_model=export_tflite, resume=resume)


# ---------------------------------------------------------------------------
# evaluation and inference
# ---------------------------------------------------------------------------


def _resolve_model_path(model) -> Path:
    if isinstance(model, TrainingResult):
        if model.onnx_path is None:
            raise ExportError("The training result has no exported ONNX model.")
        return Path(model.onnx_path)
    path = Path(model)
    if not path.is_file():
        raise ExportError(f"Model file not found: {path}")
    if path.suffix.lower() != ".onnx":
        raise ExportError("evaluate() expects an .onnx model or a TrainingResult.")
    return path


def evaluate(
    model,
    dataset: str | Path | None = None,
    *,
    split: Literal["test", "all"] = "test",
    negatives="auto",
    download_negatives: bool = False,
    clip_seconds: float | None = None,
    target_fp_per_hour: float = 1.0,
    max_negative_windows: int = 100_000,
    output_dir: str | Path | None = None,
    output_report: bool = True,
    seed: int = 42,
    **overrides: Any,
) -> EvaluationResult:
    """Evaluate an exported ONNX model on held-out recordings and negatives."""
    model_path = _resolve_model_path(model)
    if dataset is None:
        raise DatasetError("dataset is required: point it at the recordings used for evaluation.")
    metadata = load_model_metadata(model_path)
    if clip_seconds is None:
        clip_seconds = metadata.clip_seconds or 2.0

    config = TrainerConfig.from_preset(
        "quick",
        word=metadata.word or "wakeword",
        dataset=dataset,
        output_dir=output_dir or model_path.parent,
        **overrides,
    )
    config.audio.clip_seconds = float(clip_seconds)
    config.evaluation.target_fp_per_hour = float(target_fp_per_hour)
    config.evaluation.max_negative_windows = int(max_negative_windows)
    config.negatives.source = negatives if isinstance(negatives, Path) else str(negatives)
    config.negatives.download = bool(download_negatives)
    config.train.seed = int(seed)
    config.validate()

    report = scan_dataset(config.dataset, clip_seconds=config.audio.clip_seconds)
    if split == "test":
        dataset_split = split_dataset(report.valid_clips, seed=seed)
        clips = dataset_split.test or dataset_split.val or dataset_split.train
        split_summary = dataset_split.summary()
    elif split == "all":
        clips = report.valid_clips
        split_summary = {"all": len(clips)}
    else:
        raise ConfigurationError("split must be 'test' or 'all'.")

    extractor = FeatureExtractor(
        mel_model_path=config.features.mel_model_path,
        embedding_model_path=config.features.embedding_model_path,
        window_size=config.features.window_size,
        step_size=config.features.step_size,
        providers=config.features.providers,
        num_threads=config.features.num_threads,
    )
    clip_samples = config.audio.clip_samples
    frames = extractor.output_shape(clip_samples)[0]
    resolved = resolve_negatives(
        config.negatives,
        frames=frames,
        extractor=extractor,
        clip_samples=clip_samples,
    )

    augmenter = Augmenter(config.augment, config.audio)
    sampler = PositiveSampler(
        clips,
        extractor,
        augmenter,
        clip_samples=clip_samples,
        seed=seed,
    )
    features = sampler.features_for(clips, augment=False, seed=seed, pad_mode="center")
    detector = WakeWordDetector(model_path, feature_extractor=extractor, clip_seconds=clip_seconds)
    evaluator = Evaluator(
        detector.score_features,
        config.evaluation,
        word=config.word,
        model_path=model_path,
    )
    result = evaluator.evaluate(
        clips,
        features,
        resolved.validation or resolved.train,
        dataset_summary=report.summary(),
        split_summary=split_summary,
    )
    if output_report:
        result.save(Path(output_dir or model_path.parent) / "evaluation")
    return result


def predict(model, audio: str | Path | np.ndarray, *, threshold: float | None = None) -> float:
    """Return the wake word score for a file path or an int16/float waveform."""
    detector = WakeWordDetector(_resolve_model_path(model), threshold=threshold)
    if isinstance(audio, (str, Path)):
        score, _ = detector.score_file_windows(audio)
        return score
    return detector.score_audio(np.asarray(audio))


def detect(model, audio: str | Path | np.ndarray, *, threshold: float | None = None) -> DetectionResult:
    """Like :func:`predict`, but returns a :class:`DetectionResult`."""
    detector = WakeWordDetector(_resolve_model_path(model), threshold=threshold)
    if isinstance(audio, (str, Path)):
        return detector.detect_file(audio)
    return detector.detect(np.asarray(audio))


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


def _load_checkpoint(source: Path) -> tuple[torch.nn.Module, TrainerConfig, int]:
    payload = torch.load(source, map_location="cpu", weights_only=False)
    config = TrainerConfig.from_dict(payload.get("config", {}))
    frames = int(payload.get("frames") or 0)
    if frames <= 0:
        clip_samples = config.audio.clip_samples
        from .audio.features import num_embedding_frames

        frames = num_embedding_frames(clip_samples)
    model = build_model(frames, config.model)
    model.load_state_dict(payload["model"])
    return model.eval(), config, frames


def export_model(
    source,
    output: str | Path,
    *,
    format: Literal["onnx", "tflite"] = "onnx",
    threshold: float | None = None,
) -> Path:
    """Export a checkpoint, training result, or existing ONNX file.

    ``source`` may be a ``TrainingResult``, a ``.pt`` checkpoint or a ``.onnx``
    file. TFLite export needs the optional ``tflite`` extra.
    """
    output = Path(output)
    if format == "tflite":
        output = output.with_suffix(".tflite")
        if isinstance(source, (str, Path)) and Path(source).suffix.lower() == ".onnx":
            metadata = load_model_metadata(Path(source))
            return export_tflite(None, output, frames=metadata.frames or 16, onnx_path=Path(source))
        if isinstance(source, TrainingResult):
            if source.onnx_path is None:
                raise ExportError("Training result has no ONNX model to convert.")
            metadata = load_model_metadata(source.onnx_path)
            return export_tflite(None, output, frames=metadata.frames or 16, onnx_path=source.onnx_path)
        if isinstance(source, (str, Path)):
            model, config, frames = _load_checkpoint(Path(source))
            onnx_tmp = output.with_suffix(".onnx")
            export_onnx(
                model,
                onnx_tmp,
                frames=frames,
                word=config.word,
                threshold=threshold or 0.5,
                clip_seconds=config.audio.clip_seconds,
                sample_rate=config.audio.sample_rate,
            )
            return export_tflite(model, output, frames=frames, onnx_path=onnx_tmp)
        raise ExportError("Unsupported source for TFLite export.")

    output = output.with_suffix(".onnx") if output.suffix.lower() != ".onnx" else output
    if isinstance(source, TrainingResult):
        if source.onnx_path is None:
            raise ExportError("Training result has no exported ONNX model.")
        source_path = Path(source.onnx_path)
        if source_path.resolve() != output.resolve():
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, output)
        return output
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.suffix.lower() == ".onnx":
            if path.resolve() != output.resolve():
                output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, output)
            return output
        model, config, frames = _load_checkpoint(path)
        return export_onnx(
            model,
            output,
            frames=frames,
            word=config.word,
            threshold=threshold if threshold is not None else 0.5,
            clip_seconds=config.audio.clip_seconds,
            sample_rate=config.audio.sample_rate,
        )
    if isinstance(source, torch.nn.Module):
        raise ExportError("Exporting a raw torch.Module requires a config; save it with Trainer.export() instead.")
    raise ExportError("Unsupported source for ONNX export.")


# ---------------------------------------------------------------------------
# object-oriented wrapper
# ---------------------------------------------------------------------------


class WakeWordTrainer:
    """Stateful wrapper around the full pipeline."""

    def __init__(
        self,
        word: str,
        dataset: str | Path | None = None,
        output_dir: str | Path | None = None,
        *,
        preset: Preset = "balanced",
        negatives="auto",
        download_negatives: bool = False,
        **overrides: Any,
    ) -> None:
        self.config = _build_config(word, dataset, output_dir, preset, negatives, download_negatives, overrides)
        self._trainer = Trainer(self.config)
        self.result: TrainingResult | None = None

    # ------------------------------------------------------------------
    @property
    def prepared(self):
        return self._trainer.prepared

    def prepare(self, *, force: bool = False) -> dict[str, Any]:
        prepared = self._trainer.prepare(force=force)
        return prepared.summary()

    def train(
        self,
        *,
        export: bool = True,
        export_tflite: bool = False,
        resume: bool = False,
        event_handler=None,
    ) -> TrainingResult:
        if event_handler is not None:
            from .training.callbacks import FunctionCallback

            self._trainer.callbacks.add(FunctionCallback(event_handler))
        self.result = self._trainer.train(
            export=export,
            export_tflite_model=export_tflite,
            resume=resume,
        )
        return self.result

    def evaluate(self, *, use_exported: bool = True) -> EvaluationResult:
        """Evaluate on the held-out test split from :meth:`prepare`."""
        if self._trainer.prepared is None:
            self._trainer.prepare()
        prepared = self._trainer.prepared
        assert prepared is not None
        clips = prepared.test_clips
        if not clips:
            raise DatasetError("No held-out clips are available for evaluation.")

        sampler = PositiveSampler(
            clips,
            self._trainer.extractor,
            Augmenter(self.config.augment, self.config.audio),
            clip_samples=prepared.clip_samples,
            seed=self.config.train.seed,
        )
        features = sampler.features_for(clips, augment=False, pad_mode="center")

        onnx_path = self.result.onnx_path if self.result else None
        if use_exported and onnx_path and Path(onnx_path).is_file():
            detector = WakeWordDetector(onnx_path, feature_extractor=self._trainer.extractor)
            scorer = detector.score_features
            model_path = Path(onnx_path)
        else:
            model = self._trainer.model
            if model is None:
                checkpoint = self.config.output_dir / "wakeword.pt"
                if not checkpoint.is_file():
                    raise ExportError("No trained model available. Call train() first.")
                model, _config, _frames = _load_checkpoint(checkpoint)
                self._trainer.model = model.to(self._trainer.device)
            scorer = self._trainer._scorer(model)
            model_path = None

        evaluator = Evaluator(scorer, self.config.evaluation, word=self.config.word, model_path=model_path)
        result = evaluator.evaluate(
            clips,
            features,
            prepared.negatives.validation or prepared.negatives.train,
            dataset_summary=prepared.report.summary(),
            split_summary=prepared.split.summary(),
            extra_warnings=prepared.warnings,
        )
        result.save(Path(self.config.output_dir) / "evaluation")
        return result

    def predict(self, audio: str | Path | np.ndarray) -> float:
        model = self.exported_model()
        return predict(model, audio)

    def exported_model(self) -> Path:
        if self.result and self.result.onnx_path and Path(self.result.onnx_path).is_file():
            return Path(self.result.onnx_path)
        candidate = Path(self.config.output_dir) / f"{_slug(self.config.word)}.onnx"
        if candidate.is_file():
            return candidate
        raise ExportError("No exported model yet. Call train(export=True) or export() first.")

    def export(
        self,
        output: str | Path | None = None,
        *,
        tflite: bool = False,
        threshold: float | None = None,
    ) -> Path:
        if self.result is None:
            self._trainer.prepare()
        info = self._trainer.export(
            output,
            tflite=tflite,
            threshold=threshold,
            model=self._trainer.model,
        )
        path = info["onnx"]
        if self.result is not None:
            self.result.onnx_path = path
            self.result.tflite_path = info.get("tflite")
        return path

    @property
    def model_metadata(self) -> ModelMetadata:
        return load_model_metadata(self.exported_model())
