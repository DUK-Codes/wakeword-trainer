"""The training loop."""

from __future__ import annotations

import contextlib
import json
import random
import re
import time
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..audio.features import FeatureExtractor
from ..audio.io import human_duration
from ..config import TrainerConfig
from ..data.augment import Augmenter
from ..data.dataset import DatasetReport, scan_dataset
from ..data.negatives import NegativeDataError, ResolvedNegatives, resolve_negatives, save_negative_info
from ..data.sampler import NegativeSampler, PositiveSampler
from ..data.split import DatasetSplit, split_dataset
from ..evaluation.evaluator import Evaluator, evenly_spaced_indices
from ..evaluation.metrics import (
    choose_operating_point,
    default_thresholds,
    sweep_thresholds,
)
from ..evaluation.report import plot_training_history
from ..exceptions import ConfigurationError, DatasetError, ExportError, TrainingCancelled
from ..export.onnx import export_onnx
from ..export.tflite import export_tflite, tflite_available
from ..models.dnn import average_state_dicts, build_model, count_parameters
from ..version import __version__
from .callbacks import (
    Callback,
    CallbackList,
    CancellationCallback,
    CancelToken,
    FunctionCallback,
    TrainingEvent,
)
from .schedule import (
    lr_at_step,
    negative_ratio_at_step,
    validation_steps,
    warmup_hold_from_fractions,
)


def _slug(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return value or "wakeword"


def _cpu_state_dict(model: nn.Module) -> OrderedDict:
    return OrderedDict((key, value.detach().cpu().clone()) for key, value in model.state_dict().items())


@dataclass
class PreparedData:
    report: DatasetReport
    split: DatasetSplit
    negatives: ResolvedNegatives
    frames: int
    clip_samples: int
    val_clips: list
    test_clips: list
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "dataset": self.report.summary(),
            "split": self.split.summary(),
            "negatives": self.negatives.info.summary(),
            "validation_negatives": (
                self.negatives.validation_info.summary() if self.negatives.validation_info else None
            ),
            "frames": self.frames,
            "clip_samples": self.clip_samples,
            "warnings": list(self.warnings),
        }


@dataclass
class TrainingResult:
    word: str
    output_dir: Path
    best_step: int
    total_steps: int
    parameters: int
    duration_seconds: float
    best_metrics: dict[str, Any]
    history: dict[str, Any]
    onnx_path: Path | None
    tflite_path: Path | None
    evaluation_payload: dict[str, Any] | None
    warnings: list[str]
    config: TrainerConfig

    def summary(self) -> dict[str, Any]:
        return {
            "word": self.word,
            "output_dir": str(self.output_dir),
            "best_step": self.best_step,
            "total_steps": self.total_steps,
            "parameters": self.parameters,
            "duration_seconds": round(self.duration_seconds, 2),
            "duration_human": human_duration(self.duration_seconds),
            "best_metrics": _jsonable(self.best_metrics),
            "onnx_path": str(self.onnx_path) if self.onnx_path else None,
            "tflite_path": str(self.tflite_path) if self.tflite_path else None,
            "evaluation": _jsonable(self.evaluation_payload.get("evaluation") if self.evaluation_payload else None),
            "warnings": list(self.warnings),
            "preset": self.config.train.preset,
            "clip_seconds": self.config.audio.clip_seconds,
        }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


class Trainer:
    """End-to-end trainer: scan -> split -> features -> train -> evaluate -> export."""

    def __init__(
        self,
        config: TrainerConfig,
        *,
        callbacks: Sequence[Callback] | None = None,
        event_handler=None,
        stop_token: CancelToken | None = None,
        log_every: int = 20,
        extractor: FeatureExtractor | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.output_dir = Path(config.output_dir).expanduser().resolve()
        self.log_every = max(1, int(log_every))
        self.total_steps = int(config.train.steps)
        self.extractor = extractor
        self.prepared: PreparedData | None = None
        self.model: nn.Module | None = None
        self.device = torch.device("cpu")
        self.stop_token = stop_token or CancelToken()
        self.callbacks = CallbackList(list(callbacks or []))
        if event_handler is not None:
            self.callbacks.add(FunctionCallback(event_handler))
        self.callbacks.add(CancellationCallback(self.stop_token))
        self._thresholds = default_thresholds(
            config.evaluation.threshold_min,
            config.evaluation.threshold_max,
            config.evaluation.threshold_steps,
        )

    # ------------------------------------------------------------------
    def log(self, message: str) -> None:
        self.callbacks.on_log(message)

    def _event(self, kind: str, message: str = "", **payload) -> None:
        self.callbacks.on_event(TrainingEvent(kind, message=message, payload=payload))

    def cancel(self) -> None:
        self.stop_token.set()

    # ------------------------------------------------------------------
    def prepare(self, *, force: bool = False) -> PreparedData:
        """Scan the dataset, split it by group and resolve negative data."""
        if self.prepared is not None and not force:
            return self.prepared

        if self.config.dataset is None:
            raise ConfigurationError("config.dataset is required.")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._event("prepare", message="Scanning dataset", stage="scan", progress=0.0)

        def scan_progress(done: int, total: int) -> None:
            self._event(
                "prepare",
                message=f"Scanning dataset ({done}/{total})",
                stage="scan",
                progress=done / max(1, total),
            )

        report = scan_dataset(
            self.config.dataset,
            detect_duplicates=True,
            clip_seconds=self.config.audio.clip_seconds,
            progress=scan_progress,
        )
        report.save(self.output_dir / "dataset_report.json")
        for warning in report.warnings:
            self.log(f"Dataset warning: {warning}")

        split = split_dataset(
            report.valid_clips,
            seed=self.config.train.seed,
        )
        split.save(self.output_dir / "split", root=self.config.dataset)
        for warning in split.warnings:
            self.log(f"Split warning: {warning}")

        self._event("prepare", message="Loading feature models", stage="features", progress=0.35)
        if self.extractor is None:
            self.extractor = FeatureExtractor(
                mel_model_path=self.config.features.mel_model_path,
                embedding_model_path=self.config.features.embedding_model_path,
                window_size=self.config.features.window_size,
                step_size=self.config.features.step_size,
                providers=self.config.features.providers,
                num_threads=self.config.features.num_threads or self.config.train.num_threads,
            )
        clip_samples = self.config.audio.clip_samples
        frames = self.extractor.output_shape(clip_samples)[0]
        if frames < 1:
            raise ConfigurationError(
                f"clip_seconds={self.config.audio.clip_seconds} is too short to produce embedding frames."
            )

        self._event("prepare", message="Resolving negative data", stage="negatives", progress=0.5)

        def download_progress(done: int, total: int, name: str) -> None:
            fraction = done / max(1, total)
            self._event(
                "prepare",
                message=f"Downloading {name} ({fraction * 100:.1f}%)",
                stage="negatives",
                progress=0.5 + 0.4 * fraction,
            )

        negatives = resolve_negatives(
            self.config.negatives,
            frames=frames,
            extractor=self.extractor,
            clip_samples=clip_samples,
            progress=download_progress,
            log=self.log,
        )
        save_negative_info(self.output_dir / "negatives.json", negatives)
        self._event("prepare", message="Dataset ready", stage="done", progress=1.0)

        val_clips = split.val or split.test or split.train
        test_clips = split.test or split.val or split.train
        warnings = list(report.warnings) + list(split.warnings)
        if not split.val:
            warnings.append("Validation split is empty; the training split is used for model selection.")
        if not split.test:
            warnings.append("Test split is empty; evaluation metrics are not fully held out.")

        prepared = PreparedData(
            report=report,
            split=split,
            negatives=negatives,
            frames=frames,
            clip_samples=clip_samples,
            val_clips=list(val_clips),
            test_clips=list(test_clips),
            warnings=warnings,
        )
        self.prepared = prepared
        self._save_config_artifacts(prepared)
        return prepared

    def _save_config_artifacts(self, prepared: PreparedData) -> None:
        self.config.save(self.output_dir / "trainer_config.json")
        (self.output_dir / "prepared.json").write_text(
            json.dumps(_jsonable(prepared.summary()), indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # ------------------------------------------------------------------
    def _seed_everything(self) -> None:
        seed = int(self.config.train.seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(False)

    def _resolve_device(self) -> torch.device:
        requested = self.config.train.device
        if requested == "auto":
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return torch.device(requested)

    @torch.no_grad()
    def _predict_features(
        self,
        model: nn.Module,
        features: np.ndarray,
        *,
        batch_size: int = 2048,
    ) -> np.ndarray:
        features = np.asarray(features, dtype=np.float32)
        if features.size == 0:
            return np.zeros((0,), dtype=np.float32)
        if features.ndim == 2:
            features = features[None, ...]
        model.eval()
        scores = np.empty(features.shape[0], dtype=np.float32)
        for start in range(0, features.shape[0], batch_size):
            chunk = torch.from_numpy(np.ascontiguousarray(features[start : start + batch_size])).to(self.device)
            scores[start : start + chunk.shape[0]] = model(chunk).reshape(-1).float().cpu().numpy()
        return scores

    def _scorer(self, model: nn.Module):
        return lambda features: self._predict_features(model, features)

    def _predict_pool(self, model: nn.Module, pool, indices: np.ndarray) -> np.ndarray:
        scores = np.empty(len(indices), dtype=np.float32)
        batch = 2048
        for start in range(0, len(indices), batch):
            chunk = indices[start : start + batch]
            features = np.asarray(pool.get_many(chunk), dtype=np.float32)
            scores[start : start + chunk.size] = self._predict_features(model, features)
        return scores

    # ------------------------------------------------------------------
    def _validate(
        self,
        model: nn.Module,
        pos_features: np.ndarray,
        neg_pool,
        neg_indices: np.ndarray,
        step: int,
    ) -> dict[str, Any]:
        pos_scores = self._predict_features(model, pos_features)
        neg_scores = self._predict_pool(model, neg_pool, neg_indices)
        hours = len(neg_indices) * neg_pool.hop_seconds / 3600.0
        points = sweep_thresholds(pos_scores, neg_scores, hours, self._thresholds)
        operating = choose_operating_point(points, self.config.evaluation.target_fp_per_hour)
        at_half = next((point for point in points if abs(point.threshold - 0.5) < 0.005), operating)
        from ..evaluation.metrics import auc_score

        eps = 1e-6
        positive_loss = -float(np.mean(np.log(np.clip(pos_scores, eps, 1.0)))) if pos_scores.size else 0.0
        negative_loss = -float(np.mean(np.log(np.clip(1.0 - neg_scores, eps, 1.0)))) if neg_scores.size else 0.0
        val_loss = 0.5 * (positive_loss + negative_loss)

        return {
            "step": int(step),
            "recall": round(operating.recall, 4),
            "frr": round(operating.frr, 4),
            "fp_per_hour": round(operating.fp_per_hour, 4),
            "false_positives": int(operating.false_positives),
            "threshold": round(operating.threshold, 4),
            "auc": round(auc_score(pos_scores, neg_scores), 4),
            "val_loss": round(val_loss, 6),
            "negative_hours": round(hours, 4),
            "at_0_5": {
                "recall": round(at_half.recall, 4),
                "fp_per_hour": round(at_half.fp_per_hour, 4),
            },
        }

    @staticmethod
    def _selection_key(metrics: dict[str, Any]) -> tuple[float, float, float]:
        return (
            float(metrics.get("recall", 0.0)),
            -float(metrics.get("fp_per_hour", 1e9)),
            -float(metrics.get("val_loss", 1e9)),
        )

    # ------------------------------------------------------------------
    def train(
        self,
        *,
        export: bool = True,
        export_tflite_model: bool = False,
        resume: bool = False,
    ) -> TrainingResult:
        started = time.time()
        prepared = self.prepare()
        self._seed_everything()
        torch.set_num_threads(max(1, int(self.config.train.num_threads)))
        self.device = self._resolve_device()

        train_clips = prepared.split.train
        if not train_clips:
            raise DatasetError("Training split is empty.")
        if len(train_clips) < 3:
            self.log(
                f"Only {len(train_clips)} training clip(s). The model will overfit; "
                "record more examples for a usable detector."
            )

        augmenter = Augmenter(
            self.config.augment,
            self.config.audio,
            seed=self.config.train.seed,
        )
        positive_sampler = PositiveSampler(
            train_clips,
            self.extractor,
            augmenter,
            clip_samples=prepared.clip_samples,
            seed=self.config.train.seed,
        )
        if len(prepared.negatives.train) == 0:
            raise NegativeDataError(
                "Negatives are required for training. Configure negatives='auto', a .npy feature file, "
                "or a folder of negative audio."
            )
        negative_sampler = NegativeSampler(prepared.negatives.train, seed=self.config.train.seed + 1)

        val_neg_pool = prepared.negatives.validation or prepared.negatives.train
        val_indices = evenly_spaced_indices(
            len(val_neg_pool), min(self.config.train.val_windows, len(val_neg_pool)), seed=self.config.train.seed
        )
        if val_indices.size == 0:
            raise NegativeDataError("Validation negative pool is empty.")

        self.log(f"Extracting validation features for {len(prepared.val_clips)} clip(s)...")
        val_pos_features = positive_sampler.features_for(
            prepared.val_clips,
            augment=False,
            seed=self.config.train.seed + 100,
            pad_mode="center",
        )

        model = build_model(prepared.frames, self.config.model).to(self.device)
        self.model = model
        parameters = count_parameters(model)
        self.log(f"Model: {parameters:,} parameters, input ({prepared.frames}, 96), device={self.device.type}")

        total_steps = int(self.config.train.steps)
        self.total_steps = total_steps
        warmup, hold = warmup_hold_from_fractions(
            total_steps, self.config.train.warmup_fraction, self.config.train.hold_fraction
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.train.learning_rate,
            weight_decay=self.config.train.weight_decay,
        )

        history: dict[str, Any] = {"steps": [], "loss": [], "eval": []}
        start_step = 0
        last_checkpoint = self.output_dir / "last.pt"
        if resume and last_checkpoint.is_file():
            payload = torch.load(last_checkpoint, map_location="cpu", weights_only=False)
            model.load_state_dict(payload["model"])
            if "optimizer" in payload:
                try:
                    optimizer.load_state_dict(payload["optimizer"])
                except Exception:  # noqa: BLE001
                    self.log("Could not restore optimizer state; continuing with a fresh optimizer.")
            start_step = int(payload.get("step", 0))
            history_path = self.output_dir / "history.json"
            if history_path.is_file():
                with contextlib.suppress(Exception):
                    history = json.loads(history_path.read_text(encoding="utf-8"))
            self.log(f"Resuming from step {start_step}.")

        eval_steps = set(validation_steps(total_steps, self.config.train.val_every))
        best_models: list[tuple[tuple[float, float, float], int, OrderedDict, dict[str, Any]]] = []
        best_key: tuple[float, float, float] | None = None
        validations_without_improvement = 0
        rng = random.Random(self.config.train.seed + 7)
        cancelled = False

        self.callbacks.on_train_start(self, total_steps)
        try:
            for step in range(start_step, total_steps):
                if self.stop_token.cancelled:
                    raise TrainingCancelled("Training cancelled by user.")
                current_step = step + 1
                lr = lr_at_step(
                    step,
                    self.config.train.learning_rate,
                    warmup_steps=warmup,
                    hold_steps=hold,
                    total_steps=total_steps,
                )
                for group in optimizer.param_groups:
                    group["lr"] = lr

                ratio = negative_ratio_at_step(
                    step, total_steps, self.config.train.negative_ratio_start, self.config.train.negative_ratio_end
                )
                n_negative = max(0, int(round(self.config.train.batch_size * ratio)))
                pos_features = positive_sampler.sample(self.config.train.batch_size, rng=rng, augment=True)
                neg_features = (
                    negative_sampler.sample(n_negative, rng=rng)
                    if n_negative
                    else np.zeros((0, prepared.frames, 96), dtype=np.float32)
                )

                features = np.concatenate([pos_features, neg_features], axis=0)
                labels = np.concatenate(
                    [
                        np.ones(pos_features.shape[0], dtype=np.float32),
                        np.zeros(neg_features.shape[0], dtype=np.float32),
                    ]
                )
                x = torch.from_numpy(np.ascontiguousarray(features)).to(self.device)
                y = torch.from_numpy(labels).to(self.device)

                model.train()
                optimizer.zero_grad(set_to_none=True)
                probabilities = model(x).reshape(-1)
                loss = F.binary_cross_entropy(probabilities.clamp(1e-6, 1.0 - 1e-6), y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

                history["steps"].append(current_step)
                history["loss"].append(round(float(loss.detach().cpu()), 6))
                self.callbacks.on_step(self, current_step, float(loss.detach().cpu()), lr)

                if current_step in eval_steps:
                    metrics = self._validate(model, val_pos_features, val_neg_pool, val_indices, current_step)
                    metrics["loss"] = round(float(loss.detach().cpu()), 6)
                    history["eval"].append(metrics)
                    self.callbacks.on_eval(self, current_step, metrics)
                    key = self._selection_key(metrics)

                    improved = best_key is None or key > best_key
                    if improved:
                        best_key = key
                        validations_without_improvement = 0
                    else:
                        validations_without_improvement += 1

                    state = _cpu_state_dict(model)
                    best_models.append((key, current_step, state, metrics))
                    best_models.sort(key=lambda item: item[0], reverse=True)
                    keep = max(1, int(self.config.train.average_top_k))
                    del best_models[keep:]

                    self._write_history(history)
                    self.log(
                        f"Validation step {current_step}: recall={metrics['recall']:.3f} "
                        f"FP/h={metrics['fp_per_hour']:.2f} threshold={metrics['threshold']:.2f}"
                    )

                    if (
                        self.config.train.early_stop_patience
                        and validations_without_improvement >= self.config.train.early_stop_patience
                    ):
                        self.log(
                            f"Early stopping after {validations_without_improvement} validations without improvement."
                        )
                        break

                if self.config.train.checkpoint_every and current_step % self.config.train.checkpoint_every == 0:
                    self._save_checkpoint(last_checkpoint, model, optimizer, current_step, history)
                    self.callbacks.on_checkpoint(
                        self, current_step, last_checkpoint, history["eval"][-1] if history["eval"] else {}
                    )
        except TrainingCancelled:
            cancelled = True
            self._save_checkpoint(
                last_checkpoint, model, optimizer, history["steps"][-1] if history["steps"] else 0, history
            )
            self._write_history(history)
            self.log("Training cancelled; partial checkpoint saved.")
        finally:
            self._write_history(history)

        if not history["steps"]:
            raise TrainingCancelled("Training ended before the first step.")

        final_model = self._select_final_model(model, best_models, val_pos_features, val_neg_pool, val_indices)
        best_step = best_models[0][1] if best_models else int(history["steps"][-1])
        best_metrics = best_models[0][3] if best_models else (history["eval"][-1] if history["eval"] else {})

        self.model = final_model
        self._save_checkpoint(
            self.output_dir / "wakeword.pt",
            final_model,
            optimizer,
            best_step,
            {"best_metrics": best_metrics, "history": history},
        )

        evaluation_payload = None
        if not cancelled:
            evaluation_payload = self._final_evaluation(final_model, positive_sampler)
            if evaluation_payload is not None:
                best_metrics = {
                    **best_metrics,
                    "test": evaluation_payload.get("evaluation", {}),
                }

        threshold = 0.5
        if evaluation_payload:
            threshold = float(evaluation_payload["evaluation"]["operating_point"]["threshold"])
        elif best_metrics:
            threshold = float(best_metrics.get("threshold", 0.5))

        onnx_path = None
        tflite_path = None
        warnings = list(prepared.warnings)
        if export and not cancelled:
            export_info = self.export(
                tflite=export_tflite_model,
                threshold=threshold,
                model=final_model,
            )
            onnx_path = export_info.get("onnx")
            tflite_path = export_info.get("tflite")
            warnings.extend(export_info.get("warnings", []))

        result = TrainingResult(
            word=self.config.word,
            output_dir=self.output_dir,
            best_step=int(best_step),
            total_steps=total_steps,
            parameters=parameters,
            duration_seconds=time.time() - started,
            best_metrics=best_metrics,
            history=history,
            onnx_path=onnx_path,
            tflite_path=tflite_path,
            evaluation_payload=evaluation_payload,
            warnings=warnings,
            config=self.config,
        )
        (self.output_dir / "training_result.json").write_text(
            json.dumps(_jsonable(result.summary()), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        with contextlib.suppress(Exception):
            plot_training_history(history, self.output_dir / "training_curves.png")
        self.callbacks.on_train_end(self, result)
        return result

    # ------------------------------------------------------------------
    def _select_final_model(
        self,
        trained: nn.Module,
        best_models: list[tuple[tuple[float, float, float], int, OrderedDict, dict[str, Any]]],
        val_pos_features: np.ndarray,
        val_neg_pool,
        val_indices: np.ndarray,
    ) -> nn.Module:
        if not best_models:
            return trained

        best_key, best_step, best_state, _metrics = best_models[0]
        trained.load_state_dict(best_state)
        candidate = trained

        ensemble_states = [state for _key, _step, state, _metrics in best_models]
        if len(ensemble_states) > 1:
            averaged_state = average_state_dicts(ensemble_states)
            ensemble = build_model(self.prepared.frames, self.config.model).to(self.device)
            ensemble.load_state_dict(averaged_state)
            metrics = self._validate(ensemble, val_pos_features, val_neg_pool, val_indices, best_step)
            key = self._selection_key(metrics)
            self.log(
                f"Averaged {len(ensemble_states)} checkpoints: recall={metrics['recall']:.3f} "
                f"FP/h={metrics['fp_per_hour']:.2f}"
            )
            if key >= best_key:
                candidate = ensemble
                self.log("Using averaged checkpoint.")
            else:
                self.log("Averaged checkpoint scored lower; keeping the best single checkpoint.")
        return candidate

    def _final_evaluation(self, model: nn.Module, positive_sampler: PositiveSampler):
        prepared = self.prepared
        clips = prepared.test_clips
        if not clips:
            return None
        self.log(f"Evaluating on {len(clips)} held-out clip(s)...")
        features = positive_sampler.features_for(
            clips, augment=False, seed=self.config.train.seed + 200, pad_mode="center"
        )
        augmented = None
        if self.config.evaluation.augmented_test_variants > 0:
            augmented = positive_sampler.features_for(
                clips,
                augment=True,
                variants=self.config.evaluation.augmented_test_variants,
                seed=self.config.train.seed + 300,
                pad_mode="center",
            )
        evaluator = Evaluator(
            self._scorer(model),
            self.config.evaluation,
            word=self.config.word,
            model_path=None,
        )
        result = evaluator.evaluate(
            clips,
            features,
            prepared.negatives.validation or prepared.negatives.train,
            dataset_summary=prepared.report.summary(),
            split_summary=prepared.split.summary(),
            extra_warnings=prepared.warnings,
            augmented_features=augmented,
        )
        result.save(self.output_dir / "evaluation", history=None)
        self.callbacks.on_event(
            TrainingEvent(
                "metrics",
                step=self.total_steps,
                total_steps=self.total_steps,
                metrics=result.metrics,
                payload={"stage": "test"},
            )
        )
        self.log(
            f"Test: recall={result.operating_point.get('recall', 0):.3f} "
            f"FP/h={result.operating_point.get('fp_per_hour', 0):.2f} "
            f"threshold={result.operating_point.get('threshold', 0):.2f}"
        )
        return result.payload

    # ------------------------------------------------------------------
    def _save_checkpoint(
        self,
        path: Path,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        history: dict[str, Any] | None = None,
    ) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": _cpu_state_dict(model),
            "optimizer": optimizer.state_dict(),
            "step": int(step),
            "frames": int(self.prepared.frames) if self.prepared else None,
            "config": self.config.to_dict(),
            "history": history or {},
            "version": __version__,
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
        return path

    def _write_history(self, history: dict[str, Any]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "history.json").write_text(
            json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # ------------------------------------------------------------------
    def export(
        self,
        path: str | Path | None = None,
        *,
        tflite: bool = False,
        threshold: float | None = None,
        model: nn.Module | None = None,
    ) -> dict[str, Any]:
        """Export the trained model to ONNX (and optionally TFLite)."""
        if self.prepared is None:
            self.prepare()
        model = model or self.model
        if model is None:
            checkpoint = self.output_dir / "wakeword.pt"
            if not checkpoint.is_file():
                raise ExportError("No trained model is available. Call train() first.")
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model = build_model(self.prepared.frames, self.config.model)
            model.load_state_dict(payload["model"])
        model = model.eval().cpu()

        onnx_path = Path(path) if path else self.output_dir / f"{_slug(self.config.word)}.onnx"
        if onnx_path.suffix.lower() != ".onnx":
            onnx_path = onnx_path.with_suffix(".onnx")
        threshold = float(threshold if threshold is not None else 0.5)
        extra = {
            "dataset_clips": str(len(self.prepared.report.valid_clips)),
            "dataset_groups": str(len(self.prepared.report.groups)),
            "preset": self.config.train.preset,
            "steps": str(self.config.train.steps),
        }
        export_onnx(
            model,
            onnx_path,
            frames=self.prepared.frames,
            word=self.config.word,
            threshold=threshold,
            clip_seconds=self.config.audio.clip_seconds,
            sample_rate=self.config.audio.sample_rate,
            extra_metadata=extra,
        )
        warnings: list[str] = []
        tflite_path: Path | None = None
        if tflite:
            if not tflite_available():
                warnings.append(
                    "TFLite export skipped: install the optional extra `wakeword-trainer[tflite]` (ai-edge-torch)."
                )
            else:
                try:
                    tflite_path = export_tflite(
                        model,
                        onnx_path.with_suffix(".tflite"),
                        frames=self.prepared.frames,
                        onnx_path=onnx_path,
                    )
                except Exception as exc:  # noqa: BLE001
                    warnings.append(f"TFLite export failed: {exc}")

        info = {
            "word": self.config.word,
            "onnx": str(onnx_path),
            "tflite": str(tflite_path) if tflite_path else None,
            "threshold": threshold,
            "clip_seconds": self.config.audio.clip_seconds,
            "frames": self.prepared.frames,
            "sample_rate": self.config.audio.sample_rate,
            "warnings": warnings,
            "version": __version__,
        }
        (self.output_dir / "model_info.json").write_text(
            json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        self.log(f"Exported ONNX model: {onnx_path}")
        return {"onnx": onnx_path, "tflite": tflite_path, "warnings": warnings, "threshold": threshold}
