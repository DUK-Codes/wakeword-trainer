"""Held-out evaluation shared by the library API and the GUI."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..audio.features import EMBEDDING_DIM
from ..config import EvalConfig
from ..data.dataset import Clip
from ..data.negatives import NegativePool
from ..exceptions import ConfigurationError
from .metrics import (
    OperatingPoint,
    auc_score,
    choose_operating_point,
    default_thresholds,
    false_positives_per_hour,
    recall_at_threshold,
    sweep_thresholds,
)
from .report import save_evaluation_report

Scorer = Callable[[np.ndarray], np.ndarray]


@dataclass
class EvaluationResult:
    payload: dict[str, Any]
    pos_scores: np.ndarray
    neg_scores: np.ndarray
    points: list[OperatingPoint]
    report_dir: Path | None = None

    @property
    def metrics(self) -> dict[str, Any]:
        return self.payload.get("evaluation", {})

    @property
    def operating_point(self) -> dict[str, Any]:
        return self.metrics.get("operating_point", {})

    @property
    def threshold(self) -> float:
        return float(self.operating_point.get("threshold", 0.5))

    def save(self, directory: str | Path, *, history: dict | None = None) -> Path:
        self.report_dir = save_evaluation_report(
            directory,
            payload=self.payload,
            pos_scores=self.pos_scores,
            neg_scores=self.neg_scores,
            points=self.points,
            target_fp_per_hour=float(self.payload.get("target_fp_per_hour", 1.0)),
            history=history,
        )
        return self.report_dir

    def summary(self) -> dict[str, Any]:
        return {
            "positives": int(self.pos_scores.size),
            "negatives": int(self.neg_scores.size),
            **self.operating_point,
            "auc": self.metrics.get("auc", 0.0),
            "report_dir": str(self.report_dir) if self.report_dir else None,
        }


def score_features_batched(scorer: Scorer, features: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if features.size == 0:
        return np.zeros((0,), dtype=np.float32)
    if features.ndim == 2:
        features = features[None, ...]
    scores = np.empty(features.shape[0], dtype=np.float32)
    for start in range(0, features.shape[0], batch_size):
        chunk = features[start : start + batch_size]
        scores[start : start + chunk.shape[0]] = np.asarray(scorer(chunk), dtype=np.float32).reshape(-1)
    return scores


def score_pool(
    scorer: Scorer,
    pool: NegativePool,
    indices: Sequence[int] | np.ndarray,
    *,
    batch_size: int = 2048,
) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    scores = np.empty(indices.size, dtype=np.float32)
    for start in range(0, indices.size, batch_size):
        chunk = indices[start : start + batch_size]
        features = np.asarray(pool.get_many(chunk), dtype=np.float32)
        scores[start : start + chunk.size] = np.asarray(scorer(features), dtype=np.float32).reshape(-1)
    return scores


def evenly_spaced_indices(length: int, limit: int, *, seed: int = 0) -> np.ndarray:
    if length <= 0:
        return np.zeros((0,), dtype=np.int64)
    if limit <= 0 or length <= limit:
        return np.arange(length, dtype=np.int64)
    del seed
    return np.unique(np.linspace(0, length - 1, num=limit, dtype=np.int64))


class Evaluator:
    """Computes recall/FRR/FP-per-hour for held-out clips and negatives."""

    def __init__(
        self,
        scorer: Scorer,
        config: EvalConfig | None = None,
        *,
        word: str = "",
        model_path: str | Path | None = None,
    ) -> None:
        self.scorer = scorer
        self.config = config or EvalConfig()
        self.config.validate()
        self.word = word
        self.model_path = Path(model_path) if model_path else None
        self._thresholds = default_thresholds(
            self.config.threshold_min,
            self.config.threshold_max,
            self.config.threshold_steps,
        )

    def evaluate(
        self,
        clips: Sequence[Clip],
        features: np.ndarray,
        negative_pool: NegativePool,
        *,
        dataset_summary: dict | None = None,
        split_summary: dict | None = None,
        extra_warnings: Sequence[str] | None = None,
        per_clip: bool = True,
        augmented_features: np.ndarray | None = None,
        augmented_labels: np.ndarray | None = None,
    ) -> EvaluationResult:
        clips = list(clips)
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 3 or features.shape[-1] != EMBEDDING_DIM:
            raise ConfigurationError(f"Evaluator expected (N, frames, {EMBEDDING_DIM}) features, got {features.shape}.")

        pos_scores = score_features_batched(self.scorer, features)
        indices = evenly_spaced_indices(len(negative_pool), self.config.max_negative_windows, seed=0)
        neg_scores = score_pool(self.scorer, negative_pool, indices)
        neg_hours = float(indices.size) * negative_pool.hop_seconds / 3600.0

        points = sweep_thresholds(pos_scores, neg_scores, neg_hours, self._thresholds)
        operating = choose_operating_point(points, self.config.target_fp_per_hour)
        at_half = self._operating_at(0.5, pos_scores, neg_scores, neg_hours)

        warnings = list(extra_warnings or [])
        if len(clips) < 5:
            warnings.append(
                f"Only {len(clips)} held-out positive clip(s). Recall is a coarse estimate; record more data."
            )
        if neg_hours < 0.5:
            warnings.append(f"Only {neg_hours:.2f} h of negative audio. False-positive estimates are very noisy.")
        if operating.recall >= 0.999 and operating.false_positives == 0:
            warnings.append(
                "The held-out data is perfectly separated. This often means the dataset is small or "
                "too homogeneous; validate on new recordings from the target device."
            )
        if operating.fp_per_hour > self.config.target_fp_per_hour:
            warnings.append(
                "No threshold met the target false-positive rate; the reported point has the fewest false positives."
            )

        per_clip_rows: list[dict[str, Any]] = []
        if per_clip and len(clips) == pos_scores.size:
            for clip, score in zip(clips, pos_scores, strict=False):
                per_clip_rows.append(
                    {
                        "path": str(clip.path),
                        "group": clip.group,
                        "score": round(float(score), 6),
                        "detected": bool(score >= operating.threshold),
                    }
                )

        payload: dict[str, Any] = {
            "word": self.word,
            "model": str(self.model_path) if self.model_path else None,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "target_fp_per_hour": float(self.config.target_fp_per_hour),
            "dataset": dataset_summary or {},
            "split": split_summary or {},
            "evaluation": {
                "positives": int(pos_scores.size),
                "negatives": int(neg_scores.size),
                "negative_hours": round(neg_hours, 4),
                "auc": round(auc_score(pos_scores, neg_scores), 4),
                "mean_positive_score": round(float(np.mean(pos_scores)), 4) if pos_scores.size else 0.0,
                "mean_negative_score": round(float(np.mean(neg_scores)), 4) if neg_scores.size else 0.0,
                "target_fp_per_hour": float(self.config.target_fp_per_hour),
                "operating_point": operating.as_dict(),
                "at_threshold_0_5": at_half.as_dict(),
                "threshold_table": [point.as_dict() for point in points],
                "warnings": warnings,
            },
            "per_clip": per_clip_rows,
        }

        if augmented_features is not None and len(augmented_features):
            aug_scores = score_features_batched(self.scorer, augmented_features)
            aug_points = sweep_thresholds(aug_scores, neg_scores, neg_hours, self._thresholds)
            aug_operating = choose_operating_point(aug_points, self.config.target_fp_per_hour)
            payload["augmented_evaluation"] = {
                "variants": int(aug_scores.size),
                "recall": round(aug_operating.recall, 4),
                "threshold": round(aug_operating.threshold, 4),
                "fp_per_hour": round(aug_operating.fp_per_hour, 4),
                "mean_score": round(float(np.mean(aug_scores)), 4),
            }
        del augmented_labels

        return EvaluationResult(payload, pos_scores, neg_scores, points)

    @staticmethod
    def _operating_at(
        threshold: float,
        pos_scores: np.ndarray,
        neg_scores: np.ndarray,
        neg_hours: float,
    ) -> OperatingPoint:
        recall = recall_at_threshold(pos_scores, threshold)
        fph = false_positives_per_hour(neg_scores, neg_hours, threshold)
        fp = int(np.sum(np.asarray(neg_scores) >= threshold))
        return OperatingPoint(threshold, recall, 1.0 - recall, fp, fph)
