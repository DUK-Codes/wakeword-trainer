"""Detection metrics: recall, FRR, false positives per hour, threshold sweeps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _array(scores) -> np.ndarray:
    return np.asarray(scores, dtype=np.float32).reshape(-1)


@dataclass(frozen=True)
class OperatingPoint:
    threshold: float
    recall: float
    frr: float
    false_positives: int
    fp_per_hour: float

    def as_dict(self) -> dict:
        return {
            "threshold": round(float(self.threshold), 4),
            "recall": round(float(self.recall), 4),
            "frr": round(float(self.frr), 4),
            "false_positives": int(self.false_positives),
            "fp_per_hour": round(float(self.fp_per_hour), 4),
        }


def recall_at_threshold(pos_scores, threshold: float) -> float:
    scores = _array(pos_scores)
    if scores.size == 0:
        return 0.0
    return float(np.mean(scores >= threshold))


def false_positive_rate(neg_scores, threshold: float) -> float:
    scores = _array(neg_scores)
    if scores.size == 0:
        return 0.0
    return float(np.mean(scores >= threshold))


def false_positives_per_hour(neg_scores, hours: float, threshold: float) -> float:
    scores = _array(neg_scores)
    if scores.size == 0 or hours <= 0:
        return 0.0
    return float(np.sum(scores >= threshold) / hours)


def default_thresholds(minimum: float = 0.01, maximum: float = 0.99, steps: int = 99) -> np.ndarray:
    return np.linspace(float(minimum), float(maximum), int(max(2, steps)))


def sweep_thresholds(
    pos_scores,
    neg_scores,
    neg_hours: float,
    thresholds: np.ndarray | None = None,
) -> list[OperatingPoint]:
    pos = _array(pos_scores)
    neg = _array(neg_scores)
    thresholds = default_thresholds() if thresholds is None else np.asarray(thresholds, dtype=np.float32)
    points: list[OperatingPoint] = []
    for threshold in thresholds:
        recall = float(np.mean(pos >= threshold)) if pos.size else 0.0
        fp = int(np.sum(neg >= threshold)) if neg.size else 0
        fph = float(fp / neg_hours) if neg_hours > 0 else 0.0
        points.append(
            OperatingPoint(
                threshold=float(threshold),
                recall=recall,
                frr=1.0 - recall,
                false_positives=fp,
                fp_per_hour=fph,
            )
        )
    return points


def choose_operating_point(points: list[OperatingPoint], target_fp_per_hour: float) -> OperatingPoint:
    """Pick the highest-recall threshold whose FP/hour is within the target."""
    if not points:
        return OperatingPoint(0.5, 0.0, 1.0, 0, 0.0)
    acceptable = [point for point in points if point.fp_per_hour <= target_fp_per_hour]
    if not acceptable:
        # Nothing meets the target: return the point with the fewest false positives,
        # breaking ties by recall.
        return min(points, key=lambda point: (point.false_positives, -point.recall))
    return max(acceptable, key=lambda point: (point.recall, -point.fp_per_hour, -abs(point.threshold - 0.5)))


def roc_points(pos_scores, neg_scores, thresholds: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    pos = _array(pos_scores)
    neg = _array(neg_scores)
    thresholds = default_thresholds() if thresholds is None else np.asarray(thresholds, dtype=np.float32)
    tpr = np.array([np.mean(pos >= t) if pos.size else 0.0 for t in thresholds])
    fpr = np.array([np.mean(neg >= t) if neg.size else 0.0 for t in thresholds])
    return fpr, tpr


def auc_score(pos_scores, neg_scores) -> float:
    """Area under the ROC curve computed with the Mann-Whitney statistic."""
    pos = _array(pos_scores)
    neg = _array(neg_scores)
    if pos.size == 0 or neg.size == 0:
        return 0.0
    combined = np.concatenate([pos, neg])
    order = np.argsort(combined, kind="mergesort")
    ranks = np.empty(combined.size, dtype=np.float64)
    ranks[order] = np.arange(1, combined.size + 1)
    # Average ranks for ties.
    sorted_values = combined[order]
    start = 0
    for index in range(1, combined.size + 1):
        if index == combined.size or sorted_values[index] != sorted_values[start]:
            if index - start > 1:
                ranks[order[start:index]] = np.mean(ranks[order[start:index]])
            start = index
    pos_ranks = ranks[: pos.size]
    return float((pos_ranks.sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def summarize_scores(
    pos_scores,
    neg_scores,
    neg_hours: float,
    *,
    target_fp_per_hour: float = 1.0,
    thresholds: np.ndarray | None = None,
) -> dict:
    pos = _array(pos_scores)
    neg = _array(neg_scores)
    points = sweep_thresholds(pos, neg, neg_hours, thresholds)
    operating = choose_operating_point(points, target_fp_per_hour)
    return {
        "positives": int(pos.size),
        "negatives": int(neg.size),
        "negative_hours": round(float(neg_hours), 4),
        "auc": round(auc_score(pos, neg), 4),
        "mean_positive_score": round(float(np.mean(pos)), 4) if pos.size else 0.0,
        "mean_negative_score": round(float(np.mean(neg)), 4) if neg.size else 0.0,
        "target_fp_per_hour": target_fp_per_hour,
        "operating_point": operating.as_dict(),
    }
