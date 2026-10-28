"""Detection metrics and operating-point selection."""

from __future__ import annotations

import numpy as np
import pytest

from wakeword_trainer.evaluation.metrics import (
    OperatingPoint,
    auc_score,
    choose_operating_point,
    default_thresholds,
    false_positive_rate,
    false_positives_per_hour,
    recall_at_threshold,
    roc_points,
    summarize_scores,
    sweep_thresholds,
)


def test_recall_and_false_positive_rate():
    positives = [0.9, 0.8, 0.4, 0.1]
    negatives = [0.95, 0.7, 0.2, 0.05]
    assert recall_at_threshold(positives, 0.5) == 0.5
    assert recall_at_threshold(positives, 0.05) == 1.0
    assert false_positive_rate(negatives, 0.5) == 0.5
    assert false_positive_rate([], 0.5) == 0.0


def test_false_positives_per_hour():
    negatives = [0.9, 0.8, 0.1]
    assert false_positives_per_hour(negatives, 2.0, 0.5) == 1.0
    assert false_positives_per_hour(negatives, 0.0, 0.5) == 0.0
    assert false_positives_per_hour([], 2.0, 0.5) == 0.0


def test_default_thresholds_are_bounded():
    thresholds = default_thresholds(0.01, 0.99, 99)
    assert thresholds.size == 99
    assert thresholds[0] == pytest.approx(0.01)
    assert thresholds[-1] == pytest.approx(0.99)
    assert np.all(np.diff(thresholds) > 0)


def test_sweep_recall_is_monotonic():
    points = sweep_thresholds([0.1, 0.4, 0.7, 0.95], [0.2, 0.6], 0.5)
    recalls = [point.recall for point in points]
    assert recalls == sorted(recalls, reverse=True)
    for point in points:
        assert 0.0 <= point.recall <= 1.0
        assert point.frr == pytest.approx(1.0 - point.recall)


def test_choose_operating_point_prefers_target():
    points = [
        OperatingPoint(0.3, 0.99, 0.01, 10, 5.0),
        OperatingPoint(0.5, 0.95, 0.05, 1, 0.5),
        OperatingPoint(0.7, 0.90, 0.10, 0, 0.0),
    ]
    chosen = choose_operating_point(points, target_fp_per_hour=1.0)
    assert chosen.threshold == 0.5
    assert chosen.recall == 0.95


def test_choose_operating_point_tie_break_prefers_recall():
    points = [
        OperatingPoint(0.3, 0.80, 0.20, 0, 0.0),
        OperatingPoint(0.6, 0.90, 0.10, 0, 0.0),
    ]
    # Higher recall wins when FP/hour ties.
    assert choose_operating_point(points, 1.0).threshold == 0.6

    equal = [
        OperatingPoint(0.3, 0.90, 0.10, 0, 0.0),
        OperatingPoint(0.6, 0.90, 0.10, 0, 0.0),
    ]
    # Same recall and FP/hour: the threshold closer to 0.5 wins.
    assert choose_operating_point(equal, 1.0).threshold == 0.6


def test_choose_operating_point_without_acceptable_point():
    points = [
        OperatingPoint(0.3, 0.99, 0.01, 7, 7.0),
        OperatingPoint(0.9, 0.10, 0.90, 2, 2.0),
    ]
    chosen = choose_operating_point(points, target_fp_per_hour=0.5)
    assert chosen.false_positives == 2


def test_auc_perfect_and_reversed():
    assert auc_score([0.9, 0.8], [0.2, 0.1]) == pytest.approx(1.0)
    assert auc_score([0.1, 0.2], [0.8, 0.9]) == pytest.approx(0.0)
    assert auc_score([0.5], [0.5]) == pytest.approx(0.5)
    assert auc_score([], [0.5]) == 0.0


def test_roc_points_shapes():
    fpr, tpr = roc_points([0.9, 0.2], [0.1, 0.05], thresholds=np.array([0.0, 0.5, 1.0]))
    assert fpr.shape == tpr.shape == (3,)
    assert fpr[0] == 1.0 and tpr[0] == 1.0
    assert fpr[-1] == 0.0 and tpr[-1] == 0.0


def test_summarize_scores_payload():
    summary = summarize_scores([0.9, 0.8, 0.7], [0.1, 0.05, 0.0], neg_hours=3.0, target_fp_per_hour=1.0)
    assert summary["positives"] == 3
    assert summary["negatives"] == 3
    assert summary["negative_hours"] == 3.0
    assert summary["auc"] == 1.0
    assert summary["operating_point"]["recall"] == 1.0
    assert summary["operating_point"]["fp_per_hour"] == 0.0
