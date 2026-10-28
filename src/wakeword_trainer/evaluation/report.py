"""Evaluation and training reports (JSON, Markdown, PNG)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import OperatingPoint


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_threshold_sweep(
    points: Sequence[OperatingPoint],
    target_fp_per_hour: float,
    path: str | Path,
) -> Path:
    plt = _plt()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    thresholds = [point.threshold for point in points]
    recalls = [point.recall for point in points]
    fps = [point.fp_per_hour for point in points]

    figure, axis = plt.subplots(figsize=(7, 4), dpi=130)
    axis.plot(thresholds, recalls, label="Recall", color="#2f6fdf")
    axis.set_xlabel("Threshold")
    axis.set_ylabel("Recall", color="#2f6fdf")
    axis.set_ylim(0, 1.02)
    axis.grid(alpha=0.25)
    twin = axis.twinx()
    twin.plot(thresholds, fps, label="FP / hour", color="#d9534f")
    twin.set_ylabel("False positives / hour", color="#d9534f")
    twin.set_yscale("symlog", linthresh=0.01)
    twin.axhline(target_fp_per_hour, color="#d9534f", linestyle="--", linewidth=1, alpha=0.7)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path


def plot_score_distribution(pos_scores, neg_scores, path: str | Path) -> Path:
    plt = _plt()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pos = np.asarray(pos_scores, dtype=np.float32).reshape(-1)
    neg = np.asarray(neg_scores, dtype=np.float32).reshape(-1)

    figure, axis = plt.subplots(figsize=(7, 4), dpi=130)
    bins = np.linspace(0.0, 1.0, 41)
    if neg.size:
        axis.hist(neg, bins=bins, alpha=0.6, label="Negatives", color="#d9534f", density=True)
    if pos.size:
        axis.hist(pos, bins=bins, alpha=0.7, label="Positives", color="#2f6fdf", density=True)
    axis.set_xlabel("Model score")
    axis.set_ylabel("Density")
    axis.set_yscale("symlog", linthresh=0.01)
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path


def plot_training_history(history: dict, path: str | Path) -> Path:
    plt = _plt()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    steps = history.get("steps", [])
    losses = history.get("loss", [])
    evals = history.get("eval", [])

    figure, axis = plt.subplots(figsize=(8, 4), dpi=130)
    if steps and losses:
        axis.plot(steps, losses, label="Loss", color="#555555", alpha=0.6, linewidth=1)
    axis.set_xlabel("Step")
    axis.set_ylabel("Loss")
    axis.grid(alpha=0.25)

    if evals:
        twin = axis.twinx()
        eval_steps = [item.get("step", 0) for item in evals]
        recalls = [item.get("recall", 0.0) for item in evals]
        twin.plot(eval_steps, recalls, label="Val recall", color="#2f6fdf", marker="o", markersize=3)
        twin.set_ylabel("Validation recall", color="#2f6fdf")
        twin.set_ylim(0, 1.05)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path


def render_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    word = payload.get("word") or "(unnamed)"
    lines.append(f"# Wake word report: {word}")
    lines.append("")
    lines.append(f"Generated: {payload.get('created', '')}")
    lines.append("")

    dataset = payload.get("dataset") or {}
    if dataset:
        lines.append("## Dataset")
        lines.append("")
        lines.append(f"- Files used: **{dataset.get('valid_files', 0)}**")
        lines.append(f"- Independent groups: **{dataset.get('groups', 0)}**")
        lines.append(f"- Total audio: **{dataset.get('total_minutes', 0):.1f} min**")
        warnings = dataset.get("warnings") or []
        if warnings:
            lines.append("")
            lines.append("Warnings:")
            for warning in warnings:
                lines.append(f"- {warning}")
        lines.append("")

    split = payload.get("split") or {}
    if split:
        lines.append("## Split")
        lines.append("")
        lines.append("| Split | Clips | Groups | Seconds |")
        lines.append("| --- | ---: | ---: | ---: |")
        for name in ("train", "val", "test"):
            item = split.get(name, {})
            lines.append(
                f"| {name} | {item.get('clips', 0)} | {item.get('groups', 0)} | {item.get('seconds', 0):.1f} |"
            )
        lines.append("")

    training = payload.get("training") or {}
    if training:
        lines.append("## Training")
        lines.append("")
        lines.append(f"- Preset: **{training.get('preset', '-')}**")
        lines.append(f"- Steps: **{training.get('steps', 0)}**")
        lines.append(f"- Parameters: **{training.get('parameters', 0):,}**")
        lines.append(f"- Best step: **{training.get('best_step', 0)}**")
        lines.append("")

    evaluation = payload.get("evaluation")
    if evaluation:
        lines.append("## Held-out evaluation")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("| --- | ---: |")
        lines.append(f"| Positive clips | {evaluation.get('positives', 0)} |")
        lines.append(f"| Negative windows | {evaluation.get('negatives', 0)} |")
        lines.append(f"| Negative audio | {evaluation.get('negative_hours', 0):.2f} h |")
        lines.append(f"| AUC | {evaluation.get('auc', 0):.4f} |")
        point = evaluation.get("operating_point", {})
        lines.append(f"| Recommended threshold | {point.get('threshold', 0):.3f} |")
        lines.append(f"| Recall | {point.get('recall', 0):.3f} |")
        lines.append(f"| FRR | {point.get('frr', 0):.3f} |")
        lines.append(f"| False positives / hour | {point.get('fp_per_hour', 0):.3f} |")
        lines.append("")

        table = evaluation.get("threshold_table") or []
        if table:
            lines.append("### Threshold sweep")
            lines.append("")
            lines.append("| Threshold | Recall | FRR | FP / hour |")
            lines.append("| ---: | ---: | ---: | ---: |")
            for row in table:
                if row.get("threshold") in {0.1, 0.25, 0.5, 0.75, 0.9} or row is table[-1]:
                    lines.append(
                        f"| {row['threshold']:.2f} | {row['recall']:.3f} | {row['frr']:.3f} | {row['fp_per_hour']:.3f} |"
                    )
            lines.append("")

        if evaluation.get("warnings"):
            lines.append("### Warnings")
            lines.append("")
            for warning in evaluation["warnings"]:
                lines.append(f"- {warning}")
            lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append(
        "False positives per hour are estimated on held-out negative windows. "
        "Overlapping windows share context, so treat the number as an estimate, "
        "not a guarantee. Evaluate on real recordings from the target device for "
        "the final decision."
    )
    lines.append("")
    return "\n".join(lines)


def save_evaluation_report(
    directory: str | Path,
    *,
    payload: dict[str, Any],
    pos_scores,
    neg_scores,
    points: Sequence[OperatingPoint],
    target_fp_per_hour: float,
    history: dict | None = None,
) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload.setdefault("created", datetime.now(timezone.utc).isoformat(timespec="seconds"))

    json_path = directory / "evaluation.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (directory / "evaluation.md").write_text(render_markdown(payload), encoding="utf-8")

    try:
        plot_threshold_sweep(points, target_fp_per_hour, directory / "threshold_sweep.png")
        plot_score_distribution(pos_scores, neg_scores, directory / "score_distribution.png")
        if history:
            plot_training_history(history, directory / "training_curves.png")
    except Exception:  # noqa: BLE001 - plotting must never break training
        pass
    return directory
