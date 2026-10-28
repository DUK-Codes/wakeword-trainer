"""Learning-rate and sampling schedules."""

from __future__ import annotations

import math

import numpy as np


def lr_at_step(
    step: int,
    target_lr: float,
    *,
    warmup_steps: int,
    hold_steps: int,
    total_steps: int,
) -> float:
    """Linear warmup, short hold, cosine decay (the openWakeWord recipe)."""
    step = max(0, int(step))
    total_steps = max(1, int(total_steps))
    warmup_steps = max(1, int(warmup_steps))
    hold_steps = max(0, int(hold_steps))

    if step < warmup_steps:
        return float(target_lr * (step + 1) / warmup_steps)

    decay_start = warmup_steps + hold_steps
    if step <= decay_start:
        return float(target_lr)

    progress = (step - decay_start) / max(1, total_steps - decay_start)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * target_lr * (1.0 + math.cos(math.pi * progress))
    floor = target_lr * 0.01
    return float(max(floor, cosine))


def negative_ratio_at_step(
    step: int,
    total_steps: int,
    start: float = 4.0,
    end: float = 32.0,
) -> float:
    """Linearly increase the negatives-per-positive ratio during training."""
    if total_steps <= 1:
        return float(end)
    progress = min(1.0, max(0.0, step / (total_steps - 1)))
    return float(start + (end - start) * progress)


def validation_steps(
    total_steps: int,
    every: int,
    *,
    include_first: bool = False,
) -> list[int]:
    """Steps at which to validate, always including the final step."""
    total_steps = int(total_steps)
    every = max(1, int(every))
    steps = list(range(every, total_steps + 1, every))
    if include_first:
        steps = [max(1, every // 2), *steps]
    if not steps or steps[-1] != total_steps:
        steps.append(total_steps)
    return sorted(set(steps))


def warmup_hold_from_fractions(
    total_steps: int,
    warmup_fraction: float,
    hold_fraction: float,
) -> tuple[int, int]:
    warmup = max(1, int(round(total_steps * warmup_fraction)))
    hold = max(0, int(round(total_steps * hold_fraction)))
    if warmup + hold >= total_steps:
        hold = max(0, total_steps - warmup - 1)
    return warmup, hold


def make_lr_schedule(
    total_steps: int,
    target_lr: float,
    warmup_fraction: float = 0.15,
    hold_fraction: float = 0.35,
) -> np.ndarray:
    warmup, hold = warmup_hold_from_fractions(total_steps, warmup_fraction, hold_fraction)
    return np.array(
        [
            lr_at_step(step, target_lr, warmup_steps=warmup, hold_steps=hold, total_steps=total_steps)
            for step in range(total_steps)
        ],
        dtype=np.float64,
    )
