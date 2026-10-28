"""Group-aware train/validation/test splitting.

Augmented copies of one recording share a group. Splitting by group (instead
of by file) prevents the same underlying audio from appearing in both the
training set and the held-out evaluation set, which otherwise inflates recall
to unrealistically high values.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from ..exceptions import DatasetError
from .dataset import Clip, load_manifest, save_manifest


@dataclass
class DatasetSplit:
    train: list[Clip] = field(default_factory=list)
    val: list[Clip] = field(default_factory=list)
    test: list[Clip] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __iter__(self):
        yield self.train
        yield self.val
        yield self.test

    @property
    def all_clips(self) -> list[Clip]:
        return [*self.train, *self.val, *self.test]

    def summary(self) -> dict:
        def describe(clips: list[Clip]) -> dict:
            return {
                "clips": len(clips),
                "groups": len({clip.group for clip in clips}),
                "seconds": round(sum(clip.duration for clip in clips), 2),
            }

        return {
            "train": describe(self.train),
            "val": describe(self.val),
            "test": describe(self.test),
            "warnings": list(self.warnings),
        }

    def save(self, directory: str | Path, *, root: str | Path | None = None) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for name in ("train", "val", "test"):
            save_manifest(getattr(self, name), directory / f"{name}.json", root=root)
        (directory / "split.json").write_text(
            json.dumps(self.summary(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return directory

    @classmethod
    def load(cls, directory: str | Path, *, root: str | Path | None = None) -> DatasetSplit:
        directory = Path(directory)
        split = cls()
        for name in ("train", "val", "test"):
            path = directory / f"{name}.json"
            if path.is_file():
                setattr(split, name, load_manifest(path, root=root))
        return split


def _greedy_assign(
    groups: dict[str, list[Clip]],
    ratios: dict[str, float],
    rng: random.Random,
) -> dict[str, list[Clip]]:
    keys = list(groups)
    rng.shuffle(keys)
    keys.sort(key=lambda key: len(groups[key]), reverse=True)

    result: dict[str, list[Clip]] = {name: [] for name in ratios}
    counts = {name: 0 for name in ratios}
    total = sum(len(groups[key]) for key in keys)
    target = {name: max(1.0, ratio * total) for name, ratio in ratios.items()}

    remaining = list(keys)
    if len(keys) >= 3:
        # Guarantee non-empty val/test splits using the smallest groups.
        for name in ("test", "val"):
            candidate = min(remaining, key=lambda key: len(groups[key]))
            remaining.remove(candidate)
            result[name].extend(groups[candidate])
            counts[name] += len(groups[candidate])

    for position, key in enumerate(remaining):
        name = max(ratios, key=lambda n: (target[n] - counts[n]) / target[n])
        if name != "train" and not result["train"] and position == len(remaining) - 1:
            name = "train"
        result[name].extend(groups[key])
        counts[name] += len(groups[key])

    for clips in result.values():
        clips.sort(key=lambda clip: str(clip.path))
    return result


def split_dataset(
    clips: list[Clip],
    *,
    val_ratio: float = 0.12,
    test_ratio: float = 0.18,
    seed: int = 42,
    min_val: int = 1,
    min_test: int = 1,
) -> DatasetSplit:
    """Split clips into train/val/test without sharing a group across splits."""
    if not clips:
        raise DatasetError("Cannot split an empty dataset.")

    grouped: dict[str, list[Clip]] = defaultdict(list)
    for clip in clips:
        grouped[clip.group].append(clip)

    total_groups = len(grouped)
    split = DatasetSplit()

    if total_groups < 3 or len(clips) < 6:
        split.train = sorted(clips, key=lambda clip: str(clip.path))
        split.warnings.append(
            "Too few independent groups for a held-out split. All clips were used for training; "
            "evaluation metrics will be optimistic. Record more speakers/sessions or provide "
            "augmented copies with distinct source names."
        )
        return split

    test_ratio = max(0.0, min(test_ratio, 0.5))
    val_ratio = max(0.0, min(val_ratio, 0.5 - test_ratio))
    train_ratio = 1.0 - val_ratio - test_ratio

    target_test_groups = max(min_test, round(total_groups * test_ratio))
    target_val_groups = max(min_val, round(total_groups * val_ratio))
    while target_test_groups + target_val_groups >= total_groups:
        if target_test_groups >= target_val_groups and target_test_groups > min_test:
            target_test_groups -= 1
        elif target_val_groups > min_val:
            target_val_groups -= 1
        else:
            break

    ratios = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
    assigned = _greedy_assign(grouped, ratios, random.Random(seed))
    split.train = assigned["train"]
    split.val = assigned["val"]
    split.test = assigned["test"]

    if not split.train:
        raise DatasetError("Split produced an empty training set; add more recordings.")
    if not split.val:
        split.warnings.append("No validation clips; model selection will use the training loss only.")
    if not split.test:
        split.warnings.append("No test clips; the final metrics will be computed on the validation split.")
    return split
