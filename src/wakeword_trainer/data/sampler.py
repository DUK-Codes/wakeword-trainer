"""Batching utilities for training and evaluation."""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from ..audio.features import FeatureExtractor
from .augment import Augmenter, ClipBank
from .dataset import Clip
from .negatives import NegativePool


@dataclass
class FeatureBatch:
    features: np.ndarray
    labels: np.ndarray
    weights: np.ndarray | None = None

    def __len__(self) -> int:
        return int(self.features.shape[0])


class PositiveSampler:
    """Draws augmented positive windows and computes their embeddings."""

    def __init__(
        self,
        clips: Sequence[Clip],
        extractor: FeatureExtractor,
        augmenter: Augmenter,
        *,
        clip_samples: int,
        seed: int = 42,
        bank: ClipBank | None = None,
    ) -> None:
        if not clips:
            raise ValueError("PositiveSampler requires at least one clip.")
        self.clips = list(clips)
        self.extractor = extractor
        self.augmenter = augmenter
        self.clip_samples = int(clip_samples)
        self._rng = random.Random(seed)
        self._bank = bank or ClipBank(clip.path for clip in self.clips)

    def _render(
        self,
        clip: Clip,
        *,
        augment: bool,
        rng: random.Random,
        pad_mode: str | None = None,
    ) -> np.ndarray:
        return self.augmenter.render(
            self._bank.get(clip.path),
            self.clip_samples,
            augment=augment,
            rng=rng,
            pad_mode=pad_mode,
        )

    def sample_audio(self, n: int, *, rng: random.Random | None = None, augment: bool = True) -> np.ndarray:
        rng = rng or self._rng
        chosen = [self.clips[rng.randrange(len(self.clips))] for _ in range(n)]
        return np.stack([self._render(clip, augment=augment, rng=rng) for clip in chosen], axis=0)

    def sample(self, n: int, *, rng: random.Random | None = None, augment: bool = True) -> np.ndarray:
        audio = self.sample_audio(n, rng=rng, augment=augment)
        return np.asarray(self.extractor.embeddings(audio), dtype=np.float32)

    def features_for(
        self,
        clips: Iterable[Clip],
        *,
        augment: bool = False,
        variants: int = 1,
        seed: int = 0,
        pad_mode: str | None = "center",
        progress=None,
    ) -> np.ndarray:
        """Compute deterministic features for evaluation clips.

        With ``variants > 1`` each clip is rendered several times with
        different augmentation seeds; all variants are returned.
        """
        rng = random.Random(seed)
        rendered: list[np.ndarray] = []
        clips = list(clips)
        for index, clip in enumerate(clips, start=1):
            for _ in range(max(1, variants)):
                rendered.append(self._render(clip, augment=augment, rng=rng, pad_mode=pad_mode))
            if progress is not None:
                progress(index, len(clips), "features")
        if not rendered:
            return np.zeros((0, self.extractor.window_size, 0), dtype=np.float32)
        return np.asarray(self.extractor.embeddings(np.stack(rendered, axis=0)), dtype=np.float32)


class NegativeSampler:
    """Samples embedding windows from a negative pool."""

    def __init__(self, pool: NegativePool, *, seed: int = 42) -> None:
        self.pool = pool
        self._rng = random.Random(seed)
        self._cached_matrix: np.ndarray | None = None
        self._cached_indices: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.pool)

    def sample(self, n: int, *, rng: random.Random | None = None) -> np.ndarray:
        if len(self.pool) == 0:
            return np.zeros((0, self.pool.frames, 96), dtype=np.float32)
        rng = rng or self._rng
        indices = (
            rng.sample(range(len(self.pool)), n)
            if n <= len(self.pool)
            else [rng.randrange(len(self.pool)) for _ in range(n)]
        )
        return np.asarray(self.pool.get_many(indices), dtype=np.float32)

    def sample_evenly(self, n: int, *, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Return ``n`` evenly spaced windows plus their indices."""
        if len(self.pool) == 0:
            return np.zeros((0, self.pool.frames, 96), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        n = min(n, len(self.pool))
        indices = np.linspace(0, len(self.pool) - 1, num=n, dtype=np.int64)
        return np.asarray(self.pool.get_many(indices), dtype=np.float32), indices


def concat_batches(parts: Iterable[FeatureBatch]) -> FeatureBatch:
    parts = list(parts)
    features = np.concatenate([part.features for part in parts if len(part)], axis=0)
    labels = np.concatenate([part.labels for part in parts if len(part)], axis=0)
    weights_parts = [part.weights for part in parts if len(part) and part.weights is not None]
    weights = np.concatenate(weights_parts, axis=0) if weights_parts else None
    return FeatureBatch(features, labels, weights)
