"""Augmentation chain and fixed-window rendering."""

from __future__ import annotations

import random

import numpy as np
from conftest import word_signal

from wakeword_trainer.config import AudioConfig, AugmentConfig
from wakeword_trainer.data.augment import Augmenter


def _audio(seed: int = 0, seconds: float = 1.4) -> np.ndarray:
    samples = word_signal(seed, seconds=seconds)
    return (samples * 32767.0).astype(np.int16)


def test_render_yields_exact_window_without_augmentation():
    augmenter = Augmenter(AugmentConfig(enabled=False), AudioConfig(trim_silence=False))
    rendered = augmenter.render(_audio(), 16_000, augment=False, pad_mode="center", rng=random.Random(1))
    assert rendered.dtype == np.int16
    assert rendered.size == 16_000
    assert np.any(rendered != 0)


def test_render_is_deterministic_for_a_fixed_seed():
    augmenter = Augmenter(
        AugmentConfig(enabled=True, gain_prob=1.0, noise_prob=1.0, pitch_prob=0.0, speed_prob=0.0),
        AudioConfig(trim_silence=False),
    )
    first = augmenter.render(_audio(), 24_000, augment=True, rng=random.Random(7))
    second = augmenter.render(_audio(), 24_000, augment=True, rng=random.Random(7))
    np.testing.assert_array_equal(first, second)


def test_disabled_augmenter_returns_input():
    augmenter = Augmenter(AugmentConfig(enabled=False))
    samples = _audio()
    np.testing.assert_array_equal(augmenter.augment(samples), samples)


def test_augment_preserves_dtype_and_peak():
    augmenter = Augmenter(
        AugmentConfig(
            enabled=True,
            gain_prob=1.0,
            noise_prob=1.0,
            pitch_prob=1.0,
            speed_prob=1.0,
            reverb_prob=0.0,
            background_prob=0.0,
        ),
        AudioConfig(trim_silence=False),
    )
    samples = _audio(2)
    output = augmenter.augment(samples, random.Random(3))
    assert output.dtype == np.int16
    # Speed and pitch augmentation intentionally change the duration; the
    # fixed-length window is produced later by ``render``.
    assert 0.5 * samples.size <= output.size <= 1.5 * samples.size
    assert np.any(output != 0)
    assert np.max(np.abs(output.astype(np.int32))) <= 32767


def test_pitch_shift_changes_signal_and_length():
    augmenter = Augmenter(AugmentConfig(enabled=False))
    samples = _audio(4)
    shifted = augmenter._pitch_shift(samples.astype(np.float32) / 32768.0, 4.0)
    # A +4 semitone shift via resampling shortens the clip by ~2^(4/12).
    expected = samples.size / (2.0 ** (4.0 / 12.0))
    assert abs(shifted.size - expected) < 200
    assert not np.array_equal(np.round(shifted * 32767), samples)
