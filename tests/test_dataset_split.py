"""Dataset scanning and leakage-safe splitting."""

from __future__ import annotations

from pathlib import Path

import pytest

from wakeword_trainer.data.dataset import Clip, group_key, scan_dataset
from wakeword_trainer.data.split import split_dataset
from wakeword_trainer.exceptions import DatasetError


def test_scan_dataset_counts_valid_files(positives_dir: Path):
    report = scan_dataset(positives_dir, detect_duplicates=True)
    summary = report.summary()
    assert summary["valid_files"] == 12
    assert summary["invalid_files"] == 0
    assert summary["groups"] == 12
    assert summary["total_seconds"] > 10
    assert summary["sample_rates"] == {16_000: 12}
    assert summary["channels"] == {1: 12}
    assert report.ok


def test_scan_dataset_detects_duplicates(copied_audio: Path):
    report = scan_dataset(copied_audio, detect_duplicates=True)
    assert len(report.duplicates) == 1
    first, second = report.duplicates[0]
    assert first != second


def test_scan_dataset_reports_unreadable_file(tmp_path: Path, positives_dir: Path):
    import shutil

    shutil.copyfile(positives_dir / "speaker_00_session_00.wav", tmp_path / "good.wav")
    (tmp_path / "broken.wav").write_bytes(b"this is not a wav file")
    report = scan_dataset(tmp_path, detect_duplicates=False)
    assert len(report.valid_clips) == 1
    assert len(report.invalid) == 1
    assert report.invalid[0].path.name == "broken.wav"


def test_scan_dataset_requires_audio(tmp_path: Path):
    (tmp_path / "readme.txt").write_text("nothing here", encoding="utf-8")
    with pytest.raises(DatasetError):
        scan_dataset(tmp_path)


def test_group_key_strips_augmentation_suffixes(clear_isolated_audio: Path):
    root = clear_isolated_audio
    assert group_key(root / "hello.wav", root) == "hello"
    assert group_key(root / "hello_aug1.wav", root) == "hello"
    assert group_key(root / "hello_v2.wav", root) == "hello"
    assert group_key(root / "other_noise.wav", root) == "other"
    assert group_key(root / "third_aug1.wav", root) == "third"


def test_scan_dataset_groups_augmented_copies(clear_isolated_audio: Path):
    report = scan_dataset(clear_isolated_audio, detect_duplicates=False)
    groups = report.groups
    assert groups == {"hello": 3, "other": 2, "third": 2, "fourth": 2}


def test_split_never_shares_a_group(clear_isolated_audio: Path):
    report = scan_dataset(clear_isolated_audio, detect_duplicates=False)
    split = split_dataset(report.valid_clips, seed=3)

    by_name = {
        "train": {clip.group for clip in split.train},
        "val": {clip.group for clip in split.val},
        "test": {clip.group for clip in split.test},
    }
    assert by_name["train"].isdisjoint(by_name["val"])
    assert by_name["train"].isdisjoint(by_name["test"])
    assert by_name["val"].isdisjoint(by_name["test"])

    combined = sorted(str(clip.path) for clip in split.all_clips)
    original = sorted(str(clip.path) for clip in report.valid_clips)
    assert combined == original


def test_split_is_deterministic(clear_isolated_audio: Path):
    report = scan_dataset(clear_isolated_audio, detect_duplicates=False)
    first = split_dataset(report.valid_clips, seed=11)
    second = split_dataset(report.valid_clips, seed=11)
    assert [str(c.path) for c in first.train] == [str(c.path) for c in second.train]
    assert [str(c.path) for c in first.test] == [str(c.path) for c in second.test]


def test_split_saves_and_loads(tmp_path: Path, clear_isolated_audio: Path):
    report = scan_dataset(clear_isolated_audio, detect_duplicates=False)
    split = split_dataset(report.valid_clips, seed=5)
    directory = split.save(tmp_path / "split", root=clear_isolated_audio)
    loaded = type(split).load(directory, root=clear_isolated_audio)
    assert [str(c.path) for c in loaded.train] == [str(c.path) for c in split.train]
    assert [str(c.path) for c in loaded.test] == [str(c.path) for c in split.test]


def test_tiny_dataset_warns_and_train_only():
    clips = [
        Clip(path=Path("a.wav"), duration=1.0, sample_rate=16_000, channels=1, group="a"),
        Clip(path=Path("b.wav"), duration=1.0, sample_rate=16_000, channels=1, group="b"),
    ]
    split = split_dataset(clips)
    assert split.train and not split.val and not split.test
    assert split.warnings


def test_empty_split_is_rejected():
    with pytest.raises(DatasetError):
        split_dataset([])
