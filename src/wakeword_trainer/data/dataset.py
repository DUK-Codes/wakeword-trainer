"""Dataset scanning, validation and manifests."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from ..audio.io import AUDIO_EXTENSIONS, AudioInfo, probe_audio
from ..exceptions import DatasetError

_AUG_SUFFIX = re.compile(
    r"(?:[_\-.](?:aug|augment|augmented|noise|rir|reverb|pitch|speed|gain|copy|variant|var|v)\d*)$",
    re.IGNORECASE,
)


def _default_group(path: Path, root: Path) -> str:
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = Path(path.name)
    stem = _AUG_SUFFIX.sub("", rel.stem) or rel.stem
    parent = "" if rel.parent == Path(".") else rel.parent.as_posix()
    return f"{parent}/{stem}" if parent else stem


def group_key(path: str | Path, root: str | Path | None = None, group_by: str | Callable[[Path], str] = "stem") -> str:
    """Return the leakage-safe group id for an audio file.

    Files that are augmented copies of the same recording share a group, so
    they can never end up in both train and test.
    """
    path = Path(path)
    root = Path(root) if root is not None else path.parent
    if callable(group_by):
        return str(group_by(path))
    if group_by == "file":
        return path.relative_to(root).as_posix() if path.is_relative_to(root) else path.as_posix()
    if group_by in {"dir", "directory"}:
        rel = path.relative_to(root) if path.is_relative_to(root) else Path(path.name)
        return "" if rel.parent == Path(".") else rel.parent.as_posix()
    if group_by == "stem":
        return _default_group(path, root)
    raise DatasetError(f"Unknown group_by mode: {group_by!r}.")


@dataclass
class Clip:
    path: Path
    duration: float
    sample_rate: int
    channels: int
    group: str
    error: str | None = None

    @property
    def valid(self) -> bool:
        return self.error is None and self.duration > 0


@dataclass
class DatasetReport:
    root: Path
    clips: list[Clip] = field(default_factory=list)
    invalid: list[Clip] = field(default_factory=list)
    duplicates: list[tuple[Path, Path]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    rejected: list[tuple[Path, str]] = field(default_factory=list)

    @property
    def valid_clips(self) -> list[Clip]:
        return [clip for clip in self.clips if clip.valid]

    @property
    def total_duration(self) -> float:
        return sum(clip.duration for clip in self.valid_clips)

    @property
    def groups(self) -> dict[str, int]:
        return dict(Counter(clip.group for clip in self.valid_clips))

    @property
    def ok(self) -> bool:
        return bool(self.valid_clips)

    def summary(self) -> dict:
        durations = [clip.duration for clip in self.valid_clips]
        return {
            "root": str(self.root),
            "files": len(self.clips) + len(self.invalid),
            "valid_files": len(self.valid_clips),
            "invalid_files": len(self.invalid),
            "groups": len(self.groups),
            "total_seconds": round(sum(durations), 3),
            "total_minutes": round(sum(durations) / 60.0, 2),
            "min_seconds": round(min(durations), 3) if durations else 0.0,
            "max_seconds": round(max(durations), 3) if durations else 0.0,
            "mean_seconds": round(sum(durations) / len(durations), 3) if durations else 0.0,
            "sample_rates": dict(Counter(clip.sample_rate for clip in self.valid_clips)),
            "channels": dict(Counter(clip.channels for clip in self.valid_clips)),
            "duplicates": [[str(a), str(b)] for a, b in self.duplicates],
            "warnings": list(self.warnings),
            "rejected": [[str(path), reason] for path, reason in self.rejected],
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.summary(), indent=2, ensure_ascii=False), encoding="utf-8")
        return target


def iter_audio_files(
    root: str | Path,
    *,
    recursive: bool = True,
    exclude: Iterable[str] = (),
) -> list[Path]:
    root = Path(root)
    if not root.is_dir():
        raise DatasetError(f"Dataset directory does not exist: {root}")
    excluded = {Path(item).name.lower() for item in exclude}
    pattern = "**/*" if recursive else "*"
    files: list[Path] = []
    for path in root.glob(pattern):
        if not path.is_file():
            continue
        if path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        if any(part.startswith(".") for part in path.relative_to(root).parts):
            continue
        if path.name.lower() in excluded:
            continue
        files.append(path)
    return sorted(files)


def _hash_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def scan_dataset(
    root: str | Path,
    *,
    group_by: str | Callable[[Path], str] = "stem",
    recursive: bool = True,
    exclude: Iterable[str] = (),
    detect_duplicates: bool = True,
    clip_seconds: float = 2.0,
    progress: Callable[[int, int], None] | None = None,
) -> DatasetReport:
    """Scan a folder of wake word recordings and validate every file."""
    root = Path(root).expanduser().resolve()
    files = iter_audio_files(root, recursive=recursive, exclude=exclude)
    if not files:
        raise DatasetError(
            f"No audio files found in {root}. Supported extensions: {', '.join(sorted(AUDIO_EXTENSIONS))}."
        )

    report = DatasetReport(root=root)
    hashes: dict[str, Path] = {}
    group_sizes: dict[str, int] = defaultdict(int)

    for index, path in enumerate(files, start=1):
        try:
            info: AudioInfo = probe_audio(path)
            clip = Clip(
                path=path,
                duration=float(info.duration),
                sample_rate=int(info.sample_rate),
                channels=int(info.channels),
                group=group_key(path, root, group_by),
            )
        except Exception as exc:  # noqa: BLE001 - report every unreadable file
            report.invalid.append(Clip(path=path, duration=0.0, sample_rate=0, channels=0, group="", error=str(exc)))
            continue

        if clip.duration <= 0:
            clip.error = "zero-length audio"
            report.invalid.append(clip)
            continue

        report.clips.append(clip)
        group_sizes[clip.group] += 1

        if detect_duplicates and clip.duration > 0:
            try:
                digest = _hash_file(path)
            except OSError:
                digest = ""
            if digest:
                if digest in hashes:
                    report.duplicates.append((hashes[digest], path))
                else:
                    hashes[digest] = path

        if progress is not None:
            progress(index, len(files))

    durations = [clip.duration for clip in report.valid_clips]
    if durations:
        short = [c for c in report.valid_clips if c.duration < min(0.35, clip_seconds * 0.4)]
        long = [c for c in report.valid_clips if c.duration > clip_seconds * 3]
        if short:
            report.warnings.append(
                f"{len(short)} clip(s) are very short (< {min(0.35, clip_seconds * 0.4):.2f} s); "
                "trim or re-record them if the wake word is cut off."
            )
        if long:
            report.warnings.append(
                f"{len(long)} clip(s) are longer than {clip_seconds * 3:.1f} s; "
                "silence trimming / random cropping will be applied."
            )
    if len(report.valid_clips) < 20:
        report.warnings.append(
            "Fewer than 20 recordings. The model will overfit; record more varied examples for a usable detector."
        )
    if len(group_sizes) < 5:
        report.warnings.append(
            "Fewer than 5 independent groups. Held-out evaluation will be unreliable; "
            "avoid augmented copies of the same recording in the dataset."
        )
    if report.duplicates:
        report.warnings.append(f"{len(report.duplicates)} duplicate file(s) detected (identical bytes).")
    mismatched = [c for c in report.valid_clips if c.sample_rate != 16_000]
    if mismatched:
        report.warnings.append(f"{len(mismatched)} file(s) are not 16 kHz and will be resampled during training.")
    stereo = [c for c in report.valid_clips if c.channels > 1]
    if stereo:
        report.warnings.append(f"{len(stereo)} stereo file(s) will be downmixed to mono.")

    if not report.valid_clips:
        raise DatasetError(f"No readable audio files in {root}.")
    return report


def save_manifest(
    clips: Iterable[Clip],
    path: str | Path,
    *,
    root: str | Path | None = None,
) -> Path:
    """Write a JSON manifest of clips (paths relative to ``root`` when given)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    root_path = Path(root).resolve() if root is not None else None
    items = []
    for clip in clips:
        clip_path = clip.path.resolve()
        if root_path is not None:
            try:
                stored = clip_path.relative_to(root_path).as_posix()
            except ValueError:
                stored = clip_path.as_posix()
        else:
            stored = clip_path.as_posix()
        items.append(
            {
                "path": stored,
                "group": clip.group,
                "duration": round(clip.duration, 4),
                "sample_rate": clip.sample_rate,
                "channels": clip.channels,
            }
        )
    target.write_text(
        json.dumps({"root": str(root_path) if root_path else None, "clips": items}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return target


def load_manifest(path: str | Path, *, root: str | Path | None = None) -> list[Clip]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    base = Path(root) if root is not None else Path(data.get("root") or ".")
    clips: list[Clip] = []
    for item in data.get("clips", []):
        clip_path = Path(item["path"])
        if not clip_path.is_absolute():
            clip_path = base / clip_path
        clips.append(
            Clip(
                path=clip_path,
                duration=float(item.get("duration", 0.0)),
                sample_rate=int(item.get("sample_rate", 16_000)),
                channels=int(item.get("channels", 1)),
                group=str(item.get("group", clip_path.stem)),
            )
        )
    return clips
