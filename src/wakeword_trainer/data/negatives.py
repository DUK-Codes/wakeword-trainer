"""Negative (non-wake-word) data sources.

Three options are supported, matching the three most common workflows:

``"auto"``
    Download the pre-computed openWakeWord feature sets from Hugging Face
    (ACAV100M for training, the validation set for false-positive estimates).
``path/to/file.npy``
    A pre-computed embedding array, either ``(N, frames, 96)`` or a
    validation-style ``(time, 96)`` matrix.
``path/to/folder``
    A folder of negative audio. Non-overlapping windows are embedded once and
    cached next to the other feature caches.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..audio.features import EMBEDDING_DIM, FeatureExtractor
from ..audio.io import AUDIO_EXTENSIONS, estimated_hours, load_audio
from ..config import NegativeConfig
from ..exceptions import DownloadError, NegativeDataError
from ..paths import downloads_dir, features_cache_dir

ACAV_URL = (
    "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/"
    "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
)
VALIDATION_URL = (
    "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy"
)
ACAV_SIZE = 17_280_000_128
VALIDATION_SIZE = 184_836_608

ProgressFn = Callable[[int, int, str], None]


@dataclass
class NegativeSourceInfo:
    kind: str
    path: Path | None
    windows: int
    hours: float
    frames: int
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "kind": self.kind,
            "path": str(self.path) if self.path else None,
            "windows": self.windows,
            "hours": round(self.hours, 4),
            "frames": self.frames,
            "notes": list(self.notes),
        }


class NegativePool:
    """Random access to fixed-length embedding windows."""

    kind = "base"

    def __init__(self, frames: int, hop_seconds: float) -> None:
        self.frames = frames
        self.hop_seconds = hop_seconds

    def __len__(self) -> int:  # pragma: no cover - overridden
        raise NotImplementedError

    def get(self, index: int) -> np.ndarray:  # pragma: no cover - overridden
        raise NotImplementedError

    def get_many(self, indices: Iterable[int] | np.ndarray) -> np.ndarray:
        idx = np.asarray(list(indices) if not isinstance(indices, np.ndarray) else indices, dtype=np.int64)
        return np.stack([self.get(int(i)) for i in idx], axis=0)

    @property
    def hours(self) -> float:
        return estimated_hours(len(self), self.hop_seconds)

    def close(self) -> None:
        return None

    def __repr__(self) -> str:
        return f"<{type(self).__name__} windows={len(self)} frames={self.frames} hours={self.hours:.2f}>"


class ArrayPool(NegativePool):
    """In-memory or memory-mapped ``(N, frames, 96)`` array."""

    kind = "array"

    def __init__(self, data: np.ndarray, hop_seconds: float = 0.08) -> None:
        if data.ndim != 3 or data.shape[-1] != EMBEDDING_DIM:
            raise NegativeDataError(f"Expected a (N, frames, {EMBEDDING_DIM}) array, got shape {data.shape}.")
        super().__init__(data.shape[1], hop_seconds)
        self._data = data

    def __len__(self) -> int:
        return int(self._data.shape[0])

    def get(self, index: int) -> np.ndarray:
        return np.asarray(self._data[index], dtype=np.float32)

    def get_many(self, indices: Iterable[int] | np.ndarray) -> np.ndarray:
        idx = np.asarray(list(indices) if not isinstance(indices, np.ndarray) else indices, dtype=np.int64)
        return np.asarray(self._data[idx], dtype=np.float32)


class SlidingPool(NegativePool):
    """Lazy sliding-window view over a ``(time, 96)`` feature matrix."""

    kind = "sliding"

    def __init__(self, data: np.ndarray, frames: int, hop_seconds: float = 0.08) -> None:
        if data.ndim != 2 or data.shape[-1] != EMBEDDING_DIM:
            raise NegativeDataError(f"Expected a (time, {EMBEDDING_DIM}) matrix, got shape {data.shape}.")
        if data.shape[0] < frames:
            raise NegativeDataError(
                f"Negative feature file has {data.shape[0]} frames, fewer than the {frames} required."
            )
        super().__init__(frames, hop_seconds)
        self._data = data
        self._offsets = np.arange(frames, dtype=np.int64)

    def __len__(self) -> int:
        return int(self._data.shape[0] - self.frames + 1)

    def get(self, index: int) -> np.ndarray:
        return np.asarray(self._data[index : index + self.frames], dtype=np.float32)

    def get_many(self, indices: Iterable[int] | np.ndarray) -> np.ndarray:
        idx = np.asarray(list(indices) if not isinstance(indices, np.ndarray) else indices, dtype=np.int64)
        matrix = idx[:, None] + self._offsets[None, :]
        return np.asarray(self._data[matrix], dtype=np.float32)


class AcavPool(ArrayPool):
    """ACAV100M windows (16 frames per 2-second clip)."""

    kind = "acav"

    def __init__(self, data: np.ndarray, frames: int) -> None:
        if data.shape[1] < frames:
            raise NegativeDataError(
                f"ACAV features contain {data.shape[1]} frames per clip, but the configured "
                f"window needs {frames}. Use clip_seconds >= {frames * 0.08:.2f} s or a "
                "different negative source."
            )
        if data.shape[1] != frames:
            data = data[:, :frames, :]
        super().__init__(data, hop_seconds=2.0)


class EmptyPool(NegativePool):
    kind = "none"

    def __init__(self, frames: int) -> None:
        super().__init__(frames, hop_seconds=1.0)

    def __len__(self) -> int:
        return 0

    def get(self, index: int) -> np.ndarray:
        raise NegativeDataError("No negative data is configured.")

    def get_many(self, indices: Iterable[int] | np.ndarray) -> np.ndarray:
        raise NegativeDataError("No negative data is configured.")


# ---------------------------------------------------------------------------
# downloads
# ---------------------------------------------------------------------------


def download_file(
    url: str,
    destination: str | Path,
    *,
    expected_size: int | None = None,
    progress: ProgressFn | None = None,
    timeout: int = 60,
    retries: int = 3,
) -> Path:
    """Download ``url`` with resume support and optional progress reporting."""
    import requests

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and (expected_size is None or destination.stat().st_size == expected_size):
        return destination

    part = destination.with_suffix(destination.suffix + ".part")
    existing = part.stat().st_size if part.is_file() else 0

    headers = {"Range": f"bytes={existing}-"} if existing else {}
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, headers=headers, stream=True, timeout=timeout) as response:
                if response.status_code == 416 and existing:
                    # Already fully downloaded.
                    break
                response.raise_for_status()
                total = int(response.headers.get("Content-Length", 0)) + existing
                if expected_size:
                    total = expected_size
                mode = "ab" if existing else "wb"
                with part.open(mode) as handle:
                    done = existing
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        done += len(chunk)
                        if progress is not None:
                            progress(done, total, destination.name)
            break
        except Exception as exc:  # noqa: BLE001 - retried below
            if attempt == retries:
                raise DownloadError(f"Failed to download {url}: {exc}") from exc
            time.sleep(2.0 * attempt)
            existing = part.stat().st_size if part.is_file() else 0
            headers = {"Range": f"bytes={existing}-"} if existing else {}

    if expected_size is not None and part.stat().st_size != expected_size:
        actual = part.stat().st_size
        part.unlink(missing_ok=True)
        raise DownloadError(
            f"Downloaded {actual} bytes but expected {expected_size} for {destination.name}. "
            "The download was incomplete; try again."
        )
    part.replace(destination)
    return destination


def ensure_acav(progress: ProgressFn | None = None) -> Path:
    target = downloads_dir() / "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
    return download_file(ACAV_URL, target, expected_size=ACAV_SIZE, progress=progress)


def ensure_validation_features(progress: ProgressFn | None = None) -> Path:
    target = downloads_dir() / "validation_set_features.npy"
    return download_file(VALIDATION_URL, target, expected_size=VALIDATION_SIZE, progress=progress)


# ---------------------------------------------------------------------------
# audio folder -> cached features
# ---------------------------------------------------------------------------


def _directory_fingerprint(paths: list[Path], clip_samples: int, frames: int) -> str:
    digest = hashlib.sha1()
    digest.update(str(clip_samples).encode())
    digest.update(str(frames).encode())
    for path in paths:
        stat = path.stat()
        digest.update(str(path.resolve()).encode("utf-8"))
        digest.update(str(stat.st_size).encode())
        digest.update(str(int(stat.st_mtime)).encode())
    return digest.hexdigest()[:20]


def build_audio_feature_cache(
    directory: str | Path,
    extractor: FeatureExtractor,
    *,
    clip_samples: int,
    frames: int,
    max_hours: float = 5.0,
    progress: Callable[[int, int, str], None] | None = None,
    batch_size: int = 16,
) -> Path:
    files = sorted(
        path for path in Path(directory).rglob("*") if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )
    if not files:
        raise NegativeDataError(f"No audio files found in negative directory: {directory}")

    fingerprint = _directory_fingerprint(files, clip_samples, frames)
    target = features_cache_dir() / f"audio_negatives_{fingerprint}.npy"
    if target.is_file():
        return target

    tmp = target.with_suffix(".tmp.npy")
    max_windows = max(1, int(max_hours * 3600 * 16_000 / clip_samples))
    written = 0
    matrix = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(max_windows, frames, EMBEDDING_DIM))

    total_files = len(files)
    for file_index, path in enumerate(files, start=1):
        if written >= max_windows:
            break
        try:
            audio = load_audio(path)
        except Exception:  # noqa: BLE001 - skip unreadable negatives
            continue
        n_windows = audio.size // clip_samples
        if n_windows <= 0:
            continue
        batch_audio = []
        for window_index in range(n_windows):
            if written + len(batch_audio) >= max_windows:
                break
            start = window_index * clip_samples
            batch_audio.append(audio[start : start + clip_samples])
            if len(batch_audio) == batch_size:
                feats = extractor.embeddings(np.stack(batch_audio, axis=0))
                if isinstance(feats, list):
                    feats = np.stack(feats)
                take = min(len(feats), max_windows - written)
                matrix[written : written + take] = feats[:take].astype(np.float16)
                written += take
                batch_audio = []
        if batch_audio and written < max_windows:
            feats = extractor.embeddings(np.stack(batch_audio, axis=0))
            if isinstance(feats, list):
                feats = np.stack(feats)
            take = min(len(feats), max_windows - written)
            matrix[written : written + take] = feats[:take].astype(np.float16)
            written += take
        if progress is not None:
            progress(file_index, total_files, path.name)

    matrix.flush()
    del matrix
    if written == 0:
        tmp.unlink(missing_ok=True)
        raise NegativeDataError("Could not extract any negative windows from the audio folder.")
    final = np.load(tmp)[:written].astype(np.float16)
    np.save(target, final)
    del final
    tmp.unlink(missing_ok=True)
    return target


# ---------------------------------------------------------------------------
# public resolver
# ---------------------------------------------------------------------------


@dataclass
class ResolvedNegatives:
    train: NegativePool
    validation: NegativePool | None
    info: NegativeSourceInfo
    validation_info: NegativeSourceInfo | None = None


def _load_npy_pool(path: Path, frames: int) -> NegativePool:
    try:
        data = np.load(path, mmap_mode="r")
    except Exception as exc:  # noqa: BLE001
        raise NegativeDataError(f"Cannot read negative feature file {path}: {exc}") from exc

    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim == 3 and data.shape[-1] == EMBEDDING_DIM:
        if data.shape[1] > frames:
            data = data[:, :frames, :]
        elif data.shape[1] < frames:
            raise NegativeDataError(f"{path.name} stores {data.shape[1]} frames per clip, but {frames} are needed.")
        return ArrayPool(data, hop_seconds=2.0)
    if data.ndim == 2 and data.shape[-1] == EMBEDDING_DIM:
        return SlidingPool(data, frames, hop_seconds=0.08)
    raise NegativeDataError(
        f"Unsupported negative feature shape {data.shape} in {path.name}. "
        f"Expected (N, frames, {EMBEDDING_DIM}) or (time, {EMBEDDING_DIM})."
    )


def _load_acav_pool(path: Path, frames: int) -> AcavPool:
    data = np.load(path, mmap_mode="r")
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3 or data.shape[-1] != EMBEDDING_DIM:
        raise NegativeDataError(
            f"{path.name} is not an ACAV-style (N, frames, {EMBEDDING_DIM}) file (shape {data.shape})."
        )
    return AcavPool(data, frames)


def resolve_negatives(
    config: NegativeConfig,
    *,
    frames: int,
    extractor: FeatureExtractor,
    clip_samples: int,
    progress: ProgressFn | None = None,
    log: Callable[[str], None] | None = None,
) -> ResolvedNegatives:
    """Resolve the configured negative source into sampling pools."""
    say = log or (lambda _message: None)
    source = config.source

    if isinstance(source, str) and source == "none":
        pool = EmptyPool(frames)
        info = NegativeSourceInfo("none", None, 0, 0.0, frames, ["No negatives configured."])
        return ResolvedNegatives(pool, None, info)

    if isinstance(source, (str, Path)) and str(source) not in {"auto"}:
        path = Path(source).expanduser()
        if not path.exists():
            raise NegativeDataError(f"Negative source does not exist: {path}")
        if path.is_dir():
            say(f"Building negative feature cache from {path} ...")
            cache = build_audio_feature_cache(
                path,
                extractor,
                clip_samples=clip_samples,
                frames=frames,
                max_hours=config.max_audio_hours,
                progress=progress,
            )
            data = np.load(cache, mmap_mode="r")
            pool = ArrayPool(data, hop_seconds=clip_samples / 16_000)
            info = NegativeSourceInfo(
                "audio",
                path,
                len(pool),
                pool.hours,
                frames,
                [f"Features cached at {cache}"],
            )
            return ResolvedNegatives(pool, pool, info, info)
        if path.suffix.lower() != ".npy":
            raise NegativeDataError(
                "Negative source must be 'auto', 'none', a .npy feature file, or a folder of audio."
            )
        pool = _load_npy_pool(path, frames)
        kind = "validation" if isinstance(pool, SlidingPool) else "npy"
        info = NegativeSourceInfo(kind, path, len(pool), pool.hours, frames)
        return ResolvedNegatives(pool, pool, info, info)

    # "auto"
    acav_path = (
        Path(config.acav_path).expanduser()
        if config.acav_path
        else downloads_dir() / ("openwakeword_features_ACAV100M_2000_hrs_16bit.npy")
    )
    validation_path = (
        Path(config.validation_path).expanduser()
        if config.validation_path
        else downloads_dir() / "validation_set_features.npy"
    )

    if not acav_path.is_file():
        if not config.download:
            raise NegativeDataError(
                "ACAV100M negative features are not available locally. "
                f"Expected: {acav_path}\n"
                "Either pass download=True to fetch ~17 GB from Hugging Face, "
                "or set negatives='path/to/file.npy' / negatives='path/to/negative_audio'."
            )
        say("Downloading ACAV100M negative features (~17 GB, one time).")
        acav_path = ensure_acav(progress)

    train_pool = _load_acav_pool(acav_path, frames)
    train_info = NegativeSourceInfo("acav", acav_path, len(train_pool), train_pool.hours, frames)

    val_pool = None
    val_info = None
    if validation_path.is_file():
        val_pool = SlidingPool(np.load(validation_path, mmap_mode="r"), frames, hop_seconds=0.08)
        val_info = NegativeSourceInfo("validation", validation_path, len(val_pool), val_pool.hours, frames)
    elif config.download:
        say("Downloading validation-set features (~185 MB, one time).")
        validation_path = ensure_validation_features(progress)
        val_pool = SlidingPool(np.load(validation_path, mmap_mode="r"), frames, hop_seconds=0.08)
        val_info = NegativeSourceInfo("validation", validation_path, len(val_pool), val_pool.hours, frames)
    else:
        say("Validation feature file not found; false-positive rate will be estimated from ACAV100M clips.")
        val_pool = train_pool
        val_info = train_info

    return ResolvedNegatives(train_pool, val_pool, train_info, val_info)


def save_negative_info(path: str | Path, resolved: ResolvedNegatives) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "train": resolved.info.summary(),
        "validation": resolved.validation_info.summary() if resolved.validation_info else None,
    }
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return target
