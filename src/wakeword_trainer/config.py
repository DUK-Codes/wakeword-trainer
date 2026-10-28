"""Configuration objects for the training pipeline.

Everything is a plain dataclass so a config can be built in a script,
serialized to JSON, passed to a subprocess, or edited by the GUI.

Example
-------
>>> from wakeword_trainer import TrainerConfig
>>> cfg = TrainerConfig.from_preset("quick", word="hey computer")
>>> cfg.train.steps
4000
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Literal, get_args, get_origin, get_type_hints

from .exceptions import ConfigurationError

Preset = Literal["quick", "balanced", "best"]
PadMode = Literal["start", "center", "end", "random"]
NegativeSource = str | Path

PRESETS: dict[str, dict[str, Any]] = {
    "quick": {
        "steps": 4_000,
        "batch_size": 48,
        "negative_ratio_end": 16.0,
        "val_every": 500,
        "average_top_k": 1,
        "layer_dim": 32,
        "n_blocks": 1,
    },
    "balanced": {
        "steps": 15_000,
        "batch_size": 64,
        "negative_ratio_end": 32.0,
        "val_every": 1_000,
        "average_top_k": 3,
        "layer_dim": 64,
        "n_blocks": 2,
    },
    "best": {
        "steps": 40_000,
        "batch_size": 96,
        "negative_ratio_end": 48.0,
        "val_every": 2_000,
        "average_top_k": 3,
        "layer_dim": 128,
        "n_blocks": 2,
    },
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _jsonable(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _coerce(value: Any, hint: Any) -> Any:
    """Best-effort conversion of a JSON value back to a dataclass field type."""
    if value is None:
        return None
    origin = get_origin(hint)
    get_args(hint)
    if origin is Literal:
        return value
    if origin in (tuple,):
        return tuple(value) if isinstance(value, (list, tuple)) else value
    if origin in (list,):
        return list(value) if isinstance(value, (list, tuple)) else value
    if hint is Path:
        return Path(value).expanduser()
    if hint is bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
    if hint is int:
        return int(value)
    if hint is float:
        return float(value)
    return value


@dataclass
class AudioConfig:
    """How raw audio files are normalized before feature extraction."""

    sample_rate: int = 16_000
    clip_seconds: float = 2.0
    min_clip_seconds: float = 0.25
    max_clip_seconds: float = 60.0
    trim_silence: bool = True
    trim_top_db: float = 40.0
    pad_mode: PadMode = "random"
    fade_ms: float = 10.0
    normalize_peak: bool = False

    @property
    def clip_samples(self) -> int:
        return int(round(self.sample_rate * self.clip_seconds))

    def validate(self) -> None:
        if self.sample_rate != 16_000:
            raise ConfigurationError(
                f"The openWakeWord feature models require 16 kHz audio; got sample_rate={self.sample_rate}."
            )
        if not 0.5 <= self.clip_seconds <= 10.0:
            raise ConfigurationError("clip_seconds must be between 0.5 and 10 seconds.")
        if self.min_clip_seconds <= 0 or self.min_clip_seconds >= self.clip_seconds:
            raise ConfigurationError("min_clip_seconds must be > 0 and < clip_seconds.")


@dataclass
class FeatureConfig:
    """Feature extractor settings.

    ``None`` model paths resolve to the bundled ONNX models.
    """

    mel_model_path: Path | None = None
    embedding_model_path: Path | None = None
    window_size: int = 76
    step_size: int = 8
    providers: list[str] | None = None
    num_threads: int = 4

    def validate(self) -> None:
        if self.window_size < 8:
            raise ConfigurationError("window_size must be at least 8 mel frames.")
        if not 1 <= self.step_size <= self.window_size:
            raise ConfigurationError("step_size must be between 1 and window_size.")


@dataclass
class AugmentConfig:
    """On-the-fly augmentation applied to positive clips during training."""

    enabled: bool = True
    gain_prob: float = 0.5
    gain_db: tuple[float, float] = (-6.0, 6.0)
    noise_prob: float = 0.5
    noise_snr_db: tuple[float, float] = (5.0, 30.0)
    pitch_prob: float = 0.3
    pitch_semitones: tuple[float, float] = (-2.0, 2.0)
    speed_prob: float = 0.2
    speed_rate: tuple[float, float] = (0.9, 1.1)
    reverb_prob: float = 0.2
    rir_dir: Path | None = None
    background_prob: float = 0.3
    background_dir: Path | None = None
    background_snr_db: tuple[float, float] = (5.0, 20.0)

    def validate(self) -> None:
        for name in (
            "gain_prob",
            "noise_prob",
            "pitch_prob",
            "speed_prob",
            "reverb_prob",
            "background_prob",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ConfigurationError(f"augment.{name} must be in [0, 1].")
        if self.rir_dir is not None and not Path(self.rir_dir).is_dir():
            raise ConfigurationError(f"augment.rir_dir does not exist: {self.rir_dir}")
        if self.background_dir is not None and not Path(self.background_dir).is_dir():
            raise ConfigurationError(f"augment.background_dir does not exist: {self.background_dir}")


@dataclass
class ModelConfig:
    layer_dim: int = 64
    n_blocks: int = 2
    dropout: float = 0.1

    def validate(self) -> None:
        if self.layer_dim < 8:
            raise ConfigurationError("model.layer_dim must be at least 8.")
        if self.n_blocks < 0:
            raise ConfigurationError("model.n_blocks cannot be negative.")
        if not 0.0 <= self.dropout < 1.0:
            raise ConfigurationError("model.dropout must be in [0, 1).")


@dataclass
class NegativeConfig:
    """Where negative (non-wake-word) examples come from.

    ``source`` accepts:

    * ``"auto"``      – download the openWakeWord feature sets on demand;
    * ``"none"``      – no negatives (not recommended, only for smoke tests);
    * a ``.npy`` path – pre-computed embeddings;
    * a directory    – audio files used to build a negative pool.
    """

    source: NegativeSource = "auto"
    download: bool = False
    acav_path: Path | None = None
    validation_path: Path | None = None
    train_pool_limit: int | None = None
    max_audio_hours: float = 5.0

    def validate(self) -> None:
        if isinstance(self.source, str) and self.source not in {"auto", "none"}:
            path = Path(self.source).expanduser()
            if not path.exists():
                raise ConfigurationError(f"negatives.source does not exist: {path}")
            if path.suffix.lower() != ".npy" and not path.is_dir():
                raise ConfigurationError(
                    "negatives.source must be 'auto', 'none', a .npy file, or a directory of audio."
                )
        if self.max_audio_hours <= 0:
            raise ConfigurationError("negatives.max_audio_hours must be positive.")


@dataclass
class TrainConfig:
    preset: Preset = "balanced"
    steps: int = 15_000
    batch_size: int = 64
    negative_ratio_start: float = 4.0
    negative_ratio_end: float = 32.0
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    warmup_fraction: float = 0.15
    hold_fraction: float = 0.35
    val_every: int = 1_000
    val_windows: int = 20_000
    seed: int = 42
    device: str = "auto"
    num_threads: int = 4
    average_top_k: int = 3
    checkpoint_every: int = 1_000
    early_stop_patience: int | None = None

    def validate(self) -> None:
        if self.steps < 10:
            raise ConfigurationError("train.steps must be at least 10.")
        if self.batch_size < 1:
            raise ConfigurationError("train.batch_size must be positive.")
        if not 0 <= self.warmup_fraction < 1:
            raise ConfigurationError("train.warmup_fraction must be in [0, 1).")
        if not 0 <= self.hold_fraction < 1 - self.warmup_fraction:
            raise ConfigurationError("train.hold_fraction is too large.")
        if self.learning_rate <= 0:
            raise ConfigurationError("train.learning_rate must be positive.")
        if self.negative_ratio_start < 0 or self.negative_ratio_end < 0:
            raise ConfigurationError("negative ratios cannot be negative.")
        if self.val_every < 1:
            raise ConfigurationError("train.val_every must be positive.")
        if self.device not in {"auto", "cpu", "cuda"} and not self.device.startswith("cuda:"):
            raise ConfigurationError("train.device must be 'auto', 'cpu', 'cuda' or 'cuda:N'.")


@dataclass
class EvalConfig:
    target_fp_per_hour: float = 1.0
    threshold_min: float = 0.01
    threshold_max: float = 0.99
    threshold_steps: int = 99
    max_negative_windows: int = 100_000
    augmented_test_variants: int = 0

    def validate(self) -> None:
        if self.target_fp_per_hour <= 0:
            raise ConfigurationError("evaluation.target_fp_per_hour must be positive.")
        if not 0 <= self.threshold_min < self.threshold_max <= 1:
            raise ConfigurationError("evaluation thresholds must satisfy 0 <= min < max <= 1.")
        if self.threshold_steps < 2:
            raise ConfigurationError("evaluation.threshold_steps must be at least 2.")


@dataclass
class TrainerConfig:
    """Complete configuration for one wake word training run."""

    word: str = ""
    dataset: Path | None = None
    output_dir: Path = Path("wakeword_model")
    language: str = "ru"
    audio: AudioConfig = field(default_factory=AudioConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    negatives: NegativeConfig = field(default_factory=NegativeConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)

    # ------------------------------------------------------------------ presets
    @classmethod
    def from_preset(cls, preset: Preset = "balanced", **overrides: Any) -> TrainerConfig:
        """Create a config from a named preset.

        Recognized keyword overrides mirror the dataclass structure, e.g.
        ``word="hey jarvis"``, ``dataset=Path(...)``, ``steps=20000``,
        ``layer_dim=96``, ``negatives="auto"``.
        """
        if preset not in PRESETS:
            raise ConfigurationError(f"Unknown preset {preset!r}; choose from {sorted(PRESETS)}.")
        cfg = cls()
        cfg.train.preset = preset
        for key, value in PRESETS[preset].items():
            if key in {"layer_dim", "n_blocks"}:
                setattr(cfg.model, key, value)
            else:
                setattr(cfg.train, key, value)
        cfg.apply_overrides(overrides)
        return cfg

    def apply_overrides(self, overrides: dict[str, Any]) -> TrainerConfig:
        """Apply flat keyword overrides.

        Keys may be top-level (``word``), dotted (``train.steps``), or short
        names for the most common training/model fields (``steps``,
        ``layer_dim``, ``negatives``, ``batch_size``, ...).
        """
        short_train = {f.name for f in fields(TrainConfig)} - {"preset"}
        short_model = {"layer_dim", "n_blocks", "dropout"}
        short_audio = {
            "clip_seconds",
            "sample_rate",
            "trim_silence",
            "trim_top_db",
            "pad_mode",
            "normalize_peak",
            "min_clip_seconds",
        }
        short_eval = {
            "target_fp_per_hour",
            "threshold_min",
            "threshold_max",
            "threshold_steps",
            "max_negative_windows",
            "augmented_test_variants",
        }
        for key, value in overrides.items():
            if value is None:
                continue
            if key in {"word", "dataset", "output_dir", "language"}:
                setattr(self, key, _coerce(value, get_type_hints(type(self))[key]))
            elif key == "negatives":
                self.negatives.source = value if isinstance(value, Path) else str(value)
            elif key == "download":
                self.negatives.download = bool(value)
            elif key == "augment":
                for aug_key, aug_value in dict(value).items():
                    setattr(self.augment, aug_key, aug_value)
            elif key in short_model:
                setattr(self.model, key, _coerce(value, get_type_hints(ModelConfig)[key]))
            elif key in short_audio:
                setattr(self.audio, key, _coerce(value, get_type_hints(AudioConfig)[key]))
            elif key in short_eval:
                setattr(self.evaluation, key, _coerce(value, get_type_hints(EvalConfig)[key]))
            elif key in short_train:
                setattr(self.train, key, _coerce(value, get_type_hints(TrainConfig)[key]))
            elif "." in key:
                section, _, name = key.partition(".")
                target = getattr(self, section, None)
                if target is None or not hasattr(target, name):
                    raise ConfigurationError(f"Unknown configuration key: {key}")
                setattr(target, name, _coerce(value, get_type_hints(type(target))[name]))
            else:
                raise ConfigurationError(f"Unknown configuration key: {key}")
        return self

    # ------------------------------------------------------------- validation
    def validate(self) -> TrainerConfig:
        self.audio.validate()
        self.features.validate()
        self.augment.validate()
        self.model.validate()
        self.negatives.validate()
        self.train.validate()
        self.evaluation.validate()
        if not self.word.strip():
            raise ConfigurationError("word must not be empty.")
        if self.dataset is not None and not Path(self.dataset).is_dir():
            raise ConfigurationError(f"dataset directory does not exist: {self.dataset}")
        return self

    # --------------------------------------------------------- serialization
    def to_dict(self) -> dict[str, Any]:
        return {f.name: _jsonable(getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainerConfig:
        cfg = cls()
        hints = get_type_hints(cls)
        nested = {
            "audio": AudioConfig,
            "features": FeatureConfig,
            "augment": AugmentConfig,
            "model": ModelConfig,
            "negatives": NegativeConfig,
            "train": TrainConfig,
            "evaluation": EvalConfig,
        }
        for key, value in data.items():
            if key in nested:
                section = getattr(cfg, key)
                section_hints = get_type_hints(nested[key])
                for sub_key, sub_value in value.items():
                    if hasattr(section, sub_key):
                        setattr(section, sub_key, _coerce(sub_value, section_hints[sub_key]))
            elif hasattr(cfg, key):
                setattr(cfg, key, _coerce(value, hints[key]))
        return cfg

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> TrainerConfig:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def copy(self, **changes: Any) -> TrainerConfig:
        clone = replace(
            self,
            audio=replace(self.audio),
            features=replace(self.features),
            augment=replace(self.augment),
            model=replace(self.model),
            negatives=replace(self.negatives),
            train=replace(self.train),
            evaluation=replace(self.evaluation),
        )
        return clone.apply_overrides(changes)
