"""Persisted GUI state (window-independent, JSON on disk)."""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import PRESETS, TrainerConfig
from ..paths import config_dir
from .i18n import DEFAULT_LANGUAGE, LANGUAGES

STATE_FILE = "gui_state.json"


@dataclass
class GuiState:
    language: str = DEFAULT_LANGUAGE
    dataset: str = ""
    output: str = ""
    word: str = ""
    preset: str = "balanced"
    negatives_mode: str = "auto"  # auto | npy | audio | none
    negatives_path: str = ""
    download_negatives: bool = False
    export_tflite: bool = False
    resume: bool = False
    last_model: str = ""
    advanced: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    @classmethod
    def path(cls) -> Path:
        return config_dir() / STATE_FILE

    @classmethod
    def load(cls) -> GuiState:
        path = cls.path()
        if not path.is_file():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - a corrupt state file must not block startup
            return cls()
        state = cls()
        for key in (
            "language",
            "dataset",
            "output",
            "word",
            "preset",
            "negatives_mode",
            "negatives_path",
            "last_model",
        ):
            value = data.get(key)
            if isinstance(value, str):
                setattr(state, key, value)
        for key in ("download_negatives", "export_tflite", "resume"):
            if key in data:
                setattr(state, key, bool(data[key]))
        if isinstance(data.get("advanced"), dict):
            state.advanced = dict(data["advanced"])
        if state.language not in LANGUAGES:
            state.language = DEFAULT_LANGUAGE
        if state.preset not in PRESETS:
            state.preset = "balanced"
        return state

    def save(self) -> None:
        path = self.path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "language": self.language,
            "dataset": self.dataset,
            "output": self.output,
            "word": self.word,
            "preset": self.preset,
            "negatives_mode": self.negatives_mode,
            "negatives_path": self.negatives_path,
            "download_negatives": self.download_negatives,
            "export_tflite": self.export_tflite,
            "resume": self.resume,
            "last_model": self.last_model,
            "advanced": self.advanced,
        }
        with contextlib.suppress(OSError):
            path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # ------------------------------------------------------------------
    def negatives_source(self) -> str:
        if self.negatives_mode == "npy":
            return self.negatives_path or "auto"
        if self.negatives_mode == "audio":
            return self.negatives_path or "auto"
        if self.negatives_mode == "none":
            return "none"
        return "auto"

    def build_config(
        self,
        *,
        word: str | None = None,
        dataset: str | None = None,
        output: str | None = None,
    ) -> TrainerConfig:
        """Build a validated :class:`TrainerConfig` from the current state."""
        overrides = {
            key: value
            for key, value in self.advanced.items()
            if not key.startswith("_") and key not in {"background_dir", "rir_dir", "augment"}
        }
        config = TrainerConfig.from_preset(
            self.preset,  # type: ignore[arg-type]
            word=(word if word is not None else self.word).strip() or "wakeword",
            dataset=(dataset if dataset is not None else self.dataset) or None,
            output_dir=(output if output is not None else self.output) or "wakeword_models",
            **overrides,
        )
        if self.advanced.get("background_dir"):
            config.augment.background_dir = Path(self.advanced["background_dir"])
        if self.advanced.get("rir_dir"):
            config.augment.rir_dir = Path(self.advanced["rir_dir"])
        if self.advanced.get("_no_augment"):
            config.augment.enabled = False
        config.negatives.source = self.negatives_source()
        config.negatives.download = bool(self.download_negatives)
        config.validate()
        return config
