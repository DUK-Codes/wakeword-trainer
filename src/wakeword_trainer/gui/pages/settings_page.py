"""Settings page: preset, model, augmentation, negatives."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import customtkinter as ctk

from ...config import PRESETS, ConfigurationError, TrainerConfig
from ..widgets import (
    COLOR_MUTED,
    COLOR_WARN,
    Card,
    PathRow,
    Tooltip,
    font,
)
from .base import BasePage

_NUMERIC_FIELDS: list[tuple[str, str, str, Callable[[str], Any]]] = [
    ("steps", "steps", "int", int),
    ("batch_size", "batch_size", "int", int),
    ("layer_dim", "layer_dim", "int", int),
    ("n_blocks", "n_blocks", "int", int),
    ("val_every", "val_every", "int", int),
    ("checkpoint_every", "checkpoint_every", "int", int),
    ("early_stop_patience", "early_stop", "optional", int),
    ("seed", "seed", "int", int),
    ("learning_rate", "learning_rate", "float", float),
    ("clip_seconds", "clip_seconds", "float", float),
    ("target_fp_per_hour", "target_fp", "float", float),
]

# Keys controlled by a preset (everything else is user-owned).
_PRESET_KEYS = {
    "steps",
    "batch_size",
    "negative_ratio_start",
    "negative_ratio_end",
    "val_every",
    "average_top_k",
    "layer_dim",
    "n_blocks",
}


class SettingsPage(BasePage):
    def build(self) -> None:
        self._vars: dict[str, ctk.StringVar] = {}
        self._labels: dict[str, ctk.CTkLabel] = {}
        self._loading = False
        self.columnconfigure(0, weight=1)
        row = 0

        # ---------------------------------------------------------- presets
        self.preset_card = Card(self)
        self.preset_card.grid(row=row, column=0, sticky="ew", pady=(0, 12))
        row += 1
        body = self.preset_card.body
        body.grid_columnconfigure(0, weight=1)

        self.preset_var = ctk.StringVar(value=self.app.gui_state.preset)
        self.preset_buttons: dict[str, ctk.CTkRadioButton] = {}
        for index, preset in enumerate(("quick", "balanced", "best")):
            radio = ctk.CTkRadioButton(
                body,
                text=self.tr(f"preset_{preset}"),
                variable=self.preset_var,
                value=preset,
                font=font(13, "bold"),
                command=self._preset_changed,
            )
            radio.grid(row=index, column=0, sticky="w", pady=(4, 0))
            self.preset_buttons[preset] = radio
        self.preset_desc = ctk.CTkLabel(
            body, text="", font=font(12), text_color=COLOR_MUTED, anchor="w", justify="left", wraplength=780
        )
        self.preset_desc.grid(row=3, column=0, sticky="w", pady=(8, 0))

        # ------------------------------------------------------------ model
        self.model_card = Card(self)
        self.model_card.grid(row=row, column=0, sticky="ew", pady=(0, 12))
        row += 1
        body = self.model_card.body
        body.grid_columnconfigure(1, weight=1)

        self.word_label = ctk.CTkLabel(
            body, text=self.tr("word"), font=font(12, "bold"), text_color=COLOR_MUTED, anchor="w"
        )
        self.word_label.grid(row=0, column=0, sticky="w", padx=(0, 12))
        self.word_var = ctk.StringVar(value=self.app.gui_state.word)
        self.word_entry = ctk.CTkEntry(
            body, textvariable=self.word_var, font=font(13), height=34, placeholder_text=self.tr("word_placeholder")
        )
        self.word_entry.grid(row=0, column=1, columnspan=2, sticky="ew")
        self.word_var.trace_add("write", lambda *_: self._on_word_change())

        self.output_label = ctk.CTkLabel(
            body, text=self.tr("output_dir"), font=font(12, "bold"), text_color=COLOR_MUTED, anchor="w"
        )
        self.output_label.grid(row=1, column=0, sticky="w", padx=(0, 12), pady=(10, 0))
        self.output_row = PathRow(body, mode="directory", translator=self.tr, on_change=self._on_output_change)
        self.output_row.grid(row=1, column=1, columnspan=2, sticky="ew", pady=(10, 0))
        self.output_row.set(self.app.gui_state.output)

        self.output_hint = ctk.CTkLabel(
            body, text=self.tr("output_dir_hint"), font=font(11), text_color=COLOR_MUTED, anchor="w"
        )
        self.output_hint.grid(row=2, column=1, columnspan=2, sticky="w", pady=(4, 0))

        # --------------------------------------------------------- advanced
        self.advanced_card = Card(self)
        self.advanced_card.grid(row=row, column=0, sticky="ew", pady=(0, 12))
        row += 1
        body = self.advanced_card.body
        for column in range(4):
            body.grid_columnconfigure(column, weight=1, uniform="adv")

        self._build_advanced(body)

        self.augment_label = ctk.CTkLabel(
            body, text=self.tr("augment_enabled"), font=font(12, "bold"), text_color=COLOR_MUTED, anchor="w"
        )
        self.augment_label.grid(row=5, column=0, sticky="w", pady=(12, 0))
        self.augment_var = ctk.BooleanVar(value=not self.app.gui_state.advanced.get("_no_augment", False))
        self.augment_switch = ctk.CTkSwitch(body, text="", variable=self.augment_var, command=self._on_augment_change)
        self.augment_switch.grid(row=5, column=1, sticky="w", pady=(12, 0))
        self.augment_hint = ctk.CTkLabel(
            body, text=self.tr("augment_hint"), font=font(11), text_color=COLOR_MUTED, anchor="w"
        )
        self.augment_hint.grid(row=5, column=2, columnspan=2, sticky="w", pady=(12, 0))

        self.background_label = ctk.CTkLabel(
            body, text=self.tr("background_dir"), font=font(12), text_color=COLOR_MUTED, anchor="w"
        )
        self.background_label.grid(row=6, column=0, sticky="w", pady=(10, 0))
        self.background_row = PathRow(body, mode="directory", translator=self.tr, on_change=self._on_background)
        self.background_row.grid(row=6, column=1, columnspan=3, sticky="ew", pady=(10, 0))
        self.background_row.set(self.app.gui_state.advanced.get("background_dir", ""))

        self.rir_label = ctk.CTkLabel(body, text=self.tr("rir_dir"), font=font(12), text_color=COLOR_MUTED, anchor="w")
        self.rir_label.grid(row=7, column=0, sticky="w", pady=(10, 0))
        self.rir_row = PathRow(body, mode="directory", translator=self.tr, on_change=self._on_rir)
        self.rir_row.grid(row=7, column=1, columnspan=3, sticky="ew", pady=(10, 0))
        self.rir_row.set(self.app.gui_state.advanced.get("rir_dir", ""))

        # -------------------------------------------------------- negatives
        self.negatives_card = Card(self)
        self.negatives_card.grid(row=row, column=0, sticky="ew", pady=(0, 12))
        row += 1
        body = self.negatives_card.body
        body.grid_columnconfigure(1, weight=1)

        self.negatives_label = ctk.CTkLabel(
            body, text=self.tr("negatives_source"), font=font(12, "bold"), text_color=COLOR_MUTED, anchor="w"
        )
        self.negatives_label.grid(row=0, column=0, sticky="w", padx=(0, 12))
        self.negatives_mode = ctk.CTkOptionMenu(
            body,
            values=[
                self.tr("negatives_auto"),
                self.tr("negatives_npy"),
                self.tr("negatives_audio"),
                self.tr("negatives_none"),
            ],
            font=font(12),
            command=self._on_negatives_mode,
        )
        self.negatives_mode.grid(row=0, column=1, sticky="ew")
        self._mode_keys = ["auto", "npy", "audio", "none"]
        mode = self.app.gui_state.negatives_mode if self.app.gui_state.negatives_mode in self._mode_keys else "auto"
        self.negatives_mode.set(self.tr(f"negatives_{mode}"))

        self.negatives_path_label = ctk.CTkLabel(
            body, text=self.tr("negatives_path"), font=font(12), text_color=COLOR_MUTED, anchor="w"
        )
        self.negatives_path_label.grid(row=1, column=0, sticky="w", padx=(0, 12), pady=(10, 0))
        self.negatives_row = ctk.CTkFrame(body, fg_color="transparent")
        self.negatives_row.grid(row=1, column=1, sticky="ew", pady=(10, 0))
        self.negatives_row.grid_columnconfigure(0, weight=1)
        self.negatives_path_var = ctk.StringVar(value=self.app.gui_state.negatives_path)
        self.negatives_entry = ctk.CTkEntry(
            self.negatives_row, textvariable=self.negatives_path_var, font=font(12), height=32
        )
        self.negatives_entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.negatives_browse = ctk.CTkButton(
            self.negatives_row,
            text=self.tr("browse"),
            width=90,
            height=32,
            font=font(12),
            command=self._browse_negatives,
        )
        self.negatives_browse.grid(row=0, column=1)
        self.negatives_path_var.trace_add("write", lambda *_: self._on_negatives_path())

        self.download_var = ctk.BooleanVar(value=self.app.gui_state.download_negatives)
        self.download_check = ctk.CTkCheckBox(
            body,
            text=self.tr("negatives_download"),
            variable=self.download_var,
            font=font(12),
            command=self._on_download,
        )
        self.download_check.grid(row=2, column=1, sticky="w", pady=(10, 0))
        self.download_hint = ctk.CTkLabel(
            body, text=self.tr("negatives_download_hint"), font=font(11), text_color=COLOR_MUTED, anchor="w"
        )
        self.download_hint.grid(row=3, column=1, sticky="w", pady=(2, 0))

        self.negatives_warning = ctk.CTkLabel(
            body, text="", font=font(11), text_color=COLOR_WARN, anchor="w", justify="left", wraplength=780
        )
        self.negatives_warning.grid(row=4, column=0, columnspan=2, sticky="w", pady=(8, 0))

        # ----------------------------------------------------------- export
        self.export_card = Card(self)
        self.export_card.grid(row=row, column=0, sticky="ew", pady=(0, 12))
        row += 1
        body = self.export_card.body
        self.tflite_var = ctk.BooleanVar(value=self.app.gui_state.export_tflite)
        self.tflite_check = ctk.CTkCheckBox(
            body,
            text=self.tr("export_tflite"),
            variable=self.tflite_var,
            font=font(12),
            command=self._on_tflite,
        )
        self.tflite_check.grid(row=0, column=0, sticky="w")
        self.tflite_hint = ctk.CTkLabel(
            body, text=self.tr("export_tflite_hint"), font=font(11), text_color=COLOR_MUTED, anchor="w"
        )
        self.tflite_hint.grid(row=1, column=0, sticky="w", pady=(2, 0))

        self.resume_var = ctk.BooleanVar(value=self.app.gui_state.resume)
        self.resume_check = ctk.CTkCheckBox(
            body,
            text=self.tr("train_resume"),
            variable=self.resume_var,
            font=font(12),
            command=self._on_resume,
        )
        self.resume_check.grid(row=2, column=0, sticky="w", pady=(12, 0))

        self._update_negatives_warning()
        self._update_preset_desc()
        self._sync_fields()

    # ------------------------------------------------------------------
    def _build_advanced(self, parent) -> None:
        self.advanced_hint = ctk.CTkLabel(
            parent, text=self.tr("advanced"), font=font(13, "bold"), text_color=COLOR_MUTED, anchor="w"
        )
        self.advanced_hint.grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 6))

        for index, (key, label_key, kind, caster) in enumerate(_NUMERIC_FIELDS):
            row = 1 + index // 3
            column = (index % 3) * 2
            label = ctk.CTkLabel(parent, text=self.tr(label_key), font=font(11), text_color=COLOR_MUTED, anchor="w")
            label.grid(row=row, column=column, sticky="w", padx=(0, 8), pady=(6, 0))
            variable = ctk.StringVar()
            entry = ctk.CTkEntry(parent, textvariable=variable, font=font(12), height=30, width=100)
            entry.grid(row=row, column=column + 1, sticky="ew", padx=(0, 14), pady=(6, 0))
            variable.trace_add("write", lambda *_a, k=key, c=caster, kk=kind: self._on_numeric(k, c, kk))
            self._vars[key] = variable
            self._labels[key] = label
            if kind == "optional":
                Tooltip(entry, self.tr("early_stop_off"))

        # Device selector
        device_row = 1 + (len(_NUMERIC_FIELDS) + 2) // 3
        self.device_label = ctk.CTkLabel(
            parent, text=self.tr("device"), font=font(11), text_color=COLOR_MUTED, anchor="w"
        )
        self.device_label.grid(row=device_row, column=0, sticky="w", pady=(6, 0))
        self.device_var = ctk.StringVar(value=str(self.app.gui_state.advanced.get("device", "auto")))
        self.device_menu = ctk.CTkOptionMenu(
            parent,
            values=["auto", "cpu", "cuda"],
            variable=self.device_var,
            font=font(12),
            width=100,
            command=self._on_device,
        )
        self.device_menu.grid(row=device_row, column=1, sticky="w", padx=(0, 14), pady=(6, 0))
        self.early_stop_off_label = ctk.CTkLabel(
            parent, text=self.tr("early_stop_off"), font=font(10), text_color=COLOR_MUTED, anchor="w"
        )
        self.early_stop_off_label.grid(row=device_row, column=2, sticky="w")

    # ------------------------------------------------------------------
    def _save(self) -> None:
        if self._loading:
            return
        self.app.gui_state.save()

    def _on_word_change(self) -> None:
        self.app.gui_state.word = self.word_var.get()
        self._save()

    def _on_output_change(self, value: str) -> None:
        self.app.gui_state.output = value.strip()
        self._save()

    def _on_background(self, value: str) -> None:
        self._set_advanced("background_dir", value.strip() or None)

    def _on_rir(self, value: str) -> None:
        self._set_advanced("rir_dir", value.strip() or None)

    def _set_advanced(self, key: str, value) -> None:
        if self._loading:
            return
        if value in (None, ""):
            self.app.gui_state.advanced.pop(key, None)
        else:
            self.app.gui_state.advanced[key] = value
        self._save()

    def _on_numeric(self, key: str, caster, kind: str) -> None:
        if self._loading or key not in self._vars:
            return
        text = self._vars[key].get().strip()
        if not text:
            if kind == "optional":
                self.app.gui_state.advanced.pop(key, None)
                self._save()
            return
        try:
            value = caster(text)
        except (TypeError, ValueError):
            return
        self.app.gui_state.advanced[key] = value
        self._save()

    def _on_device(self, value: str) -> None:
        self.app.gui_state.advanced["device"] = value
        self._save()

    def _on_augment_change(self) -> None:
        enabled = bool(self.augment_var.get())
        self.app.gui_state.advanced["_no_augment"] = not enabled
        if not enabled:
            self.app.gui_state.advanced["augment"] = {"enabled": False}
        else:
            self.app.gui_state.advanced.pop("augment", None)
        self._save()

    def _on_tflite(self) -> None:
        self.app.gui_state.export_tflite = bool(self.tflite_var.get())
        self._save()

    def _on_resume(self) -> None:
        self.app.gui_state.resume = bool(self.resume_var.get())
        self._save()

    def _on_download(self) -> None:
        self.app.gui_state.download_negatives = bool(self.download_var.get())
        self._save()

    def _on_negatives_mode(self, _value: str) -> None:
        index = self.negatives_mode.current() if hasattr(self.negatives_mode, "current") else 0
        index = max(0, min(index, len(self._mode_keys) - 1))
        self.app.gui_state.negatives_mode = self._mode_keys[index]
        self._save()
        self._update_negatives_warning()

    def _on_negatives_path(self) -> None:
        self.app.gui_state.negatives_path = self.negatives_path_var.get().strip()
        self._save()

    def _browse_negatives(self) -> None:
        from tkinter import filedialog

        mode = self.app.gui_state.negatives_mode
        if mode == "audio":
            path = filedialog.askdirectory()
        else:
            path = filedialog.askopenfilename(filetypes=[("NumPy features", "*.npy"), ("All files", "*.*")])
        if path:
            self.negatives_path_var.set(path)

    def _update_negatives_warning(self) -> None:
        mode = self.app.gui_state.negatives_mode
        if mode == "none":
            self.negatives_warning.configure(text=self.tr("negatives_none_warning"))
        elif mode in {"npy", "audio"} and not self.app.gui_state.negatives_path:
            self.negatives_warning.configure(text=self.tr("negatives_path"))
        else:
            self.negatives_warning.configure(text="")

    # ------------------------------------------------------------------
    def _preset_changed(self) -> None:
        preset = self.preset_var.get()
        if preset not in PRESETS:
            return
        self.app.gui_state.preset = preset
        preset_values = dict(PRESETS[preset])
        for key in list(self.app.gui_state.advanced):
            if key in _PRESET_KEYS:
                self.app.gui_state.advanced.pop(key, None)
        for key, value in preset_values.items():
            self.app.gui_state.advanced.setdefault(key, value)
        self._save()
        self._update_preset_desc()
        self._sync_fields()

    def _update_preset_desc(self) -> None:
        preset = self.preset_var.get()
        self.preset_desc.configure(text=self.tr(f"preset_{preset}_desc"))

    def _sync_fields(self) -> None:
        """Refresh entry values from the current preset + advanced overrides."""
        self._loading = True
        try:
            preset_values = dict(PRESETS.get(self.app.gui_state.preset, PRESETS["balanced"]))
            merged = {**preset_values, **self.app.gui_state.advanced}

            defaults = {
                "steps": 15_000,
                "batch_size": 64,
                "layer_dim": 64,
                "n_blocks": 2,
                "val_every": 1_000,
                "checkpoint_every": 1_000,
                "early_stop_patience": None,
                "seed": 42,
                "learning_rate": 3e-4,
                "clip_seconds": 2.0,
                "target_fp_per_hour": 1.0,
            }
            for key, _label, kind, _caster in _NUMERIC_FIELDS:
                if key not in self._vars:
                    continue
                value = merged.get(key, defaults.get(key))
                if kind == "optional" and not value:
                    self._vars[key].set("")
                elif key == "learning_rate":
                    self._vars[key].set(f"{float(value):g}")
                elif isinstance(value, float):
                    self._vars[key].set(f"{value:g}")
                else:
                    self._vars[key].set(str(value))

            self.device_var.set(str(merged.get("device", "auto")))
            augment_cfg = merged.get("augment") or {}
            self.augment_var.set(not (augment_cfg.get("enabled") is False or merged.get("_no_augment")))
            self.background_row.set(str(merged.get("background_dir") or ""))
            self.rir_row.set(str(merged.get("rir_dir") or ""))
        finally:
            self._loading = False

    # ------------------------------------------------------------------
    def config(self) -> TrainerConfig:
        """Build and validate a :class:`TrainerConfig` from the page."""
        state = self.app.gui_state
        overrides = {
            key: value
            for key, value in state.advanced.items()
            if not key.startswith("_") and key not in {"background_dir", "rir_dir", "augment"}
        }
        config = TrainerConfig.from_preset(
            state.preset,  # type: ignore[arg-type]
            word=self.word_var.get().strip() or "wakeword",
            dataset=self.app.gui_state.dataset or None,
            output_dir=self.output_row.get() or "wakeword_models",
            **overrides,
        )
        if state.advanced.get("background_dir"):
            config.augment.background_dir = Path(state.advanced["background_dir"])
        if state.advanced.get("rir_dir"):
            config.augment.rir_dir = Path(state.advanced["rir_dir"])
        if state.advanced.get("_no_augment"):
            config.augment.enabled = False
        config.negatives.source = state.negatives_source()
        config.negatives.download = bool(self.download_var.get())
        config.validate()
        return config

    def validate_settings(self) -> tuple[TrainerConfig | None, str | None]:
        try:
            config = self.config()
        except ConfigurationError as exc:
            return None, str(exc)
        except Exception as exc:  # noqa: BLE001
            return None, f"{type(exc).__name__}: {exc}"
        return config, None

    # ------------------------------------------------------------------
    def on_show(self) -> None:
        self._sync_fields()

    def retranslate(self) -> None:
        self.preset_card.set_title(self.tr("preset"))
        for preset, radio in self.preset_buttons.items():
            radio.configure(text=self.tr(f"preset_{preset}"))
        self._update_preset_desc()
        self.model_card.set_title(self.tr("settings_title"))
        self.word_label.configure(text=self.tr("word"))
        self.word_entry.configure(placeholder_text=self.tr("word_placeholder"))
        self.output_label.configure(text=self.tr("output_dir"))
        self.output_hint.configure(text=self.tr("output_dir_hint"))
        self.advanced_card.set_title(self.tr("advanced"))
        self.advanced_hint.configure(text=self.tr("advanced"))
        self.augment_label.configure(text=self.tr("augment_enabled"))
        self.augment_hint.configure(text=self.tr("augment_hint"))
        self.background_label.configure(text=self.tr("background_dir"))
        self.rir_label.configure(text=self.tr("rir_dir"))
        self.negatives_card.set_title(self.tr("negatives_source"))
        self.negatives_label.configure(text=self.tr("negatives_source"))
        self.negatives_path_label.configure(text=self.tr("negatives_path"))
        self.negatives_browse.configure(text=self.tr("browse"))
        self.download_check.configure(text=self.tr("negatives_download"))
        self.download_hint.configure(text=self.tr("negatives_download_hint"))
        self.export_card.set_title(self.tr("results_export"))
        self.tflite_check.configure(text=self.tr("export_tflite"))
        self.tflite_hint.configure(text=self.tr("export_tflite_hint"))
        self.resume_check.configure(text=self.tr("train_resume"))
        self.device_label.configure(text=self.tr("device"))
        self.early_stop_off_label.configure(text=self.tr("early_stop_off"))
        for key, _label, _kind, _caster in _NUMERIC_FIELDS:
            if key in self._labels:
                self._labels[key].configure(text=self.tr(_label))
        mode = self.app.gui_state.negatives_mode
        self.negatives_mode.configure(
            values=[
                self.tr("negatives_auto"),
                self.tr("negatives_npy"),
                self.tr("negatives_audio"),
                self.tr("negatives_none"),
            ]
        )
        self.negatives_mode.set(self.tr(f"negatives_{mode}"))
        self._update_negatives_warning()
