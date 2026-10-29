"""Components page: optional speech engines, negatives and data folders.

The desktop build ships without speech engines. This page installs and removes
them, shows the state of the downloaded training data, and opens the folders
where voices, plugins and caches are kept.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import customtkinter as ctk

from ... import components
from ...data.negatives import ACAV_SIZE, VALIDATION_SIZE, ensure_acav, ensure_validation_features
from ...paths import bundled_model_paths, cache_dir, config_dir, downloads_dir, tts_plugins_dir
from ...tts import registry
from ..widgets import (
    COLOR_ERROR,
    COLOR_MUTED,
    COLOR_OK,
    COLOR_TEXT,
    Card,
    font,
    open_path,
)
from .base import BasePage

_ENGINES = ("tts-piper",)

_NEGATIVES = {
    "validation": ("validation_set_features.npy", VALIDATION_SIZE),
    "acav": ("openwakeword_features_ACAV100M_2000_hrs_16bit.npy", ACAV_SIZE),
}


class ComponentsPage(BasePage):
    def build(self) -> None:
        self._busy = False
        self._progress = 0.0
        self._progress_text = ""
        self._tick_id: str | None = None
        self._engines: dict[str, dict[str, object]] = {}
        self._negatives: dict[str, dict[str, object]] = {}

        self.columnconfigure(0, weight=1)
        row = 0

        # --------------------------------------------------------- progress
        self.progress_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.progress_frame.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        self.progress_frame.grid_columnconfigure(0, weight=1)
        self.progress_bar = ctk.CTkProgressBar(self.progress_frame, height=10)
        self.progress_bar.grid(row=0, column=0, sticky="ew")
        self.progress_bar.set(0.0)
        self.progress_label = ctk.CTkLabel(
            self.progress_frame, text="", font=font(11), text_color=COLOR_MUTED, anchor="w"
        )
        self.progress_label.grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.progress_frame.grid_remove()
        row += 1

        # ---------------------------------------------------------- engines
        self.engines_card = Card(
            self, title=self.tr("components_engines_title"), subtitle=self.tr("components_engines_hint")
        )
        self.engines_card.grid(row=row, column=0, sticky="ew", pady=(0, 12))
        row += 1
        body = self.engines_card.body
        body.grid_columnconfigure(0, weight=1)

        for index, component_id in enumerate(_ENGINES):
            self._build_engine_row(body, index, component_id)

        self.coqui_note = ctk.CTkLabel(
            body,
            text=self.tr("components_coqui_note"),
            font=font(11),
            text_color=COLOR_MUTED,
            anchor="w",
            justify="left",
            wraplength=760,
        )
        self.coqui_note.grid(row=len(_ENGINES), column=0, sticky="w", pady=(14, 0))

        # -------------------------------------------------------- negatives
        self.negatives_card = Card(
            self, title=self.tr("components_negatives_title"), subtitle=self.tr("components_negatives_hint")
        )
        self.negatives_card.grid(row=row, column=0, sticky="ew", pady=(0, 12))
        row += 1
        body = self.negatives_card.body
        body.grid_columnconfigure(0, weight=1)

        for index, key in enumerate(_NEGATIVES):
            label_key = "components_neg_validation" if key == "validation" else "components_neg_acav"
            self._negatives[key] = self._build_data_row(
                body,
                index,
                self.tr(label_key),
                command=lambda k=key: self.download_negatives(k),
            )

        # ----------------------------------------------------------- models
        self.models_card = Card(self, title=self.tr("components_models_title"))
        self.models_card.grid(row=row, column=0, sticky="ew", pady=(0, 12))
        row += 1
        self.models_label = ctk.CTkLabel(
            self.models_card.body,
            text=self.tr("components_models_ok"),
            font=font(12),
            text_color=COLOR_OK,
            anchor="w",
            justify="left",
            wraplength=760,
        )
        self.models_label.grid(row=0, column=0, sticky="w")

        # ------------------------------------------------------------ paths
        self.paths_card = Card(
            self, title=self.tr("components_paths_title"), subtitle=self.tr("components_paths_hint")
        )
        self.paths_card.grid(row=row, column=0, sticky="ew")
        body = self.paths_card.body
        body.grid_columnconfigure(0, weight=1)

        self.cache_path_label, self.cache_path_value = self._build_path_row(
            body, 0, self.tr("components_cache_dir"), cache_dir()
        )
        self.config_path_label, self.config_path_value = self._build_path_row(
            body, 1, self.tr("components_config_dir"), config_dir()
        )
        self.plugins_button = ctk.CTkButton(
            body,
            text=self.tr("tts_install_plugins"),
            font=font(12),
            height=30,
            width=190,
            fg_color="#333a46",
            hover_color="#3d4552",
            command=lambda: open_path(tts_plugins_dir()),
        )
        self.plugins_button.grid(row=2, column=0, sticky="w", pady=(12, 0))

    # ------------------------------------------------------------------ ui
    def _size(self, size: int | float) -> str:
        """Format a byte count with localized units."""
        size = float(size)
        gb, mb, kb = ("ГБ", "МБ", "КБ") if self.tr.language == "ru" else ("GB", "MB", "KB")
        if size >= 1e9:
            return f"{size / 1e9:.1f} {gb}"
        if size >= 1e6:
            return f"{size / 1e6:.0f} {mb}"
        return f"{size / 1e3:.0f} {kb}"

    def _build_engine_row(self, body, index: int, component_id: str) -> None:
        frame = ctk.CTkFrame(body, fg_color="transparent")
        frame.grid(row=index, column=0, sticky="ew", pady=(0 if index == 0 else 12, 0))
        frame.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(
            frame,
            text=self.tr(f"components_{components.COMPONENTS[component_id].engine}_title"),
            font=font(13, "bold"),
            text_color=COLOR_TEXT,
            anchor="w",
        )
        title.grid(row=0, column=0, sticky="w")
        description = ctk.CTkLabel(
            frame,
            text=self.tr(f"components_{components.COMPONENTS[component_id].engine}_desc"),
            font=font(11),
            text_color=COLOR_MUTED,
            anchor="w",
            justify="left",
            wraplength=560,
        )
        description.grid(row=1, column=0, sticky="w")
        status = ctk.CTkLabel(frame, text="", font=font(11), text_color=COLOR_MUTED, anchor="e")
        status.grid(row=0, column=1, rowspan=2, sticky="e", padx=(12, 12))
        button = ctk.CTkButton(frame, text="", font=font(12), height=32, width=120)
        button.grid(row=0, column=2, rowspan=2, sticky="e")

        self._engines[component_id] = {
            "title": title,
            "description": description,
            "status": status,
            "button": button,
        }

    def _build_data_row(self, body, index: int, text: str, *, command) -> dict[str, object]:
        frame = ctk.CTkFrame(body, fg_color="transparent")
        frame.grid(row=index, column=0, sticky="ew", pady=(0 if index == 0 else 10, 0))
        frame.grid_columnconfigure(0, weight=1)

        label = ctk.CTkLabel(frame, text=text, font=font(12, "bold"), text_color=COLOR_TEXT, anchor="w")
        label.grid(row=0, column=0, sticky="w")
        status = ctk.CTkLabel(frame, text="", font=font(11), text_color=COLOR_MUTED, anchor="e")
        status.grid(row=0, column=1, sticky="e", padx=(12, 12))
        button = ctk.CTkButton(frame, text=self.tr("components_download"), font=font(12), height=30, width=120)
        button.configure(command=command)
        button.grid(row=0, column=2, sticky="e")
        return {"label": label, "status": status, "button": button}

    def _build_path_row(self, body, index: int, text: str, path: Path):
        frame = ctk.CTkFrame(body, fg_color="transparent")
        frame.grid(row=index, column=0, sticky="ew", pady=(0 if index == 0 else 8, 0))
        frame.grid_columnconfigure(1, weight=1)

        label = ctk.CTkLabel(frame, text=text, font=font(12, "bold"), text_color=COLOR_TEXT, anchor="w", width=170)
        label.grid(row=0, column=0, sticky="w")
        value = ctk.CTkLabel(
            frame, text=str(path), font=font(11), text_color=COLOR_MUTED, anchor="w", justify="left", wraplength=560
        )
        value.grid(row=0, column=1, sticky="w", padx=(0, 12))
        button = ctk.CTkButton(
            frame,
            text=self.tr("components_open"),
            font=font(12),
            height=28,
            width=90,
            fg_color="#333a46",
            hover_color="#3d4552",
            command=lambda: open_path(path),
        )
        button.grid(row=0, column=2, sticky="e")
        return label, value

    # -------------------------------------------------------------- state
    def _refresh_engines(self) -> None:
        infos = registry.list_backends()
        for component_id, widgets in self._engines.items():
            engine = components.COMPONENTS[component_id].engine
            state = components.component_state(component_id)
            available = bool(infos.get(engine, {}).get("available"))
            button = widgets["button"]
            status = widgets["status"]

            if state.installed:
                status.configure(text=f"{self.tr('components_installed')} · {self._size(state.size_bytes)}")
                status.configure(text_color=COLOR_OK)
                button.configure(
                    text=self.tr("components_remove"),
                    command=lambda key=component_id: self.remove_engine(key),
                    state="normal" if state.removable else "disabled",
                )
                button.grid()
            elif available:
                status.configure(text=self.tr("components_installed"), text_color=COLOR_OK)
                button.grid_remove()
            else:
                size = components.COMPONENTS[component_id].size_mb
                status.configure(
                    text=f"{self.tr('components_not_installed')} · ~{self._size(size * 1024 * 1024)}",
                    text_color=COLOR_MUTED,
                )
                button.configure(
                    text=self.tr("components_install"),
                    command=lambda key=component_id: self.install_engine(key),
                    state="normal",
                )
                button.grid()

    def _refresh_negatives(self) -> None:
        for key, widgets in self._negatives.items():
            filename, expected = _NEGATIVES[key]
            path = downloads_dir() / filename
            present = path.is_file() and path.stat().st_size == expected
            if present:
                widgets["status"].configure(text=self.tr("components_present"), text_color=COLOR_OK)
                widgets["button"].grid_remove()
            else:
                widgets["status"].configure(
                    text=f"{self.tr('components_missing')} · ~{self._size(expected)}", text_color=COLOR_MUTED
                )
                widgets["button"].configure(state="normal")
                widgets["button"].grid()

    def _refresh_models(self) -> None:
        try:
            bundled_model_paths()
        except Exception:  # noqa: BLE001 - only the message matters
            self.models_label.configure(text=self.tr("error"), text_color=COLOR_ERROR)
        else:
            self.models_label.configure(text=self.tr("components_models_ok"), text_color=COLOR_OK)

    def on_show(self) -> None:
        self._refresh_engines()
        self._refresh_negatives()
        self._refresh_models()

    # ---------------------------------------------------------- operations
    def install_engine(self, component_id: str) -> None:
        if self._busy:
            self.app.show_info(self.tr("components_engines_title"), self.tr("components_busy"))
            return
        self._busy = True
        self._begin_progress(self.tr("components_downloading"))

        def work():
            return components.download_component(component_id, progress=self._progress_callback)

        def done(_path) -> None:
            self._busy = False
            self._finish_progress()
            registry.refresh_plugins()
            self._refresh_engines()
            self.app.set_status(self.tr("components_engine_ready"))

        def failed(exc) -> None:
            self._busy = False
            self._finish_progress()
            self._refresh_engines()
            self.app.show_error(
                self.tr("components_engines_title"),
                self.tr("components_engine_failed", error=str(exc)),
            )

        self.app.run_task(work, done, failed)

    def remove_engine(self, component_id: str) -> None:
        if self._busy:
            self.app.show_info(self.tr("components_engines_title"), self.tr("components_busy"))
            return
        try:
            components.remove_component(component_id)
        except Exception as exc:  # noqa: BLE001 - shown in the dialog
            self.app.show_error(self.tr("error"), str(exc))
            return
        registry.refresh_plugins()
        self._refresh_engines()
        self.app.set_status(self.tr("components_removed"))

    def download_negatives(self, key: str) -> None:
        if self._busy:
            self.app.show_info(self.tr("components_negatives_title"), self.tr("components_busy"))
            return
        if key == "acav" and not self._confirm_acav():
            return

        self._busy = True
        self._begin_progress(self.tr("components_downloading"))
        worker = ensure_acav if key == "acav" else ensure_validation_features

        def work():
            return worker(progress=self._progress_callback)

        def done(path) -> None:
            self._busy = False
            self._finish_progress()
            self._refresh_negatives()
            self.app.set_status(self.tr("components_acav_done", path=str(path)))

        def failed(exc) -> None:
            self._busy = False
            self._finish_progress()
            self.app.show_error(self.tr("components_negatives_title"), str(exc))

        self.app.run_task(work, done, failed)

    def _confirm_acav(self) -> bool:
        from tkinter import messagebox

        return bool(
            messagebox.askyesno(
                self.tr("components_acav_confirm_title"),
                self.tr("components_acav_confirm"),
                parent=self,
            )
        )

    # ------------------------------------------------------------ progress
    def _begin_progress(self, text: str) -> None:
        self._progress = 0.0
        self._progress_text = text
        self.progress_frame.grid()
        self.progress_label.configure(text=text, text_color=COLOR_MUTED)
        self._stop_ticker()
        self._tick()

    def _progress_callback(self, done: int, total: int, name: str) -> None:
        if total:
            self._progress = min(done / total, 1.0)
            self._progress_text = f"{name} — {self._size(done)} / {self._size(total)}"
        else:  # pragma: no cover - servers without Content-Length
            self._progress = 0.0
            self._progress_text = f"{name} — {self._size(done)}"

    def _tick(self) -> None:
        self.progress_bar.set(max(self._progress, 0.01))
        self.progress_label.configure(text=self._progress_text)
        self._tick_id = self.after(150, self._tick)

    def _stop_ticker(self) -> None:
        if self._tick_id is not None:
            with contextlib.suppress(Exception):
                self.after_cancel(self._tick_id)
            self._tick_id = None

    def _finish_progress(self) -> None:
        self._stop_ticker()
        self.progress_bar.set(1.0)
        self.progress_frame.grid_remove()

    # -------------------------------------------------------- retranslate
    def retranslate(self) -> None:
        self.engines_card.set_title(self.tr("components_engines_title"), self.tr("components_engines_hint"))
        self.negatives_card.set_title(
            self.tr("components_negatives_title"), self.tr("components_negatives_hint")
        )
        self.models_card.set_title(self.tr("components_models_title"))
        self.paths_card.set_title(self.tr("components_paths_title"), self.tr("components_paths_hint"))
        self.models_label.configure(text=self.tr("components_models_ok"))
        self.coqui_note.configure(text=self.tr("components_coqui_note"))
        self.cache_path_label.configure(text=self.tr("components_cache_dir"))
        self.config_path_label.configure(text=self.tr("components_config_dir"))
        self.plugins_button.configure(text=self.tr("tts_install_plugins"))
        self._refresh_engines()
        self._refresh_negatives()
