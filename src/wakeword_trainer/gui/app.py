"""CustomTkinter desktop application for the full training cycle."""

from __future__ import annotations

import contextlib
import queue
import threading
from collections.abc import Callable
from typing import Any

import customtkinter as ctk

from ..version import __version__
from .i18n import LANGUAGE_NAMES, Translator
from .pages.dataset_page import DatasetPage
from .pages.results_page import ResultsPage
from .pages.settings_page import SettingsPage
from .pages.test_page import TestPage
from .pages.train_page import TrainPage
from .pages.tts_page import TTSPage
from .state import GuiState
from .widgets import (
    COLOR_BG,
    COLOR_CARD,
    COLOR_ERROR,
    COLOR_MUTED,
    COLOR_SIDEBAR,
    COLOR_TEXT,
    font,
    open_path,
)


class WakeWordTrainerApp(ctk.CTk):
    """Main window; owns shared state, navigation and background tasks."""

    def __init__(self, state: GuiState | None = None) -> None:
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        super().__init__(fg_color=COLOR_BG)

        self.gui_state = state or GuiState.load()
        self.tr = Translator(self.gui_state.language)
        self.dataset_report = None
        self._nav_blocked = False
        self._task_queue: queue.Queue = queue.Queue()
        self._closing = False

        self.title(self.tr("app_title"))
        self.geometry("1180x820")
        self.minsize(1020, 700)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_header()
        self._build_pages()
        self._build_statusbar()

        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self._show_page("dataset")
        self._poll_tasks()

    # ------------------------------------------------------------------
    def _build_sidebar(self) -> None:
        self.sidebar = ctk.CTkFrame(self, fg_color=COLOR_SIDEBAR, corner_radius=0, width=232)
        self.sidebar.grid(row=0, column=0, rowspan=3, sticky="nsw")
        self.sidebar.grid_propagate(False)
        self.sidebar.grid_rowconfigure(8, weight=1)

        self.brand = ctk.CTkLabel(
            self.sidebar,
            text="Wake Word\nTrainer",
            font=font(22, "bold"),
            text_color=COLOR_TEXT,
            anchor="w",
            justify="left",
        )
        self.brand.grid(row=0, column=0, sticky="w", padx=20, pady=(24, 2))
        self.brand_sub = ctk.CTkLabel(
            self.sidebar,
            text=self.tr("app_subtitle"),
            font=font(11),
            text_color=COLOR_MUTED,
            anchor="w",
            justify="left",
            wraplength=190,
        )
        self.brand_sub.grid(row=1, column=0, sticky="w", padx=20, pady=(0, 22))

        self.nav_buttons: dict[str, ctk.CTkButton] = {}
        items = [
            ("dataset", "nav_dataset"),
            ("settings", "nav_settings"),
            ("train", "nav_train"),
            ("results", "nav_results"),
            ("test", "nav_test"),
            ("tts", "nav_tts"),
        ]
        for offset, (key, label_key) in enumerate(items):
            button = ctk.CTkButton(
                self.sidebar,
                text=self.tr(label_key),
                font=font(14),
                anchor="w",
                height=40,
                corner_radius=8,
                fg_color="transparent",
                hover_color="#2b303a",
                text_color=COLOR_TEXT,
                command=lambda k=key: self._show_page(k),
            )
            button.grid(row=2 + offset, column=0, sticky="ew", padx=14, pady=2)
            self.nav_buttons[key] = button

        self.sidebar_spacer = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.sidebar_spacer.grid(row=8, column=0, sticky="nsew")

        self.language_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.language_frame.grid(row=9, column=0, sticky="ew", padx=14, pady=(0, 8))
        self.language_label = ctk.CTkLabel(
            self.language_frame,
            text=self.tr("language"),
            font=font(11),
            text_color=COLOR_MUTED,
            anchor="w",
        )
        self.language_label.pack(anchor="w")
        switch = ctk.CTkSegmentedButton(
            self.language_frame,
            values=[LANGUAGE_NAMES["ru"], LANGUAGE_NAMES["en"]],
            font=font(12),
            command=self._on_language,
        )
        switch.set(LANGUAGE_NAMES[self.gui_state.language])
        switch.pack(fill="x", pady=(4, 0))
        self.language_switch = switch

        self.version_label = ctk.CTkLabel(
            self.sidebar,
            text=f"v{__version__}",
            font=font(10),
            text_color="#5f6775",
            anchor="w",
        )
        self.version_label.grid(row=10, column=0, sticky="w", padx=20, pady=(0, 14))

    def _build_header(self) -> None:
        self.header = ctk.CTkFrame(self, fg_color="transparent", height=54)
        self.header.grid(row=0, column=1, sticky="ew", padx=24, pady=(18, 4))
        self.header.grid_columnconfigure(0, weight=1)

        self.page_title = ctk.CTkLabel(self.header, text="", font=font(24, "bold"), text_color=COLOR_TEXT, anchor="w")
        self.page_title.grid(row=0, column=0, sticky="w")

        self.output_button = ctk.CTkButton(
            self.header,
            text=self.tr("open_output"),
            font=font(12),
            height=32,
            width=140,
            fg_color=COLOR_CARD,
            hover_color="#2d333d",
            command=lambda: open_path(self.gui_state.output) if self.gui_state.output else None,
        )
        self.output_button.grid(row=0, column=1, sticky="e")

    def _build_pages(self) -> None:
        self.container = ctk.CTkFrame(self, fg_color="transparent")
        self.container.grid(row=1, column=1, sticky="nsew", padx=24, pady=(0, 8))
        self.container.grid_columnconfigure(0, weight=1)
        self.container.grid_rowconfigure(0, weight=1)

        self.pages: dict[str, Any] = {
            "dataset": DatasetPage(self.container, self),
            "settings": SettingsPage(self.container, self),
            "train": TrainPage(self.container, self),
            "results": ResultsPage(self.container, self),
            "test": TestPage(self.container, self),
            "tts": TTSPage(self.container, self),
        }
        self.dataset_page = self.pages["dataset"]
        self.settings_page = self.pages["settings"]
        self.train_page = self.pages["train"]
        self.results_page = self.pages["results"]
        self.test_page = self.pages["test"]
        self.tts_page = self.pages["tts"]
        for page in self.pages.values():
            page.grid(row=0, column=0, sticky="nsew")
            page.grid_remove()

        self.current_page = ""

    def _build_statusbar(self) -> None:
        self.statusbar = ctk.CTkFrame(self, fg_color="transparent", height=26)
        self.statusbar.grid(row=2, column=1, sticky="ew", padx=24, pady=(0, 10))
        self.statusbar.grid_columnconfigure(0, weight=1)
        self.status_label = ctk.CTkLabel(
            self.statusbar, text=self.tr("ready"), font=font(11), text_color=COLOR_MUTED, anchor="w"
        )
        self.status_label.grid(row=0, column=0, sticky="w")

    # ------------------------------------------------------------------
    def _show_page(self, key: str) -> None:
        if key not in self.pages or key == self.current_page:
            return
        if self._nav_blocked and key != "train":
            return
        if self.current_page:
            # Page cleanup must never break navigation.
            with contextlib.suppress(Exception):
                self.pages[self.current_page].on_hide()
        for name, page in self.pages.items():
            page.grid_remove()
            self.nav_buttons[name].configure(fg_color="transparent")
        page = self.pages[key]
        page.grid()
        self.nav_buttons[key].configure(fg_color="#2b303a")
        self.current_page = key
        self.page_title.configure(text=self.tr(f"nav_{key}"))
        try:
            page.on_show()
        except Exception as exc:  # noqa: BLE001
            self.show_error(self.tr("error"), str(exc))

    def _on_language(self, label: str) -> None:
        language = "en" if label == LANGUAGE_NAMES["en"] else "ru"
        if language == self.tr.language:
            return
        self.tr.set_language(language)
        self.gui_state.language = language
        self.gui_state.save()
        self._retranslate()

    def _retranslate(self) -> None:
        self.title(self.tr("app_title"))
        self.brand_sub.configure(text=self.tr("app_subtitle"))
        self.language_label.configure(text=self.tr("language"))
        self.output_button.configure(text=self.tr("open_output"))
        for key, button in self.nav_buttons.items():
            button.configure(text=self.tr(f"nav_{key}"))
        if self.current_page:
            self.page_title.configure(text=self.tr(f"nav_{self.current_page}"))
        self.status_label.configure(text=self.tr("ready"))
        for page in self.pages.values():
            # One broken page must not break the language switch.
            with contextlib.suppress(Exception):
                page.retranslate()

    # ------------------------------------------------------------------
    def set_status(self, message: str, *, error: bool = False) -> None:
        self.status_label.configure(text=message, text_color=COLOR_ERROR if error else COLOR_MUTED)
        self.output_button.configure(
            command=lambda: open_path(self.gui_state.output) if self.gui_state.output else None
        )

    def block_navigation(self, blocked: bool) -> None:
        self._nav_blocked = bool(blocked)
        for key, button in self.nav_buttons.items():
            if key == "train":
                continue
            button.configure(state="disabled" if blocked else "normal")

    # ------------------------------------------------------------------
    def show_info(self, title: str, message: str) -> None:
        from tkinter import messagebox

        messagebox.showinfo(title, message, parent=self)

    def show_error(self, title: str, message: str) -> None:
        from tkinter import messagebox

        self.set_status(str(message).splitlines()[0][:180], error=True)
        messagebox.showerror(title, message, parent=self)

    # ------------------------------------------------------------------
    def run_task(
        self,
        func: Callable[[], Any],
        on_success: Callable[[Any], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
    ) -> None:
        """Run ``func`` in a thread and deliver the result on the Tk thread."""

        def worker() -> None:
            try:
                result = func()
            except Exception as exc:  # noqa: BLE001 - delivered to the UI
                self._task_queue.put((on_error, None, exc))
            else:
                self._task_queue.put((on_success, result, None))

        threading.Thread(target=worker, daemon=True).start()

    def _poll_tasks(self) -> None:
        while True:
            try:
                callback, result, error = self._task_queue.get_nowait()
            except queue.Empty:
                break
            if error is not None:
                if callback is not None:
                    with contextlib.suppress(Exception):
                        callback(error)
            elif callback is not None:
                try:
                    callback(result)
                except Exception as exc:  # noqa: BLE001
                    self.show_error(self.tr("error"), str(exc))
        if not self._closing:
            self.after(80, self._poll_tasks)

    # ------------------------------------------------------------------
    def state_config_clip_seconds(self) -> float:
        try:
            return float(self.gui_state.advanced.get("clip_seconds", 2.0))
        except (TypeError, ValueError):
            return 2.0

    def on_training_done(self, event, *, cancelled: bool = False) -> None:
        payload = event.payload or {}
        output_dir = payload.get("output_dir") or self.gui_state.output
        onnx_path = payload.get("onnx_path")
        if onnx_path:
            self.gui_state.last_model = str(onnx_path)
        if output_dir:
            self.gui_state.output = str(output_dir)
        self.gui_state.save()
        if not cancelled and output_dir:
            loaded = self.results_page.load(output_dir)
            if loaded:
                self._show_page("results")
                self.set_status(self.tr("train_done"))
        if not cancelled:
            self.train_page.log.append(self.tr("train_done"))

    # ------------------------------------------------------------------
    def on_close(self) -> None:
        if self.train_page.run is not None and self.train_page.run.running:
            from tkinter import messagebox

            if not messagebox.askyesno(
                self.tr("train_title"),
                self.tr("train_stopping") + "\n\n" + self.tr("close") + "?",
                parent=self,
            ):
                return
        self._closing = True
        for page in self.pages.values():
            with contextlib.suppress(Exception):
                page.on_hide()
        with contextlib.suppress(Exception):
            self.train_page.shutdown()
        with contextlib.suppress(Exception):
            self.test_page.shutdown()
        self.gui_state.save()
        self.destroy()


def main() -> None:
    import sys

    state = GuiState.load()
    # ``--lang`` is a convenience for shortcuts, not a scripting interface.
    for index, argument in enumerate(sys.argv):
        if argument == "--lang" and index + 1 < len(sys.argv):
            language = sys.argv[index + 1].lower()
            if language in {"ru", "en"}:
                state.language = language
    app = WakeWordTrainerApp(state)
    app.mainloop()


if __name__ == "__main__":
    main()
