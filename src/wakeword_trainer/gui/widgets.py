"""Reusable CustomTkinter widgets and small helpers."""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import customtkinter as ctk

from .i18n import Translator

FONT_FAMILY = "Segoe UI"
COLOR_BG = "#16181d"
COLOR_SIDEBAR = "#1d2026"
COLOR_CARD = "#22262e"
COLOR_CARD_ALT = "#282d36"
COLOR_ACCENT = "#4f8cff"
COLOR_ACCENT_HOVER = "#3d76e0"
COLOR_OK = "#3fbf7f"
COLOR_WARN = "#e0a83f"
COLOR_ERROR = "#e05f5f"
COLOR_TEXT = "#e8ecf3"
COLOR_MUTED = "#98a2b3"


def font(size: int = 13, weight: str = "normal") -> ctk.CTkFont:
    return ctk.CTkFont(family=FONT_FAMILY, size=size, weight=weight)


def open_path(path: str | Path) -> None:
    """Open a file or folder with the system default application."""
    target = str(Path(path).expanduser())
    if not Path(target).exists():
        return
    try:
        if sys.platform.startswith("win"):
            os.startfile(target)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", target])
        else:
            subprocess.Popen(["xdg-open", target])
    except Exception:  # noqa: BLE001 - opening a folder must never crash the app
        pass


class Card(ctk.CTkFrame):
    """Rounded panel with an optional header row."""

    def __init__(self, master, title: str = "", subtitle: str = "", **kwargs) -> None:
        kwargs.setdefault("fg_color", COLOR_CARD)
        kwargs.setdefault("corner_radius", 12)
        super().__init__(master, **kwargs)
        self.grid_columnconfigure(0, weight=1)
        self._row = 0

        self.header = ctk.CTkFrame(self, fg_color="transparent")
        self.header.grid_columnconfigure(0, weight=1)

        self.title_label = ctk.CTkLabel(
            self.header,
            text="",
            font=font(15, "bold"),
            text_color=COLOR_TEXT,
            anchor="w",
        )
        self.subtitle_label = ctk.CTkLabel(
            self.header,
            text="",
            font=font(12),
            text_color=COLOR_MUTED,
            anchor="w",
            justify="left",
            wraplength=760,
        )

        self.body = ctk.CTkFrame(self, fg_color="transparent")
        self.body.grid(row=1, column=0, sticky="nsew", padx=16, pady=(10, 14))
        self.body.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self.set_title(title, subtitle)

    def set_title(self, title: str, subtitle: str | None = None) -> None:
        """Set the header; the header row disappears when both parts are empty."""
        subtitle = self.subtitle_label.cget("text") if subtitle is None else subtitle
        self.title_label.configure(text=title)
        self.subtitle_label.configure(text=subtitle)

        if title:
            self.title_label.grid(row=0, column=0, sticky="w")
        else:
            self.title_label.grid_remove()
        if subtitle:
            self.subtitle_label.grid(row=1 if title else 0, column=0, sticky="w", pady=(2, 0) if title else 0)
        else:
            self.subtitle_label.grid_remove()

        if title or subtitle:
            self.header.grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 0))
            self.body.grid_configure(pady=(10, 14))
        else:
            self.header.grid_remove()
            self.body.grid_configure(pady=(14, 14))

    def add(self, widget, **grid_kwargs) -> None:
        self._row += 1
        widget.grid(row=self._row, column=0, sticky=grid_kwargs.pop("sticky", "ew"), **grid_kwargs)


class PathRow(ctk.CTkFrame):
    """Entry + browse button for a folder or file."""

    def __init__(
        self,
        master,
        *,
        placeholder: str = "",
        mode: str = "directory",
        filetypes: list[tuple[str, str]] | None = None,
        translator: Translator | None = None,
        on_change: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(master, fg_color="transparent")
        self.grid_columnconfigure(0, weight=1)
        self.mode = mode
        self.filetypes = filetypes or [("All files", "*.*")]
        self.on_change = on_change
        self._tr = translator

        self.var = ctk.StringVar(value="")
        self.entry = ctk.CTkEntry(
            self,
            textvariable=self.var,
            placeholder_text=placeholder,
            font=font(13),
            height=34,
        )
        self.entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.button = ctk.CTkButton(
            self,
            text=(translator("browse") if translator else "Browse…"),
            width=96,
            height=34,
            font=font(13),
            command=self.browse,
        )
        self.button.grid(row=0, column=1)
        self.var.trace_add("write", self._changed)

    def _changed(self, *_args) -> None:
        if self.on_change is not None:
            self.on_change(self.var.get())

    def browse(self) -> None:
        from tkinter import filedialog

        if self.mode == "directory":
            path = filedialog.askdirectory(initialdir=self.var.get() or None)
        else:
            path = filedialog.askopenfilename(
                initialdir=str(Path(self.var.get()).parent) if self.var.get() else None,
                filetypes=self.filetypes,
            )
        if path:
            self.var.set(path)

    def get(self) -> str:
        return self.var.get().strip()

    def set(self, value: str | Path | None) -> None:
        self.var.set(str(value) if value else "")


class Metric(ctk.CTkFrame):
    """Large number with a caption."""

    def __init__(self, master, caption: str, value: str = "—", *, color: str = COLOR_TEXT) -> None:
        super().__init__(master, fg_color=COLOR_CARD_ALT, corner_radius=10)
        self.grid_columnconfigure(0, weight=1)
        self.caption_label = ctk.CTkLabel(self, text=caption, font=font(11), text_color=COLOR_MUTED, anchor="w")
        self.caption_label.grid(row=0, column=0, sticky="w", padx=12, pady=(10, 0))
        self.value_label = ctk.CTkLabel(self, text=value, font=font(20, "bold"), text_color=color, anchor="w")
        self.value_label.grid(row=1, column=0, sticky="w", padx=12, pady=(0, 10))

    def set_caption(self, caption: str) -> None:
        self.caption_label.configure(text=caption)

    def set_value(self, value: str, color: str | None = None) -> None:
        self.value_label.configure(text=value)
        if color:
            self.value_label.configure(text_color=color)


class LogBox(ctk.CTkFrame):
    """Read-only scrolling log."""

    def __init__(self, master, *, height: int = 180) -> None:
        super().__init__(master, fg_color="transparent")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.text = ctk.CTkTextbox(
            self,
            height=height,
            font=ctk.CTkFont(family="Consolas", size=12),
            fg_color="#12141a",
            text_color=COLOR_TEXT,
            wrap="word",
        )
        self.text.grid(row=0, column=0, sticky="nsew")
        self.text.configure(state="disabled")
        self._max_lines = 2000
        self._lines = 0

    def append(self, message: str) -> None:
        self.text.configure(state="normal")
        self.text.insert("end", message.rstrip() + "\n")
        self._lines += 1
        if self._lines > self._max_lines:
            self.text.delete("1.0", f"{self._lines - self._max_lines}.0")
            self._lines = self._max_lines
        self.text.see("end")
        self.text.configure(state="disabled")

    def clear(self) -> None:
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.configure(state="disabled")
        self._lines = 0


class Subtitle(ctk.CTkLabel):
    def __init__(self, master, text: str = "") -> None:
        super().__init__(
            master,
            text=text,
            font=font(12),
            text_color=COLOR_MUTED,
            anchor="w",
            justify="left",
            wraplength=820,
        )


def labeled(parent, text: str, row: int, *, column: int = 0, columnspan: int = 1) -> ctk.CTkLabel:
    label = ctk.CTkLabel(parent, text=text, font=font(12, "bold"), text_color=COLOR_MUTED, anchor="w")
    label.grid(row=row, column=column, columnspan=columnspan, sticky="w", pady=(8, 2))
    return label


class Tooltip:
    """Very small tooltip helper (CustomTkinter has none built in)."""

    def __init__(self, widget, text: str, delay: int = 450) -> None:
        self.widget = widget
        self.text = text
        self.delay = delay
        self._after_id: str | None = None
        self._window: ctk.CTkToplevel | None = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _event=None) -> None:
        self._cancel()
        self._after_id = self.widget.after(self.delay, self._show)

    def _cancel(self) -> None:
        if self._after_id is not None:
            with contextlib.suppress(Exception):
                self.widget.after_cancel(self._after_id)
            self._after_id = None

    def _show(self) -> None:
        if self._window is not None or not self.text:
            return
        x = self.widget.winfo_rootx() + 12
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        window = ctk.CTkToplevel(self.widget)
        window.wm_overrideredirect(True)
        window.wm_geometry(f"+{x}+{y}")
        window.attributes("-topmost", True)
        label = ctk.CTkLabel(
            window,
            text=self.text,
            font=font(11),
            fg_color=COLOR_CARD_ALT,
            corner_radius=6,
            padx=8,
            pady=4,
            justify="left",
            wraplength=320,
        )
        label.pack()
        self._window = window

    def _hide(self, _event=None) -> None:
        self._cancel()
        if self._window is not None:
            try:
                self._window.destroy()
            finally:
                self._window = None
