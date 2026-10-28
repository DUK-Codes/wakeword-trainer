"""Shared page base class."""

from __future__ import annotations

from typing import TYPE_CHECKING

import customtkinter as ctk

if TYPE_CHECKING:  # pragma: no cover
    from .app import WakeWordTrainerApp


class BasePage(ctk.CTkScrollableFrame):
    """Scrollable page with a build/retranslate lifecycle."""

    def __init__(self, master, app: WakeWordTrainerApp) -> None:
        super().__init__(master, fg_color="transparent", corner_radius=0)
        self.app = app
        self.tr = app.tr
        self.grid_columnconfigure(0, weight=1)
        self.build()

    # ------------------------------------------------------------------
    def build(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def retranslate(self) -> None:  # pragma: no cover - overridden
        return None

    def on_show(self) -> None:
        return None

    def on_hide(self) -> None:
        return None
