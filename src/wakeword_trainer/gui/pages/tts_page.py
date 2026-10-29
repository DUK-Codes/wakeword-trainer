"""TTS page: optional speech synthesis to extend the positive dataset."""

from __future__ import annotations

import threading
import traceback
from pathlib import Path

import customtkinter as ctk

from ...tts import registry
from ..widgets import (
    COLOR_ERROR,
    COLOR_MUTED,
    COLOR_OK,
    COLOR_WARN,
    Card,
    LogBox,
    font,
    open_path,
)
from .base import BasePage

_LANGUAGE_FILTERS = {"all": "Все / All", "ru": "Русский", "en": "English"}


class TTSPage(BasePage):
    def build(self) -> None:
        self._backend_names: list[str] = []
        self._voices: list = []
        self._generating = False
        self._download_busy = False

        self.columnconfigure(0, weight=1)
        row = 0

        # --------------------------------------------------------- settings
        self.setup_card = Card(self, title=self.tr("tts_title"), subtitle=self.tr("tts_hint"))
        self.setup_card.grid(row=row, column=0, sticky="ew", pady=(0, 12))
        row += 1
        body = self.setup_card.body
        body.grid_columnconfigure(1, weight=1)
        body.grid_columnconfigure(3, weight=1)

        self.backend_label = ctk.CTkLabel(
            body, text=self.tr("tts_backend"), font=font(12, "bold"), text_color=COLOR_MUTED, anchor="w"
        )
        self.backend_label.grid(row=0, column=0, sticky="w", padx=(0, 10))
        self.backend_menu = ctk.CTkOptionMenu(body, values=["—"], font=font(12), command=self._on_backend)
        self.backend_menu.grid(row=0, column=1, sticky="ew", padx=(0, 18))

        self.language_label = ctk.CTkLabel(
            body, text=self.tr("tts_language"), font=font(12, "bold"), text_color=COLOR_MUTED, anchor="w"
        )
        self.language_label.grid(row=0, column=2, sticky="w", padx=(0, 10))
        self.language_menu = ctk.CTkOptionMenu(
            body,
            values=list(_LANGUAGE_FILTERS.values()),
            font=font(12),
            command=self._on_language,
        )
        self.language_menu.grid(row=0, column=3, sticky="ew")
        self.language_menu.set(_LANGUAGE_FILTERS["all"])

        self.voice_label = ctk.CTkLabel(
            body, text=self.tr("tts_voice"), font=font(12, "bold"), text_color=COLOR_MUTED, anchor="w"
        )
        self.voice_label.grid(row=1, column=0, sticky="w", padx=(0, 10), pady=(10, 0))
        self.voice_menu = ctk.CTkOptionMenu(body, values=["—"], font=font(12), command=self._on_voice)
        self.voice_menu.grid(row=1, column=1, columnspan=3, sticky="ew", pady=(10, 0))

        self.backend_info = ctk.CTkLabel(
            body, text="", font=font(11), text_color=COLOR_MUTED, anchor="w", justify="left", wraplength=600
        )
        self.backend_info.grid(row=2, column=0, columnspan=3, sticky="w", pady=(8, 0))
        self.install_engine_button = ctk.CTkButton(
            body,
            text=self.tr("tts_install_engine"),
            font=font(12),
            height=30,
            width=160,
            fg_color="#333a46",
            hover_color="#3d4552",
            command=lambda: self.app.open_page("components"),
        )
        self.install_engine_button.grid(row=2, column=3, sticky="e", pady=(8, 0))
        self.install_engine_button.grid_remove()

        self.license_label = ctk.CTkLabel(
            body, text=self.tr("tts_license_note"), font=font(11), text_color=COLOR_WARN, anchor="w"
        )
        self.license_label.grid(row=3, column=0, columnspan=4, sticky="w", pady=(4, 0))

        # ------------------------------------------------------------- text
        self.text_card = Card(self, title=self.tr("tts_text"))
        self.text_card.grid(row=row, column=0, sticky="ew", pady=(0, 12))
        row += 1
        body = self.text_card.body
        body.grid_columnconfigure(0, weight=1)
        self.text_box = ctk.CTkTextbox(body, height=120, font=font(13), wrap="word")
        self.text_box.grid(row=0, column=0, sticky="ew")
        self.text_box.insert("1.0", self._default_phrases())

        self.actions = ctk.CTkFrame(body, fg_color="transparent")
        self.actions.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        self.download_button = ctk.CTkButton(
            self.actions,
            text=self.tr("tts_download"),
            height=34,
            width=170,
            font=font(12),
            fg_color="#333a46",
            hover_color="#3d4552",
            command=self.download_voice,
        )
        self.download_button.pack(side="left")
        self.generate_button = ctk.CTkButton(
            self.actions,
            text=self.tr("tts_generate"),
            height=34,
            width=210,
            font=font(13, "bold"),
            command=self.generate,
        )
        self.generate_button.pack(side="left", padx=(8, 0))
        self.progress = ctk.CTkProgressBar(self.actions, height=12, width=160)
        self.progress.pack(side="left", padx=(14, 0))
        self.progress.set(0.0)

        self.status = ctk.CTkLabel(self.actions, text="", font=font(12), text_color=COLOR_MUTED, anchor="w")
        self.status.pack(side="left", padx=(12, 0))

        # ---------------------------------------------------------- plugins
        self.plugin_card = Card(self, title=self.tr("tts_plugins_title"))
        self.plugin_card.grid(row=row, column=0, sticky="ew", pady=(0, 12))
        row += 1
        body = self.plugin_card.body
        body.grid_columnconfigure(0, weight=1)
        plugin_actions = ctk.CTkFrame(body, fg_color="transparent")
        plugin_actions.grid(row=0, column=0, sticky="ew")
        self.plugin_open = ctk.CTkButton(
            plugin_actions,
            text=self.tr("tts_install_plugins"),
            height=32,
            width=190,
            font=font(12),
            fg_color="#333a46",
            hover_color="#3d4552",
            command=self.open_plugins,
        )
        self.plugin_open.pack(side="left")
        self.plugin_template = ctk.CTkButton(
            plugin_actions,
            text=self.tr("tts_install_template"),
            height=32,
            width=190,
            font=font(12),
            fg_color="#333a46",
            hover_color="#3d4552",
            command=self.create_template,
        )
        self.plugin_template.pack(side="left", padx=(8, 0))
        self.plugin_reload = ctk.CTkButton(
            plugin_actions,
            text="↻",
            height=32,
            width=40,
            font=font(14),
            fg_color="#333a46",
            hover_color="#3d4552",
            command=self.reload_backends,
        )
        self.plugin_reload.pack(side="left", padx=(8, 0))

        self.plugin_errors_label = ctk.CTkLabel(
            body, text="", font=font(11), text_color=COLOR_ERROR, anchor="w", justify="left", wraplength=780
        )
        self.plugin_errors_label.grid(row=1, column=0, sticky="w", pady=(6, 0))

        # --------------------------------------------------------------- log
        self.log_card = Card(self, title=self.tr("train_log"))
        self.log_card.grid(row=row, column=0, sticky="ew")
        self.log_card.body.grid_columnconfigure(0, weight=1)
        self.log = LogBox(self.log_card.body, height=140)
        self.log.grid(row=0, column=0, sticky="ew")

        self.after(100, self.reload_backends)

    # ------------------------------------------------------------------
    def _default_phrases(self) -> str:
        word = (self.app.gui_state.word or "hey computer").strip()
        return "\n".join(
            [
                word,
                f"эй, {word}",
                f"слушай, {word}",
                f"{word}, пожалуйста",
                f"окей, {word}",
            ]
        )

    # ------------------------------------------------------------------
    def reload_backends(self) -> None:
        errors = registry.refresh_plugins()
        infos = registry.list_backends()
        self._backend_names = list(infos)
        labels = [name for name in self._backend_names]
        if labels:
            self.backend_menu.configure(values=labels)
            current = self.backend_menu.get()
            if current not in labels:
                # Prefer an installed backend.
                preferred = next((name for name in labels if infos[name]["available"]), labels[0])
                self.backend_menu.set(preferred)
            self._on_backend(self.backend_menu.get())
        else:
            self.backend_menu.configure(values=["—"])
            self.backend_menu.set("—")

        if errors:
            text = self.tr("tts_plugin_errors") + ": " + ", ".join(errors)
            self.plugin_errors_label.configure(text=text)
        else:
            self.plugin_errors_label.configure(text="")

    def _current_backend(self) -> str:
        return self.backend_menu.get()

    def _on_backend(self, name: str) -> None:
        infos = registry.list_backends()
        info = infos.get(name, {})
        if info.get("available"):
            self.backend_info.configure(
                text=(info.get("description") or "") + "\n" + (info.get("license") or ""),
                text_color=COLOR_MUTED,
            )
            self.install_engine_button.grid_remove()
        else:
            text = self.tr(f"tts_missing_{name}")
            if text == f"tts_missing_{name}":
                text = self.tr("tts_not_installed", error=info.get("error") or "")
            self.backend_info.configure(text=text, text_color=COLOR_WARN)
            self.install_engine_button.grid()
        self._refresh_voices()

    def _on_language(self, label: str) -> None:
        self._refresh_voices()

    def _language_code(self) -> str | None:
        label = self.language_menu.get()
        for code, value in _LANGUAGE_FILTERS.items():
            if value == label:
                return None if code == "all" else code
        return None

    def _refresh_voices(self) -> None:
        backend = self._current_backend()
        if not backend or backend == "—":
            return
        try:
            self._voices = registry.list_voices(backend, language=self._language_code())
        except Exception as exc:  # noqa: BLE001
            self.log.append(str(exc))
            self._voices = []
        if not self._voices:
            self.voice_menu.configure(values=["—"])
            self.voice_menu.set("—")
            self.voice_menu.configure(state="disabled")
            return
        labels = [f"{voice.name} [{voice.language}]" for voice in self._voices]
        self.voice_menu.configure(values=labels, state="readonly")
        self.voice_menu.set(labels[0])

    def _current_voice(self):
        if not self._voices:
            return None
        label = self.voice_menu.get()
        for voice in self._voices:
            if f"{voice.name} [{voice.language}]" == label:
                return voice
        return self._voices[0]

    def _on_voice(self, _label: str) -> None:
        voice = self._current_voice()
        if voice is not None and voice.id == "__clone__":
            self.status.configure(text=self.tr("tts_voice_needs_download"), text_color=COLOR_MUTED)

    # ------------------------------------------------------------------
    def open_plugins(self) -> None:
        open_path(registry.plugin_directory())

    def create_template(self) -> None:
        try:
            path = registry.install_plugin_template(overwrite=False)
        except Exception as exc:  # noqa: BLE001
            if "already exists" in str(exc):
                path = registry.plugin_directory() / "my_tts_backend.py"
            else:
                self.app.show_error(self.tr("tts_install_template"), str(exc))
                return
        self.app.show_info(self.tr("tts_install_template"), self.tr("tts_template_created", path=path))
        open_path(path)

    # ------------------------------------------------------------------
    def download_voice(self) -> None:
        if self._download_busy:
            return
        voice = self._current_voice()
        backend = self._current_backend()
        if voice is None or not backend or backend == "—":
            return
        self._download_busy = True
        self.download_button.configure(state="disabled")
        self.status.configure(text=self.tr("tts_downloading"), text_color=COLOR_MUTED)
        self.log.append(f"download {backend}/{voice.id}")

        def work():
            return registry.download_voice(backend, voice.id)

        def done(path):
            self._download_busy = False
            self.download_button.configure(state="normal")
            self.status.configure(text=self.tr("tts_downloaded", path=path), text_color=COLOR_OK)
            self.log.append(self.tr("tts_downloaded", path=path))

        def failed(exc):
            self._download_busy = False
            self.download_button.configure(state="normal")
            self.status.configure(text=str(exc), text_color=COLOR_ERROR)
            self.log.append(f"ERROR: {exc}")

        self.app.run_task(work, done, failed)

    # ------------------------------------------------------------------
    def generate(self) -> None:
        if self._generating:
            return
        folder = self.app.gui_state.dataset
        if not folder or not Path(folder).is_dir():
            self.app.show_error(self.tr("tts_generate"), self.tr("tts_need_dataset"))
            return
        text = self.text_box.get("1.0", "end").strip()
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            self.app.show_error(self.tr("tts_generate"), self.tr("tts_need_text"))
            return
        backend = self._current_backend()
        voice = self._current_voice()
        if not backend or backend == "—" or voice is None:
            self.app.show_error(self.tr("tts_generate"), self.tr("tts_no_voices"))
            return

        info = registry.list_backends().get(backend, {})
        if not info.get("available"):
            self.app.show_error(self.tr("tts_generate"), self.tr("tts_not_installed", error=info.get("error") or ""))
            return

        self._generating = True
        self.generate_button.configure(state="disabled")
        self.progress.set(0.0)
        self.status.configure(text=self.tr("tts_generating", done=0, total=len(lines)), text_color=COLOR_MUTED)
        state = {"done": 0, "total": len(lines), "finished": False, "error": None, "results": []}

        def progress(done: int, total: int, name: str) -> None:
            state["done"] = done
            state["total"] = total
            state["last"] = name

        def work():
            try:
                results = registry.synthesize_many(
                    lines,
                    folder,
                    backend=backend,
                    voice=voice.id,
                    language=voice.language if voice.language != "multi" else None,
                    prefix="tts",
                    progress=progress,
                )
                state["results"] = results
            except Exception as exc:  # noqa: BLE001
                state["error"] = exc
                # The desktop build has no console; the traceback goes to the
                # log file set up by ``runtime.ensure_std_streams``.
                traceback.print_exc()
            finally:
                state["finished"] = True

        threading.Thread(target=work, daemon=True).start()
        self._poll_generation(state, folder)

    def _poll_generation(self, state: dict, folder: str) -> None:
        done = int(state["done"])
        total = max(1, int(state["total"]))
        self.progress.set(min(1.0, done / total))
        if not state["finished"]:
            self.status.configure(text=self.tr("tts_generating", done=done, total=total), text_color=COLOR_MUTED)
            self.after(150, lambda: self._poll_generation(state, folder))
            return

        self._generating = False
        self.generate_button.configure(state="normal")
        if state["error"] is not None:
            self.status.configure(text=str(state["error"]), text_color=COLOR_ERROR)
            self.log.append(f"ERROR: {state['error']}")
            return
        count = len(state["results"])
        self.progress.set(1.0)
        self.status.configure(text=self.tr("tts_done", count=count, folder=folder), text_color=COLOR_OK)
        self.log.append(self.tr("tts_done", count=count, folder=folder))
        self.app.set_status(self.tr("tts_done", count=count, folder=folder))

    # ------------------------------------------------------------------
    def on_show(self) -> None:
        # Pick up engines installed on the Components tab since the last visit.
        self.reload_backends()

    def retranslate(self) -> None:
        self.setup_card.set_title(self.tr("tts_title"), self.tr("tts_hint"))
        self.backend_label.configure(text=self.tr("tts_backend"))
        self.language_label.configure(text=self.tr("tts_language"))
        self.voice_label.configure(text=self.tr("tts_voice"))
        self.license_label.configure(text=self.tr("tts_license_note"))
        self.text_card.set_title(self.tr("tts_text"))
        self.download_button.configure(text=self.tr("tts_download"))
        self.generate_button.configure(text=self.tr("tts_generate"))
        self.plugin_card.set_title(self.tr("tts_plugins_title"))
        self.plugin_open.configure(text=self.tr("tts_install_plugins"))
        self.plugin_template.configure(text=self.tr("tts_install_template"))
        self.log_card.set_title(self.tr("train_log"))
