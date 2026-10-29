"""Live microphone test with an exported ONNX model."""

from __future__ import annotations

import contextlib
import queue
from pathlib import Path

import customtkinter as ctk
import numpy as np

from ..widgets import (
    COLOR_ERROR,
    COLOR_MUTED,
    COLOR_OK,
    COLOR_TEXT,
    COLOR_WARN,
    Card,
    PathRow,
    font,
)
from .base import BasePage


class TestPage(BasePage):
    def build(self) -> None:
        self.detector = None
        self.stream = None
        self._queue: queue.Queue = queue.Queue(maxsize=64)
        self._timer: str | None = None
        self._detected_until = 0.0
        self._hits = 0
        self._model_ready = False

        self.columnconfigure(0, weight=1)
        row = 0

        # ------------------------------------------------------------ model
        self.model_card = Card(self, title=self.tr("test_title"), subtitle=self.tr("test_hint"))
        self.model_card.grid(row=row, column=0, sticky="ew", pady=(0, 12))
        row += 1
        body = self.model_card.body
        body.grid_columnconfigure(1, weight=1)

        self.model_label = ctk.CTkLabel(
            body, text=self.tr("test_model"), font=font(12, "bold"), text_color=COLOR_MUTED, anchor="w"
        )
        self.model_label.grid(row=0, column=0, sticky="w", padx=(0, 12))
        self.model_row = PathRow(
            body,
            mode="file",
            filetypes=[("ONNX model", "*.onnx"), ("All files", "*.*")],
            translator=self.tr,
            on_change=self._on_model_path,
        )
        self.model_row.grid(row=0, column=1, sticky="ew")
        default_model = self.app.gui_state.last_model
        if not default_model and self.app.gui_state.output:
            candidate = Path(self.app.gui_state.output) / "model_info.json"
            if candidate.is_file():
                try:
                    import json

                    info = json.loads(candidate.read_text(encoding="utf-8"))
                    default_model = info.get("onnx") or ""
                except Exception:  # noqa: BLE001
                    default_model = ""
        self.model_row.set(default_model)

        self.load_button = ctk.CTkButton(
            body, text=self.tr("test_choose_model"), height=34, width=170, font=font(12), command=self.load_model
        )
        self.load_button.grid(row=1, column=1, sticky="w", pady=(10, 0))

        self.info_label = ctk.CTkLabel(
            body, text="", font=font(12), text_color=COLOR_MUTED, anchor="w", justify="left", wraplength=760
        )
        self.info_label.grid(row=2, column=1, sticky="w", pady=(6, 0))

        # ------------------------------------------------------------- mic
        self.mic_card = Card(self, title=self.tr("test_title"))
        self.mic_card.grid(row=row, column=0, sticky="ew", pady=(0, 12))
        row += 1
        body = self.mic_card.body
        body.grid_columnconfigure(0, weight=1)

        self.score_bar = ctk.CTkProgressBar(body, height=26, corner_radius=8)
        self.score_bar.grid(row=0, column=0, sticky="ew")
        self.score_bar.set(0.0)

        score_row = ctk.CTkFrame(body, fg_color="transparent")
        score_row.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        self.score_label = ctk.CTkLabel(
            score_row, text="0.000", font=font(24, "bold"), text_color=COLOR_MUTED, anchor="w"
        )
        self.score_label.pack(side="left")
        self.state_label = ctk.CTkLabel(
            score_row, text=self.tr("test_waiting"), font=font(14, "bold"), text_color=COLOR_MUTED, anchor="e"
        )
        self.state_label.pack(side="right")

        threshold_row = ctk.CTkFrame(body, fg_color="transparent")
        threshold_row.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        self.threshold_text = ctk.CTkLabel(
            threshold_row, text=self.tr("test_threshold"), font=font(12), text_color=COLOR_MUTED, anchor="w"
        )
        self.threshold_text.pack(side="left")
        self.threshold_value = ctk.CTkLabel(
            threshold_row, text="0.50", font=font(12, "bold"), text_color=COLOR_MUTED, anchor="w", width=44
        )
        self.threshold_value.pack(side="right")
        self.threshold_slider = ctk.CTkSlider(
            threshold_row, from_=0.01, to=0.99, number_of_steps=98, command=self._on_threshold
        )
        self.threshold_slider.set(0.5)
        self.threshold_slider.pack(side="left", fill="x", expand=True, padx=12)

        buttons = ctk.CTkFrame(body, fg_color="transparent")
        buttons.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        self.mic_button = ctk.CTkButton(
            buttons,
            text=self.tr("test_mic_start"),
            height=38,
            width=190,
            font=font(13, "bold"),
            command=self.toggle_mic,
            state="disabled",
        )
        self.mic_button.pack(side="left")
        self.hits_label = ctk.CTkLabel(buttons, text="", font=font(12), text_color=COLOR_MUTED, anchor="w")
        self.hits_label.pack(side="left", padx=(14, 0))

        # ------------------------------------------------------------ file
        self.file_card = Card(self, title=self.tr("test_file"))
        self.file_card.grid(row=row, column=0, sticky="ew")
        row += 1
        body = self.file_card.body
        body.grid_columnconfigure(0, weight=1)
        self.file_row = PathRow(
            body,
            mode="file",
            filetypes=[("Audio", "*.wav *.flac *.ogg *.mp3 *.m4a"), ("All files", "*.*")],
            translator=self.tr,
        )
        self.file_row.grid(row=0, column=0, sticky="ew")
        self.file_button = ctk.CTkButton(
            body, text=self.tr("test_file"), height=34, width=150, font=font(12), command=self.test_file
        )
        self.file_button.grid(row=1, column=0, sticky="w", pady=(10, 0))
        self.file_result = ctk.CTkLabel(body, text="", font=font(13), text_color=COLOR_MUTED, anchor="w")
        self.file_result.grid(row=2, column=0, sticky="w", pady=(6, 0))

        if self.model_row.get() and Path(self.model_row.get()).is_file():
            self.after(200, self.load_model)

    # ------------------------------------------------------------------
    def _on_model_path(self, value: str) -> None:
        self._model_ready = False
        if hasattr(self, "mic_button"):
            self.mic_button.configure(state="disabled")
        self.app.gui_state.last_model = value.strip()
        self.app.gui_state.save()

    def _on_threshold(self, value: float) -> None:
        self.threshold_value.configure(text=f"{float(value):.2f}")
        if self.detector is not None:
            self.detector.threshold = float(value)

    def load_model(self) -> None:
        path = self.model_row.get()
        if not path or not Path(path).is_file():
            self.app.show_error(self.tr("test_no_model"), path or "—")
            return
        self.load_button.configure(state="disabled")
        self.info_label.configure(text="…", text_color=COLOR_MUTED)

        def work():
            from ...models.inference import WakeWordDetector

            detector = WakeWordDetector(path, threshold=float(self.threshold_slider.get()))
            return detector, detector.metadata

        def done(result):
            detector, metadata = result
            self.detector = detector
            self._model_ready = True
            self.load_button.configure(state="normal")
            self.mic_button.configure(state="normal")
            self.info_label.configure(
                text=(
                    f"{self.tr('test_model_loaded', word=metadata.word or '—')}\n"
                    f"frames={metadata.frames or '—'} · clip={metadata.clip_seconds:.2f}s · "
                    f"threshold={metadata.threshold:.2f}"
                ),
                text_color=COLOR_OK,
            )
            self.app.gui_state.last_model = path
            self.app.gui_state.save()

        def failed(exc):
            self._model_ready = False
            self.load_button.configure(state="normal")
            self.info_label.configure(text=self.tr("test_load_failed", error=exc), text_color=COLOR_ERROR)

        self.app.run_task(work, done, failed)

    # ------------------------------------------------------------------
    def toggle_mic(self) -> None:
        if self.stream is not None:
            self.stop_mic()
            return
        if self.detector is None:
            self.app.show_error(self.tr("test_mic_start"), self.tr("test_no_model"))
            return
        try:
            import sounddevice as sd
        except Exception as exc:  # noqa: BLE001
            self.app.show_error(self.tr("test_mic_start"), self.tr("test_no_device") + f"\n{exc}")
            return

        def callback(indata, frames, time_info, status) -> None:  # noqa: ANN001, ARG001
            with contextlib.suppress(queue.Full):
                self._queue.put_nowait(bytes(indata))

        try:
            self.stream = sd.RawInputStream(
                samplerate=16_000,
                channels=1,
                dtype="int16",
                blocksize=1280,
                callback=callback,
            )
            self.stream.start()
        except Exception as exc:  # noqa: BLE001
            self.stream = None
            self.app.show_error(self.tr("test_mic_start"), str(exc))
            return

        from ...models.inference import StreamingDetector

        self._streaming = StreamingDetector(self.detector, hop_seconds=0.08)
        self._hits = 0
        self.hits_label.configure(text="")
        self.mic_button.configure(text=self.tr("test_mic_stop"), fg_color=COLOR_ERROR, hover_color="#c94f4f")
        self.state_label.configure(text=self.tr("test_waiting"), text_color=COLOR_MUTED)
        self._poll()

    def stop_mic(self) -> None:
        if self._timer is not None:
            with contextlib.suppress(Exception):
                self.after_cancel(self._timer)
            self._timer = None
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:  # noqa: BLE001
                pass
            self.stream = None
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self.mic_button.configure(text=self.tr("test_mic_start"), fg_color="#4f8cff", hover_color="#3d76e0")
        self.score_bar.set(0.0)
        self.score_label.configure(text="0.000", text_color=COLOR_MUTED)
        self.state_label.configure(text=self.tr("test_waiting"), text_color=COLOR_MUTED)

    # ------------------------------------------------------------------
    def _poll(self) -> None:
        self._timer = None
        if self.stream is None or self.detector is None:
            return

        score = None
        blocks = 0
        while blocks < 8:
            try:
                raw = self._queue.get_nowait()
            except queue.Empty:
                break
            blocks += 1
            block = np.frombuffer(raw, dtype=np.int16)
            value = self._streaming.push(block)
            if value is not None:
                score = value

        import time

        now = time.time()
        if score is not None:
            threshold = float(self.detector.threshold)
            self.score_bar.set(max(0.001, float(score)))
            self.score_label.configure(text=f"{score:.3f}", text_color=COLOR_TEXT if score < threshold else COLOR_OK)
            if score >= threshold:
                self._hits += 1
                self._detected_until = now + 1.2
                self.hits_label.configure(text=f"×{self._hits}")

        if now < self._detected_until:
            self.state_label.configure(text=self.tr("test_detected"), text_color=COLOR_OK)
        else:
            self.state_label.configure(text=self.tr("test_waiting"), text_color=COLOR_MUTED)

        self._timer = self.after(60, self._poll)

    # ------------------------------------------------------------------
    def test_file(self) -> None:
        path = self.file_row.get()
        if not path or not Path(path).is_file():
            self.app.show_error(self.tr("test_file"), "—")
            return
        if self.detector is None:
            self.app.show_error(self.tr("test_file"), self.tr("test_no_model"))
            return
        self.file_result.configure(text="…", text_color=COLOR_MUTED)
        detector = self.detector

        def work():
            score, _scores = detector.score_file_windows(path)
            return score

        def done(score):
            threshold = float(detector.threshold)
            detected = score >= threshold
            self.file_result.configure(
                text=self.tr("test_file_score", score=score) + ("  ✓" if detected else ""),
                text_color=COLOR_OK if detected else COLOR_WARN,
            )

        def failed(exc):
            self.file_result.configure(text=str(exc), text_color=COLOR_ERROR)

        self.app.run_task(work, done, failed)

    # ------------------------------------------------------------------
    def on_hide(self) -> None:
        if self.stream is not None:
            self.stop_mic()

    def shutdown(self) -> None:
        self.stop_mic()

    def retranslate(self) -> None:
        self.model_card.set_title(self.tr("test_title"), self.tr("test_hint"))
        self.model_label.configure(text=self.tr("test_model"))
        self.load_button.configure(text=self.tr("test_choose_model"))
        self.mic_card.set_title(self.tr("test_title"))
        self.threshold_text.configure(text=self.tr("test_threshold"))
        self.mic_button.configure(
            text=self.tr("test_mic_stop") if self.stream is not None else self.tr("test_mic_start")
        )
        self.state_label.configure(text=self.tr("test_waiting"))
        self.file_card.set_title(self.tr("test_file"))
        self.file_button.configure(text=self.tr("test_file"))
        self.model_row.button.configure(text=self.tr("browse"))
        self.file_row.button.configure(text=self.tr("browse"))
