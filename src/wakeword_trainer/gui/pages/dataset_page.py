"""Dataset page: validation report, microphone recording."""

from __future__ import annotations

import contextlib
from pathlib import Path

import customtkinter as ctk

from ..recording import InputDevice, MicrophoneUnavailable, Recorder, input_devices, save_recording
from ..widgets import (
    COLOR_ERROR,
    COLOR_MUTED,
    COLOR_OK,
    COLOR_WARN,
    Card,
    LogBox,
    Metric,
    PathRow,
    font,
)
from .base import BasePage


class DatasetPage(BasePage):
    def build(self) -> None:
        self.columnconfigure(0, weight=1)
        row = 0

        # ---------------------------------------------------------- folder
        # No card title here: the page header already says "Dataset".
        self.folder_card = Card(self, subtitle=self.tr("dataset_hint"))
        self.folder_card.grid(row=row, column=0, sticky="ew", pady=(0, 12))
        row += 1
        body = self.folder_card.body
        body.grid_columnconfigure(0, weight=1)

        self.path_row = PathRow(
            body,
            placeholder=self.tr("dataset_placeholder"),
            mode="directory",
            translator=self.tr,
            on_change=self._on_path_change,
        )
        self.path_row.grid(row=0, column=0, sticky="ew")
        self.path_row.set(self.app.gui_state.dataset)

        self.actions = ctk.CTkFrame(body, fg_color="transparent")
        self.actions.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        self.validate_button = ctk.CTkButton(
            self.actions,
            text=self.tr("dataset_check"),
            font=font(13, "bold"),
            height=36,
            command=self.validate,
        )
        self.validate_button.pack(side="left")

        self.status_label = ctk.CTkLabel(self.actions, text="", font=font(12), text_color=COLOR_MUTED, anchor="w")
        self.status_label.pack(side="left", padx=(12, 0))

        # --------------------------------------------------------- metrics
        self.metrics_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.metrics_frame.grid(row=row, column=0, sticky="ew", pady=(0, 12))
        row += 1
        for column in range(4):
            self.metrics_frame.grid_columnconfigure(column, weight=1, uniform="metric")

        self.metric_files = Metric(self.metrics_frame, self.tr("dataset_valid"))
        self.metric_groups = Metric(self.metrics_frame, self.tr("dataset_groups"))
        self.metric_duration = Metric(self.metrics_frame, self.tr("dataset_duration"))
        self.metric_invalid = Metric(self.metrics_frame, self.tr("dataset_invalid"))
        for column, widget in enumerate(
            (self.metric_files, self.metric_groups, self.metric_duration, self.metric_invalid)
        ):
            widget.grid(row=0, column=column, sticky="ew", padx=(0 if column == 0 else 8, 0))

        # -------------------------------------------------------- warnings
        self.warnings_card = Card(self, title=self.tr("dataset_warnings"))
        self.warnings_card.grid(row=row, column=0, sticky="ew", pady=(0, 12))
        self.warnings_card.grid_remove()
        row += 1
        self.warnings_box = ctk.CTkTextbox(
            self.warnings_card.body,
            height=96,
            font=font(12),
            fg_color="#12141a",
            text_color=COLOR_WARN,
            wrap="word",
        )
        self.warnings_box.grid(row=0, column=0, sticky="ew")
        self.warnings_box.configure(state="disabled")

        # ------------------------------------------------------- recording
        self.record_card = Card(
            self, title=self.tr("dataset_record_title"), subtitle=self.tr("dataset_record_hint")
        )
        self.record_card.grid(row=row, column=0, sticky="ew", pady=(0, 12))
        row += 1
        body = self.record_card.body
        body.grid_columnconfigure(1, weight=1)

        self.device_label = ctk.CTkLabel(
            body, text=self.tr("record_device"), font=font(12, "bold"), text_color=COLOR_MUTED, anchor="w"
        )
        self.device_label.grid(row=0, column=0, sticky="w", padx=(0, 10))
        self.device_combo = ctk.CTkComboBox(
            body, values=[self.tr("record_default_device")], font=font(12), height=32, state="readonly"
        )
        self.device_combo.grid(row=0, column=1, sticky="ew")
        self.device_combo.set(self.tr("record_default_device"))

        self.level_label = ctk.CTkLabel(
            body, text=self.tr("record_level"), font=font(12, "bold"), text_color=COLOR_MUTED, anchor="w"
        )
        self.level_label.grid(row=1, column=0, sticky="w", padx=(0, 10), pady=(10, 0))
        self.level_bar = ctk.CTkProgressBar(body, height=14)
        self.level_bar.grid(row=1, column=1, sticky="ew", pady=(10, 0))
        self.level_bar.set(0.0)

        self.record_actions = ctk.CTkFrame(body, fg_color="transparent")
        self.record_actions.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        self.record_button = ctk.CTkButton(
            self.record_actions,
            text=self.tr("record_start"),
            font=font(13, "bold"),
            height=36,
            command=self.toggle_recording,
        )
        self.record_button.pack(side="left")
        self.discard_button = ctk.CTkButton(
            self.record_actions,
            text=self.tr("record_discard"),
            font=font(13),
            height=36,
            width=120,
            fg_color="#333a46",
            hover_color="#3d4552",
            command=self.discard_recording,
            state="disabled",
        )
        self.discard_button.pack(side="left", padx=(8, 0))
        self.record_status = ctk.CTkLabel(
            self.record_actions, text="", font=font(12), text_color=COLOR_MUTED, anchor="w"
        )
        self.record_status.pack(side="left", padx=(12, 0))

        self.log_card = Card(self, title=self.tr("train_log"))
        self.log_card.grid(row=row, column=0, sticky="ew")
        self.log_card.body.grid_columnconfigure(0, weight=1)
        self.log = LogBox(self.log_card.body, height=120)
        self.log.grid(row=0, column=0, sticky="ew")

        # ----------------------------------------------------------- state
        self.recorder = Recorder()
        self._devices: list[InputDevice] = []
        self._record_timer: str | None = None
        self._level_value = 0.0
        self._mic_available = True
        self._scanning = False
        self._load_devices()

    # ------------------------------------------------------------------
    def _load_devices(self) -> None:
        try:
            self._devices = input_devices()
        except MicrophoneUnavailable as exc:
            self._mic_available = False
            self._devices = []
            self.device_combo.configure(values=["—"], state="disabled")
            self.device_combo.set("—")
            self.record_button.configure(state="disabled")
            self.record_status.configure(text=str(exc), text_color=COLOR_ERROR)
            return

        labels = [self.tr("record_default_device")]
        for device in self._devices:
            if device.is_default:
                labels[0] = f"{self.tr('record_default_device')} — {device.name}"
            else:
                labels.append(device.name)
        self.device_combo.configure(values=labels, state="readonly")
        self.device_combo.set(labels[0])

    def _selected_device(self) -> int | None:
        if not self._devices:
            return None
        index = self.device_combo.current()
        if index <= 0:
            return None
        return self._devices[index].index

    # ------------------------------------------------------------------
    def _on_path_change(self, value: str) -> None:
        self.app.gui_state.dataset = value.strip()
        self.app.gui_state.save()

    def validate(self) -> None:
        if self._scanning:
            return
        folder = self.path_row.get()
        if not folder:
            self.app.show_error(self.tr("dataset_check"), self.tr("dataset_placeholder"))
            return
        if not Path(folder).is_dir():
            self.app.show_error(self.tr("dataset_check"), f"{folder}")
            return

        self._scanning = True
        self.validate_button.configure(state="disabled")
        self.status_label.configure(text=self.tr("dataset_checking"), text_color=COLOR_MUTED)
        self.log.clear()

        from ...api import inspect_dataset

        def work():
            return inspect_dataset(folder, clip_seconds=self.app.gui_state_config_clip_seconds())

        def done(report):
            self._scanning = False
            self.validate_button.configure(state="normal")
            self.app.dataset_report = report
            self.app.gui_state.dataset = folder
            self.app.gui_state.save()
            self._render_report(report)
            self.status_label.configure(text=self.tr("dataset_check"), text_color=COLOR_MUTED)
            self.app.set_status(
                f"{report.summary()['valid_files']} files · {report.summary()['total_minutes']:.1f} min"
            )

        def failed(exc):
            self._scanning = False
            self.validate_button.configure(state="normal")
            self.status_label.configure(text=str(exc), text_color=COLOR_ERROR)
            self.app.show_error(self.tr("dataset_check"), str(exc))

        self.app.run_task(work, done, failed)

    def _render_report(self, report) -> None:
        summary = report.summary()
        self.metric_files.set_value(str(summary["valid_files"]), COLOR_OK if summary["valid_files"] else COLOR_ERROR)
        self.metric_groups.set_value(str(summary["groups"]), COLOR_OK if summary["groups"] >= 5 else COLOR_WARN)
        minutes = summary["total_minutes"]
        self.metric_duration.set_value(f"{minutes:.1f} мин" if self.tr.language == "ru" else f"{minutes:.1f} min")
        invalid = summary["invalid_files"]
        self.metric_invalid.set_value(str(invalid), COLOR_ERROR if invalid else COLOR_MUTED)

        self.warnings_box.configure(state="normal")
        self.warnings_box.delete("1.0", "end")
        warnings = list(summary.get("warnings") or [])
        if summary.get("duplicates"):
            warnings.append(self.tr("dataset_duplicates") + f": {len(summary['duplicates'])}")
        if not warnings:
            self.warnings_box.configure(text_color=COLOR_OK)
            self.warnings_box.insert("end", self.tr("dataset_no_warnings"))
        else:
            self.warnings_box.configure(text_color=COLOR_WARN)
            for warning in warnings:
                self.warnings_box.insert("end", f"• {warning}\n")
        self.warnings_box.configure(state="disabled")
        self.warnings_card.grid()

        for path, reason in summary.get("rejected", [])[:20]:
            self.log.append(f"{path} — {reason}")
        self.log.append(
            f"{summary['valid_files']} / {summary['files']} files, "
            f"{summary['total_minutes']:.1f} min, {summary['groups']} groups"
        )

    # ------------------------------------------------------------------
    def toggle_recording(self) -> None:
        if not self._mic_available:
            self.app.show_error(self.tr("record_start"), self.tr("record_no_device"))
            return
        if self.recorder.recording:
            self.finish_recording()
            return
        folder = self.path_row.get()
        if not folder:
            self.app.show_error(self.tr("record_start"), self.tr("record_need_folder"))
            return
        try:
            self.recorder.start(
                self._selected_device(),
                level_callback=self._set_level_threadsafe,
            )
        except MicrophoneUnavailable as exc:
            self.app.show_error(self.tr("record_start"), str(exc))
            return

        self.record_button.configure(text=self.tr("record_stop"), fg_color=COLOR_ERROR, hover_color="#c94f4f")
        self.discard_button.configure(state="normal")
        self.record_status.configure(text=self.tr("recording"), text_color=COLOR_ERROR)
        self._tick_recording()

    def _set_level_threadsafe(self, level: float) -> None:
        self._level_value = level

    def _tick_recording(self) -> None:
        if not self.recorder.recording:
            return
        self.level_bar.set(max(self._level_value, 0.01))
        self.record_status.configure(
            text=f"{self.tr('recording')} {self.recorder.duration:.1f} s", text_color=COLOR_ERROR
        )
        self._record_timer = self.after(50, self._tick_recording)

    def _stop_timer(self) -> None:
        if self._record_timer is not None:
            with contextlib.suppress(Exception):
                self.after_cancel(self._record_timer)
            self._record_timer = None

    def _reset_record_ui(self) -> None:
        self._stop_timer()
        self.level_bar.set(0.0)
        self.record_button.configure(text=self.tr("record_start"), fg_color="#4f8cff", hover_color="#3d76e0")
        self.discard_button.configure(state="disabled")
        self.record_status.configure(text="", text_color=COLOR_MUTED)

    def finish_recording(self) -> None:
        folder = self.path_row.get()
        samples = self.recorder.stop()
        self._reset_record_ui()
        if samples.size < int(0.2 * 16_000):
            self.record_status.configure(text=self.tr("record_no_device"), text_color=COLOR_WARN)
            return
        try:
            path = save_recording(samples, folder, prefix="rec")
        except Exception as exc:  # noqa: BLE001
            self.app.show_error(self.tr("record_start"), str(exc))
            return
        self.record_status.configure(text=self.tr("record_saved", name=path.name), text_color=COLOR_OK)
        self.log.append(self.tr("record_saved", name=str(path)))
        self.app.set_status(self.tr("record_saved", name=path.name))

    def discard_recording(self) -> None:
        self.recorder.abort()
        self._reset_record_ui()

    # ------------------------------------------------------------------
    def on_show(self) -> None:
        if not self.path_row.get() and self.app.gui_state.dataset:
            self.path_row.set(self.app.gui_state.dataset)

    def on_hide(self) -> None:
        if self.recorder.recording:
            self.discard_recording()

    def retranslate(self) -> None:
        self.folder_card.set_title("", self.tr("dataset_hint"))
        self.warnings_card.set_title(self.tr("dataset_warnings"))
        self.validate_button.configure(text=self.tr("dataset_check"))
        self.metric_files.set_caption(self.tr("dataset_valid"))
        self.metric_groups.set_caption(self.tr("dataset_groups"))
        self.metric_duration.set_caption(self.tr("dataset_duration"))
        self.metric_invalid.set_caption(self.tr("dataset_invalid"))
        self.record_card.set_title(self.tr("dataset_record_title"), self.tr("dataset_record_hint"))
        self.device_label.configure(text=self.tr("record_device"))
        self.level_label.configure(text=self.tr("record_level"))
        self.record_button.configure(
            text=self.tr("record_stop") if self.recorder.recording else self.tr("record_start")
        )
        self.discard_button.configure(text=self.tr("record_discard"))
        self.path_row.entry.configure(placeholder_text=self.tr("dataset_placeholder"))
        self.path_row.button.configure(text=self.tr("browse"))
        self.log_card.set_title(self.tr("train_log"))
