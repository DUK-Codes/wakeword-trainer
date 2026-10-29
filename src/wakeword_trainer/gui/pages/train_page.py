"""Training page: live progress, metrics, chart and log."""

from __future__ import annotations

import contextlib
from pathlib import Path

import customtkinter as ctk

from ..charts import ChartFrame
from ..runner import TrainingRun
from ..widgets import (
    COLOR_MUTED,
    COLOR_OK,
    COLOR_WARN,
    Card,
    LogBox,
    Metric,
    font,
)
from .base import BasePage


class TrainPage(BasePage):
    def build(self) -> None:
        self.run: TrainingRun | None = None
        self._poll_id: str | None = None
        self._chart_counter = 0
        self._total_steps = 0
        self._last_step = 0
        self._training = False
        self._cancel_seen = False
        self._finished = False

        self.columnconfigure(0, weight=1)
        row = 0

        # -------------------------------------------------------- controls
        self.control_card = Card(self, title=self.tr("train_title"))
        self.control_card.grid(row=row, column=0, sticky="ew", pady=(0, 12))
        row += 1
        body = self.control_card.body
        body.grid_columnconfigure(0, weight=1)

        buttons = ctk.CTkFrame(body, fg_color="transparent")
        buttons.grid(row=0, column=0, sticky="ew")
        self.start_button = ctk.CTkButton(
            buttons,
            text=self.tr("train_start"),
            font=font(14, "bold"),
            height=40,
            width=190,
            command=self.start_training,
        )
        self.start_button.pack(side="left")
        self.stop_button = ctk.CTkButton(
            buttons,
            text=self.tr("train_stop"),
            font=font(13),
            height=40,
            width=140,
            fg_color="#3a3f4b",
            hover_color="#4a5060",
            state="disabled",
            command=self.stop_training,
        )
        self.stop_button.pack(side="left", padx=(10, 0))
        self.open_button = ctk.CTkButton(
            buttons,
            text=self.tr("open_output"),
            font=font(12),
            height=40,
            width=140,
            fg_color="#333a46",
            hover_color="#3d4552",
            command=self._open_output,
        )
        self.open_button.pack(side="right")

        self.progress = ctk.CTkProgressBar(body, height=16)
        self.progress.grid(row=1, column=0, sticky="ew", pady=(14, 0))
        self.progress.set(0.0)

        info = ctk.CTkFrame(body, fg_color="transparent")
        info.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        self.step_label = ctk.CTkLabel(info, text="—", font=font(12), text_color=COLOR_MUTED, anchor="w")
        self.step_label.pack(side="left")
        self.time_label = ctk.CTkLabel(info, text="", font=font(12), text_color=COLOR_MUTED, anchor="e")
        self.time_label.pack(side="right")

        # --------------------------------------------------------- metrics
        metrics = ctk.CTkFrame(self, fg_color="transparent")
        metrics.grid(row=row, column=0, sticky="ew", pady=(0, 12))
        row += 1
        for column in range(4):
            metrics.grid_columnconfigure(column, weight=1, uniform="tm")
        self.metric_loss = Metric(metrics, self.tr("train_loss"))
        self.metric_recall = Metric(metrics, self.tr("train_recall"))
        self.metric_fp = Metric(metrics, self.tr("train_fp"))
        self.metric_threshold = Metric(metrics, self.tr("train_threshold"))
        for column, widget in enumerate((self.metric_loss, self.metric_recall, self.metric_fp, self.metric_threshold)):
            widget.grid(row=0, column=column, sticky="ew", padx=(0 if column == 0 else 8, 0))

        # ----------------------------------------------------------- chart
        self.chart = ChartFrame(self, height=260)
        self.chart.grid(row=row, column=0, sticky="ew", pady=(0, 12))
        row += 1

        # ------------------------------------------------------------- log
        self.log_card = Card(self, title=self.tr("train_log"))
        self.log_card.grid(row=row, column=0, sticky="ew")
        self.log_card.body.grid_columnconfigure(0, weight=1)
        self.log = LogBox(self.log_card.body, height=200)
        self.log.grid(row=0, column=0, sticky="ew")

    # ------------------------------------------------------------------
    def _open_output(self) -> None:
        from ..widgets import open_path

        output = self.app.gui_state.output
        if output:
            open_path(output)

    def _set_running(self, running: bool) -> None:
        self._training = running
        self.start_button.configure(state="disabled" if running else "normal")
        self.stop_button.configure(state="normal" if running else "disabled")

    # ------------------------------------------------------------------
    def start_training(self) -> None:
        if self._training:
            self.app.show_info(self.tr("train_title"), self.tr("train_already_running"))
            return
        if not self.app.gui_state.dataset:
            self.app.show_error(self.tr("train_title"), self.tr("train_need_dataset"))
            return
        if not Path(self.app.gui_state.dataset).is_dir():
            self.app.show_error(self.tr("train_title"), self.tr("train_need_dataset"))
            return
        if not self.app.gui_state.word.strip():
            self.app.show_error(self.tr("train_title"), self.tr("train_need_word"))
            return
        if not self.app.gui_state.output.strip():
            self.app.show_error(self.tr("train_title"), self.tr("train_output_missing"))
            return

        config, error = self.app.settings_page.validate_settings()
        if config is None:
            self.app.show_error(self.tr("train_title"), self.tr("settings_invalid", error=error or "?"))
            return

        if config.negatives.source == "auto" and not self.app.gui_state.download_negatives:
            self.app.show_info(self.tr("train_title"), self.tr("train_first_run_download"))

        self.log.clear()
        self.chart.clear(self.tr("train_curve"))
        self.progress.set(0.0)
        self._total_steps = int(config.train.steps)
        self._last_step = 0
        self._chart_counter = 0
        self._cancel_seen = False
        self._finished = False
        for widget in (self.metric_loss, self.metric_recall, self.metric_fp, self.metric_threshold):
            widget.set_value("—", COLOR_MUTED)
        self.step_label.configure(text=f"{self.tr('train_step')} 0 / {self._total_steps}")
        self.time_label.configure(text="")

        try:
            self.run = TrainingRun(config, tflite=self.app.gui_state.export_tflite, resume=self.app.gui_state.resume)
            self.run.start()
        except Exception as exc:  # noqa: BLE001
            self.app.show_error(self.tr("train_title"), self.tr("train_python_missing", error=exc))
            self.run = None
            return

        self._set_running(True)
        self.app.set_status(self.tr("train_running"))
        self.app.block_navigation(True)
        self.log.append(f"python: {self.run.pid}")
        self.log.append(f"config: {self.run.config_path}")
        self.log.append(f"events: {self.run.events_path}")
        self._schedule_poll()

    # ------------------------------------------------------------------
    def _schedule_poll(self, delay: int = 150) -> None:
        self._poll_id = self.after(delay, self._poll)

    def _poll(self) -> None:
        self._poll_id = None
        if self.run is None:
            return

        for event in self.run.read_events():
            self._handle_event(event)

        if self.run.running:
            self._update_timing()
            self._schedule_poll()
            return

        # Process finished: drain remaining output and finalize once.
        for event in self.run.drain():
            self._handle_event(event)
        if not self._finished:
            self._finalize()

    def _update_timing(self) -> None:
        if self.run is None:
            return
        elapsed = self.run.elapsed
        if self._total_steps > 0 and self._last_step > 0:
            eta = elapsed * (self._total_steps - self._last_step) / max(1, self._last_step)
            self.time_label.configure(
                text=f"{self.tr('train_elapsed')}: {self._fmt(elapsed)}   {self.tr('train_eta')}: {self._fmt(eta)}"
            )
        else:
            self.time_label.configure(text=f"{self.tr('train_elapsed')}: {self._fmt(elapsed)}")

    @staticmethod
    def _fmt(seconds: float) -> str:
        seconds = max(0, int(seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            return f"{hours}ч {minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    # ------------------------------------------------------------------
    def _handle_event(self, event) -> None:
        if event.kind == "start":
            self._total_steps = event.total_steps or self._total_steps
            self.log.append(f"start · {self._total_steps} steps")
        elif event.kind == "prepare":
            self.progress.set(min(0.02, event.payload.get("progress", 0.0) * 0.02))
            stage = event.payload.get("stage", "")
            self.step_label.configure(text=f"{self.tr('train_preparing')}: {stage}")
            self.log.append(event.message or f"prepare: {stage}")
        elif event.kind == "step":
            step = event.step
            loss = event.metrics.get("loss")
            self._last_step = max(self._last_step, step)
            if self._total_steps:
                self.progress.set(min(1.0, self._last_step / self._total_steps))
            if loss is not None:
                self.metric_loss.set_value(f"{loss:.4f}")
                self.chart.add_step(step, loss)
            self.step_label.configure(text=f"{self.tr('train_step')} {self._last_step} / {self._total_steps}")
            self._chart_counter += 1
            if self._chart_counter % 10 == 0:
                self.chart.redraw()
        elif event.kind == "metrics":
            self._handle_metrics(event)
        elif event.kind == "checkpoint":
            self.log.append(f"checkpoint · step {event.step} · {event.payload.get('path', '')}")
        elif event.kind == "log":
            if event.message:
                self.log.append(event.message)
        elif event.kind == "done":
            self._finished = True
            self.app.on_training_done(
                event, cancelled=self._cancel_seen or bool(self.run and self.run.cancel_requested)
            )
        elif event.kind == "error":
            self.log.append(f"ERROR: {event.message}")
            details = event.payload.get("details")
            if details:
                for line in str(details).strip().splitlines()[-8:]:
                    self.log.append(f"  {line}")

    def _handle_metrics(self, event) -> None:
        metrics = event.metrics or {}
        stage = event.payload.get("stage")
        if stage == "test":
            self.log.append(
                f"test · recall={metrics.get('operating_point', {}).get('recall', 0):.3f} "
                f"FP/h={metrics.get('operating_point', {}).get('fp_per_hour', 0):.2f}"
            )
            return
        recall = metrics.get("recall")
        fp = metrics.get("fp_per_hour")
        threshold = metrics.get("threshold")
        if recall is not None:
            self.metric_recall.set_value(f"{recall:.3f}", COLOR_OK if recall >= 0.9 else COLOR_WARN)
        if fp is not None:
            self.metric_fp.set_value(f"{fp:.2f}", COLOR_OK if fp <= 1.0 else COLOR_WARN)
        if threshold is not None:
            self.metric_threshold.set_value(f"{threshold:.2f}")
        if event.step:
            self.chart.add_eval(event.step, float(recall or 0.0), float(fp or 0.0))
            self.chart.redraw()
        self.log.append(f"eval step {event.step}: recall={recall} fp/hour={fp} threshold={threshold}")

    # ------------------------------------------------------------------
    def _finalize(self) -> None:
        if self.run is None:
            return
        self._finished = True
        code = self.run.returncode
        cancelled = self._cancel_seen or self.run.cancel_requested
        self._set_running(False)
        self.app.block_navigation(False)

        if code == 0:
            self.progress.set(1.0)
            if cancelled:
                self.app.set_status(self.tr("train_cancelled"))
                self.log.append(self.tr("train_cancelled"))
            else:
                self.app.set_status(self.tr("train_done"))
                self.log.append(self.tr("train_done"))
        elif cancelled or code in (130, -15):
            self.app.set_status(self.tr("train_cancelled"))
            self.log.append(self.tr("train_cancelled"))
        else:
            message = self.run.failure_message() or f"exit code {code}"
            self.app.set_status(self.tr("train_failed"))
            self.log.append(self.tr("train_failed") + f": {message}")
            self.app.show_error(self.tr("train_failed"), message)

        self.chart.redraw()

    # ------------------------------------------------------------------
    def stop_training(self) -> None:
        if self.run is None or not self.run.running:
            return
        self._cancel_seen = True
        self.run.cancel()
        self.stop_button.configure(state="disabled", text=self.tr("train_stopping"))
        self.app.set_status(self.tr("train_stopping"))
        self.log.append(self.tr("train_stopping"))
        # Give the worker a few seconds to stop cleanly, then force it.
        self.after(6000, self._force_stop)

    def _force_stop(self) -> None:
        if self.run is not None and self.run.running:
            self.log.append("worker did not stop in time; terminating")
            self.run.terminate()

    def shutdown(self) -> None:
        if self._poll_id is not None:
            with contextlib.suppress(Exception):
                self.after_cancel(self._poll_id)
            self._poll_id = None
        if self.run is not None:
            if self.run.running:
                self.run.cancel()
                self.run.terminate()
            self.run.close()

    def on_show(self) -> None:
        if not self._training and self.run is not None and self._finished:
            self.start_button.configure(state="normal")

    def retranslate(self) -> None:
        self.control_card.set_title(self.tr("train_title"))
        self.start_button.configure(text=self.tr("train_start"))
        self.stop_button.configure(text=self.tr("train_stop") if not self._cancel_seen else self.tr("train_stopping"))
        self.open_button.configure(text=self.tr("open_output"))
        self.metric_loss.set_caption(self.tr("train_loss"))
        self.metric_recall.set_caption(self.tr("train_recall"))
        self.metric_fp.set_caption(self.tr("train_fp"))
        self.metric_threshold.set_caption(self.tr("train_threshold"))
        self.chart.set_caption(self.tr("train_curve"))
        self.log_card.set_title(self.tr("train_log"))
        if not self._training:
            self.step_label.configure(text=self.tr("train_idle"))
