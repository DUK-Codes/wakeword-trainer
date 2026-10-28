"""Results page: held-out metrics, plots, export and per-file scores."""

from __future__ import annotations

import json
from pathlib import Path

import customtkinter as ctk

from ..charts import ChartFrame, ImagePanel
from ..widgets import (
    COLOR_MUTED,
    COLOR_OK,
    COLOR_WARN,
    Card,
    LogBox,
    Metric,
    font,
    open_path,
)
from .base import BasePage


class ResultsPage(BasePage):
    def build(self) -> None:
        self.output_dir: Path | None = None
        self.training: dict = {}
        self.evaluation: dict = {}
        self.model_info: dict = {}
        self.per_clip: list[dict] = []
        self._busy = False

        self.columnconfigure(0, weight=1)
        row = 0

        # ------------------------------------------------------------ header
        self.header_card = Card(self)
        self.header_card.grid(row=row, column=0, sticky="ew", pady=(0, 12))
        row += 1
        body = self.header_card.body
        body.grid_columnconfigure(0, weight=1)

        self.empty_label = ctk.CTkLabel(
            body, text=self.tr("results_none"), font=font(13), text_color=COLOR_MUTED, anchor="w"
        )
        self.empty_label.grid(row=0, column=0, sticky="w")

        self.model_label = ctk.CTkLabel(
            body, text="", font=font(12), text_color=COLOR_MUTED, anchor="w", wraplength=760, justify="left"
        )
        self.model_label.grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.model_label.grid_remove()

        buttons = ctk.CTkFrame(body, fg_color="transparent")
        buttons.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        self.open_dir_button = ctk.CTkButton(
            buttons, text=self.tr("results_open_dir"), height=34, font=font(12), command=self._open_dir
        )
        self.open_dir_button.pack(side="left")
        self.open_report_button = ctk.CTkButton(
            buttons,
            text=self.tr("results_open_report"),
            height=34,
            font=font(12),
            fg_color="#333a46",
            hover_color="#3d4552",
            command=self._open_report,
        )
        self.open_report_button.pack(side="left", padx=(8, 0))
        self.export_onnx_button = ctk.CTkButton(
            buttons,
            text=self.tr("results_export_onnx"),
            height=34,
            font=font(12),
            fg_color="#333a46",
            hover_color="#3d4552",
            command=lambda: self._export("onnx"),
        )
        self.export_onnx_button.pack(side="left", padx=(8, 0))
        self.export_tflite_button = ctk.CTkButton(
            buttons,
            text=self.tr("results_export_tflite"),
            height=34,
            font=font(12),
            fg_color="#333a46",
            hover_color="#3d4552",
            command=lambda: self._export("tflite"),
        )
        self.export_tflite_button.pack(side="left", padx=(8, 0))

        # ----------------------------------------------------------- metrics
        self.metrics_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.metrics_frame.grid(row=row, column=0, sticky="ew", pady=(0, 12))
        row += 1
        for column in range(4):
            self.metrics_frame.grid_columnconfigure(column, weight=1, uniform="rm")

        self.metric_recall = Metric(self.metrics_frame, self.tr("results_recall"))
        self.metric_frr = Metric(self.metrics_frame, self.tr("results_frr"))
        self.metric_fp = Metric(self.metrics_frame, self.tr("results_fp"))
        self.metric_threshold = Metric(self.metrics_frame, self.tr("results_threshold"))
        self.metric_auc = Metric(self.metrics_frame, self.tr("results_auc"))
        self.metric_neg_hours = Metric(self.metrics_frame, self.tr("results_neg_hours"))
        self.metric_positives = Metric(self.metrics_frame, self.tr("results_positives"))
        self.metric_best_step = Metric(self.metrics_frame, self.tr("train_step"))
        self._metrics = [
            self.metric_recall,
            self.metric_frr,
            self.metric_fp,
            self.metric_threshold,
            self.metric_auc,
            self.metric_neg_hours,
            self.metric_positives,
            self.metric_best_step,
        ]
        for index, widget in enumerate(self._metrics):
            widget.grid(
                row=index // 4,
                column=index % 4,
                sticky="ew",
                padx=(0 if index % 4 == 0 else 8, 0),
                pady=(0 if index < 4 else 8, 0),
            )

        # ------------------------------------------------------------- chart
        self.sweep_chart = ChartFrame(self, height=250)
        self.sweep_chart.grid(row=row, column=0, sticky="ew", pady=(0, 12))
        row += 1

        # ------------------------------------------------------------ images
        self.images_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.images_frame.grid(row=row, column=0, sticky="ew", pady=(0, 12))
        row += 1
        for column in range(2):
            self.images_frame.grid_columnconfigure(column, weight=1, uniform="img")
        self.scores_image = ImagePanel(self.images_frame, height=260)
        self.scores_image.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.curves_image = ImagePanel(self.images_frame, height=260)
        self.curves_image.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

        # ---------------------------------------------------------- warnings
        self.warnings_card = Card(self)
        self.warnings_card.grid(row=row, column=0, sticky="ew", pady=(0, 12))
        row += 1
        self.warnings_box = ctk.CTkTextbox(
            self.warnings_card.body,
            height=90,
            font=font(12),
            fg_color="#12141a",
            text_color=COLOR_WARN,
            wrap="word",
        )
        self.warnings_box.grid(row=0, column=0, sticky="ew")
        self.warnings_box.configure(state="disabled")

        # ----------------------------------------------------------- table
        self.table_card = Card(self)
        self.table_card.grid(row=row, column=0, sticky="ew")
        self.table_card.body.grid_columnconfigure(0, weight=1)
        self.table = LogBox(self.table_card.body, height=200)
        self.table.grid(row=0, column=0, sticky="ew")

        self._set_buttons(False)

    # ------------------------------------------------------------------
    def _set_buttons(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for button in (
            self.open_dir_button,
            self.open_report_button,
            self.export_onnx_button,
            self.export_tflite_button,
        ):
            button.configure(state=state)

    def _open_dir(self) -> None:
        if self.output_dir:
            open_path(self.output_dir)

    def _open_report(self) -> None:
        if self.output_dir:
            report = Path(self.output_dir) / "evaluation" / "evaluation.md"
            if report.is_file():
                open_path(report)
            else:
                open_path(self.output_dir)

    # ------------------------------------------------------------------
    def load(self, output_dir: str | Path) -> bool:
        """Load artifacts written by a training run."""
        output_dir = Path(output_dir)
        training_path = output_dir / "training_result.json"
        evaluation_path = output_dir / "evaluation" / "evaluation.json"
        model_info_path = output_dir / "model_info.json"

        training = self._read_json(training_path)
        if training is None:
            self.output_dir = output_dir
            self._set_buttons(False)
            return False

        self.output_dir = output_dir
        self.training = training
        self.evaluation = self._read_json(evaluation_path) or {}
        self.model_info = self._read_json(model_info_path) or {}
        self.per_clip = list(self.evaluation.get("per_clip") or [])
        self._render()
        return True

    @staticmethod
    def _read_json(path: Path) -> dict | None:
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return None

    # ------------------------------------------------------------------
    def _render(self) -> None:
        training = self.training
        evaluation = (self.evaluation.get("evaluation") or {}) if self.evaluation else {}
        operating = evaluation.get("operating_point") or {}
        if not operating and training.get("best_metrics"):
            metrics = training["best_metrics"]
            operating = {
                "recall": metrics.get("recall"),
                "frr": metrics.get("frr"),
                "fp_per_hour": metrics.get("fp_per_hour"),
                "threshold": metrics.get("threshold"),
            }
        self.empty_label.grid_remove()
        self.model_label.grid()

        word = training.get("word") or self.evaluation.get("word") or ""
        onnx_path = training.get("onnx_path") or self.model_info.get("onnx") or ""
        self.model_label.configure(
            text=f"{self.tr('results_model')}: {onnx_path or '—'}" + (f"\n{word}" if word else "")
        )
        self._set_buttons(True)

        def number(value, fmt="{:.3f}", color=COLOR_MUTED):
            if value is None:
                return "—", COLOR_MUTED
            try:
                return fmt.format(float(value)), color
            except (TypeError, ValueError):
                return str(value), COLOR_MUTED

        recall = operating.get("recall")
        self.metric_recall.set_value(*number(recall, color=COLOR_OK if (recall or 0) >= 0.9 else COLOR_WARN))
        self.metric_frr.set_value(*number(operating.get("frr")))
        fp = operating.get("fp_per_hour")
        self.metric_fp.set_value(*number(fp, "{:.2f}", COLOR_OK if (fp or 0) <= 1.0 else COLOR_WARN))
        self.metric_threshold.set_value(*number(operating.get("threshold"), "{:.2f}"))
        self.metric_auc.set_value(*number(evaluation.get("auc"), "{:.4f}"))
        self.metric_neg_hours.set_value(*number(evaluation.get("negative_hours"), "{:.2f}"))
        self.metric_positives.set_value(str(evaluation.get("positives", "—")))
        self.metric_best_step.set_value(str(training.get("best_step", "—")))

        self.sweep_chart.set_caption(self.tr("results_threshold_sweep"))
        rows = evaluation.get("threshold_table") or []
        target_fp = float(evaluation.get("target_fp_per_hour", 1.0) or 1.0)
        self.sweep_chart.plot_threshold_rows(rows, self.tr("results_threshold_sweep"), target_fp)

        base = Path(self.output_dir) if self.output_dir else Path(".")
        self.scores_image.show(base / "evaluation" / "score_distribution.png", self.tr("results_scores"))
        self.curves_image.show(base / "evaluation" / "training_curves.png", self.tr("results_curves"))

        warnings = list(evaluation.get("warnings") or []) + list(training.get("warnings") or [])
        self.warnings_box.configure(state="normal")
        self.warnings_box.delete("1.0", "end")
        if warnings:
            self.warnings_box.configure(text_color=COLOR_WARN)
            for warning in dict.fromkeys(warnings):
                self.warnings_box.insert("end", f"• {warning}\n")
        else:
            self.warnings_box.configure(text_color=COLOR_OK)
            self.warnings_box.insert("end", self.tr("results_no_warnings"))
        self.warnings_box.configure(state="disabled")

        self.table.clear()
        if self.per_clip:
            header = f"{'score':>7}  {'THR':^3}  {'group':<24}  file"
            self.table.append(header)
            for item in sorted(self.per_clip, key=lambda entry: float(entry.get("score", 0)), reverse=True):
                score = float(item.get("score", 0.0))
                mark = "✓" if item.get("detected") else " "
                group = str(item.get("group", ""))[:24]
                self.table.append(f"{score:7.4f}  {mark:^3}  {group:<24}  {item.get('path', '')}")
        else:
            self.table.append(self.tr("results_none"))

    # ------------------------------------------------------------------
    def _export(self, fmt: str) -> None:
        if self._busy or self.output_dir is None:
            return
        from tkinter import filedialog

        onnx_path = self.training.get("onnx_path") or self.model_info.get("onnx")
        checkpoint = Path(self.output_dir) / "wakeword.pt"
        source = onnx_path or (str(checkpoint) if checkpoint.is_file() else None)
        if not source:
            self.app.show_error(self.tr("results_export"), self.tr("results_none"))
            return

        suffix = ".onnx" if fmt == "onnx" else ".tflite"
        word = self.training.get("word") or "wakeword"
        slug = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in word.lower()).strip("-") or "wakeword"
        target = filedialog.asksaveasfilename(
            defaultextension=suffix,
            initialfile=f"{slug}{suffix}",
            filetypes=[(fmt.upper(), f"*{suffix}")],
        )
        if not target:
            return

        from ...api import export_model

        self._busy = True
        self._set_buttons(False)
        self.app.set_status(f"{self.tr('results_export')} → {target}")

        def work():
            return export_model(source, target, format=fmt)  # type: ignore[arg-type]

        def done(path):
            self._busy = False
            self._set_buttons(True)
            self.app.set_status(self.tr("results_exported", path=path))
            self.app.show_info(self.tr("results_export"), self.tr("results_exported", path=path))

        def failed(exc):
            self._busy = False
            self._set_buttons(True)
            self.app.show_error(self.tr("results_export_failed", error=exc), str(exc))

        self.app.run_task(work, done, failed)

    # ------------------------------------------------------------------
    def on_show(self) -> None:
        if self.output_dir is None and self.app.gui_state.output:
            self.load(self.app.gui_state.output)

    def set_empty(self) -> None:
        self.output_dir = None
        self._set_buttons(False)
        self.model_label.grid_remove()
        self.empty_label.grid()
        for metric in self._metrics:
            metric.set_value("—", COLOR_MUTED)
        self.sweep_chart.clear(self.tr("results_threshold_sweep"))
        self.scores_image.show(None)
        self.curves_image.show(None)
        self.table.clear()

    def retranslate(self) -> None:
        self.header_card.set_title(self.tr("results_title"))
        self.empty_label.configure(text=self.tr("results_none"))
        self.open_dir_button.configure(text=self.tr("results_open_dir"))
        self.open_report_button.configure(text=self.tr("results_open_report"))
        self.export_onnx_button.configure(text=self.tr("results_export_onnx"))
        self.export_tflite_button.configure(text=self.tr("results_export_tflite"))
        self.metric_recall.set_caption(self.tr("results_recall"))
        self.metric_frr.set_caption(self.tr("results_frr"))
        self.metric_fp.set_caption(self.tr("results_fp"))
        self.metric_threshold.set_caption(self.tr("results_threshold"))
        self.metric_auc.set_caption(self.tr("results_auc"))
        self.metric_neg_hours.set_caption(self.tr("results_neg_hours"))
        self.metric_positives.set_caption(self.tr("results_positives"))
        self.metric_best_step.set_caption(self.tr("train_step"))
        self.sweep_chart.set_caption(self.tr("results_threshold_sweep"))
        self.warnings_card.set_title(self.tr("results_warnings"))
        self.table_card.set_title(self.tr("results_per_clip"))
        if self.training:
            self._render()
