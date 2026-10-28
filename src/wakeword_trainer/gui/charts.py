"""Chart widgets for live training curves and evaluation plots."""

from __future__ import annotations

from pathlib import Path

import customtkinter as ctk

from .widgets import COLOR_CARD, COLOR_MUTED, font


def _mpl():
    import matplotlib

    matplotlib.use("Agg", force=False)
    from matplotlib.figure import Figure

    return Figure


class ChartFrame(ctk.CTkFrame):
    """A matplotlib canvas embedded in the Tk layout."""

    def __init__(self, master, *, height: int = 240, **kwargs) -> None:
        kwargs.setdefault("fg_color", COLOR_CARD)
        kwargs.setdefault("corner_radius", 10)
        super().__init__(master, **kwargs)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self._height = height
        self._mpl_canvas = None

        self.caption = ctk.CTkLabel(self, text="", font=font(11), text_color=COLOR_MUTED, anchor="w")
        self.caption.grid(row=0, column=0, sticky="ew", padx=10, pady=(6, 0))
        self._holder = ctk.CTkFrame(self, fg_color="transparent", height=height)
        self._holder.grid(row=1, column=0, sticky="nsew", padx=6, pady=6)
        self._holder.grid_columnconfigure(0, weight=1)
        self._holder.grid_rowconfigure(0, weight=1)
        self._placeholder = ctk.CTkLabel(self._holder, text="—", font=font(12), text_color=COLOR_MUTED)
        self._placeholder.grid(row=0, column=0)

        self._steps: list[int] = []
        self._losses: list[float] = []
        self._eval_steps: list[int] = []
        self._recalls: list[float] = []
        self._eval_fp: list[float] = []

    # ------------------------------------------------------------------
    def _ensure_canvas(self):
        if self._mpl_canvas is not None:
            return self._mpl_canvas
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        Figure = _mpl()
        self._figure = Figure(figsize=(6.4, self._height / 110), dpi=100, facecolor=COLOR_CARD)
        self._axis = self._figure.add_subplot(111)
        self._style_axis()
        self._mpl_canvas = FigureCanvasTkAgg(self._figure, master=self._holder)
        widget = self._mpl_canvas.get_tk_widget()
        widget.configure(background=COLOR_CARD, highlightthickness=0)
        widget.grid(row=0, column=0, sticky="nsew")
        self._placeholder.grid_remove()
        return self._mpl_canvas

    def _style_axis(self) -> None:
        axis = self._axis
        axis.set_facecolor(COLOR_CARD)
        axis.tick_params(colors="#98a2b3", labelsize=8)
        for spine in axis.spines.values():
            spine.set_color("#3a4150")
        axis.grid(color="#3a4150", alpha=0.35, linewidth=0.6)
        axis.yaxis.label.set_color("#98a2b3")
        axis.xaxis.label.set_color("#98a2b3")

    def clear(self, caption: str = "") -> None:
        self._steps.clear()
        self._losses.clear()
        self._eval_steps.clear()
        self._recalls.clear()
        self._eval_fp.clear()
        self.caption.configure(text=caption)
        if self._mpl_canvas is not None:
            self._axis.clear()
            self._style_axis()
            if hasattr(self, "_twin"):
                self._twin = None
            self._mpl_canvas.draw_idle()
        self._placeholder.grid()

    def set_caption(self, caption: str) -> None:
        self.caption.configure(text=caption)

    # ------------------------------------------------------------------
    def add_step(self, step: int, loss: float) -> None:
        self._steps.append(int(step))
        self._losses.append(float(loss))

    def add_eval(self, step: int, recall: float, fp_per_hour: float) -> None:
        self._eval_steps.append(int(step))
        self._recalls.append(float(recall))
        self._eval_fp.append(float(fp_per_hour))

    def redraw(self, *, log_scale: bool = False) -> None:
        if not self._steps and not self._eval_steps:
            return
        self._ensure_canvas()
        self._axis.clear()
        self._style_axis()
        if self._steps:
            self._axis.plot(
                self._steps,
                self._losses,
                color="#9aa4b2",
                linewidth=1.0,
                alpha=0.8,
                label="Loss",
            )
            self._axis.set_xlabel("Step")
            self._axis.set_ylabel("Loss")
        if log_scale:
            self._axis.set_yscale("symlog", linthresh=1e-3)
        if self._eval_steps:
            twin = self._axis.twinx()
            twin.set_facecolor("none")
            twin.tick_params(colors="#98a2b3", labelsize=8)
            twin.set_ylabel("Recall / FP per hour", color="#98a2b3")
            twin.plot(
                self._eval_steps,
                self._recalls,
                color="#4f8cff",
                marker="o",
                markersize=3,
                linewidth=1.2,
                label="Recall",
            )
            twin.set_ylim(-0.02, 1.05)
            twin.yaxis.label.set_color("#98a2b3")
            self._twin = twin
        self._figure.tight_layout()
        self._mpl_canvas.draw_idle()

    # ------------------------------------------------------------------
    def plot_history(self, history: dict, caption: str = "") -> None:
        """Populate from a training history dict (``steps``/``loss``/``eval``)."""
        self.clear(caption)
        for step, loss in zip(history.get("steps", []), history.get("loss", []), strict=False):
            self.add_step(step, loss)
        for item in history.get("eval", []):
            self.add_eval(item.get("step", 0), item.get("recall", 0.0), item.get("fp_per_hour", 0.0))
        self.redraw()

    def plot_threshold_rows(self, rows: list[dict], caption: str = "", target_fp: float = 1.0) -> None:
        """Plot recall and FP/hour against threshold from a report table."""
        if not rows:
            self.clear(caption)
            return
        self._ensure_canvas()
        self.caption.configure(text=caption)
        self._axis.clear()
        self._style_axis()
        thresholds = [float(row.get("threshold", 0.0)) for row in rows]
        recalls = [float(row.get("recall", 0.0)) for row in rows]
        fps = [float(row.get("fp_per_hour", 0.0)) for row in rows]
        self._axis.plot(thresholds, recalls, color="#4f8cff", label="Recall", linewidth=1.4)
        self._axis.set_xlabel("Threshold")
        self._axis.set_ylabel("Recall")
        self._axis.set_ylim(-0.02, 1.05)
        twin = self._axis.twinx()
        twin.set_facecolor("none")
        twin.tick_params(colors="#98a2b3", labelsize=8)
        twin.plot(thresholds, fps, color="#e05f5f", linewidth=1.2, label="FP/hour")
        twin.set_yscale("symlog", linthresh=0.01)
        twin.axhline(target_fp, color="#e0a83f", linestyle="--", linewidth=0.9)
        twin.set_ylabel("FP / hour", color="#98a2b3")
        self._figure.tight_layout()
        self._mpl_canvas.draw_idle()


class ImagePanel(ctk.CTkFrame):
    """Displays a PNG report figure, scaled to the panel."""

    def __init__(self, master, *, height: int = 220) -> None:
        super().__init__(master, fg_color=COLOR_CARD, corner_radius=10, height=height)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._height = height
        self._image = None
        self._path: Path | None = None
        self.label = ctk.CTkLabel(self, text="—", font=font(12), text_color=COLOR_MUTED)
        self.label.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        self.bind("<Configure>", self._on_resize)

    def show(self, path: str | Path | None, fallback: str = "—") -> None:
        self._path = Path(path) if path else None
        if self._path is None or not self._path.is_file():
            self._image = None
            self.label.configure(image=None, text=fallback)
            return
        self._render()

    def _on_resize(self, _event=None) -> None:
        if self._path is not None and self._path.is_file():
            self._render()

    def _render(self) -> None:
        try:
            from PIL import Image
        except Exception:  # noqa: BLE001
            return
        if self._path is None:
            return
        try:
            width = max(200, self.winfo_width() - 20)
            height = max(120, self.winfo_height() - 20)
            with Image.open(self._path) as image:
                image = image.convert("RGB")
                ratio = min(width / image.width, height / image.height)
                size = (max(1, int(image.width * ratio)), max(1, int(image.height * ratio)))
                resized = image.resize(size, Image.LANCZOS)
            self._image = ctk.CTkImage(light_image=resized, dark_image=resized, size=size)
            self.label.configure(image=self._image, text="")
        except Exception:  # noqa: BLE001
            self.label.configure(text=self._path.name)
