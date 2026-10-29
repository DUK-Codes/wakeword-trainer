"""Tests for :mod:`wakeword_trainer.runtime`.

The windowed desktop build runs without stdio; ``torch.hub`` and other loaders
write to ``sys.stderr`` and must not crash there.
"""

from __future__ import annotations

import sys
from pathlib import Path

from wakeword_trainer import runtime


def test_ensure_std_streams_keeps_existing_streams(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("WAKEWORD_TRAINER_CACHE_DIR", str(tmp_path))
    assert runtime.ensure_std_streams() is None


def test_ensure_std_streams_redirects_missing_streams(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("WAKEWORD_TRAINER_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    monkeypatch.setattr(sys, "stdin", None)

    path = runtime.ensure_std_streams()

    assert path is not None
    assert path.is_file()
    assert sys.stdin is not None and sys.stdin.read() == ""
    # Libraries such as tqdm write here; this is the frozen-exe failure mode.
    print("loading model", file=sys.stderr)
    print("progress", file=sys.stdout)
    sys.stderr.flush()
    sys.stdout.flush()
    text = path.read_text(encoding="utf-8")
    assert "loading model" in text
    assert "progress" in text
