"""Runtime helpers for the windowed desktop build.

A PyInstaller executable built with ``console=False`` starts with ``sys.stdout``,
``sys.stderr`` and ``sys.stdin`` set to ``None``. Anything that writes to a
stream then fails: ``torch.hub``, ``tqdm`` and libraries that warn through
``stderr`` all raise ``'NoneType' object has no attribute 'write'`` even when
the code itself is correct.

:func:`ensure_std_streams` points the missing streams at a log file inside the
user cache directory. Library output then keeps working and stays available for
diagnostics instead of being thrown away.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import TextIO

LOG_FILE_NAME = "desktop.log"
LOG_LIMIT_BYTES = 5 * 1024 * 1024


class _NullStream(io.TextIOBase):
    """Fallback sink used when the cache directory is not writable."""

    def write(self, text: str) -> int:
        return len(text)

    def flush(self) -> None:
        return None


def log_path() -> Path:
    """Location of the desktop log file (``<cache>/logs/desktop.log``)."""
    from .paths import cache_dir

    return cache_dir() / "logs" / LOG_FILE_NAME


def ensure_std_streams() -> Path | None:
    """Give the process usable ``stdout``/``stderr``/``stdin``.

    Returns the log file when streams had to be redirected, ``None`` when the
    process already has them (a normal console or pip-installed run).
    """
    if sys.stdout is not None and sys.stderr is not None and sys.stdin is not None:
        return None

    path: Path | None = None
    handle: TextIO
    try:
        path = log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file() and path.stat().st_size > LOG_LIMIT_BYTES:
            path.unlink()
        handle = path.open("a", encoding="utf-8", errors="replace", buffering=1)
    except Exception:  # noqa: BLE001 - a broken cache dir must not stop the app
        path = None
        handle = _NullStream()

    if sys.stdout is None:
        sys.stdout = handle
    if sys.stderr is None:
        sys.stderr = handle
    if sys.stdin is None:
        # Tk never reads stdin; an empty buffer keeps libraries that probe it happy.
        sys.stdin = io.StringIO("")
    return path
