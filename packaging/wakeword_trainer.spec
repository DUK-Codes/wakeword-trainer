# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the standalone Windows desktop app.

Build with ``packaging/build_exe.ps1`` (Windows) or::

    pyinstaller --clean --noconfirm packaging/wakeword_trainer.spec

The executable bundles the library, the GUI and the ONNX feature models, but
not PyPI/wheel metadata. Use a CPU-only PyTorch build to keep the file within
GitHub's 2 GB release-asset limit.
"""

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files

ROOT = Path(SPECPATH).resolve().parent

datas = collect_data_files("wakeword_trainer")
datas += collect_data_files("customtkinter")
# ``collect_data_files`` skips Python files, but the TTS plugin template is a
# Python file shipped as data.
datas.append(
    (
        str(ROOT / "src" / "wakeword_trainer" / "tts" / "templates" / "custom_backend.py"),
        "wakeword_trainer/tts/templates",
    )
)

hiddenimports = [
    "wakeword_trainer.gui.worker",
    "wakeword_trainer.gui.app",
    "matplotlib.backends.backend_tkagg",
    "onnxruntime",
    "onnxruntime.capi.onnxruntime_pybind11_state",
    "sounddevice",
    "PIL._tkinter_finder",
]

a = Analysis(
    [str(ROOT / "packaging" / "gui_entry.py")],
    pathex=[str(ROOT / "src")],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "pytest",
        "pytest_cov",
        "ruff",
        "build",
        "twine",
        "setuptools",
        "pip",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="wakeword-trainer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
