# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Windows desktop application (one-folder build).

Build with ``packaging/build_win.ps1`` (Windows) or::

    pyinstaller --clean --noconfirm packaging/wakeword_trainer.spec

The output is ``dist/WakeWordTrainer/`` — the folder the installer deploys and
the portable archive contains. Speech engines are *not* part of it: they are
separate component archives (see ``build_components.ps1``) so the base download
stays small. Use a CPU-only PyTorch build to stay within GitHub's 2 GB limit.
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
    # Loaded through importlib in ``tts.registry``; PyInstaller cannot see them.
    "wakeword_trainer.tts.piper",
    "wakeword_trainer.tts.coqui",
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
    [],
    exclude_binaries=True,
    name="WakeWordTrainer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ROOT / "packaging" / "app.ico"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="WakeWordTrainer",
)
