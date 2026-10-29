"""Optional components that can be added after installation.

The desktop build ships without speech engines. Everything optional lives in a
*component*: a portable archive with the same layout as the folder it is
extracted to::

    <root>/<component-id>/component.json
    <root>/<component-id>/site-packages/<python packages>

Two kinds of roots are scanned:

* ``<folder with the application>/components`` — what the Windows installer
  deployed, next to ``WakeWordTrainer.exe``;
* ``<user config>/wakeword-trainer/components`` — what the Components page
  downloaded later.

``site-packages`` of every installed component is appended to ``sys.path`` (and
to the Windows DLL search path) at import time, so a backend can simply do
``import piper``. Removing a component only affects the user-level root; files
deployed by the installer are managed by the uninstaller.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import os
import shutil
import sys
import tempfile
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .exceptions import DownloadError
from .paths import components_dir

ProgressFn = Callable[[int, int, str], None]

MANIFEST_NAME = "component.json"
SITE_PACKAGES = "site-packages"
RELEASE_ASSET_BASE = "https://github.com/DUK-Codes/wakeword-trainer/releases/latest/download"
COMPONENTS_ENV = "WAKEWORD_TRAINER_COMPONENTS_DIR"


@dataclass(frozen=True)
class Component:
    """A component that can be installed from an archive."""

    id: str
    asset: str
    size_mb: float
    engine: str = ""


COMPONENTS: dict[str, Component] = {
    "tts-piper": Component(id="tts-piper", asset="component-tts-piper.zip", size_mb=33, engine="piper"),
}


@dataclass(frozen=True)
class ComponentState:
    component: Component
    installed: bool
    root: Path | None = None
    version: str = ""
    size_bytes: int = 0
    removable: bool = False

    @property
    def size_mb(self) -> float:
        return round(self.size_bytes / (1024 * 1024), 1)


# ---------------------------------------------------------------------------
# locations
# ---------------------------------------------------------------------------


def application_dir() -> Path | None:
    """Folder with ``WakeWordTrainer.exe`` in a packaged build."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return None


def component_roots() -> list[Path]:
    """Existing component roots, most trusted first."""
    roots: list[Path] = []
    override = os.environ.get(COMPONENTS_ENV)
    if override:
        roots.append(Path(override).expanduser())
    app = application_dir()
    if app is not None:
        roots.append(app / "components")
    roots.append(components_dir())

    unique: list[Path] = []
    for root in roots:
        if root not in unique and root.is_dir():
            unique.append(root)
    return unique


def writable_root() -> Path:
    """Root used for components downloaded or installed by the user."""
    path = components_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_component(component_id: str) -> tuple[Path, str] | None:
    """Return ``(directory, manifest version)`` for an installed component."""
    for root in component_roots():
        directory = root / component_id
        manifest = directory / MANIFEST_NAME
        if not manifest.is_file():
            continue
        try:
            # ``utf-8-sig`` also accepts manifests written by Windows tools that
            # add a BOM; ``json`` rejects those bytes otherwise.
            data = json.loads(manifest.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            data = {}
        return directory, str(data.get("version") or "")
    return None


def is_installed(component_id: str) -> bool:
    return find_component(component_id) is not None


# ---------------------------------------------------------------------------
# sys.path activation
# ---------------------------------------------------------------------------

_activated = False


def activate() -> list[Path]:
    """Append installed components to ``sys.path``; return their folders.

    Folders are *appended* on purpose: components only provide the packages the
    application does not bundle, and the bundled copies must always win.
    """
    global _activated
    added: list[Path] = []
    for root in component_roots():
        for directory in sorted(root.iterdir()):
            site = directory / SITE_PACKAGES
            if not site.is_dir():
                continue
            resolved = str(site)
            added.append(site)
            if resolved not in sys.path:
                sys.path.append(resolved)
            if os.name == "nt" and hasattr(os, "add_dll_directory"):
                with contextlib.suppress(OSError):  # pragma: no cover - depends on the folder
                    os.add_dll_directory(resolved)
    importlib.invalidate_caches()
    _activated = True
    return added


def activated() -> bool:
    return _activated


# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------


def _directory_size(path: Path) -> int:
    total = 0
    for entry in path.rglob("*"):
        try:
            if entry.is_file():
                total += entry.stat().st_size
        except OSError:  # pragma: no cover - vanished during the scan
            continue
    return total


def component_state(component_id: str) -> ComponentState:
    component = COMPONENTS[component_id]
    found = find_component(component_id)
    if found is None:
        return ComponentState(component=component, installed=False)
    directory, version = found
    removable = directory.parent == writable_root()
    return ComponentState(
        component=component,
        installed=True,
        root=directory,
        version=version,
        size_bytes=_directory_size(directory),
        removable=removable,
    )


def all_states() -> dict[str, ComponentState]:
    return {component_id: component_state(component_id) for component_id in COMPONENTS}


# ---------------------------------------------------------------------------
# install / remove
# ---------------------------------------------------------------------------


def _safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    base = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if base != target and base not in target.parents:
            raise DownloadError(f"Component archive contains an unsafe path: {member.filename!r}.")
    archive.extractall(destination)


def install_archive(archive_path: str | Path, *, expected_id: str | None = None) -> Path:
    """Extract a component archive into the user component folder."""
    archive_path = Path(archive_path)
    if not archive_path.is_file():
        raise DownloadError(f"Component archive not found: {archive_path}")

    with tempfile.TemporaryDirectory(prefix="wwt-component-") as temporary:
        staging = Path(temporary)
        with zipfile.ZipFile(archive_path) as archive:
            _safe_extract(archive, staging)

        manifest_path = staging / MANIFEST_NAME
        if not manifest_path.is_file():
            # Tolerate archives that wrap everything in a single top-level folder.
            candidates = [path for path in staging.iterdir() if (path / MANIFEST_NAME).is_file()]
            if len(candidates) != 1:
                raise DownloadError("Component archive has no component.json manifest.")
            staging = candidates[0]
            manifest_path = staging / MANIFEST_NAME

        data = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        component_id = str(data.get("id") or "")
        if not component_id:
            raise DownloadError("Component manifest does not declare an id.")
        if expected_id and component_id != expected_id:
            raise DownloadError(f"Expected component {expected_id!r} but the archive contains {component_id!r}.")

        target = writable_root() / component_id
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        shutil.move(str(staging), str(target))

    activate()
    return target


def download_component(component_id: str, *, progress: ProgressFn | None = None) -> Path:
    """Download a component from the latest GitHub release and install it."""
    component = COMPONENTS.get(component_id)
    if component is None:
        raise DownloadError(f"Unknown component: {component_id!r}.")
    from .data.negatives import download_file
    from .paths import downloads_dir

    url = f"{RELEASE_ASSET_BASE}/{component.asset}"
    archive = download_file(url, downloads_dir() / component.asset, progress=progress)
    return install_archive(archive, expected_id=component_id)


def remove_component(component_id: str) -> bool:
    """Delete a user-installed component. Installer-managed files stay."""
    found = find_component(component_id)
    if found is None:
        return False
    directory, _version = found
    if directory.parent != writable_root():
        raise DownloadError(
            "This component was installed with the application; remove it from Windows settings instead."
        )
    shutil.rmtree(directory, ignore_errors=True)
    return True


__all__ = [
    "COMPONENTS",
    "Component",
    "ComponentState",
    "activate",
    "activated",
    "all_states",
    "application_dir",
    "component_roots",
    "component_state",
    "download_component",
    "find_component",
    "install_archive",
    "is_installed",
    "remove_component",
    "writable_root",
]
