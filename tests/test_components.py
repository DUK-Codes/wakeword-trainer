"""Optional component archives: layout, activation and removal."""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pytest

from wakeword_trainer import components
from wakeword_trainer.exceptions import DownloadError


def make_archive(
    path: Path,
    *,
    component_id: str = "tts-piper",
    module: str = "demo_component_pkg",
    version: str = "1.2.3",
) -> Path:
    """Build a synthetic component archive with one importable package."""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "component.json",
            json.dumps({"id": component_id, "version": version, "python": "3.10"}),
        )
        archive.writestr(f"site-packages/{module}/__init__.py", "MARKER = 'component'\n")
    return path


@pytest.fixture
def user_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate both component roots inside the test tmp folder."""
    root = tmp_path / "user-components"
    root.mkdir()
    monkeypatch.setattr(components, "components_dir", lambda: root)
    monkeypatch.setattr(components, "application_dir", lambda: None)
    monkeypatch.delenv(components.COMPONENTS_ENV, raising=False)
    yield root
    for name in [name for name in sys.modules if name == "demo_component_pkg"]:
        sys.modules.pop(name, None)
    sys.path[:] = [entry for entry in sys.path if str(root) not in entry]


def test_install_archive_registers_component(tmp_path: Path, user_root: Path):
    archive = make_archive(tmp_path / "component.zip")
    target = components.install_archive(archive, expected_id="tts-piper")

    assert target == user_root / "tts-piper"
    assert (target / "component.json").is_file()
    assert components.is_installed("tts-piper")
    assert components.find_component("tts-piper") == (target, "1.2.3")

    state = components.component_state("tts-piper")
    assert state.installed
    assert state.removable
    assert state.version == "1.2.3"
    assert state.size_bytes > 0


def test_activate_makes_component_importable_without_shadowing(tmp_path: Path, user_root: Path):
    components.install_archive(make_archive(tmp_path / "component.zip"))

    added = components.activate()
    site = user_root / "tts-piper" / "site-packages"
    assert site in added
    # Components are appended: bundled packages must always win.
    assert sys.path[-1] == str(site)
    assert sys.path[0] != str(site)

    import demo_component_pkg

    assert demo_component_pkg.MARKER == "component"


def test_install_rejects_wrong_component_id(tmp_path: Path, user_root: Path):
    archive = make_archive(tmp_path / "component.zip", component_id="tts-other")
    with pytest.raises(DownloadError, match="tts-piper"):
        components.install_archive(archive, expected_id="tts-piper")
    assert not (user_root / "tts-other").exists()


def test_install_rejects_archive_without_manifest(tmp_path: Path, user_root: Path):
    archive = tmp_path / "broken.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("site-packages/readme.txt", "no manifest\n")
    with pytest.raises(DownloadError, match="component.json"):
        components.install_archive(archive)


def test_install_rejects_unsafe_paths(tmp_path: Path, user_root: Path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("component.json", json.dumps({"id": "tts-demo"}))
        handle.writestr("../evil.txt", "escape\n")
    with pytest.raises(DownloadError, match="unsafe path"):
        components.install_archive(archive)
    assert not (user_root.parent / "evil.txt").exists()


def test_remove_component(tmp_path: Path, user_root: Path):
    target = components.install_archive(make_archive(tmp_path / "component.zip"))
    assert target.is_dir()

    assert components.remove_component("tts-piper") is True
    assert not target.exists()
    assert components.remove_component("tts-piper") is False


def test_installer_managed_component_is_not_removable(
    tmp_path: Path, user_root: Path, monkeypatch: pytest.MonkeyPatch
):
    app = tmp_path / "app"
    deployed = app / "components" / "tts-piper"
    deployed.mkdir(parents=True)
    (deployed / "component.json").write_text(json.dumps({"id": "tts-piper", "version": "0.1.2"}), encoding="utf-8")

    monkeypatch.setattr(components, "application_dir", lambda: app)

    state = components.component_state("tts-piper")
    assert state.installed
    assert state.root == deployed
    assert not state.removable
    with pytest.raises(DownloadError, match="uninstaller|Windows"):
        components.remove_component("tts-piper")
    assert deployed.is_dir()


def test_download_component_rejects_unknown_id(user_root: Path):
    with pytest.raises(DownloadError, match="Unknown component"):
        components.download_component("does-not-exist")


def test_component_roots_include_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, user_root: Path):
    override = tmp_path / "portable" / "components"
    override.mkdir(parents=True)
    monkeypatch.setenv(components.COMPONENTS_ENV, str(override))

    roots = components.component_roots()
    assert roots[0] == override


def test_install_archive_accepts_manifest_with_bom(tmp_path: Path, user_root: Path):
    """Windows tools write JSON with a BOM; the archive must still install."""
    archive = tmp_path / "component-bom.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(
            "component.json",
            b"\xef\xbb\xbf" + json.dumps({"id": "tts-piper", "version": "0.1.2"}).encode("utf-8"),
        )
        handle.writestr("site-packages/demo_component_pkg/__init__.py", "MARKER = 'component'\n")

    target = components.install_archive(archive, expected_id="tts-piper")

    assert target == user_root / "tts-piper"
    assert components.find_component("tts-piper") == (target, "0.1.2")
