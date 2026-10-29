# Releasing

Both delivery channels are produced by `.github/workflows/release.yml` from the same commit:

- **PyPI** — `sdist` + wheel for `pip install wakeword-trainer`;
- **GitHub Releases** — the same archives plus the Windows delivery:
  - `wakeword-trainer-<version>-windows-x64-setup.exe` — Inno Setup installer;
  - `wakeword-trainer-<version>-windows-x64-portable.zip` — application folder;
  - `component-tts-piper.zip` — the optional speech engine downloaded by the
    **Components** page.

The component archives intentionally have **no version in the file name**: the
application resolves them through `releases/latest/download/...`, so every
release must upload them again.

## Steps

1. Bump `version` in `pyproject.toml` and `src/wakeword_trainer/version.py`.
2. Commit and push to `main`; wait for CI.
3. Create and push a tag:

   ```bash
   git tag v0.1.2
   git push origin v0.1.2
   ```

4. The workflow builds the package on Linux, builds the installer on Windows,
   publishes to PyPI, and creates the GitHub release with every artifact attached.

5. After the release, check the PyPI description and the release assets:

   ```powershell
   gh release view v0.1.2 --repo DUK-Codes/wakeword-trainer
   gh release download v0.1.2 --repo DUK-Codes/wakeword-trainer --pattern "*.exe"
   ```

## Version consistency checks

The Windows job fails early if:

- `dist/WakeWordTrainer/WakeWordTrainer.exe` is missing;
- the setup `.exe` or the portable `.zip` is missing;
- either component archive is missing;
- `dist/WakeWordTrainer/components` exists — the base build must not contain
  speech engines;
- the portable archive does not contain `WakeWordTrainer/WakeWordTrainer.exe`.

## README rules

`README.md` becomes the PyPI project description, so:

- use **absolute** links for anything inside the repository (`README.ru.md`, `docs/*.md`, `LICENSE`, `NOTICE`) — PyPI resolves relative links against <https://pypi.org/project/wakeword-trainer/>, which produces 404s;
- the description is part of the uploaded metadata, so a README-only fix still requires a version bump and a new release.

`workflow_dispatch` builds the package and the Windows delivery without
publishing (the publish jobs only run for tags).

## One-time PyPI setup (trusted publishing)

No API token is stored in the repository. Configure a pending publisher on PyPI:

| Field | Value |
| --- | --- |
| PyPI project name | `wakeword-trainer` |
| Owner | `DUK-Codes` |
| Repository | `wakeword-trainer` |
| Workflow name | `release.yml` |
| Environment | `pypi` |

Create it at <https://pypi.org/manage/account/publishing/>. After the first successful release the publisher is attached to the project.

If you prefer an API token instead, create one at <https://pypi.org/manage/account/token/> and add it as a repository secret named `PYPI_API_TOKEN`:

```powershell
& "C:\Program Files\GitHub CLI\gh.exe" secret set PYPI_API_TOKEN --repo DUK-Codes/wakeword-trainer
```

Paste the token at the hidden prompt; never put it on the command line or in a file inside the repository. The release workflow uses the secret when it is present and falls back to Trusted Publishing when it is empty.

The **first** token must be account-scoped (`Entire account`), because the project does not exist on PyPI yet. After the first successful upload, create a project-scoped token for `wakeword-trainer`, update the secret, and revoke the account-scoped token.

To publish an already-built release manually:

```powershell
$env:TWINE_USERNAME = "__token__"
$env:TWINE_PASSWORD = "pypi-..."   # paste at the prompt instead of saving it
python -m twine upload dist/*
```

## Local Windows build

Requirements:

- a virtualenv with a **CPU-only** PyTorch, otherwise the folder includes CUDA
  libraries and can exceed GitHub's 2 GB release-asset limit;
- **Inno Setup 6** (`winget install JRSoftware.InnoSetup`) for the installer.

```powershell
py -3.12 -m venv .venv-build
.\.venv-build\Scripts\python -m pip install "torch==2.6.0" --index-url https://download.pytorch.org/whl/cpu
.\.venv-build\Scripts\python -m pip install -e ".[gui]" "pyinstaller>=6,<7"
.\.venv-build\Scripts\python packaging\make_icon.py
packaging\build_win.ps1 -Python ".\.venv-build\Scripts\python.exe" -Clean
```

`build_win.ps1` performs the whole pipeline:

1. PyInstaller one-folder build → `dist/WakeWordTrainer/`;
2. `build_components.ps1` → `dist/components/tts-*/` plus `dist/component-*.zip`;
3. portable archive → `dist/wakeword-trainer-<version>-windows-x64-portable.zip`;
4. Inno Setup compile → `dist/wakeword-trainer-<version>-windows-x64-setup.exe`.

Useful switches: `-SkipInstaller` (application and portable archive only),
`-SkipComponents` (reuse existing component folders).

The base build must not contain a `components` folder. `build_components.ps1`
installs engine dependencies with `--no-deps` and an explicit package list, so
numpy/onnxruntime/torch from the application are not duplicated.

## Installer notes

`packaging/installer.iss`:

- per-user install by default (`PrivilegesRequired=lowest`), elevation available;
- three setup types: `core`, `withtts`, `custom`;
- the `piper` component copies from `dist/components/tts-piper`;
- Russian and English wizard languages;
- **not** signed — SmartScreen may warn.

To compile the installer alone:

```powershell
iscc /DAppVersion=0.1.2 packaging\installer.iss
```

## Testing a release build locally

```powershell
# Silent per-user install
Start-Process -Wait "dist\wakeword-trainer-0.1.2-windows-x64-setup.exe" `
    -ArgumentList "/VERYSILENT", "/TYPE=withtts", "/SUPPRESSMSGBOXES"

# The app can open a specific page (also useful for screenshots)
& "$env:LOCALAPPDATA\Programs\WakeWord Trainer\WakeWordTrainer.exe" --page components
```

Check that `%LOCALAPPDATA%\Programs\WakeWord Trainer\components\tts-piper` exists
for `withtts`, that the Components page reports both engines as installed, and
that the Speech page can download a voice and synthesize a sample.
