# Releasing

Both delivery channels are produced by `.github/workflows/release.yml` from the same commit:

- **PyPI** — `sdist` + wheel for `pip install wakeword-trainer`;
- **GitHub Releases** — the same archives plus `wakeword-trainer-<version>-windows-x64.exe`.

## Steps

1. Bump `version` in `pyproject.toml` (and `src/wakeword_trainer/version.py` if it is not derived).
2. Commit and push to `main`; wait for CI.
3. Create and push a tag:

   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```

4. The workflow builds the package on Linux, builds the executable on Windows, publishes to PyPI, and creates the GitHub release with every artifact attached.

`workflow_dispatch` builds the package and executable without publishing (the publish jobs only run for tags).

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

## Local executable build

Use a dedicated virtualenv with a CPU-only PyTorch, otherwise the executable includes CUDA libraries and can exceed GitHub's 2 GB release-asset limit:

```powershell
py -3.10 -m venv .venv-build
.\.venv-build\Scripts\python -m pip install "torch==2.6.0"
.\.venv-build\Scripts\python -m pip install -e ".[gui]" "pyinstaller>=6,<7"
packaging\build_exe.ps1 -Python ".\.venv-build\Scripts\python.exe" -Clean
```

Output: `dist/wakeword-trainer.exe`.
