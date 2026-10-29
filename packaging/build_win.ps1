<#
.SYNOPSIS
    Build the Windows desktop delivery: application folder, portable archive,
    component archives and the Inno Setup installer.

.DESCRIPTION
    Run it from a dedicated build virtualenv with a CPU-only PyTorch installed,
    otherwise the folder will include CUDA libraries and the installer can
    exceed GitHub's 2 GB release-asset limit.

    Example:
        py -3.12 -m venv .venv-build
        .venv-build\Scripts\python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
        .venv-build\Scripts\python -m pip install -e ".[gui]" pyinstaller
        packaging\build_win.ps1 -Clean

    Output in ``dist``:
        WakeWordTrainer\                                  application folder
        wakeword-trainer-<version>-windows-x64-portable.zip
        wakeword-trainer-<version>-windows-x64-setup.exe  (needs Inno Setup 6)
        component-tts-piper.zip
#>

param(
    [string]$Python = "python",
    [switch]$Clean,
    [switch]$SkipInstaller,
    [switch]$SkipComponents
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$dist = Join-Path $root "dist"

function Invoke-Native {
    param([string]$FilePath, [string[]]$Arguments)
    # Windows PowerShell 5.1 turns native stderr into a terminating error while
    # $ErrorActionPreference is "Stop", so relax it for the duration of the call.
    $ErrorActionPreference = "Continue"
    & $FilePath @Arguments 2>&1 | ForEach-Object { Write-Host $_ }
    $code = $LASTEXITCODE
    if ($code -ne 0) { throw "$FilePath failed with exit code $code" }
}

Push-Location $root
try {
    $ErrorActionPreference = "Continue"
    $version = (& $Python -c "import re,pathlib; print(re.search(r'^version = .([^\x27\x22]+).', pathlib.Path('pyproject.toml').read_text(encoding='utf-8'), re.M).group(1))" 2>&1 | Out-String).Trim()
    $ErrorActionPreference = "Stop"
    if ($version -notmatch '^\d+\.\d+') { throw "Could not read the version: $version" }
    Write-Host "Building wakeword-trainer $version"

    # ---------------------------------------------------------- application
    $arguments = @(
        "-m", "PyInstaller",
        "--noconfirm",
        "--distpath", $dist,
        "--workpath", (Join-Path $root "build"),
        (Join-Path $PSScriptRoot "wakeword_trainer.spec")
    )
    if ($Clean) { $arguments = @("-m", "PyInstaller", "--clean") + $arguments[2..($arguments.Count - 1)] }
    Invoke-Native -FilePath $Python -Arguments $arguments

    $appDir = Join-Path $dist "WakeWordTrainer"
    $exe = Join-Path $appDir "WakeWordTrainer.exe"
    if (-not (Test-Path $exe)) { throw "Missing build output: $exe" }
    $sizeMb = [math]::Round((Get-ChildItem $appDir -Recurse -File | Measure-Object Length -Sum).Sum / 1MB, 1)
    Write-Host "Application folder: $appDir ($sizeMb MB)"

    # ----------------------------------------------------------- components
    if (-not $SkipComponents) {
        & (Join-Path $PSScriptRoot "build_components.ps1") -Python $Python -Version $version
        if ($LASTEXITCODE -ne 0) { throw "Component build failed" }
    }

    # ------------------------------------------------------------- portable
    $portable = Join-Path $dist "wakeword-trainer-$version-windows-x64-portable.zip"
    Remove-Item -Force $portable -ErrorAction SilentlyContinue
    Compress-Archive -Path $appDir -DestinationPath $portable
    Write-Host "Portable archive: $portable"

    # ------------------------------------------------------------ installer
    if ($SkipInstaller) { return }

    $iscc = (Get-Command iscc -ErrorAction SilentlyContinue).Source
    if (-not $iscc) {
        $iscc = @(
            (Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6\ISCC.exe"),
            "C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
            "C:\Program Files\Inno Setup 6\ISCC.exe"
        ) | Where-Object { Test-Path $_ } | Select-Object -First 1
    }
    if (-not $iscc) {
        throw "Inno Setup 6 not found. Install it or pass -SkipInstaller."
    }

    Invoke-Native -FilePath $iscc -Arguments @("/DAppVersion=$version", (Join-Path $PSScriptRoot "installer.iss"))

    $setup = Join-Path $dist "wakeword-trainer-$version-windows-x64-setup.exe"
    if (-not (Test-Path $setup)) { throw "Missing installer: $setup" }
    $setupMb = [math]::Round((Get-Item $setup).Length / 1MB, 1)
    Write-Host "Installer: $setup ($setupMb MB)"
}
finally {
    Pop-Location
}
