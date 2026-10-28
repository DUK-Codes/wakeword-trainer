<#
.SYNOPSIS
    Build the standalone Windows executable for wakeword-trainer.

.DESCRIPTION
    Uses PyInstaller with the spec in this folder. Run it from a dedicated
    build virtualenv that has a CPU-only PyTorch installed, otherwise the
    executable will include CUDA libraries and can exceed GitHub's 2 GB
    release-asset limit.

    Example:
        py -3.10 -m venv .venv-build
        .venv-build\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
        .venv-build\Scripts\python -m pip install -e ".[gui]" pyinstaller
        packaging\build_exe.ps1 -Clean
#>

param(
    [string]$Python = "python",
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

$arguments = @(
    "-m", "PyInstaller",
    "--noconfirm",
    "--distpath", (Join-Path $root "dist"),
    "--workpath", (Join-Path $root "build"),
    (Join-Path $PSScriptRoot "wakeword_trainer.spec")
)
if ($Clean) {
    $arguments = @("-m", "PyInstaller", "--clean") + $arguments[2..($arguments.Count - 1)]
}

Push-Location $root
try {
    & $Python @arguments
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed with exit code $LASTEXITCODE" }

    $exe = Join-Path $root "dist/wakeword-trainer.exe"
    if (-not (Test-Path $exe)) { throw "Missing build output: $exe" }
    $sizeMb = [math]::Round((Get-Item $exe).Length / 1MB, 1)
    Write-Host "Built $exe ($sizeMb MB)"
}
finally {
    Pop-Location
}
