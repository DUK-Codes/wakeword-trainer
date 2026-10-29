<#
.SYNOPSIS
    Build the optional component archives (speech engines).

.DESCRIPTION
    Each component is a folder with a manifest and a ``site-packages`` tree that
    the application appends to ``sys.path``. The same folder is zipped for the
    Components page (downloaded from the latest GitHub release) and copied by
    the Windows installer when the matching option is selected.

    Dependencies that are already part of the application bundle (PyTorch,
    onnxruntime, numpy) are deliberately *not* installed into a component:
    duplicating them would triple the download and could shadow the bundled
    versions. The components are therefore installed with ``--no-deps`` and
    list their extra dependencies explicitly.

    Components must be built with the same Python version as the application
    bundle (3.12 in CI) so compiled wheels match.

    Example:
        packaging\build_components.ps1 -Version 0.1.2
#>

param(
    [string]$Python = "python",
    [string]$Version = "0.0.0"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$dist = Join-Path $root "dist"
$output = Join-Path $dist "components"

function Install-ComponentPackages {
    param([string]$Site, [string[]]$Packages)

    # pip writes notices to stderr; Windows PowerShell 5.1 would treat them as
    # terminating errors while $ErrorActionPreference is "Stop".
    $ErrorActionPreference = "Continue"
    & $Python -m pip install --disable-pip-version-check --no-warn-script-location `
        --no-deps --no-compile --target $Site @Packages 2>&1 | ForEach-Object { Write-Host $_ }
    $code = $LASTEXITCODE
    $ErrorActionPreference = "Stop"
    if ($code -ne 0) { throw "pip install failed: $($Packages -join ', ')" }
}

function Remove-StrayFiles {
    param([string]$Site)

    # ``pip --target`` leaves pip's own bookkeeping in the target folder.
    Get-ChildItem $Site -Filter "*.whl" -File | Remove-Item -Force
    Get-ChildItem $Site -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
    foreach ($name in @("bin", "share", "pydevd_plugins")) {
        $path = Join-Path $Site $name
        if (Test-Path $path) { Remove-Item -Recurse -Force $path }
    }
}

function New-Component {
    param(
        [string]$Id,
        [string[]]$Packages,
        [string]$Description
    )

    $target = Join-Path $output $Id
    Remove-Item -Recurse -Force $target -ErrorAction SilentlyContinue
    $site = Join-Path $target "site-packages"
    New-Item -ItemType Directory -Force -Path $site | Out-Null

    Write-Host "Installing $Id ($($Packages -join ', '))"
    Install-ComponentPackages -Site $site -Packages $Packages
    Remove-StrayFiles -Site $site

    $pythonVersion = (& $Python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>&1 | Out-String).Trim()
    $manifest = [ordered]@{
        id          = $Id
        version     = $Version
        description = $Description
        python      = $pythonVersion
        packages    = $Packages
    }
    # Windows PowerShell 5.1 writes a BOM with -Encoding utf8; JSON readers reject it.
    $manifestJson = $manifest | ConvertTo-Json -Depth 4
    [System.IO.File]::WriteAllText(
        (Join-Path $target "component.json"),
        $manifestJson,
        (New-Object System.Text.UTF8Encoding($false)))

    $archive = Join-Path $dist "component-$Id.zip"
    Remove-Item -Force $archive -ErrorAction SilentlyContinue
    Compress-Archive -Path (Join-Path $target "*") -DestinationPath $archive
    $sizeMb = [math]::Round((Get-Item $archive).Length / 1MB, 1)
    Write-Host "  -> $archive ($sizeMb MB)"
}

# Piper: onnxruntime/numpy come from the application, so only the engine and
# its pure-Python helper are added. The espeak-ng data ships inside the wheel.
New-Component -Id "tts-piper" `
    -Packages @("piper-tts==1.8.0", "pathvalidate>=3,<4") `
    -Description "Piper speech engine (voices are downloaded separately)."

Write-Host "Components are in $output"
