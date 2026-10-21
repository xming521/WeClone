<#
.SYNOPSIS
    Copy WeClone into .\app and install dependencies (uv if available, otherwise python -m venv + pip).

.DESCRIPTION
    ASCII-only on purpose: Windows PowerShell 5.1 reads BOM-less .ps1 as ANSI.

.PARAMETER Install
    Create .venv and install WeClone base deps + presidio + zh spaCy model.

.PARAMETER Full
    Additionally install the [main] group (torch cu126 + LLaMA-Factory, ~10GB). Requires uv.

.PARAMETER Force
    Overwrite a non-empty destination.

.PARAMETER Mirror
    PyPI mirror used by pip/uv (default: Tsinghua, faster in mainland China).
#>
[CmdletBinding()]
param(
    [string]$Source = "",
    [string]$Dest = "",
    [switch]$Install,
    [switch]$Full,
    [switch]$Force,
    [switch]$SkipCopy,
    [string]$Mirror = "https://pypi.tuna.tsinghua.edu.cn/simple"
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
if (-not $Dest) { $Dest = Join-Path $Root "app" }

# Inside a WeClone checkout (i.e. this folder lives at WeClone/weclone-cloud),
# the parent directory already IS WeClone: reuse it instead of copying/cloning.
$ParentRoot = Split-Path -Parent $Root
if (Test-Path (Join-Path $ParentRoot "weclone\__init__.py")) {
    Write-Host ">> parent directory is a WeClone checkout, reusing it" -ForegroundColor Cyan
    $SkipCopy = $true
    if (-not $Source) { $Source = $ParentRoot }
    $Dest = $ParentRoot
}

$VenvDir = Join-Path $Root ".venv"
$VenvPy = Join-Path $VenvDir "Scripts\python.exe"

function Die([string]$Text) {
    Write-Host ""
    Write-Host "   FAILED: $Text" -ForegroundColor Red
    exit 1
}

function Get-PythonVersion([string]$Exe, [string[]]$Prefix) {
    # Native programs write "not found" to stderr; with ErrorActionPreference=Stop
    # PowerShell 5.1 turns that into a terminating error, so relax it locally.
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $probe = $Prefix + @("-c", "import sys;print('%d.%d'%sys.version_info[:2])")
        $output = & $Exe @probe 2>$null
        if ($LASTEXITCODE -ne 0) { return "" }
        $text = "$output".Trim()
        if ($text -match '^\d+\.\d+$') { return $text }
        return ""
    } catch {
        return ""
    } finally {
        $ErrorActionPreference = $previous
    }
}

function Find-Python312 {
    $seen = @()
    $candidates = @(
        @{ Exe = "python"; Args = @() },
        @{ Exe = "py"; Args = @("-3.12") },
        @{ Exe = "py"; Args = @("-3") }
    )
    foreach ($candidate in $candidates) {
        if (-not (Get-Command $candidate.Exe -ErrorAction SilentlyContinue)) { continue }
        $version = Get-PythonVersion $candidate.Exe $candidate.Args
        if ($version -eq "3.12") { return $candidate }
        if ($version) { $seen += "$($candidate.Exe) $($candidate.Args -join ' ') => $version" }
    }
    if ($seen.Count -gt 0) {
        Write-Host "   found these Python runtimes:" -ForegroundColor Yellow
        foreach ($line in $seen) { Write-Host "     $line" -ForegroundColor Yellow }
    }
    return $null
}

# --- locate the WeClone source --------------------------------------------
if (-not $Source) {
    $candidates = @(
        (Join-Path (Split-Path -Parent $Root) "WeClone-master"),
        (Join-Path ([Environment]::GetFolderPath("Desktop")) "WeClone-master")
    )
    $Source = ($candidates | Where-Object { Test-Path (Join-Path $_ "pyproject.toml") } | Select-Object -First 1)
    if (-not $Source) {
        if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
            Die "WeClone not found. Clone https://github.com/xming521/WeClone and pass -Source <path>"
        }
        $Source = Join-Path $Root "WeClone-master"
        Write-Host ">> WeClone not found locally, cloning from GitHub" -ForegroundColor Cyan
        & git clone --depth 1 https://github.com/xming521/WeClone $Source
        if ($LASTEXITCODE -ne 0) { Die "git clone WeClone failed" }
        # The bundle ships its own nested copy of the sources; keep the root one.
        foreach ($nested in @("WeClone-master")) {
            $p = Join-Path $Source $nested
            if (Test-Path $p) { Remove-Item $p -Recurse -Force }
        }
    }
    Write-Host "   source: $Source" -ForegroundColor DarkGray
}

# --- validate source -------------------------------------------------------
$pyproject = Join-Path $Source "pyproject.toml"
if (-not (Test-Path $pyproject)) {
    Die "pyproject.toml not found under $Source (use -Source to point at the WeClone repo)"
}
$raw = Get-Content $pyproject -Raw
if ($raw -notmatch 'name\s*=\s*"WeClone"') {
    Die "$Source is not the WeClone repository root"
}

$appReady = Test-Path (Join-Path $Dest "weclone\__init__.py")
if ($appReady -and -not $Force) { $SkipCopy = $true }
if ($SkipCopy) {
    if (-not $appReady) {
        Die "$Dest is missing or incomplete (no weclone\__init__.py); remove it and run without -SkipCopy"
    }
} elseif ((Test-Path $Dest) -and -not $Force) {
    $existing = Get-ChildItem $Dest -Force | Select-Object -First 1
    if ($existing) {
        Die "$Dest is not empty; add -Force or choose another -Dest"
    }
}

# --- copy ------------------------------------------------------------------
if ($SkipCopy -and $appReady) {
    Write-Host "   ..  copy skipped (app/ already has WeClone)" -ForegroundColor DarkGray
} else {
    Write-Host ">> copying WeClone" -ForegroundColor Cyan
    Write-Host "   $Source  ->  $Dest" -ForegroundColor DarkGray
    New-Item -ItemType Directory -Path $Dest -Force | Out-Null

    $excludeDirs = @(".git", "models", "node_modules", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache")
    $robocopyArgs = @($Source, $Dest, "/E", "/NFL", "/NDL", "/NJH", "/NJS", "/NP", "/R:2", "/W:2")
    foreach ($dir in $excludeDirs) { $robocopyArgs += @("/XD", (Join-Path $Source $dir)) }
    $robocopyArgs += @("/XF", "*.pyc", "*.pyo")

    & robocopy @robocopyArgs | Out-Null
    $code = $LASTEXITCODE
    if ($code -ge 8) { Die "robocopy failed (exit $code)" }
    Write-Host "   OK  source copied (robocopy exit $code; 0-7 is success)" -ForegroundColor Green
}

$configTarget = Join-Path $Root "cloud.jsonc"
if (-not (Test-Path $configTarget)) {
    Copy-Item (Join-Path $Root "cloud.example.jsonc") $configTarget
    Write-Host "   OK  cloud.jsonc created" -ForegroundColor Green
}

# --- locate uv (PATH, .venv\Scripts, or python -m uv) ----------------------
$uvExe = ""
$uvPre = @()
if (Get-Command uv -ErrorAction SilentlyContinue) {
    $uvExe = "uv"
} elseif (Test-Path (Join-Path $VenvDir "Scripts\uv.exe")) {
    $uvExe = Join-Path $VenvDir "Scripts\uv.exe"
    Write-Host "   ..  using $uvExe" -ForegroundColor DarkGray
} else {
    $anyPython = ""
    if (Test-Path $VenvPy) { $anyPython = $VenvPy }
    elseif (Get-Command python -ErrorAction SilentlyContinue) { $anyPython = "python" }
    elseif (Get-Command py -ErrorAction SilentlyContinue) { $anyPython = "py" }
    if ($anyPython) {
        $previous = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        & $anyPython -m uv --version *> $null
        $uvOk = ($LASTEXITCODE -eq 0)
        $ErrorActionPreference = $previous
        if ($uvOk) {
            $uvExe = $anyPython
            $uvPre = @("-m", "uv")
            Write-Host "   ..  using '$anyPython -m uv'" -ForegroundColor DarkGray
        }
    }
}

# --- install ---------------------------------------------------------------
if ($Install) {
    if ($uvExe) {
        Write-Host ">> creating .venv with uv (python 3.12)" -ForegroundColor Cyan
        & $uvExe @($uvPre + @("venv", $VenvDir, "--python=3.12"))
        if ($LASTEXITCODE -ne 0) { Die "uv venv failed (could it not download Python 3.12? check network)" }
        $env:VIRTUAL_ENV = $VenvDir

        & $uvExe @($uvPre + @("pip", "install", "--index-url", $Mirror, "-e", $Dest))
        if ($LASTEXITCODE -ne 0) { Die "installing WeClone base deps failed" }

        & $uvExe @($uvPre + @("pip", "install", "--index-url", $Mirror, "presidio_analyzer", "presidio_anonymizer"))
        if ($LASTEXITCODE -ne 0) { Die "installing presidio failed" }

        & $uvExe @($uvPre + @("pip", "install", "https://github.com/explosion/spacy-models/releases/download/zh_core_web_sm-3.8.0/zh_core_web_sm-3.8.0-py3-none-any.whl"))
        if ($LASTEXITCODE -ne 0) { Write-Host "   !!  zh spaCy model failed; Chinese PII masking may not work" -ForegroundColor Yellow }

        & $uvExe @($uvPre + @("pip", "install", "--index-url", $Mirror, "-e", $Root))
        if ($LASTEXITCODE -ne 0) { Die "installing weclone-cloud failed" }
    } else {
        $py = Find-Python312
        if (-not $py) {
            Die "No Python 3.12 and no uv found.`n   Fastest:  python -m pip install uv   (then rerun; uv will fetch 3.12 itself)`n   Or:       winget install --id Python.Python.3.12 -e`n   Check:    py -0p"
        }
        Write-Host ">> creating .venv with $($py.Exe) $($py.Args -join ' ')" -ForegroundColor Cyan
        $venvArgs = $py.Args + @("-m", "venv", $VenvDir)
        & $py.Exe @venvArgs
        if ($LASTEXITCODE -ne 0) { Die "venv creation failed" }

        & $VenvPy -m pip install --upgrade pip --index-url $Mirror
        & $VenvPy -m pip install --index-url $Mirror -e $Dest
        if ($LASTEXITCODE -ne 0) { Die "installing WeClone base deps failed" }

        & $VenvPy -m pip install --index-url $Mirror presidio_analyzer presidio_anonymizer
        if ($LASTEXITCODE -ne 0) { Die "installing presidio failed" }

        & $VenvPy -m pip install "https://github.com/explosion/spacy-models/releases/download/zh_core_web_sm-3.8.0/zh_core_web_sm-3.8.0-py3-none-any.whl"
        if ($LASTEXITCODE -ne 0) { Write-Host "   !!  zh spaCy model failed; Chinese PII masking may not work" -ForegroundColor Yellow }

        & $VenvPy -m pip install --index-url $Mirror -e $Root
        if ($LASTEXITCODE -ne 0) { Die "installing weclone-cloud failed" }
    }
    Write-Host "   OK  install finished (light mode)" -ForegroundColor Green
}

if ($Full) {
    if (-not $uvExe) {
        Die "Full install needs uv. Easiest: run install-train.cmd (it installs uv into .venv first)."
    }
    if (-not (Test-Path $VenvDir)) { Die "-Full requires -Install first" }
    Write-Host ">> installing [main] group (torch cu126 + LLaMA-Factory, ~10GB)" -ForegroundColor Cyan
    $env:VIRTUAL_ENV = $VenvDir
    & $uvExe @($uvPre + @("pip", "install", "--group", "main", "-e", $Dest))
    if ($LASTEXITCODE -ne 0) { Die "installing [main] group failed" }
    Write-Host "   OK  local training deps ready" -ForegroundColor Green
}

Write-Host ""
Write-Host "Next:" -ForegroundColor Yellow
Write-Host "  1) edit cloud.jsonc if needed"
Write-Host "  2) `$env:WECLONE_API_KEY = 'sk-...'"
Write-Host "  3) .\.venv\Scripts\Activate.ps1 ; weclone-cloud doctor"
