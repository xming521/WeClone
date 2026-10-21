<#
.SYNOPSIS
    Assemble a clean, uploadable copy of this project on the Desktop.

.DESCRIPTION
    Copies only the source that belongs in a public repository:
      * Python package, tools, docs, tests, launcher scripts
    Excludes anything personal, secret, generated or third-party:
      * .venv, app (WeClone copy), langbot-src, wechatpad-src
      * secrets.local.cmd, cloud.jsonc, persona_system.md (contains real chats)
      * datasets, logs, caches, build output
    Then scans the result for API keys / wxids and reports what it finds.

.PARAMETER Dest
    Output folder. Default: <Desktop>\weclone-cloud

.PARAMETER Force
    Delete an existing destination folder first.
#>
param(
    [string]$Dest = "",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
if (-not $Dest) { $Dest = Join-Path ([Environment]::GetFolderPath("Desktop")) "weclone-cloud" }

Write-Host "source : $Root"
Write-Host "target : $Dest"
Write-Host ""

if (Test-Path $Dest) {
    if ($Force) {
        Write-Host "removing existing target ..."
        Remove-Item $Dest -Recurse -Force
    } else {
        Write-Host "target already exists; use -Force to replace it." -ForegroundColor Yellow
        exit 1
    }
}

$excludeDirs = @(
    ".venv", "app", "langbot-src", "wechatpad-src",
    ".langbot-data", "langbot-data", ".uv-cache", ".cache", "logs", "build",
    "finetune_data", "dataset", "__pycache__", ".pytest_cache", ".ruff_cache", ".git",
    "*.egg-info"
)
$excludeFiles = @(
    "secrets.local.cmd", "cloud.jsonc", "finetune_state.json",
    "persona_system.md", "*.report.json", "scored.json", "*.zip", "*.tar.gz",
    # Obsolete one-off local scripts: keep them out of the public package
    "start-crlf.cmd", "start.cmd", "start-all.cmd", "run-all.ps1",
    "install-train.cmd", "train-cpu.cmd", "train-cpu.ps1"
)
$excludePaths = @("*\tools\_wechatpad_note.txt")

function Test-Excluded([System.IO.FileSystemInfo]$Item) {
    if ($Item.PSIsContainer) {
        foreach ($d in $excludeDirs) { if ($Item.Name -like $d) { return $true } }
        return $false
    }
    foreach ($f in $excludeFiles) {
        if ($Item.Name -like $f) { return $true }
    }
    foreach ($p in $excludePaths) {
        if ($Item.FullName -like $p) { return $true }
    }
    return $false
}

function Copy-Tree([string]$From, [string]$To) {
    New-Item -ItemType Directory -Path $To -Force | Out-Null
    foreach ($item in (Get-ChildItem $From -Force)) {
        if (Test-Excluded $item) { continue }
        $target = Join-Path $To $item.Name
        if ($item.PSIsContainer) {
            Copy-Tree $item.FullName $target
        } else {
            Copy-Item $item.FullName $target -Force
        }
    }
}

Write-Host "copying ..."
Copy-Tree $Root $Dest
if (-not (Test-Path $Dest)) {
    Write-Host "copy failed: $Dest was not created" -ForegroundColor Red
    exit 1
}

# Never ship these even if an exclusion was missed.
foreach ($name in @("secrets.local.cmd", "cloud.jsonc", "persona_system.md", "finetune_state.json")) {
    $p = Join-Path $Dest $name
    if (Test-Path $p) {
        Remove-Item $p -Force
        Write-Host "removed stray file: $name" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "scanning for personal data ..." -ForegroundColor Cyan
$textExt = @(".md", ".py", ".ps1", ".cmd", ".jsonc", ".json", ".toml", ".html", ".txt", ".yml", ".yaml", ".example")
$patterns = @{
    "API key" = 'sk-[A-Za-z0-9]{16,}'
    "wxid"    = 'wxid_[A-Za-z0-9]{8,}'
    "phone"   = '1[3-9]\d{9}'
}
$hits = @()
Get-ChildItem $Dest -Recurse -File | Where-Object { $textExt -contains $_.Extension.ToLower() } | ForEach-Object {
    $file = $_
    $content = Get-Content $file.FullName -Raw -Encoding UTF8
    foreach ($label in $patterns.Keys) {
        if ($content -match $patterns[$label]) {
            $hits += "$label -> $($file.FullName.Substring($Dest.Length + 1))"
        }
    }
}

if ($hits.Count -gt 0) {
    Write-Host ""
    Write-Host "potential personal data found:" -ForegroundColor Red
    $hits | Sort-Object -Unique | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
    Write-Host ""
    Write-Host "Inspect these before pushing. The folder was created anyway." -ForegroundColor Yellow
} else {
    Write-Host "clean: no api keys, wxids or phone numbers found." -ForegroundColor Green
}

$files = (Get-ChildItem $Dest -Recurse -File | Measure-Object).Count
$size = [math]::Round(((Get-ChildItem $Dest -Recurse -File | Measure-Object Length -Sum).Sum / 1KB), 1)
Write-Host ""
Write-Host "done: $files files, $size KB" -ForegroundColor Green
Write-Host ""
Write-Host "next:"
Write-Host "  cd `"$Dest`""
Write-Host "  git init"
Write-Host "  git add ."
Write-Host "  git status --short     # confirm no secrets"
Write-Host "  git commit -m `"feat: weclone-cloud`""
Write-Host "  git remote add origin https://github.com/<user>/<repo>.git"
Write-Host "  git push -u origin main"
