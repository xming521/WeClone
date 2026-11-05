<#
.SYNOPSIS
    Download the LangBot repository (official Docker Hub images by default).

.DESCRIPTION
    * Uses git when available, otherwise downloads the GitHub zip (master/main).
    * Official images are kept unless -Mirror is given.
    * Idempotent: safe to run again.

.PARAMETER Root
    Target directory. Default: <project>\langbot-src

.PARAMETER Mirror
    Rewrite image names to docker.langbot.app (only useful in mainland China).
#>
param(
    [string]$Root = "",
    [switch]$Mirror
)

$ErrorActionPreference = "Stop"
if (-not $Root) { $Root = Join-Path (Split-Path -Parent $PSScriptRoot) "langbot-src" }
$Root = [IO.Path]::GetFullPath($Root)
$compose = Join-Path $Root "docker\docker-compose.yaml"

function Write-Step([string]$Text) { Write-Host ">> $Text" -ForegroundColor Cyan }

if (-not (Test-Path $compose)) {
    $git = Get-Command git -ErrorAction SilentlyContinue
    if ($git) {
        Write-Step "git clone LangBot"
        & git clone --depth 1 https://github.com/langbot-app/LangBot $Root
        if ($LASTEXITCODE -ne 0) { Write-Host "!! git clone failed, falling back to zip" -ForegroundColor Yellow }
    }
}

if (-not (Test-Path $compose)) {
    $zip = Join-Path $env:TEMP "langbot-src.zip"
    $staging = "$Root-download"
    foreach ($branch in @("master", "main")) {
        try {
            Write-Step "download $branch zip"
            Invoke-WebRequest -Uri "https://codeload.github.com/langbot-app/LangBot/zip/refs/heads/$branch" -OutFile $zip -UseBasicParsing
            if (Test-Path $staging) { Remove-Item $staging -Recurse -Force }
            Expand-Archive -Path $zip -DestinationPath $staging -Force
            $inner = Get-ChildItem $staging -Directory | Select-Object -First 1
            if (-not $inner) { throw "archive has no folder" }
            if (Test-Path $Root) { Remove-Item $Root -Recurse -Force }
            Move-Item $inner.FullName $Root
            break
        } catch {
            Write-Host "!! $branch failed: $_" -ForegroundColor Yellow
        }
    }
}

if (-not (Test-Path $compose)) {
    Write-Host "FAILED: could not obtain LangBot sources." -ForegroundColor Red
    Write-Host "Clone it manually, then run this script again:"
    Write-Host "  git clone https://github.com/langbot-app/LangBot `"$Root`""
    exit 1
}

if ($Mirror) {
    $text = [IO.File]::ReadAllText($compose)
    $new = $text.Replace("rockchin/langbot", "docker.langbot.app/langbot-public/rockchin/langbot")
    if ($new -ne $text) {
        [IO.File]::WriteAllText($compose, $new, (New-Object System.Text.UTF8Encoding($false)))
        Write-Step "mirror applied (docker.langbot.app)"
    } else {
        Write-Host ".. mirror already applied" -ForegroundColor DarkGray
    }
} else {
    Write-Host ".. using official Docker Hub images" -ForegroundColor DarkGray
}

Write-Host "OK  $compose" -ForegroundColor Green
