<#
.SYNOPSIS
    Clone and start WeChatPadPro (the WeChat protocol side) for LangBot.

.DESCRIPTION
    LangBot's WeChatPad adapter needs a running WeChatPad service. This script:
      * clones the simplified docker bundle (fdc310/WeChatPad-Docker)
      * starts it with docker compose (MySQL + Redis + WeChatPad)
      * prints the generated adminKey from the container logs

    The service is published on http://localhost:9090 (container port 8849).
    From inside the LangBot container it is http://host.docker.internal:9090.

    WARNING: personal WeChat protocol adapters can get the account banned.
    Use a spare account, and run it in the same region as the phone.
#>
param(
    [string]$Root = "",
    # Pinned to the commit this project was verified against. Use -Ref main to
    # track upstream, then re-run the checks in docs/maintenance.md.
    [string]$Ref = "c6d1b7a6a5997eeaaf8fcef801c03f70338d81b9"
)

$ErrorActionPreference = "Stop"
if (-not $Root) { $Root = Join-Path (Split-Path -Parent $PSScriptRoot) "wechatpad-src" }
$Root = [IO.Path]::GetFullPath($Root)

function Write-Step([string]$Text) { Write-Host ">> $Text" -ForegroundColor Cyan }

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "FAILED: docker not found. Start Docker Desktop first." -ForegroundColor Red
    exit 1
}
& docker info *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "FAILED: docker engine is not running." -ForegroundColor Red
    exit 1
}

if (-not (Test-Path (Join-Path $Root "docker-compose.yml")) -and -not (Test-Path (Join-Path $Root "docker-compose.yaml"))) {
    Write-Step "cloning WeChatPad-Docker"
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & git clone --depth 1 https://github.com/fdc310/WeChatPad-Docker $Root
        $cloned = $LASTEXITCODE -eq 0
        if ($cloned -and $Ref) {
            Write-Step "pinning to $Ref"
            & git -C $Root checkout --quiet $Ref
            if ($LASTEXITCODE -ne 0) {
                Write-Host "   !! commit not reachable in the shallow clone, staying on the default branch" -ForegroundColor Yellow
                Write-Host "   !! run: git -C `"$Root`" fetch --depth 50 origin && git -C `"$Root`" checkout $Ref"
            }
        }
    } finally {
        $ErrorActionPreference = $previous
    }
    if (-not (Test-Path (Join-Path $Root "docker-compose.yml")) -and -not (Test-Path (Join-Path $Root "docker-compose.yaml"))) {
        Write-Host "FAILED: git clone failed. Clone manually into $Root" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host ".. $Root already exists, keeping the current checkout" -ForegroundColor DarkGray
}

function Pull-Image([string]$Image, [int]$Tries = 4) {
    for ($i = 1; $i -le $Tries; $i++) {
        Write-Host "   pull $Image (attempt $i/$Tries)"
        & docker pull $Image
        if ($LASTEXITCODE -eq 0) { return $true }
        if ($i -lt $Tries) { Start-Sleep -Seconds (5 * $i) }
    }
    Write-Host "   !! could not pull $Image" -ForegroundColor Yellow
    return $false
}

# Docker Hub often drops large layers on flaky links; pull them one by one with
# retries so a single EOF does not abort the whole compose run.
Write-Step "pre-pulling base images (retries on network errors)"
Pull-Image "mysql:8.0" 4 | Out-Null
Pull-Image "redis:alpine" 4 | Out-Null
Pull-Image "alpine:latest" 4 | Out-Null

Write-Step "docker compose up -d"
$code = 1
for ($attempt = 1; $attempt -le 3; $attempt++) {
    Push-Location $Root
    try {
        & docker compose up -d
        $code = $LASTEXITCODE
    } finally {
        Pop-Location
    }
    if ($code -eq 0) { break }
    Write-Host "   compose attempt $attempt failed (exit $code), retrying..." -ForegroundColor Yellow
    Start-Sleep -Seconds 8
}
if ($code -ne 0) {
    Write-Host "FAILED: docker compose exited with $code" -ForegroundColor Red
    Write-Host "If it keeps failing on 'EOF' while pulling, configure a registry mirror:"
    Write-Host "  Docker Desktop -> Settings -> Docker Engine, add:"
    Write-Host '  "registry-mirrors": ["https://docker.m.daocloud.io"]'
    exit $code
}

Write-Step "waiting for the service to boot (20s)"
Start-Sleep -Seconds 20

Write-Host ""
Write-Step "admin key (from container logs)"
# Native programs write logs to stderr; with ErrorActionPreference=Stop
# PowerShell 5.1 turns that into a terminating NativeCommandError.
$previousPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
    $logs = (& docker logs wechatpad 2>&1 | Out-String)
} finally {
    $ErrorActionPreference = $previousPreference
}
$keyLine = $logs -split "`r?`n" | Select-String -Pattern "ADMIN_KEY|adminKey" | Select-Object -First 5
if ($keyLine) {
    $keyLine | ForEach-Object { Write-Host "  $_" -ForegroundColor Green }
} else {
    # The simplified bundle ships the key in app\.env instead of printing it.
    $envFile = Join-Path $Root "app\.env"
    if (Test-Path $envFile) {
        $line = Get-Content $envFile -Encoding UTF8 | Select-String -Pattern '^\s*ADMIN_KEY\s*=' | Select-Object -First 1
        if ($line) {
            Write-Host "  (from app\.env) $($line.Line.Trim())" -ForegroundColor Green
        } else {
            Write-Host "  not found; check $envFile" -ForegroundColor Yellow
        }
    } else {
        Write-Host "  not found; run: docker logs wechatpad 2>&1 | Select-String ADMIN_KEY" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Step "containers"
$previousPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
    & docker ps --filter "name=wechatpad" --filter "name=mysql_wxpad" --filter "name=redis_wxpad" --format "{{.Names}}  {{.Status}}  {{.Ports}}"
} finally {
    $ErrorActionPreference = $previousPreference
}

Write-Host ""
Write-Host "Next:" -ForegroundColor Yellow
Write-Host "  1. Open http://localhost:9090 (Swagger) and log in with the adminKey above"
Write-Host "  2. In LangBot: bot 'WeChatPad' -> adapter settings:"
Write-Host "       service URL : http://host.docker.internal:9090"
Write-Host "       admin key   : (the key above)"
Write-Host "  3. Save, enable, then scan the QR code with the spare WeChat account"
