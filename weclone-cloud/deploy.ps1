<#
.SYNOPSIS
    Start the weclone-cloud server locally (foreground or background) with logs.

.DESCRIPTION
    ASCII-only on purpose (PowerShell 5.1 ANSI parsing of BOM-less files).

.EXAMPLE
    .\deploy.ps1 -Persona persona_system.md
    .\deploy.ps1 -Port 8010 -Persona persona_system.md -Background
    .\deploy.ps1 -BindHost 0.0.0.0 -Port 8010 -Background -OpenFirewall
#>
[CmdletBinding()]
param(
    [string]$BindHost = "127.0.0.1",
    [int]$Port = 8010,
    [string]$Persona = "",
    [string]$Model = "",
    [switch]$Background,
    [switch]$OpenFirewall
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location $Root

function Die([string]$Text) {
    Write-Host ""
    Write-Host "   FAILED: $Text" -ForegroundColor Red
    exit 1
}

$exe = Join-Path $Root ".venv\Scripts\weclone-cloud.exe"
if (-not (Test-Path $exe)) { Die "$exe not found; run .\bootstrap.ps1 -Install first" }
if (-not (Test-Path (Join-Path $Root "cloud.jsonc"))) { Die "cloud.jsonc missing; copy cloud.example.jsonc" }

if ($Persona) {
    if (-not (Test-Path $Persona)) { Die "persona file not found: $Persona (run weclone-cloud persona first)" }
    $Persona = (Resolve-Path $Persona).Path
}

$logs = Join-Path $Root "logs"
New-Item -ItemType Directory -Path $logs -Force | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$logFile = Join-Path $logs "serve-$stamp.log"

$cliArgs = @("serve", "--host", $BindHost, "--port", $Port)
if ($Persona) { $cliArgs += @("--persona", $Persona) }
if ($Model) { $cliArgs += @("--model", $Model) }

$lanIp = ""
$addr = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
    Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254.*" } |
    Select-Object -First 1
if ($addr) { $lanIp = $addr.IPAddress }

Write-Host ">> weclone-cloud $($cliArgs -join ' ')" -ForegroundColor Cyan
Write-Host "   local: http://127.0.0.1:$Port/v1/chat/completions"
if ($BindHost -eq "0.0.0.0" -and $lanIp) {
    Write-Host "   LAN:   http://$($lanIp):$Port/v1/chat/completions" -ForegroundColor Green
}
Write-Host "   log:   $logFile"

if ($OpenFirewall) {
    $isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)
    if (-not $isAdmin) {
        Write-Host "   !!  -OpenFirewall needs an elevated shell, skipped" -ForegroundColor Yellow
    } else {
        $ruleName = "weclone-$Port"
        if (-not (Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue)) {
            New-NetFirewallRule -DisplayName $ruleName -Direction Inbound -Protocol TCP `
                -LocalPort $Port -Action Allow -Profile Private | Out-Null
            Write-Host "   OK  firewall rule $ruleName added (private profile)" -ForegroundColor Green
        } else {
            Write-Host "   ..  firewall rule $ruleName already exists" -ForegroundColor DarkGray
        }
    }
}

if ($Background) {
    $proc = Start-Process -FilePath $exe -ArgumentList $cliArgs -WorkingDirectory $Root `
        -RedirectStandardOutput $logFile -RedirectStandardError "$logFile.err" `
        -WindowStyle Hidden -PassThru
    $proc.Id | Set-Content (Join-Path $logs "serve.pid")
    Start-Sleep -Seconds 3
    if ($proc.HasExited) {
        Write-Host "   FAILED to start; last lines:" -ForegroundColor Red
        Get-Content "$logFile.err" -Tail 30 -ErrorAction SilentlyContinue
        Get-Content $logFile -Tail 30 -ErrorAction SilentlyContinue
        exit 1
    }
    Write-Host "   OK  running in background, PID=$($proc.Id)" -ForegroundColor Green
    Write-Host "   stop: Stop-Process -Id (Get-Content logs\serve.pid)"
    Write-Host "   check: curl http://127.0.0.1:$Port/healthz"
} else {
    Write-Host "   (Ctrl+C to stop)" -ForegroundColor DarkGray
    & $exe @cliArgs 2>&1 | Tee-Object -FilePath $logFile
}
