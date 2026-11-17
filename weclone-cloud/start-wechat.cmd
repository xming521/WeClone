@echo off
chcp 65001 >nul
title weclone-cloud - wechat stack
cd /d "%~dp0"

rem ASCII-only: cmd parses .cmd with the OEM codepage, so non-ASCII text inside
rem parenthesised blocks can split lines and break the script.

echo ==========================================================
echo   weclone-cloud  wechat stack
echo   order: persona -^> WeChatPad -^> LangBot
echo ==========================================================
echo.

where docker >nul 2>nul
if errorlevel 1 (
  echo [ERROR] docker not found. Install and start Docker Desktop first.
  echo.
  pause
  exit /b 1
)
docker info >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Docker engine is not ready. Wait until the whale icon is green.
  echo.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] dependencies missing. Run setup.cmd first.
  pause
  exit /b 1
)

echo [1/4] starting persona in a new window (keep it open) ...
start "weclone persona" cmd /k "%~dp0start-persona-docker.cmd"
timeout /t 8 >nul

echo [2/4] starting WeChatPad ...
if not exist "wechatpad-src\docker-compose.yml" (
  echo        first run: cloning WeChatPadPro ...
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\prepare_wechatpad.ps1"
) else (
  pushd "wechatpad-src"
  docker compose up -d
  popd
)

echo [3/4] starting LangBot ...
if not exist "langbot-src\docker\docker-compose.yaml" (
  echo        first run: cloning LangBot ...
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\prepare_langbot.ps1"
)
if exist "langbot-src\docker\docker-compose.yaml" (
  ".venv\Scripts\python.exe" "%~dp0tools\fix_langbot_compose.py" "langbot-src\docker\docker-compose.yaml" >nul 2>nul
  pushd "langbot-src\docker"
  docker compose up -d
  popd
)

echo [4/4] status ...
echo.
docker ps --format "{{.Names}}  {{.Status}}"
echo.
echo ==========================================================
echo   persona : http://127.0.0.1:8010/
echo   LangBot : http://localhost:5300
echo   WeChat login: see docs\runbook.md section 3
echo.
echo   First run still needs the model + WeChatPad adapter configured
echo   inside LangBot; see docs\install.md sections 5 and 6.
echo ==========================================================
echo.
pause
