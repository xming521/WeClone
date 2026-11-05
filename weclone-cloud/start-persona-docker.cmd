@echo off
chcp 65001 >nul
title WeClone persona (LAN / Docker access)
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] .venv not found. Run start.cmd once first.
  echo.
  pause
  exit /b 1
)

if exist "%~dp0secrets.local.cmd" call "%~dp0secrets.local.cmd"

echo ==========================================================
echo   WeClone persona API, listening on 0.0.0.0:8010
echo   Needed when LangBot runs in Docker (host.docker.internal).
echo   Keep this window open while using the bot.
echo ==========================================================
echo.

".venv\Scripts\python.exe" "tools\launch.py" --no-langbot --host 0.0.0.0 %*

echo.
echo Stopped. Press any key to close...
pause >nul
