@echo off
chcp 65001 >nul
title WeClone persona (no LangBot)
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] .venv not found. Run start.cmd once first.
  echo.
  pause
  exit /b 1
)

if exist "%~dp0secrets.local.cmd" call "%~dp0secrets.local.cmd"

echo ==========================================================
echo   WeClone persona API
echo   chat UI : http://127.0.0.1:8010/
echo   API     : http://127.0.0.1:8010/v1
echo   close this window to stop
echo ==========================================================
echo.

".venv\Scripts\python.exe" "tools\launch.py" --no-langbot %*

echo.
echo Stopped. Press any key to close...
pause >nul
