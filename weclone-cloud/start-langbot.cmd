@echo off
chcp 65001 >nul
title LangBot (WeClone persona provider)
cd /d "%~dp0"

if not exist "%LOCALAPPDATA%\weclone-langbot" mkdir "%LOCALAPPDATA%\weclone-langbot"

set "UVEXE="
where uv >nul 2>nul && set "UVEXE=uv"
if not defined UVEXE if exist ".venv\Scripts\uv.exe" set "UVEXE=%~dp0.venv\Scripts\uv.exe"

if not defined UVEXE (
  echo [ERROR] uv not found.
  echo Install it once, then run this file again:
  echo    C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe -m pip install uv
  echo.
  pause
  exit /b 1
)

echo ==========================================================
echo   Starting LangBot (first run downloads dependencies)
echo   Web UI:    http://localhost:5300
echo   Data dir:  %LOCALAPPDATA%\weclone-langbot
echo   Keep this window open. Ctrl+C to stop.
echo ==========================================================
echo.

cd /d "%LOCALAPPDATA%\weclone-langbot"
"%UVEXE%" tool run langbot@latest

echo.
echo LangBot stopped. Press any key to close...
pause >nul
