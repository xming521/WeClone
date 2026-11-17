@echo off
chcp 65001 >nul
title weclone-cloud - chat
cd /d "%~dp0"

rem ASCII-only: cmd parses .cmd with the OEM codepage, non-ASCII inside
rem parenthesised blocks can split lines and break the script.

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] dependencies missing. Run setup.cmd first.
  echo.
  pause
  exit /b 1
)

if not exist "secrets.local.cmd" (
  echo [ERROR] secrets.local.cmd missing.
  echo         copy secrets.example.cmd to secrets.local.cmd and set WECLONE_API_KEY
  echo.
  pause
  exit /b 1
)

echo starting... the browser will open http://127.0.0.1:8010/
echo close this window to stop the service.
echo.
start "" /b cmd /c "timeout /t 4 >nul & start http://127.0.0.1:8010/"

call "%~dp0start-persona.cmd"

pause >nul
