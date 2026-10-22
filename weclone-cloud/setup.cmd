@echo off
chcp 65001 >nul
title weclone-cloud - setup
cd /d "%~dp0"

rem ASCII-only on purpose: Windows cmd reads .cmd files with the OEM codepage,
rem so non-ASCII lines can be split and break parsing.

echo ==========================================================
echo   weclone-cloud  first-time setup
echo   - check Python 3.12 / uv / git
echo   - fetch WeClone sources into app\
echo   - create .venv and install deps (about 300 MB, 5-10 min)
echo ==========================================================
echo.

where git >nul 2>nul
if errorlevel 1 (
  echo [ERROR] git not found: https://git-scm.com/download/win
  echo.
  pause
  exit /b 1
)

where uv >nul 2>nul
if errorlevel 1 (
  echo [INFO] uv not found, falling back to python -m venv + pip
  echo        install uv for speed: powershell -c "irm https://astral.sh/uv/install.ps1 ^| iex"
  echo.
)

set "MIRROR="
if /i "%~1"=="--aliyun" set "MIRROR=https://mirrors.aliyun.com/pypi/simple"
if /i "%~1"=="--pypi"   set "MIRROR=https://pypi.org/simple"

if defined MIRROR (
  echo using mirror: %MIRROR%
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0bootstrap.ps1" -Install -Mirror "%MIRROR%"
) else (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0bootstrap.ps1" -Install
)
if errorlevel 1 (
  echo.
  echo [FAILED] setup did not finish.
  echo   network / mirror problem? try one of:
  echo     setup.cmd --aliyun
  echo     setup.cmd --pypi
  echo.
  pause
  exit /b 1
)

if not exist "secrets.local.cmd" (
  copy /y "secrets.example.cmd" "secrets.local.cmd" >nul
  echo.
  echo [NEXT] secrets.local.cmd created; open it and set WECLONE_API_KEY.
  notepad "secrets.local.cmd"
)

echo.
echo ==========================================================
echo   setup finished
echo.
echo   1) edit secrets.local.cmd  -^> WECLONE_API_KEY
echo   2) double-click start-chat.cmd    (web chat only)
echo   3) double-click start-wechat.cmd  (WeChat, needs Docker Desktop)
echo ==========================================================
echo.
pause
