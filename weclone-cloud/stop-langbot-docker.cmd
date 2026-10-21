@echo off
chcp 65001 >nul
title Stop LangBot (Docker)
cd /d "%~dp0"

set "SRC=%CD%\langbot-src"

if not exist "%SRC%\docker\docker-compose.yaml" (
  echo LangBot sources not found, nothing to stop.
  echo.
  pause
  exit /b 0
)

where docker >nul 2>nul
if errorlevel 1 (
  echo [ERROR] docker command not found.
  pause
  exit /b 1
)

echo Stopping LangBot containers ...
pushd "%SRC%\docker"
docker compose down
popd

echo.
echo Done. Press any key to close...
pause >nul
