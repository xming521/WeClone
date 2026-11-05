@echo off
chcp 65001 >nul
title Install and start LangBot (Docker)
cd /d "%~dp0"

set "SRC=%CD%\langbot-src"
set "MIRROR="

rem Usage: start-langbot-docker.cmd --mirror
rem   Use docker.langbot.app instead of Docker Hub when auth.docker.io is blocked.
if /i "%~1"=="--mirror" set "MIRROR=-Mirror"
if /i "%~1"=="/mirror" set "MIRROR=-Mirror"

where docker >nul 2>nul
if errorlevel 1 (
  echo [ERROR] docker command not found.
  echo Open Docker Desktop once and make sure it finished starting.
  echo.
  pause
  exit /b 1
)

docker info >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Docker engine is not running.
  echo Start Docker Desktop, wait until the whale icon is green, then run this again.
  echo.
  pause
  exit /b 1
)

echo [1/3] fetching LangBot sources ...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\prepare_langbot.ps1" -Root "%SRC%" %MIRROR%
if errorlevel 1 (
  echo.
  echo [FAILED] could not download LangBot.
  pause
  exit /b 1
)

echo.
echo [1.5/3] patching compose for Windows paths ...
".venv\Scripts\python.exe" "%~dp0tools\fix_langbot_compose.py" "%SRC%\docker\docker-compose.yaml"
if errorlevel 1 (
  echo.
  echo [FAILED] could not patch docker-compose.yaml
  pause
  exit /b 1
)

echo.
echo [2/3] docker compose up (core + plugin runtime; Box sandbox skipped) ...
pushd "%SRC%\docker"
docker compose up -d
set "RC=%errorlevel%"
popd
if not "%RC%"=="0" (
  echo.
  echo [FAILED] docker compose exited with %RC%
  echo Try:  docker compose -f "%SRC%\docker\docker-compose.yaml" logs --tail 80
  pause
  exit /b 1
)

echo.
echo [3/3] container status ...
pushd "%SRC%\docker"
docker compose ps
popd

echo.
echo ==========================================================
echo   LangBot should now be reachable at http://localhost:5300
echo   First visit: create the admin account.
echo.
echo   Model provider settings:
echo     Base URL : http://host.docker.internal:8010/v1
echo     API Key  : the "serve key" printed by start-persona-docker.cmd
echo     Model    : deepseek-flash
echo     Tool use : OFF
echo.
echo   Keep start-persona-docker.cmd running in another window.
echo   Stop LangBot: double-click stop-langbot-docker.cmd
echo ==========================================================
echo.
echo Press any key to close...
pause >nul
