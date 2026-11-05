@echo off
chcp 65001 >nul
title Check LangBot -> WeClone connection
cd /d "%~dp0"

echo ==========================================================
echo  [1] Is anything listening on 8010, and on which address?
echo      (need a row starting with 0.0.0.0; 127.0.0.1 will NOT work for Docker)
echo ==========================================================
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 8010 -State Listen -ErrorAction SilentlyContinue | Select-Object LocalAddress,LocalPort,OwningProcess | Format-Table -AutoSize"
if errorlevel 1 echo   (query failed)

echo.
echo ==========================================================
echo  [2] Can the LangBot container reach the host?
echo ==========================================================
docker exec langbot python -c "import urllib.request;print(urllib.request.urlopen('http://host.docker.internal:8010/healthz',timeout=5).read().decode())"
if errorlevel 1 (
  echo   FAILED - see the table above: if 8010 is missing, start start-persona-docker.cmd
  echo   if it shows 127.0.0.1 only, you started the wrong window
)

echo.
echo ==========================================================
echo  [3] LangBot containers
echo ==========================================================
docker ps --filter name=langbot --format "{{.Names}}  {{.Status}}  {{.Ports}}"

echo.
echo Copy everything above and send it back.
echo.
pause
