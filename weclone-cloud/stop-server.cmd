@echo off
chcp 65001 >nul
title Stop WeClone server
cd /d "%~dp0"

set "PID="
if exist "logs\serve.pid" set /p PID=<"logs\serve.pid"

if defined PID (
  echo Stopping weclone server ^(PID %PID%^) ...
  taskkill /PID %PID% /F >nul 2>nul
  if errorlevel 1 (
    echo   process not running anymore.
  ) else (
    echo   OK  stopped.
  )
  del /q "logs\serve.pid" >nul 2>nul
) else (
  echo No logs\serve.pid found ^(server may not be running in background^).
)

echo.
echo Checking port 8010 ...
set "FOUND="
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8010" ^| findstr "LISTENING"') do (
  echo   killing PID %%p on port 8010
  taskkill /PID %%p /F >nul 2>nul
  set "FOUND=1"
)
if not defined FOUND echo   nothing is listening on 8010.

echo.
echo Done. Press any key to close...
pause >nul
