@echo off
chcp 65001 >nul
title Stop WeChatPadPro
cd /d "%~dp0"

if not exist "wechatpad-src" (
  echo Nothing to stop.
  pause
  exit /b 0
)

pushd "wechatpad-src"
docker compose down
popd

echo.
echo Stopped. Press any key to close...
pause >nul
