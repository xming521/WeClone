@echo off
chcp 65001 >nul
title WeClone 数字分身
cd /d "%~dp0"

if not exist "persona_system.md" (
  echo 还没生成人设文件 persona_system.md
  echo 请先运行 start.cmd
  echo.
  pause
  exit /b 1
)

echo 启动服务…（首次启动约 5 秒）
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0deploy.ps1" -Persona "persona_system.md" -Port 8010 -Background
if %errorlevel% neq 0 (
  echo 启动失败，检查上面的输出。
  pause
  exit /b 1
)

start "" "http://127.0.0.1:8010/"
echo.
echo 服务地址：http://127.0.0.1:8010/
echo 停止服务：Stop-Process -Id (Get-Content logs\serve.pid)
echo 日志目录：logs\
echo.
echo 按任意键关闭本窗口（服务继续在后台运行）…
pause >nul
