@echo off
chcp 65001 >nul
title weclone-cloud - chat
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [错误] 还没安装依赖，请先双击 setup.cmd
  echo.
  pause
  exit /b 1
)

if not exist "secrets.local.cmd" (
  echo [错误] 缺少 secrets.local.cmd
  echo        请复制 secrets.example.cmd 为 secrets.local.cmd，并填入 DeepSeek key
  echo.
  pause
  exit /b 1
)

echo 启动中... 浏览器会自动打开 http://127.0.0.1:8010/
echo 关闭本窗口即停止服务。
echo.
start "" /b cmd /c "timeout /t 4 >nul & start http://127.0.0.1:8010/"

call "%~dp0start-persona.cmd"

pause >nul
