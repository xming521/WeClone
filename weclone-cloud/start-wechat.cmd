@echo off
chcp 65001 >nul
title weclone-cloud - wechat stack
cd /d "%~dp0"

echo ==========================================================
echo   weclone-cloud  接入微信
echo   顺序：分身 -^> 微信协议端 -^> LangBot
echo ==========================================================
echo.

where docker >nul 2>nul
if errorlevel 1 (
  echo [错误] 没找到 docker，请先安装 Docker Desktop 并启动它。
  echo.
  pause
  exit /b 1
)
docker info >nul 2>nul
if errorlevel 1 (
  echo [错误] Docker 引擎还没就绪，请等托盘图标变绿后重试。
  echo.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo [错误] 还没安装依赖，请先双击 setup.cmd
  pause
  exit /b 1
)

echo [1/4] 启动分身（新窗口，必须保持打开）...
start "weclone persona" cmd /k "%~dp0start-persona-docker.cmd"
timeout /t 8 >nul

echo [2/4] 启动微信协议端 ...
if not exist "wechatpad-src\docker-compose.yml" (
  echo       首次运行，需要克隆 WeChatPadPro ...
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\prepare_wechatpad.ps1"
) else (
  pushd "wechatpad-src"
  docker compose up -d
  popd
)

echo [3/4] 启动 LangBot ...
if not exist "langbot-src\docker\docker-compose.yaml" (
  echo       首次运行，需要克隆 LangBot ...
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\prepare_langbot.ps1"
)
if exist "langbot-src\docker\docker-compose.yaml" (
  ".venv\Scripts\python.exe" "%~dp0tools\fix_langbot_compose.py" "langbot-src\docker\docker-compose.yaml" >nul 2>nul
  pushd "langbot-src\docker"
  docker compose up -d
  popd
)

echo [4/4] 状态 ...
echo.
docker ps --format "{{.Names}}  {{.Status}}"
echo.
echo ==========================================================
echo   分身 : http://127.0.0.1:8010/
echo   LangBot : http://localhost:5300
echo   微信扫码 : 见 docs\runbook.md 第三节
echo.
echo   首次使用还需要在 LangBot 里配置模型和微信适配器，
echo   步骤见 docs\install.md 第 5、6 节。
echo ==========================================================
echo.
pause
