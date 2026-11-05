@echo off
chcp 65001 >nul
title weclone-cloud - setup
cd /d "%~dp0"

echo ==========================================================
echo   weclone-cloud  第一次安装
echo   - 检查 Python 3.12 / uv / git
echo   - 拉取 WeClone 源码到 app\
echo   - 创建 .venv 并安装依赖（约 300MB，5-10 分钟）
echo ==========================================================
echo.

where git >nul 2>nul
if errorlevel 1 (
  echo [错误] 没找到 git，请先安装： https://git-scm.com/download/win
  echo.
  pause
  exit /b 1
)

where uv >nul 2>nul
if errorlevel 1 (
  echo [提示] 没找到 uv，将回退到 python -m venv + pip 安装（会慢一些）
  echo        推荐安装 uv： powershell -c "irm https://astral.sh/uv/install.ps1 ^| iex"
  echo.
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0bootstrap.ps1" -Install
if errorlevel 1 (
  echo.
  echo [失败] 安装没有完成，请把上面的报错发到 issue。
  pause
  exit /b 1
)

if not exist "secrets.local.cmd" (
  copy /y "secrets.example.cmd" "secrets.local.cmd" >nul
  echo.
  echo [下一步] 已生成 secrets.local.cmd，请用记事本打开，
  echo          把 WECLONE_API_KEY 换成你自己的 DeepSeek key。
  notepad "secrets.local.cmd"
)

echo.
echo ==========================================================
echo   安装完成
echo.
echo   1) 编辑 secrets.local.cmd 填入 DeepSeek key
echo   2) 双击 start-chat.cmd    只用网页聊天
echo   3) 双击 start-wechat.cmd  接入微信（需要 Docker Desktop）
echo ==========================================================
echo.
pause
