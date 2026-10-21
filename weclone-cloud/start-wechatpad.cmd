@echo off
chcp 65001 >nul
title WeChatPadPro (WeChat protocol side)
cd /d "%~dp0"

echo ==========================================================
echo   WeChatPadPro for LangBot
echo   WARNING: personal WeChat protocol adapters risk a ban.
echo            Use a spare account and the same region as the phone.
echo ==========================================================
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\prepare_wechatpad.ps1" %*
echo.
echo Press any key to close...
pause >nul
