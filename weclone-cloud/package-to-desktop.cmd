@echo off
chcp 65001 >nul
title Package weclone-cloud for GitHub
cd /d "%~dp0"

echo ==========================================================
echo   Building a clean copy on your Desktop
echo   - includes: source, tools, docs, launcher scripts
echo   - excludes: secrets, datasets, persona, .venv, app, caches
echo ==========================================================
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\package_for_github.ps1" -Force
echo.
echo Press any key to close...
pause >nul
