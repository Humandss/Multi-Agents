@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================
echo   NPC Dialogue Server
echo   (Ctrl+C to stop, close window to kill)
echo ============================================
echo.

python -m uv run python scripts/run_server.py

echo.
echo === server stopped ===
pause
