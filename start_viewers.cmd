@echo off
setlocal
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
where uv >nul 2>nul
if not errorlevel 1 (
    uv run --offline --no-sync --project "%~dp0ultralytics" python -u -X utf8 "%~dp0launch_viewers.py" %*
) else if exist "%~dp0ultralytics\.venv\Scripts\python.exe" (
    "%~dp0ultralytics\.venv\Scripts\python.exe" -u -X utf8 "%~dp0launch_viewers.py" %*
) else (
    echo Cannot find uv or the existing project virtual environment.
    echo Please run this launcher from the original project environment.
    pause
    exit /b 1
)
if errorlevel 1 pause
