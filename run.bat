@echo off
setlocal
cd /d "%~dp0"

if exist "python\pythonw.exe" (
    start "" "python\pythonw.exe" remwmgui.py
) else if exist "python\python.exe" (
    start "" "python\python.exe" remwmgui.py
) else (
    echo [X] Local Python environment not found. Run setup.bat first.
    pause
)
