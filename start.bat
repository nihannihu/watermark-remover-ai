@echo off
setlocal
cd /d "%~dp0"

:: Launch WatermarkRemover-AI GUI via run.bat
call "%~dp0run.bat" %*
