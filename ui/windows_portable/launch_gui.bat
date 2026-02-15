@echo off
setlocal
cd /d "%~dp0"
set "APP_ROOT=%CD%"
set "NODE_DIR=%APP_ROOT%\node\node-v20.19.0-win-x64"
if exist "%NODE_DIR%\node.exe" set "PATH=%NODE_DIR%;%PATH%"
set "PYTHONHOME=%APP_ROOT%\python"
set "PYTHONUTF8=1"

powershell -NoProfile -ExecutionPolicy Bypass -File "%APP_ROOT%\converter_gui.ps1"
endlocal
