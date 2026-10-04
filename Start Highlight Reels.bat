@echo off
rem Starts the Highlight Reels app and opens it in your browser.
rem If it's already running, it just opens the browser.
rem The app keeps running in a minimised window called "Highlight Reels": closing that window stops it.
cd /d "%~dp0"
powershell -NoProfile -Command "try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 http://127.0.0.1:8765/api/reel/status | Out-Null; exit 0 } catch { exit 1 }"
if %errorlevel%==0 (
  start "" http://127.0.0.1:8765/
  exit /b
)
start "Highlight Reels (closing this window stops the app)" /min ".venv\Scripts\python.exe" -m hlv review
