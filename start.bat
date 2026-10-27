@echo off
setlocal
title NBA Predictor
cd /d "%~dp0"

rem --- Find Python: prefer "python", fall back to "py" ---
set "PY=python"
where python >nul 2>nul
if errorlevel 1 (
  where py >nul 2>nul
  if errorlevel 1 (
    echo.
    echo Python was not found. Install Python 3.10 or newer from https://www.python.org/downloads/
    echo Check "Add Python to PATH" during the install, then run this again.
    echo.
    pause
    exit /b 1
  )
  set "PY=py"
)

rem --- Install packages the first time (or if any are missing) ---
%PY% -c "import lightgbm, fastapi, uvicorn, sklearn, scipy, pandas" >nul 2>nul
if errorlevel 1 (
  echo Installing required packages the first time. This takes a few minutes...
  %PY% -m pip install -r requirements.txt
  if errorlevel 1 goto :fail
)

rem --- Starter data on first run, then refresh games, lines and injuries ---
%PY% -m scripts.bootstrap
if errorlevel 1 goto :fail

rem --- Close an older copy of the app that is still running on port 8000 ---
for /f "tokens=5" %%p in ('netstat -ano ^| findstr /R /C:":8000 .*LISTENING"') do (
  tasklist /FI "PID eq %%p" | findstr /I "python" >nul && (
    echo Closing an older NBA Predictor that was still running...
    taskkill /PID %%p /F >nul
  )
)
echo.
echo Starting NBA Predictor at http://localhost:8000
echo Keep this window open while you use the app. Close it to stop.
echo.
start "" cmd /c "timeout /t 6 >nul & start http://localhost:8000"
%PY% -m uvicorn app.api:app --port 8000 --reload --reload-dir app
goto :eof

:fail
echo.
echo Setup failed. Scroll up, or open data\update.log, for the error.
pause
exit /b 1
