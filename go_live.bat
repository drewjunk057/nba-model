@echo off
title NBA Predictor - switch to real data
cd /d "%~dp0"
echo This downloads five seasons of real NBA games from ESPN (about 6,000 box scores),
echo replaces the demo data, and retrains the models. It takes 10-25 minutes.
echo Progress is also written to data\update.log
echo.
python -c "import lightgbm, fastapi, uvicorn, sklearn, scipy, pandas" 2>nul || python -m pip install -r requirements.txt
python -m scripts.update_data --go-live
if errorlevel 1 (
  echo.
  echo Something went wrong. The last lines of data\update.log explain what.
  pause
  exit /b 1
)
echo.
echo Done. Starting the app...
call "%~dp0start.bat"
