@echo off
rem Nightly job: download new results, lines, injuries and rosters from ESPN,
rem score yesterday's predictions, predict today, retrain weekly.
rem Schedule in Task Scheduler (see README). Output goes to data\update.log
cd /d "%~dp0"
python -m scripts.update_data >nul 2>&1
