@echo off
setlocal
title Publish NBA Predictor to GitHub
cd /d "%~dp0"
set "REPO=https://github.com/drewjunk057/nba-model.git"

where git >nul 2>nul
if errorlevel 1 (
  echo Git is not installed. Get it from https://git-scm.com/download/win and run this again.
  pause
  exit /b 1
)

rem --- Refresh the starter data your friend's first run uses ---
echo Packing the current models and game data into seed\ ...
python -c "import json,shutil,pathlib as P; s=P.Path('seed'); [shutil.rmtree(s/d,ignore_errors=True) for d in ('raw','models')]; (s/'raw').mkdir(parents=True); [shutil.copy2(f,s/'raw'/f.name) for f in P.Path('data/raw').glob('*.json.gz')]; v=json.loads(P.Path('models/current.json').read_text())['model_version']; shutil.copytree(P.Path('models')/v, s/'models'/v); shutil.copy2('models/current.json', s/'models'/'current.json'); print('  model', v)"
if errorlevel 1 goto :fail

if not exist ".git" (
  git init -b main
  git remote add origin %REPO%
)
git config user.name >nul 2>nul || git config user.name "PBS"
git config user.email >nul 2>nul || git config user.email "drewjunk057@gmail.com"

git add -A
git commit -m "Update NBA Predictor" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01CTDBkTT7bM3LkUXCnmxoLC"
git push -u origin main
if errorlevel 1 goto :pushfail

echo.
echo Published to https://github.com/drewjunk057/nba-model
pause
exit /b 0

:pushfail
echo.
echo The push didn't go through. If the repository doesn't exist yet:
echo   1. Go to https://github.com/new
echo   2. Name it nba-model, choose Private, and leave every box unchecked
echo   3. Click Create repository, then run this file again
echo If a sign-in window appeared, finish signing in and run this again.
pause
exit /b 1

:fail
echo Packing the starter data failed. Run start.bat once first so the models exist.
pause
exit /b 1
