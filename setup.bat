@echo off
title PhishShift Setup
cd /d "%~dp0"
echo.
echo  ==========================================
echo    PhishShift - Liquid Glass Dashboard
echo  ==========================================
echo.
where py >nul 2>nul
if %errorlevel%==0 (
  set PY=py
) else (
  set PY=python
)
if not exist ".venv\Scripts\python.exe" (
  echo [1/4] Creating virtual environment...
  %PY% -m venv .venv
  if errorlevel 1 goto error
)
echo [2/4] Installing packages...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto error
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto error
echo [3/4] Preparing dataset and models...
".venv\Scripts\python.exe" -c "from pathlib import Path; from src.pipeline import load_dataset,train_models; p=Path('data/raw/messages.csv'); d=load_dataset(p); m,e,t=train_models(d,'models','reports'); print(e.to_string(index=False))"
if errorlevel 1 goto error
echo [4/4] Starting dashboard...
".venv\Scripts\python.exe" -m streamlit run app.py
goto end
:error
echo.
echo Setup failed. Read the error above. Check Python installation and internet access.
pause
:end
