@echo off
title PhishShift Dashboard
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo First run setup.bat once to install the requirements.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run app.py
