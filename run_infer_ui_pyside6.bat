@echo off
setlocal
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python infer_ui_pyside6.py
endlocal
