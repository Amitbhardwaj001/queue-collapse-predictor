@echo off
rem One-click launcher: sets up on first run, starts the dashboard and opens it in the browser.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Setting up for the first time - this needs internet once...
    python -m venv .venv
    .venv\Scripts\python.exe -m pip install -r requirements.txt
)
if not exist "models\calls_models.joblib" (
    echo Training the prediction models for all four domains - a few minutes, only once...
    .venv\Scripts\python.exe -m queue_predictor.train
)
start "" cmd /c "timeout /t 8 >nul & start http://127.0.0.1:5000"
echo Dashboard starting at http://127.0.0.1:5000  (close this window to stop it)
.venv\Scripts\python.exe run.py
pause
