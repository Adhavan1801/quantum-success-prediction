@echo off
echo ============================================
echo  Quantum ML Success Prediction - Setup
echo ============================================
echo.

REM --- Check Python is available ---
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.9+ and try again.
    pause
    exit /b 1
)

REM --- Create virtual environment ---
echo [1/4] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)

REM --- Activate virtual environment ---
echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat

REM --- Upgrade pip ---
echo [3/4] Upgrading pip...
python -m pip install --upgrade pip

REM --- Install dependencies ---
echo [4/4] Installing project dependencies...
pip install -r requirements.txt

echo.
echo ============================================
echo  Setup Complete!
echo ============================================
echo.
echo To activate the environment, run:
echo   venv\Scripts\activate.bat
echo.
echo To start Jupyter, run:
echo   jupyter notebook
echo.
pause
