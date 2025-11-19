@echo off
REM Quick start script for Python Build Agent

echo ╔════════════════════════════════════════════════╗
echo ║     Python Build Agent - Quick Start          ║
echo ╚════════════════════════════════════════════════╝
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found! Please install Python 3.8+
    pause
    exit /b 1
)

REM Check if .env exists
if not exist .env (
    echo Error: .env file not found!
    echo.
    echo Please create .env file from .env.example:
    echo   1. Copy .env.example to .env
    echo   2. Add your ANTHROPIC_API_KEY or OPENAI_API_KEY
    echo.
    pause
    exit /b 1
)

REM Install dependencies if needed
if not exist venv\ (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing dependencies...
pip install -q -r requirements.txt

echo.
echo Starting Build Agent...
echo.

python main.py %*
