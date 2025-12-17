@echo off
REM ============================================================================
REM ByNoemie RAG Chatbot - Quick Start Setup (Windows)
REM ============================================================================
REM Run this script after extracting the zip file:
REM   quickstart.bat
REM ============================================================================

echo ==============================================
echo   ByNoemie RAG Chatbot - Quick Start Setup
echo ==============================================
echo.

REM Check Python
echo 1. Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo X Python not found. Please install Python 3.8+
    pause
    exit /b 1
)
echo [OK] Python found
echo.

REM Create virtual environment
echo 2. Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo [OK] Virtual environment created
) else (
    echo [OK] Virtual environment already exists
)
echo.

REM Activate virtual environment
echo 3. Activating virtual environment...
call venv\Scripts\activate.bat
echo [OK] Virtual environment activated
echo.

REM Install dependencies
echo 4. Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt
echo [OK] Dependencies installed
echo.

REM Setup environment variables
echo 5. Setting up environment...
if not exist ".env" (
    copy .env.example .env
    echo [OK] Created .env file from template
    echo.
    echo *** IMPORTANT: Edit .env and add your API key! ***
    echo     Get FREE Groq API key at: https://console.groq.com
    echo.
) else (
    echo [OK] .env file already exists
)
echo.

REM Run check
echo 6. Checking configuration...
python env_setup.py --check
echo.

echo ==============================================
echo   Setup Complete!
echo ==============================================
echo.
echo Next steps:
echo.
echo   1. Add your API key to .env file:
echo      GROQ_API_KEY=gsk_xxxxxxxxxxxx
echo.
echo   2. Run the Streamlit demo:
echo      venv\Scripts\activate
echo      streamlit run app.py
echo.
echo   3. Or process products:
echo      python scripts\process_products.py --csv data\products\sample_products.csv
echo.
echo ==============================================
pause
