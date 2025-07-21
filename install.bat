@echo off
echo Installing Fashion Recommender System...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not in PATH.
    echo Please install Python 3.8+ and try again.
    pause
    exit /b 1
)

echo Python found. Installing requirements...
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo Error installing requirements. Trying with --user flag...
    pip install --user -r requirements.txt
)

echo.
echo Installation complete!
echo.
echo To run the application, execute:
echo streamlit run enhanced_fashion_app.py
echo.
pause
