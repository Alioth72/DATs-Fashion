@echo off
echo Starting Fashion Recommender System...
echo.

REM Check if streamlit is installed
streamlit --version >nul 2>&1
if errorlevel 1 (
    echo Streamlit is not installed. Running installation first...
    call install.bat
)

echo Starting the application...
echo.
echo The application will open in your default web browser.
echo If it doesn't open automatically, navigate to: http://localhost:8501
echo.
echo Press Ctrl+C to stop the application.
echo.

streamlit run enhanced_fashion_app.py
