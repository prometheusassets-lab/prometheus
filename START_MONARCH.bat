@echo off
title MONARCH PRO
color 06
cls
cd /d "%~dp0"

echo.
echo  ====================================================
echo   MONARCH PRO — EQUITY TERMINAL  [port 8001]
echo  ====================================================
echo.
echo   Pages:
echo     Screener     ^> http://127.0.0.1:8001/
echo     Options      ^> http://127.0.0.1:8001/options
echo     Fundamentals ^> http://127.0.0.1:8001/fundamentals
echo     ML Predictor ^> http://127.0.0.1:8001/ml
echo.

python --version >nul 2>&1
if errorlevel 1 ( echo [ERROR] Python not found. & pause & exit /b 1 )

echo   Checking dependencies...
pip install -q fastapi "uvicorn[standard]" pandas numpy requests yfinance scikit-learn scipy >nul 2>&1
echo   Dependencies OK.
echo.
echo   Starting server on http://127.0.0.1:8001 ...
echo   Browser opens in 5 seconds.
echo.

start /B cmd /c "timeout /t 5 /nobreak >nul && start http://127.0.0.1:8001/login"

python main.py

echo.
echo  MONARCH PRO stopped.
pause
