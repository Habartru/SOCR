@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:: SuperOCR Optimized Installer
:: Clean, simple installation script

cd /d "%~dp0"

title SuperOCR Installer v0.16.7

color 0A
echo.
echo ===========================================
echo    SuperOCR Optimized Installer
echo    Surya OCR 0.16.7 + CUDA Support
echo ===========================================
echo.

:: Check Python presence and version
echo [INFO] Checking Python version...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found! Please install Python 3.10.x
    echo.
    echo *******************************************************
    echo * PYTHON NOT FOUND                                    *
    echo * Please install Python 3.10.x from python.org        *
    echo *******************************************************
    echo.
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%i in ('python --version 2^>nul') do set PYTHON_VERSION=%%i
echo Python version: %PYTHON_VERSION%

echo %PYTHON_VERSION% | findstr /r "^3\.10\." >nul
if %errorlevel% neq 0 (
    echo [WARNING] Python 3.10.x recommended, found version %PYTHON_VERSION%!
    echo Continuing anyway...
    echo.
)

:: Remove existing virtual environment if it exists
echo [INFO] Cleaning up existing virtual environment...
if exist "%~dp0surya_env" (
    rd /s /q "%~dp0surya_env" 2>nul
)

:: Create virtual environment
echo [INFO] Creating virtual environment...
python -m venv "%~dp0surya_env"
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create virtual environment!
    echo.
    pause
    exit /b 1
)
echo [SUCCESS] Virtual environment created

:: Activate virtual environment
echo [INFO] Activating virtual environment...
if exist "%~dp0surya_env\Scripts\activate.bat" (
    call "%~dp0surya_env\Scripts\activate.bat"
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to activate virtual environment!
        echo.
        pause
        exit /b 1
    )
    echo [SUCCESS] Virtual environment activated
) else (
    echo [ERROR] Activation script not found!
    echo.
    pause
    exit /b 1
)

:: Upgrade pip
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip >nul 2>&1

:: Install dependencies from requirements.txt
echo [INFO] Installing dependencies from requirements.txt...
pip install -r requirements.txt >nul 2>&1

:: Finalize
echo [SUCCESS] SuperOCR installation completed successfully!
echo.
echo *******************************************************
echo * INSTALLATION COMPLETED SUCCESSFULLY                 *
echo *                                                     *
echo * To run the application, use launch.bat              *
echo *******************************************************
echo.
pause