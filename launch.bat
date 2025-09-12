@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:: SuperOCR Optimized Launcher
:: Clean, simple launcher script

cd /d "%~dp0"

echo ==========================================
echo    SuperOCR Application Launcher
echo ==========================================
echo.

:: Check if virtual environment exists
if not exist "%~dp0surya_env\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found!
    echo Please run install_optimized.bat first.
    echo.
    pause
    exit /b 1
)

:: Activate virtual environment
echo Activating virtual environment...
call "%~dp0surya_env\Scripts\activate.bat"
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment!
    echo.
    pause
    exit /b 1
)

:: Set GPU environment variables for optimal performance
echo Setting GPU environment variables...
set RECOGNITION_BATCH_SIZE=64
set DETECTOR_BATCH_SIZE=16
set TORCH_DEVICE=cuda
set CUDA_VISIBLE_DEVICES=0

echo GPU settings applied:
echo RECOGNITION_BATCH_SIZE=%RECOGNITION_BATCH_SIZE%
echo DETECTOR_BATCH_SIZE=%DETECTOR_BATCH_SIZE%
echo TORCH_DEVICE=%TORCH_DEVICE%
echo CUDA_VISIBLE_DEVICES=%CUDA_VISIBLE_DEVICES%
echo.

:: Check CUDA availability
echo Checking CUDA availability...
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'None')" 2>nul

echo.
echo Starting SuperOCR application...
echo.

:: Run the main application
python gui_run.py

if %errorlevel% neq 0 (
    echo.
    echo Error occurred while running the application
    echo Error code: %errorlevel%
    echo.
    echo Please check if all dependencies are installed correctly.
    echo Try running install_optimized.bat if issues persist.
    echo.
)

echo.
pause