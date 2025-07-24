@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:: Get current directory first
cd /d "%~dp0"

title SuperOCR GPU Installer - Fixed Version

color 0A
echo.
echo ===========================================
echo    SuperOCR GPU Installer - Fixed
echo    Advanced OCR with CUDA Acceleration
echo ===========================================
echo.

:: Create log file
echo [%DATE% %TIME%] Installation started > install_log.txt

:: Check Python presence and version
echo [INFO] Checking Python version... >> install_log.txt
echo [INFO] Checking Python version...
python --version 2>> install_log.txt
if %errorlevel% neq 0 (
    echo [ERROR] Python not found! Please install Python 3.10.x >> install_log.txt
    echo [ERROR] Python not found! Please install Python 3.10.x
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%i in ('python --version 2^>nul') do set PYTHON_VERSION=%%i
echo Python version: %PYTHON_VERSION% >> install_log.txt
echo Python version: %PYTHON_VERSION%

echo %PYTHON_VERSION% | findstr /r "^3\.10\." >nul
if %errorlevel% neq 0 (
    echo [ERROR] Python 3.10.x required, found version %PYTHON_VERSION%! >> install_log.txt
    echo [ERROR] Python 3.10.x required, found version %PYTHON_VERSION%!
    pause
    exit /b 1
)
echo.

:: Create virtual environment
echo [INFO] Creating virtual environment... >> install_log.txt
echo [INFO] Creating virtual environment...
python -m venv "%~dp0surya_env" >> install_log.txt 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create virtual environment! >> install_log.txt
    echo [ERROR] Failed to create virtual environment!
    pause
    exit /b 1
)

:: Activate virtual environment
echo [INFO] Activating virtual environment... >> install_log.txt
echo [INFO] Activating virtual environment...
call "%~dp0surya_env\Scripts\activate.bat" >> install_log.txt 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment! >> install_log.txt
    echo [ERROR] Failed to activate virtual environment!
    pause
    rd /s /q "%~dp0surya_env" 2>> install_log.txt
    exit /b 1
)
echo [SUCCESS] Virtual environment activated. >> install_log.txt
echo [SUCCESS] Virtual environment activated.

:: Install dependencies from requirements.txt
echo [INFO] Installing dependencies from requirements.txt... >> install_log.txt
echo [INFO] Installing dependencies from requirements.txt...
echo This may take several minutes, please wait... >> install_log.txt
echo This may take several minutes, please wait...
pip install -r requirements.txt >> install_log.txt 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies! >> install_log.txt
    echo [ERROR] Failed to install dependencies!
    pause
    call "%~dp0surya_env\Scripts\deactivate.bat" 2>> install_log.txt
    rd /s /q "%~dp0surya_env" 2>> install_log.txt
    exit /b 1
)

:: Install Surya OCR without dependencies
echo [INFO] Installing Surya OCR without dependencies... >> install_log.txt
echo [INFO] Installing Surya OCR without dependencies...
pip install surya-ocr==0.14.6 --no-deps >> install_log.txt 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install Surya OCR! >> install_log.txt
    echo [ERROR] Failed to install Surya OCR!
    pause
    call "%~dp0surya_env\Scripts\deactivate.bat" 2>> install_log.txt
    rd /s /q "%~dp0surya_env" 2>> install_log.txt
    exit /b 1
)

:: Install missing Surya dependencies
echo [INFO] Installing missing Surya dependencies... >> install_log.txt
echo [INFO] Installing missing Surya dependencies...
pip install pydantic pydantic-settings filetype pre-commit >> install_log.txt 2>&1
if %errorlevel% neq 0 (
    echo [WARN] Some Surya dependencies failed to install, continuing... >> install_log.txt
    echo [WARN] Some Surya dependencies failed to install, continuing...
)

:: Update PyTorch to compatible version
echo [INFO] Updating PyTorch to compatible version... >> install_log.txt
echo [INFO] Updating PyTorch to compatible version...
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118 >> install_log.txt 2>&1
if %errorlevel% neq 0 (
    echo [WARN] Failed to update PyTorch, trying CPU version... >> install_log.txt
    echo [WARN] Failed to update PyTorch, trying CPU version...
    pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 >> install_log.txt 2>&1
)

:: Test installation
echo [INFO] Testing installation... >> install_log.txt
echo [INFO] Testing installation...
echo Testing PyTorch... >> install_log.txt
echo Testing PyTorch...
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())" >> install_log.txt 2>&1
if %errorlevel% neq 0 (
    echo [WARN] PyTorch test failed >> install_log.txt
    echo [WARN] PyTorch test failed
)

echo Testing Surya OCR... >> install_log.txt
echo Testing Surya OCR...
python -c "import surya; print('Surya OCR imported successfully')" >> install_log.txt 2>&1
if %errorlevel% neq 0 (
    echo [WARN] Surya OCR test failed >> install_log.txt
    echo [WARN] Surya OCR test failed
)

:: Create launch script
echo [INFO] Creating launch script... >> install_log.txt
echo [INFO] Creating launch script...
echo @echo off > launch.bat
echo chcp 65001 ^>nul >> launch.bat
echo cd /d "%%~dp0" >> launch.bat
echo call "%%~dp0surya_env\Scripts\activate.bat" >> launch.bat
echo echo Starting SuperOCR application... >> launch.bat
echo python "%%~dp0gui_run.py" >> launch.bat
echo if %%errorlevel%% neq 0 ^( >> launch.bat
echo     echo Error occurred while running the application >> launch.bat
echo     pause >> launch.bat
echo ^) else ^( >> launch.bat
echo     echo Application closed successfully >> launch.bat
echo ^) >> launch.bat

:: Check for gui_run.py
echo [INFO] Checking for gui_run.py... >> install_log.txt
echo [INFO] Checking for gui_run.py...
if not exist "%~dp0gui_run.py" (
    echo [ERROR] File gui_run.py not found! >> install_log.txt
    echo [ERROR] File gui_run.py not found!
    pause
    call "%~dp0surya_env\Scripts\deactivate.bat" 2>> install_log.txt
    rd /s /q "%~dp0surya_env" 2>> install_log.txt
    exit /b 1
)

echo.
echo [SUCCESS] Installation completed! Logs saved to install_log.txt. >> install_log.txt
echo [SUCCESS] Installation completed! Logs saved to install_log.txt.
echo.
echo [INFO] Installation finished successfully! >> install_log.txt
echo [INFO] Installation finished successfully!
echo.
echo To run the application use launch.bat file
echo.

pause
