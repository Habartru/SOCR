@echo off
setlocal enabledelayedexpansion

title SuperOCR GPU Installer - Windows

color 0A
echo.
echo ===========================================
echo    SuperOCR GPU Installer
echo    Advanced OCR with CUDA Acceleration
echo ===========================================
echo.

:: Check if running as administrator
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ This script requires administrator privileges.
    echo.
    echo 🔧 Please right-click on INSTALL.bat and select "Run as administrator"
    echo.
    pause
    exit /b 1
)

:: Force Python 3.10.x
set PYTHON_CMD=py -3.10
echo 🔍 Checking Python 3.10.x...
%PYTHON_CMD% --version 2>&1 | findstr /C:"Python 3.10" >nul
if %errorlevel% neq 0 (
    echo ❌ Python 3.10.x not found!
    echo.
    echo 🔄 Downloading and installing Python 3.10.11...
    powershell -Command "Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe' -OutFile 'python-3.10.11.exe'"
    echo 📦 Installing Python 3.10.11...
    python-3.10.11.exe /quiet InstallAllUsers=1 PrependPath=1
    del python-3.10.11.exe
    echo ✅ Python 3.10.11 installed!
    set PYTHON_CMD=python
    timeout /t 5
)

:: Check Python version
%PYTHON_CMD% --version
echo.

:: Create virtual environment
echo 📁 Creating virtual environment...
if exist "surya_env" (
    echo ⚠️  Virtual environment already exists, removing...
    rmdir /s /q surya_env
)
%PYTHON_CMD% -m venv surya_env

:: Activate virtual environment
echo 🔧 Activating virtual environment...
call surya_env\Scripts\activate.bat

:: Upgrade pip
echo 📦 Upgrading pip...
python -m pip install --upgrade pip

:: Install PyTorch CUDA first
echo 🚀 Installing PyTorch with CUDA 11.8...
pip install torch==2.3.1+cu118 torchvision==0.18.1+cu118 torchaudio==2.3.1+cu118 --index-url https://download.pytorch.org/whl/cu118

:: Install Surya OCR and dependencies
echo 📦 Installing Surya OCR and dependencies...
pip install -r requirements.txt

:: Test installation
echo 🧪 Testing installation...
echo Testing CUDA availability...
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())"

echo Testing Surya OCR...
python -c "from surya.detection import DetectionPredictor; print('✅ Surya OCR ready!')"

:: Get current directory
cd /d "%~dp0"

:: Create desktop shortcut
echo 🖥️  Creating desktop shortcut...
set DESKTOP=%USERPROFILE%\Desktop
set SHORTCUT=%DESKTOP%\SuperOCR.lnk

:: Create batch file for launching
echo @echo off > launch.bat
echo cd /d "%~dp0" >> launch.bat
echo call surya_env\Scripts\activate.bat >> launch.bat
echo python gui_run.py >> launch.bat
echo pause >> launch.bat

:: Create desktop shortcut (PowerShell)
powershell -Command "$WshShell = New-Object -ComObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%SHORTCUT%'); $Shortcut.TargetPath = '%cd%\launch.bat'; $Shortcut.WorkingDirectory = '%cd%'; $Shortcut.IconLocation = '%SystemRoot%\System32\SHELL32.dll,14'; $Shortcut.Save()"

echo.
echo 🎉 Installation completed successfully!
echo.
echo 🚀 Launching SuperOCR...
call surya_env\Scripts\activate.bat
python gui_run.py

pause
