@echo off 
chcp 65001 >nul 
cd /d "%~dp0" 
call "%~dp0surya_env\Scripts\activate.bat" 
echo Starting SuperOCR application... 
python "%~dp0gui_run.py" 
if %errorlevel% neq 0 ( 
    echo Error occurred while running the application 
    pause 
) else ( 
    echo Application closed successfully 
) 
