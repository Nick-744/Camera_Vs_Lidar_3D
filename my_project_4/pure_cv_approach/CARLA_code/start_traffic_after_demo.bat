@echo off
setlocal enabledelayedexpansion

:: Get the current directory (where the .bat is placed)
set "CURRENT_DIR=%~dp0"

:: Define paths relative to the current directory
set "CARLA_DIR=%CURRENT_DIR%..\..\CARLA_0.9.11\WindowsNoEditor"
set "CARLA_EXE=%CARLA_DIR%\CarlaUE4.exe"
set "PYTHON_SCRIPT=%CURRENT_DIR%my_generate_traffic.py"

:: Normalize paths
for %%I in ("%CARLA_DIR%") do set "CARLA_DIR=%%~fI"
for %%I in ("%PYTHON_SCRIPT%") do set "PYTHON_SCRIPT=%%~fI"

:: Validate CARLA executable path
if not exist "%CARLA_EXE%" (
    echo [ERROR] CARLA executable not found at %CARLA_EXE%.
    exit /b 1
)

:: Validate Python script path
if not exist "%PYTHON_SCRIPT%" (
    echo [ERROR] Python script not found at %PYTHON_SCRIPT%.
    exit /b 1
)

:: Activate Conda environment (try user install first, fallback to global)
if exist "C:\Users\nick1\anaconda3\Scripts\activate.bat" (
    call "C:\Users\nick1\anaconda3\Scripts\activate.bat" carla-911
) else if exist "C:\ProgramData\anaconda3\Scripts\activate.bat" (
    call "C:\ProgramData\anaconda3\Scripts\activate.bat" carla-911
) else (
    echo [ERROR] Could not find Anaconda activate.bat
    exit /b 1
)

:: Run the Python Script
echo.
echo [INFO] Running Python script at %PYTHON_SCRIPT% ...
echo.
python "%PYTHON_SCRIPT%"

pause
