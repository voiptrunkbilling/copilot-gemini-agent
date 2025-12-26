@echo off
:: Copilot-Gemini Agent - Windows Installer
:: This batch file launches the PowerShell installer script.

echo.
echo ========================================
echo   Copilot-Gemini Agent Installer
echo ========================================
echo.

:: Check for admin rights (needed for winget)
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo [WARN] Not running as Administrator.
    echo        Tesseract installation may require admin rights.
    echo.
    echo To run as Administrator:
    echo   Right-click this file and select "Run as administrator"
    echo.
    pause
)

:: Get script directory
set "SCRIPT_DIR=%~dp0"

:: Check PowerShell version
powershell -NoProfile -Command "exit 0" >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] PowerShell is required but not available.
    pause
    exit /b 1
)

:: Run the PowerShell installer
powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%install_windows.ps1" %*

:: Keep window open if there was an error
if %errorLevel% neq 0 (
    echo.
    echo Installation encountered errors. See above for details.
    pause
)
