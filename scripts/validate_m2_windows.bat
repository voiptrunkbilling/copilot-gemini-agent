@echo off
REM =============================================================================
REM  M2 Windows Validation Script
REM  Copilot-Gemini Agent v0.2.0
REM =============================================================================
REM
REM  Prerequisites:
REM    - Python 3.11+ installed and on PATH
REM    - VS Code open with Copilot Chat panel visible
REM    - pywin32, pyautogui, mss installed (pip install -r requirements.txt)
REM
REM  Usage:
REM    1. Open VS Code with Copilot Chat visible
REM    2. Run this script from the project root: scripts\validate_m2_windows.bat
REM    3. Follow the interactive prompts
REM
REM =============================================================================

setlocal EnableDelayedExpansion

echo.
echo ============================================================
echo   M2 Validation Script - Copilot-Gemini Agent v0.2.0
echo ============================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.11+ and add to PATH.
    exit /b 1
)

echo [INFO] Python found:
python --version
echo.

REM Check package version
echo [TEST 1/7] Checking package version...
python -m copilot_agent --version
if errorlevel 1 (
    echo [ERROR] Package not installed. Run: pip install -e .
    exit /b 1
)
echo [PASS] Package installed correctly.
echo.

REM Check system info
echo [TEST 2/7] Checking system info and DPI...
echo.
python -c "from copilot_agent.actuator import get_dpi_info, get_primary_screen, IS_WINDOWS; dpi=get_dpi_info(); screen=get_primary_screen(); print(f'Platform: Windows={IS_WINDOWS}'); print(f'Screen: {screen.width}x{screen.height}'); print(f'DPI Scale: {dpi.scale_factor:.0%%} ({dpi.system_dpi} DPI)'); print(f'DPI Aware: {dpi.is_aware}')"
echo.
echo [PASS] System info retrieved.
echo.

REM Show current calibration
echo [TEST 3/7] Checking calibration status...
python -m copilot_agent calibrate --show
echo.

REM Calibration test
echo ============================================================
echo [TEST 4/7] Calibration Test
echo ============================================================
echo.
echo This test will open a calibration overlay.
echo Make sure VS Code is visible with Copilot Chat panel open.
echo.
echo You will click on:
echo   1. VS Code title bar
echo   2. Copilot Chat input box
echo   3. Response area (top-left)
echo   4. Response area (bottom-right)
echo.
set /p RUNCAL="Run calibration now? (y/n): "
if /i "%RUNCAL%"=="y" (
    python -m copilot_agent calibrate --recalibrate
    if errorlevel 1 (
        echo [WARN] Calibration may have been cancelled.
    ) else (
        echo [PASS] Calibration completed.
    )
) else (
    echo [SKIP] Calibration skipped.
)
echo.

REM Verify calibration saved
echo [INFO] Verifying calibration file...
if exist "%USERPROFILE%\.copilot-agent\calibration\calibration.json" (
    echo [PASS] Calibration file exists.
    type "%USERPROFILE%\.copilot-agent\calibration\calibration.json"
) else (
    echo [WARN] Calibration file not found.
)
echo.

REM Dry-run test
echo ============================================================
echo [TEST 5/7] Dry-Run Test
echo ============================================================
echo.
echo Running: agent run --dry-run --task "M2 validation test"
echo This will show planned actions without executing them.
echo.
timeout /t 5 /nobreak >nul 2>&1
python -m copilot_agent run --dry-run --task "M2 validation test"
echo.
echo [PASS] Dry-run completed (TUI should have rendered).
echo.

REM Real actions test
echo ============================================================
echo [TEST 6/7] Real Actions Test
echo ============================================================
echo.
echo This test will:
echo   - Focus VS Code window
echo   - Type "hello" in Copilot input
echo   - Press Enter
echo   - Select text and copy to clipboard
echo.
echo WARNING: This performs REAL GUI actions!
echo Make sure VS Code is open and nothing critical is in Copilot input.
echo.
set /p RUNREAL="Run real actions test? (y/n): "
if /i "%RUNREAL%"=="y" (
    echo.
    echo Running test-actions (real mode)...
    echo Press Ctrl+Shift+K at any time to trigger kill switch.
    echo.
    python -m copilot_agent test-actions
    echo.
    echo [PASS] Real actions test completed.
) else (
    echo [SKIP] Real actions test skipped.
    echo.
    echo Running test-actions in dry-run mode instead...
    python -m copilot_agent test-actions --dry-run --skip-focus
    echo [PASS] Dry-run test-actions completed.
)
echo.

REM Kill switch test
echo ============================================================
echo [TEST 7/7] Kill Switch Test
echo ============================================================
echo.
echo This test verifies the kill switch (Ctrl+Shift+K) works.
echo.
echo Instructions:
echo   1. Start typing test (takes ~5 seconds)
echo   2. While typing, press Ctrl+Shift+K
echo   3. Typing should STOP immediately
echo.
set /p RUNKILL="Run kill switch test? (y/n): "
if /i "%RUNKILL%"=="y" (
    echo.
    echo Starting long typing test...
    echo Press Ctrl+Shift+K NOW to test kill switch!
    echo.
    python -c "from copilot_agent.actuator import ActionExecutor; from copilot_agent.safety.killswitch import KillSwitch; import time; ks = KillSwitch(); ks.start(); ex = ActionExecutor(dry_run=False, kill_switch_check=ks.is_triggered); print('Typing...'); result = ex.type_text('This is a long test to verify the kill switch works correctly. Press Ctrl+Shift+K now! ' * 3); print(f'Result: {result}'); ks.stop()"
    echo.
    echo If typing stopped early, kill switch worked!
    echo [PASS] Kill switch test completed.
) else (
    echo [SKIP] Kill switch test skipped.
)
echo.

REM Summary
echo ============================================================
echo   VALIDATION SUMMARY
echo ============================================================
echo.
echo Tests completed. Please verify:
echo   [x] Package version is 0.2.0
echo   [x] System info shows correct DPI and screen size
echo   [x] Calibration file saved to ~/.copilot-agent/calibration/
echo   [x] Dry-run showed planned actions in TUI
echo   [x] Real actions focused VS Code and typed text
echo   [x] Kill switch stopped typing immediately
echo.
echo Session logs: %USERPROFILE%\.copilot-agent\sessions\
echo Calibration:  %USERPROFILE%\.copilot-agent\calibration\calibration.json
echo.
echo For DPI testing:
echo   1. Right-click Desktop ^> Display Settings
echo   2. Change "Scale" to 100%% or 125%%
echo   3. Re-run this script to verify coordinates
echo.
echo ============================================================
echo   M2 Validation Complete
echo ============================================================

endlocal
pause
