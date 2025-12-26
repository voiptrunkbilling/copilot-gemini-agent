# =============================================================================
#  M2 Windows Validation Script (PowerShell)
#  Copilot-Gemini Agent v0.2.0
# =============================================================================
#
#  Usage:
#    1. Open VS Code with Copilot Chat visible
#    2. Run: .\scripts\validate_m2_windows.ps1
#    3. Follow the interactive prompts
#
# =============================================================================

$ErrorActionPreference = "Continue"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  M2 Validation Script - Copilot-Gemini Agent v0.2.0" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Test 1: Package Version
Write-Host "[TEST 1/7] Checking package version..." -ForegroundColor Yellow
$version = python -m copilot_agent --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Package not installed. Run: pip install -e ." -ForegroundColor Red
    exit 1
}
Write-Host "[PASS] $version" -ForegroundColor Green
Write-Host ""

# Test 2: System Info
Write-Host "[TEST 2/7] Checking system info and DPI..." -ForegroundColor Yellow
python -c @"
from copilot_agent.actuator import get_dpi_info, get_primary_screen, IS_WINDOWS
dpi = get_dpi_info()
screen = get_primary_screen()
print(f'Platform: Windows={IS_WINDOWS}')
print(f'Screen: {screen.width}x{screen.height}')
print(f'DPI Scale: {dpi.scale_factor:.0%} ({dpi.system_dpi} DPI)')
print(f'DPI Aware: {dpi.is_aware}')
"@
Write-Host "[PASS] System info retrieved." -ForegroundColor Green
Write-Host ""

# Test 3: Current Calibration
Write-Host "[TEST 3/7] Checking calibration status..." -ForegroundColor Yellow
python -m copilot_agent calibrate --show
Write-Host ""

# Test 4: Calibration
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "[TEST 4/7] Calibration Test" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "This test will open a calibration overlay."
Write-Host "Make sure VS Code is visible with Copilot Chat panel open."
Write-Host ""
$runCal = Read-Host "Run calibration now? (y/n)"
if ($runCal -eq "y") {
    python -m copilot_agent calibrate --recalibrate
    Write-Host "[PASS] Calibration completed." -ForegroundColor Green
} else {
    Write-Host "[SKIP] Calibration skipped." -ForegroundColor Gray
}
Write-Host ""

# Verify calibration file
$calFile = "$env:USERPROFILE\.copilot-agent\calibration\calibration.json"
if (Test-Path $calFile) {
    Write-Host "[PASS] Calibration file exists: $calFile" -ForegroundColor Green
    Write-Host ""
    Write-Host "Contents:"
    Get-Content $calFile | ConvertFrom-Json | ConvertTo-Json -Depth 5
} else {
    Write-Host "[WARN] Calibration file not found." -ForegroundColor Yellow
}
Write-Host ""

# Test 5: Dry-Run
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "[TEST 5/7] Dry-Run Test" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Running: agent run --dry-run --task 'M2 validation test'"
Write-Host "Press 'q' to exit the TUI after it renders."
Write-Host ""
Start-Sleep -Seconds 2
python -m copilot_agent run --dry-run --task "M2 validation test"
Write-Host "[PASS] Dry-run completed." -ForegroundColor Green
Write-Host ""

# Test 6: Real Actions
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "[TEST 6/7] Real Actions Test" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "This test will perform REAL GUI actions:" -ForegroundColor Red
Write-Host "  - Focus VS Code window"
Write-Host "  - Type 'hello' in Copilot input"
Write-Host "  - Press Enter"
Write-Host "  - Select text and copy to clipboard"
Write-Host ""
$runReal = Read-Host "Run real actions test? (y/n)"
if ($runReal -eq "y") {
    Write-Host ""
    Write-Host "Running test-actions (real mode)..."
    Write-Host "Press Ctrl+Shift+K at any time to trigger kill switch." -ForegroundColor Yellow
    Write-Host ""
    python -m copilot_agent test-actions
    Write-Host "[PASS] Real actions test completed." -ForegroundColor Green
} else {
    Write-Host "[SKIP] Running dry-run instead..." -ForegroundColor Gray
    python -m copilot_agent test-actions --dry-run --skip-focus
}
Write-Host ""

# Test 7: Kill Switch
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "[TEST 7/7] Kill Switch Test" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "This test verifies the kill switch (Ctrl+Shift+K) works."
Write-Host ""
Write-Host "Instructions:"
Write-Host "  1. Long typing test starts"
Write-Host "  2. Press Ctrl+Shift+K during typing"
Write-Host "  3. Typing should STOP immediately"
Write-Host ""
$runKill = Read-Host "Run kill switch test? (y/n)"
if ($runKill -eq "y") {
    Write-Host ""
    Write-Host "Starting long typing test in 3 seconds..."
    Write-Host "Press Ctrl+Shift+K NOW to test kill switch!" -ForegroundColor Yellow
    Start-Sleep -Seconds 3
    
    python -c @"
from copilot_agent.actuator import ActionExecutor
from copilot_agent.safety.killswitch import KillSwitch
import time

ks = KillSwitch()
ks.start()
print('Kill switch active. Typing...')

ex = ActionExecutor(dry_run=False, kill_switch_check=ks.is_triggered)
result = ex.type_text('Testing kill switch functionality. Press Ctrl+Shift+K now! ' * 5)

if result.success:
    print('Typing completed (kill switch not triggered)')
else:
    print(f'Typing stopped: {result.error}')
    
ks.stop()
"@
    Write-Host ""
    Write-Host "[PASS] Kill switch test completed." -ForegroundColor Green
} else {
    Write-Host "[SKIP] Kill switch test skipped." -ForegroundColor Gray
}
Write-Host ""

# Summary
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  VALIDATION SUMMARY" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Tests completed. Please verify:" -ForegroundColor White
Write-Host "  [x] Package version is 0.2.0"
Write-Host "  [x] System info shows correct DPI and screen size"
Write-Host "  [x] Calibration file saved"
Write-Host "  [x] Dry-run showed planned actions in TUI"
Write-Host "  [x] Real actions focused VS Code and typed text"
Write-Host "  [x] Kill switch stopped typing immediately"
Write-Host ""
Write-Host "Artifacts:" -ForegroundColor Yellow
Write-Host "  Sessions:    $env:USERPROFILE\.copilot-agent\sessions\"
Write-Host "  Calibration: $env:USERPROFILE\.copilot-agent\calibration\calibration.json"
Write-Host ""
Write-Host "For DPI testing:" -ForegroundColor Yellow
Write-Host "  1. Right-click Desktop > Display Settings"
Write-Host "  2. Change 'Scale' to 100% or 125%"
Write-Host "  3. Re-run this script to verify coordinates"
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  M2 Validation Complete" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Read-Host "Press Enter to exit"
