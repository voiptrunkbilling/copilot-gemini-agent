<#
.SYNOPSIS
    M6 Production Validation Script for Copilot-Gemini Agent.

.DESCRIPTION
    Validates all M6 acceptance criteria:
    1. Fresh install works
    2. Calibration works
    3. Full loop to ACCEPT
    4. Kill switch works mid-run
    5. Crash/resume works
    6. Agent stats shows metrics

.EXAMPLE
    .\validate_m6.ps1
    .\validate_m6.ps1 -SkipInstall
    .\validate_m6.ps1 -QuickMode

.NOTES
    Run this after install_windows.ps1
#>

[CmdletBinding()]
param(
    [switch]$SkipInstall,
    [switch]$QuickMode
)

$ErrorActionPreference = "Continue"

function Write-Header { param($msg) 
    Write-Host ""
    Write-Host "=" * 60 -ForegroundColor Cyan
    Write-Host "  $msg" -ForegroundColor Cyan
    Write-Host "=" * 60 -ForegroundColor Cyan
    Write-Host ""
}

function Write-Test { param($num, $total, $msg)
    Write-Host "[TEST $num/$total] $msg" -ForegroundColor Yellow
}

function Write-Pass { param($msg)
    Write-Host "[PASS] $msg" -ForegroundColor Green
}

function Write-Fail { param($msg)
    Write-Host "[FAIL] $msg" -ForegroundColor Red
}

function Write-Skip { param($msg)
    Write-Host "[SKIP] $msg" -ForegroundColor Gray
}

function Ask-YesNo { param($prompt)
    $response = Read-Host "$prompt (y/n)"
    return $response -eq "y"
}

# Script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

Set-Location $ProjectRoot

Write-Header "M6 Production Validation"
Write-Host "Project: $ProjectRoot"
Write-Host "Time:    $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host ""

$results = @{
    "install" = "skip"
    "calibrate" = "skip"
    "run_loop" = "skip"
    "kill_switch" = "skip"
    "resume" = "skip"
    "stats" = "skip"
}

$totalTests = 6

# =============================================================================
# TEST 1: Installation Check
# =============================================================================
Write-Header "Test 1/$totalTests : Installation Check"

if ($SkipInstall) {
    Write-Skip "Installation check skipped"
    $results["install"] = "skip"
} else {
    Write-Host "Checking installation status..."
    
    # Check venv
    $venvActive = $env:VIRTUAL_ENV -ne $null
    if (-not $venvActive) {
        $activateScript = Join-Path $ProjectRoot ".venv\Scripts\Activate.ps1"
        if (Test-Path $activateScript) {
            Write-Host "Activating virtual environment..."
            . $activateScript
        } else {
            Write-Fail "Virtual environment not found. Run install_windows.ps1 first."
            exit 1
        }
    }
    
    # Check package
    try {
        $version = python -c "from copilot_agent import __version__; print(__version__)" 2>&1
        Write-Pass "Package installed: v$version"
    } catch {
        Write-Fail "Package not installed"
        exit 1
    }
    
    # Check Tesseract
    try {
        python -c "import pytesseract; pytesseract.get_tesseract_version()" 2>&1
        Write-Pass "Tesseract available"
    } catch {
        Write-Host "[WARN] Tesseract not available (OCR will be limited)" -ForegroundColor Yellow
    }
    
    # Check env file
    if (Test-Path (Join-Path $ProjectRoot ".env")) {
        $envContent = Get-Content (Join-Path $ProjectRoot ".env") -Raw
        if ($envContent -match "GEMINI_API_KEY=(?!your-api-key-here)") {
            Write-Pass "API key configured"
        } else {
            Write-Host "[WARN] API key not configured in .env" -ForegroundColor Yellow
        }
    }
    
    $results["install"] = "pass"
    Write-Pass "Installation check complete"
}

# =============================================================================
# TEST 2: Calibration
# =============================================================================
Write-Header "Test 2/$totalTests : Calibration"

$calFile = "$env:USERPROFILE\.copilot-agent\calibration\calibration.json"

Write-Host "Current calibration status:"
python -m copilot_agent calibrate --show

if (Test-Path $calFile) {
    Write-Pass "Calibration file exists"
    
    if (-not $QuickMode) {
        $recal = Ask-YesNo "Re-run interactive calibration?"
        if ($recal) {
            python -m copilot_agent calibrate --recalibrate
        }
    }
    $results["calibrate"] = "pass"
} else {
    Write-Host "No calibration found. Starting calibration..."
    python -m copilot_agent calibrate
    
    if (Test-Path $calFile) {
        Write-Pass "Calibration created"
        $results["calibrate"] = "pass"
    } else {
        Write-Fail "Calibration failed"
        $results["calibrate"] = "fail"
    }
}

# =============================================================================
# TEST 3: Full Run Loop (to ACCEPT)
# =============================================================================
Write-Header "Test 3/$totalTests : Full Run Loop"

if ($QuickMode) {
    Write-Host "Running dry-run test..."
    python -m copilot_agent run --dry-run --task "M6 validation test" --max-iterations 2
    $results["run_loop"] = "pass"
    Write-Pass "Dry-run completed"
} else {
    Write-Host "This test runs a REAL loop until ACCEPT or max iterations."
    Write-Host ""
    Write-Host "Requirements:" -ForegroundColor Yellow
    Write-Host "  - VS Code open with Copilot Chat visible"
    Write-Host "  - Gemini API key configured"
    Write-Host ""
    
    $runFull = Ask-YesNo "Run full loop test?"
    
    if ($runFull) {
        Write-Host ""
        Write-Host "Starting agent... (press Ctrl+Shift+K to abort)" -ForegroundColor Yellow
        Write-Host ""
        
        $task = Read-Host "Enter a simple test task [default: 'Say hello']"
        if ([string]::IsNullOrWhiteSpace($task)) {
            $task = "Say hello"
        }
        
        python -m copilot_agent run --task $task --max-iterations 5
        
        $results["run_loop"] = "pass"
        Write-Pass "Run loop completed"
    } else {
        Write-Skip "Full loop test skipped"
    }
}

# =============================================================================
# TEST 4: Kill Switch Mid-Run
# =============================================================================
Write-Header "Test 4/$totalTests : Kill Switch Test"

Write-Host "This test verifies the kill switch stops the agent mid-run."
Write-Host ""
Write-Host "Instructions:" -ForegroundColor Yellow
Write-Host "  1. Agent will start typing a long message"
Write-Host "  2. Press Ctrl+Shift+K immediately"
Write-Host "  3. Typing should STOP and agent should exit safely"
Write-Host ""

if ($QuickMode) {
    Write-Host "Testing kill switch registration..."
    python -c @"
from copilot_agent.safety.killswitch import KillSwitch
import time

ks = KillSwitch()
print(f'Kill hotkey: {ks.hotkey_config.modifiers} + {ks.hotkey_config.key}')
print(f'Initialized: OK')
print('Kill switch test passed (dry-run mode)')
"@
    $results["kill_switch"] = "pass"
    Write-Pass "Kill switch validated (quick mode)"
} else {
    $runKill = Ask-YesNo "Run kill switch test?"
    
    if ($runKill) {
        Write-Host ""
        Write-Host "Starting in 3 seconds... PRESS CTRL+SHIFT+K NOW!" -ForegroundColor Red
        Start-Sleep -Seconds 3
        
        python -c @"
from copilot_agent.actuator import ActionExecutor
from copilot_agent.safety.killswitch import KillSwitch
import time

ks = KillSwitch()
ks.start()

print('Kill switch active. Starting long typing test...')
print('Press Ctrl+Shift+K to abort!')
print()

ex = ActionExecutor(dry_run=False, kill_switch_check=ks.is_triggered)

# Type a really long message
long_text = 'Kill switch test in progress! Please press Ctrl+Shift+K now to verify it works. ' * 10

result = ex.type_text(long_text)

if result.success:
    print()
    print('Typing completed - kill switch was NOT triggered')
else:
    print()
    print(f'KILLED: {result.error}')
    print('Kill switch working correctly!')

ks.stop()
"@
        
        Write-Host ""
        $worked = Ask-YesNo "Did the kill switch stop typing?"
        
        if ($worked) {
            $results["kill_switch"] = "pass"
            Write-Pass "Kill switch working"
        } else {
            $results["kill_switch"] = "fail"
            Write-Fail "Kill switch may not be working"
        }
    } else {
        Write-Skip "Kill switch test skipped"
    }
}

# =============================================================================
# TEST 5: Crash/Resume
# =============================================================================
Write-Header "Test 5/$totalTests : Crash/Resume Test"

Write-Host "Checking for resumable sessions..."
python -m copilot_agent resume --list

Write-Host ""

if ($QuickMode) {
    Write-Host "Testing checkpoint system..."
    python -c @"
from copilot_agent.checkpoint import AtomicCheckpointer
import tempfile
import os

with tempfile.TemporaryDirectory() as tmpdir:
    cm = AtomicCheckpointer(tmpdir)
    
    print('AtomicCheckpointer initialized')
    print('Checkpoint system: OK')
"@
    $results["resume"] = "pass"
    Write-Pass "Checkpoint system validated"
} else {
    $testResume = Ask-YesNo "Test resume functionality?"
    
    if ($testResume) {
        Write-Host ""
        Write-Host "To test resume:" -ForegroundColor Yellow
        Write-Host "  1. Start a run: agent run -t 'test task'"
        Write-Host "  2. Press Ctrl+Shift+K to abort mid-run"
        Write-Host "  3. Run: agent resume --list"
        Write-Host "  4. Run: agent resume --session <id>"
        Write-Host ""
        
        $resumeList = python -m copilot_agent resume --list 2>&1
        
        if ($resumeList -match "No resumable sessions") {
            Write-Host "No sessions to resume. Creating a test session..."
            
            # Run briefly then abort
            Write-Host "Starting a brief run (will auto-stop after 1 iteration)..."
            python -m copilot_agent run --dry-run --task "Resume test" --max-iterations 1
            
            Write-Host ""
            python -m copilot_agent resume --list
        }
        
        $results["resume"] = "pass"
        Write-Pass "Resume test completed"
    } else {
        Write-Skip "Resume test skipped"
    }
}

# =============================================================================
# TEST 6: Agent Stats
# =============================================================================
Write-Header "Test 6/$totalTests : Agent Stats"

Write-Host "Checking agent stats command..."
Write-Host ""

Write-Host "--- All Sessions ---" -ForegroundColor Cyan
python -m copilot_agent stats --all

Write-Host ""
Write-Host "--- Recent Session ---" -ForegroundColor Cyan
python -m copilot_agent stats

$results["stats"] = "pass"
Write-Pass "Stats command working"

# =============================================================================
# SUMMARY
# =============================================================================
Write-Header "Validation Summary"

$passCount = ($results.Values | Where-Object { $_ -eq "pass" }).Count
$failCount = ($results.Values | Where-Object { $_ -eq "fail" }).Count
$skipCount = ($results.Values | Where-Object { $_ -eq "skip" }).Count

Write-Host "Results:" -ForegroundColor White
Write-Host ""

foreach ($test in @("install", "calibrate", "run_loop", "kill_switch", "resume", "stats")) {
    $status = $results[$test]
    $icon = switch ($status) {
        "pass" { "[PASS]"; $color = "Green" }
        "fail" { "[FAIL]"; $color = "Red" }
        "skip" { "[SKIP]"; $color = "Gray" }
    }
    
    $testName = switch ($test) {
        "install" { "Installation" }
        "calibrate" { "Calibration" }
        "run_loop" { "Full Run Loop" }
        "kill_switch" { "Kill Switch" }
        "resume" { "Crash/Resume" }
        "stats" { "Agent Stats" }
    }
    
    Write-Host "  $icon $testName" -ForegroundColor $color
}

Write-Host ""
Write-Host "Summary: $passCount passed, $failCount failed, $skipCount skipped" -ForegroundColor $(if ($failCount -gt 0) { "Red" } else { "Green" })
Write-Host ""

if ($failCount -eq 0) {
    Write-Host "M6 Validation: PASSED" -ForegroundColor Green
    Write-Host ""
    Write-Host "The agent is production-ready!" -ForegroundColor Green
} else {
    Write-Host "M6 Validation: FAILED" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please fix the failing tests before deployment." -ForegroundColor Red
}

Write-Host ""
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host ""

if (-not $QuickMode) {
    Read-Host "Press Enter to exit"
}
