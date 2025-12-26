<#
.SYNOPSIS
    Complete Windows installer for Copilot-Gemini Agent.

.DESCRIPTION
    This script performs a complete installation:
    1. Creates Python virtual environment
    2. Installs Python dependencies
    3. Installs Tesseract OCR (via winget)
    4. Sets up sample calibration
    5. Creates environment configuration
    6. Validates the installation

.PARAMETER SkipTesseract
    Skip Tesseract installation if already installed.

.PARAMETER SkipCalibration
    Skip sample calibration setup.

.PARAMETER VenvPath
    Path for virtual environment (default: .venv in project root).

.EXAMPLE
    .\install_windows.ps1
    .\install_windows.ps1 -SkipTesseract
    .\install_windows.ps1 -VenvPath "C:\venvs\copilot-agent"

.NOTES
    Requires:
    - Windows 10/11
    - Python 3.10 or later
    - PowerShell 5.1 or later
    - Administrator rights (for winget)
#>

[CmdletBinding()]
param(
    [switch]$SkipTesseract,
    [switch]$SkipCalibration,
    [string]$VenvPath = ".venv"
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# Colors for output
function Write-Step { param($msg) Write-Host "===> $msg" -ForegroundColor Cyan }
function Write-Success { param($msg) Write-Host "[OK] $msg" -ForegroundColor Green }
function Write-Warn { param($msg) Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Write-Err { param($msg) Write-Host "[ERROR] $msg" -ForegroundColor Red }

# Get script directory (project root)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Copilot-Gemini Agent Installer" -ForegroundColor Cyan
Write-Host "  Windows Setup Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Change to project root
Set-Location $ProjectRoot
Write-Host "Project root: $ProjectRoot"
Write-Host ""

# Step 1: Check Python
Write-Step "Checking Python installation..."

$pythonCmd = $null
foreach ($cmd in @("python", "python3", "py -3")) {
    try {
        $version = & $cmd.Split()[0] $cmd.Split()[1..$cmd.Split().Length] --version 2>$null
        if ($version -match "Python (3\.\d+)") {
            $pythonCmd = $cmd
            Write-Success "Found Python: $version"
            break
        }
    } catch {}
}

if (-not $pythonCmd) {
    Write-Err "Python 3.10+ not found. Please install Python first."
    Write-Host "Download from: https://www.python.org/downloads/"
    exit 1
}

# Step 2: Create virtual environment
Write-Step "Creating virtual environment at $VenvPath..."

if (Test-Path $VenvPath) {
    Write-Warn "Virtual environment already exists. Skipping creation."
} else {
    & $pythonCmd.Split()[0] $pythonCmd.Split()[1..$pythonCmd.Split().Length] -m venv $VenvPath
    if ($LASTEXITCODE -ne 0) {
        Write-Err "Failed to create virtual environment"
        exit 1
    }
    Write-Success "Virtual environment created"
}

# Activate venv
$activateScript = Join-Path $VenvPath "Scripts\Activate.ps1"
if (-not (Test-Path $activateScript)) {
    Write-Err "Activation script not found: $activateScript"
    exit 1
}

Write-Step "Activating virtual environment..."
. $activateScript
Write-Success "Virtual environment activated"

# Step 3: Upgrade pip
Write-Step "Upgrading pip..."
python -m pip install --upgrade pip -q
Write-Success "Pip upgraded"

# Step 4: Install dependencies
Write-Step "Installing Python dependencies..."

if (Test-Path "requirements.txt") {
    pip install -r requirements.txt -q
    if ($LASTEXITCODE -ne 0) {
        Write-Err "Failed to install requirements.txt"
        exit 1
    }
    Write-Success "Core dependencies installed"
} else {
    Write-Warn "requirements.txt not found, installing from pyproject.toml"
    pip install -e . -q
}

if (Test-Path "requirements-dev.txt") {
    pip install -r requirements-dev.txt -q
    Write-Success "Dev dependencies installed"
}

# Step 5: Install Tesseract OCR
Write-Step "Checking Tesseract OCR..."

$tesseractPath = $null
$tesseractLocations = @(
    "C:\Program Files\Tesseract-OCR\tesseract.exe",
    "C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    "$env:LOCALAPPDATA\Programs\Tesseract-OCR\tesseract.exe"
)

foreach ($loc in $tesseractLocations) {
    if (Test-Path $loc) {
        $tesseractPath = $loc
        break
    }
}

if ($tesseractPath) {
    Write-Success "Tesseract found: $tesseractPath"
} elseif ($SkipTesseract) {
    Write-Warn "Tesseract not found (skipped by user)"
} else {
    Write-Step "Installing Tesseract via winget..."
    
    # Check if winget is available
    $wingetAvailable = Get-Command winget -ErrorAction SilentlyContinue
    
    if ($wingetAvailable) {
        try {
            winget install --id UB-Mannheim.TesseractOCR --accept-package-agreements --accept-source-agreements -h
            Write-Success "Tesseract installed via winget"
            Write-Warn "You may need to restart your terminal for PATH changes"
        } catch {
            Write-Warn "Winget install failed. Please install Tesseract manually:"
            Write-Host "  https://github.com/UB-Mannheim/tesseract/wiki"
        }
    } else {
        Write-Warn "Winget not available. Please install Tesseract manually:"
        Write-Host "  https://github.com/UB-Mannheim/tesseract/wiki"
    }
}

# Step 6: Setup sample calibration
Write-Step "Setting up calibration..."

$calibrationDir = Join-Path $env:USERPROFILE ".copilot-agent"
$calibrationFile = Join-Path $calibrationDir "calibration.json"
$sampleCalibration = Join-Path $ScriptDir "sample_calibration.json"

if (-not (Test-Path $calibrationDir)) {
    New-Item -ItemType Directory -Path $calibrationDir -Force | Out-Null
}

if (Test-Path $calibrationFile) {
    Write-Warn "Calibration already exists. Skipping."
} elseif ($SkipCalibration) {
    Write-Warn "Calibration setup skipped by user"
} elseif (Test-Path $sampleCalibration) {
    Copy-Item $sampleCalibration $calibrationFile
    Write-Success "Sample calibration installed (run 'agent calibrate' to customize)"
} else {
    Write-Warn "Sample calibration not found. Run 'agent calibrate' after install."
}

# Step 7: Create .env file if not exists
Write-Step "Setting up environment configuration..."

$envFile = Join-Path $ProjectRoot ".env"
$envExample = Join-Path $ProjectRoot ".env.example"

if (Test-Path $envFile) {
    Write-Warn ".env file already exists"
} elseif (Test-Path $envExample) {
    Copy-Item $envExample $envFile
    Write-Success ".env file created from template"
    Write-Warn "IMPORTANT: Add your GEMINI_API_KEY to .env file"
} else {
    @"
# Copilot-Gemini Agent Environment Variables
# Get your API key from: https://aistudio.google.com/apikey

GEMINI_API_KEY=your-api-key-here

# Optional: Override default model
# GEMINI_MODEL=gemma-3-27b-it
"@ | Out-File -FilePath $envFile -Encoding UTF8
    Write-Success ".env template created"
    Write-Warn "IMPORTANT: Add your GEMINI_API_KEY to .env file"
}

# Step 8: Validate installation
Write-Step "Validating installation..."

$validationErrors = @()

# Check package import
try {
    python -c "from copilot_agent import __version__; print(f'Package version: {__version__}')"
    Write-Success "Package imports correctly"
} catch {
    $validationErrors += "Package import failed"
}

# Check Tesseract
try {
    python -c "import pytesseract; print(f'Tesseract: {pytesseract.get_tesseract_version()}')"
    Write-Success "Tesseract accessible from Python"
} catch {
    $validationErrors += "Tesseract not accessible from Python"
}

# Check pynput
try {
    python -c "from pynput import keyboard; print('pynput: OK')"
    Write-Success "pynput available"
} catch {
    $validationErrors += "pynput not available"
}

# Check mss
try {
    python -c "import mss; print('mss: OK')"
    Write-Success "mss (screenshot) available"
} catch {
    $validationErrors += "mss not available"
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Installation Summary" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

if ($validationErrors.Count -eq 0) {
    Write-Success "Installation completed successfully!"
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "  1. Add your GEMINI_API_KEY to .env file"
    Write-Host "  2. Run: .\$VenvPath\Scripts\Activate.ps1"
    Write-Host "  3. Run: agent calibrate"
    Write-Host "  4. Run: agent run -t 'Your task here'"
    Write-Host ""
    Write-Host "Quick test:" -ForegroundColor Yellow
    Write-Host "  agent --version"
    Write-Host "  agent test-actions --dry-run"
} else {
    Write-Err "Installation completed with warnings:"
    foreach ($err in $validationErrors) {
        Write-Host "  - $err" -ForegroundColor Yellow
    }
    Write-Host ""
    Write-Host "The agent may still work, but some features may be limited."
}

Write-Host ""
