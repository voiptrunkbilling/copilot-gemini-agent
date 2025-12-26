# Install Tesseract OCR on Windows
# Run this script in PowerShell as Administrator

# Check if Chocolatey is installed
if (!(Get-Command choco -ErrorAction SilentlyContinue)) {
    Write-Host "Installing Chocolatey..."
    Set-ExecutionPolicy Bypass -Scope Process -Force
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
    iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
}

# Install Tesseract
Write-Host "Installing Tesseract OCR..."
choco install tesseract -y

# Verify installation
$tesseractPath = "C:\Program Files\Tesseract-OCR\tesseract.exe"
if (Test-Path $tesseractPath) {
    Write-Host "Tesseract installed successfully at: $tesseractPath"
    & $tesseractPath --version
} else {
    Write-Host "Warning: Tesseract not found at expected path."
    Write-Host "You may need to add it to your PATH manually."
}

Write-Host ""
Write-Host "Installation complete. You may need to restart your terminal."
