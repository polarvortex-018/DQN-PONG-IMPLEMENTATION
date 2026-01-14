# Setup Script for DQN Pong Project
# This script creates a virtual environment and installs all dependencies

Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "DQN PONG PROJECT - SETUP SCRIPT" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""

# Step 1: Create virtual environment
Write-Host "[1/3] Creating virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "  Virtual environment already exists. Skipping creation." -ForegroundColor Green
} else {
    python -m venv venv
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  Virtual environment created successfully!" -ForegroundColor Green
    } else {
        Write-Host "  ERROR: Failed to create virtual environment." -ForegroundColor Red
        exit 1
    }
}

Write-Host ""

# Step 2: Activate virtual environment
Write-Host "[2/3] Activating virtual environment..." -ForegroundColor Yellow
Write-Host "  Run this command to activate:" -ForegroundColor Cyan
Write-Host "  .\venv\Scripts\activate" -ForegroundColor White
Write-Host ""

# Step 3: Install dependencies
Write-Host "[3/3] Installing dependencies..." -ForegroundColor Yellow
Write-Host "  After activating the virtual environment, run:" -ForegroundColor Cyan
Write-Host "  pip install -r requirements.txt" -ForegroundColor White
Write-Host ""

Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "SETUP INSTRUCTIONS" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""
Write-Host "To complete setup, run these commands:" -ForegroundColor Yellow
Write-Host ""
Write-Host "  1. .\venv\Scripts\activate" -ForegroundColor White
Write-Host "  2. pip install -r requirements.txt" -ForegroundColor White
Write-Host ""
Write-Host "Then you can start training with:" -ForegroundColor Yellow
Write-Host "  python src\train.py" -ForegroundColor White
Write-Host ""
Write-Host "Happy learning! 🚀" -ForegroundColor Green
