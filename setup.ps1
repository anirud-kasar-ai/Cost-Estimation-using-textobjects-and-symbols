# One-shot local setup for Windows (PowerShell).
# Run from the repository root:
#   .\setup.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$App = Join-Path $Root "hvac-cost-estimator"

Write-Host "==> App directory: $App" -ForegroundColor Cyan
Set-Location $App

if (-not (Test-Path ".venv")) {
    Write-Host "==> Creating Python venv..." -ForegroundColor Cyan
    python -m venv .venv
}

$Pip = Join-Path $App ".venv\Scripts\pip.exe"
$Python = Join-Path $App ".venv\Scripts\python.exe"
if (-not (Test-Path $Pip)) {
    throw "venv pip not found at $Pip"
}

Write-Host "==> Installing Python requirements..." -ForegroundColor Cyan
& $Pip install -r requirements.txt

if (-not (Test-Path ".env")) {
    Write-Host "==> Copying .env.example -> .env" -ForegroundColor Cyan
    Copy-Item ".env.example" ".env"
} else {
    Write-Host "==> .env already exists (left unchanged)" -ForegroundColor Yellow
}

Write-Host "==> Installing frontend npm packages..." -ForegroundColor Cyan
Set-Location (Join-Path $App "frontend")
npm install

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "Start API:  cd hvac-cost-estimator\backend ; ..\.venv\Scripts\activate ; uvicorn main:app --reload"
Write-Host "Start UI:   cd hvac-cost-estimator\frontend ; npm run dev"
Write-Host "Dashboard:  http://localhost:5173"
Write-Host "API docs:   http://localhost:8000/docs"
