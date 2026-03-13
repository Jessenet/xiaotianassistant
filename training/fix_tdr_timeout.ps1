# Fix Windows TDR (Timeout Detection and Recovery) timeout settings
# This allows GPU tasks to run longer without being interrupted by the system
# Requires Administrator privileges

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "  Fix Windows TDR Timeout" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

# Check Administrator privileges
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "ERROR: This script requires Administrator privileges" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please right-click PowerShell and select 'Run as Administrator', then execute:" -ForegroundColor Yellow
    Write-Host "  Set-Location '$PSScriptRoot'" -ForegroundColor White
    Write-Host "  .\fix_tdr_timeout.ps1" -ForegroundColor White
    Write-Host ""
    Write-Host "Or run:" -ForegroundColor Yellow
    Write-Host "  Start-Process powershell -Verb RunAs -ArgumentList '-NoExit', '-File', '$PSCommandPath'" -ForegroundColor White
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "Administrator privileges confirmed" -ForegroundColor Green
Write-Host ""

# Registry path
$regPath = "HKLM:\System\CurrentControlSet\Control\GraphicsDrivers"

# Ensure registry path exists
if (-not (Test-Path $regPath)) {
    Write-Host "Registry path does not exist, creating..." -ForegroundColor Yellow
    New-Item -Path $regPath -Force | Out-Null
    Write-Host "Registry path created" -ForegroundColor Green
} else {
    Write-Host "Registry path exists" -ForegroundColor Green
}

Write-Host ""
Write-Host "Modifying TDR timeout settings..." -ForegroundColor Cyan

# Backup current value (if exists)
$currentValue = Get-ItemProperty -Path $regPath -Name "TdrDelay" -ErrorAction SilentlyContinue
if ($currentValue) {
    Write-Host "  Current TdrDelay value: $($currentValue.TdrDelay)" -ForegroundColor Yellow
} else {
    Write-Host "  TdrDelay not set (using system default of 2 seconds)" -ForegroundColor Yellow
}

# Set TdrDelay = 60 seconds
try {
    Set-ItemProperty -Path $regPath -Name "TdrDelay" -Value 60 -Type DWord -Force
    Write-Host "TdrDelay set to 60 seconds" -ForegroundColor Green
    
    # Verify setting
    $newValue = Get-ItemProperty -Path $regPath -Name "TdrDelay"
    Write-Host "  Verified: TdrDelay = $($newValue.TdrDelay)" -ForegroundColor Green
} catch {
    Write-Host "Failed to set value: $_" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "======================================" -ForegroundColor Cyan
Write-Host "  Additional TDR Optimization Settings" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "TdrLevel: Controls TDR recovery behavior (0=off, default=3)" -ForegroundColor Yellow
Write-Host "TdrDdiDelay: DDI call timeout (default=5 seconds)" -ForegroundColor Yellow
Write-Host ""

$applyAdditional = Read-Host "Apply additional optimization settings? (y/N)"

if ($applyAdditional -eq 'y' -or $applyAdditional -eq 'Y') {
    Write-Host ""
    Write-Host "Applying additional optimizations..." -ForegroundColor Cyan
    
    # TdrDdiDelay: DDI call timeout (seconds)
    Set-ItemProperty -Path $regPath -Name "TdrDdiDelay" -Value 60 -Type DWord -Force
    Write-Host "TdrDdiDelay = 60 seconds" -ForegroundColor Green
    
    Write-Host "Additional optimizations applied" -ForegroundColor Green
} else {
    Write-Host "Skipping additional optimizations" -ForegroundColor Gray
}

Write-Host ""
Write-Host "======================================" -ForegroundColor Cyan
Write-Host "  Configuration Complete" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Modified registry location:" -ForegroundColor Cyan
Write-Host "  HKEY_LOCAL_MACHINE\System\CurrentControlSet\Control\GraphicsDrivers" -ForegroundColor White
Write-Host ""
Write-Host "Settings applied:" -ForegroundColor Cyan
Write-Host "  TdrDelay = 60 (GPU task timeout: 60 seconds)" -ForegroundColor White
if ($applyAdditional -eq 'y' -or $applyAdditional -eq 'Y') {
    Write-Host "  TdrDdiDelay = 60 (DDI call timeout: 60 seconds)" -ForegroundColor White
}
Write-Host ""
Write-Host "IMPORTANT: You must restart your computer for changes to take effect!" -ForegroundColor Yellow
Write-Host ""

$restart = Read-Host "Restart computer now? (y/N)"

if ($restart -eq 'y' -or $restart -eq 'Y') {
    Write-Host ""
    Write-Host "Save your work! Computer will restart in 30 seconds..." -ForegroundColor Yellow
    Write-Host "Press Ctrl+C to cancel restart" -ForegroundColor Gray
    Write-Host ""
    
    shutdown /r /t 30 /c "Applying GPU TDR timeout settings, system will restart"
    
    Write-Host "Restart scheduled" -ForegroundColor Green
    Write-Host "Run 'shutdown /a' to cancel restart" -ForegroundColor Gray
} else {
    Write-Host ""
    Write-Host "Please restart your computer manually later to apply changes" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "After restart, continue training with:" -ForegroundColor Cyan
    Write-Host "  cd E:\devAI\boltassistant\training" -ForegroundColor White
    Write-Host "  python finetune_gemma.py --config training_config.yaml" -ForegroundColor White
}

Write-Host ""
Read-Host "Press Enter to exit"
