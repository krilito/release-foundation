# Auto-generated duplicate driver cleanup script
# Run as Administrator! Each batch requires reboot to verify.

$ErrorActionPreference = "Stop"

Write-Host "=== Duplicate Driver Cleanup ===" -ForegroundColor Cyan
Write-Host ""

# Batch 1: GPU drivers (NVIDIA, AMD)
Write-Host "Batch 1: GPU Drivers" -ForegroundColor Yellow
$batch1 = @(
    "oem27.inf",  # nvpcf.inf - old NVIDIA PnP
    "oem26.inf",  # nvpcf.inf - old NVIDIA PnP  
    "oem55.inf",  # nvpcf.inf - old NVIDIA PnP
    "oem32.inf",  # nvhda.inf - old NVIDIA HDMI Audio
    "oem33.inf",  # nvhda.inf - old NVIDIA HDMI Audio
    "oem43.inf",  # nvvad.inf - old NVIDIA Virtual Audio
    "oem1.inf",   # ahflt.inf - old audio filter
    "oem25.inf",  # amdacpbus.inf - old AMD ACPI
    "oem73.inf"   # rt68cx21x64.inf - old Realtek Ethernet
)

foreach ($oem in $batch1) {
    Write-Host "  Deleting $oem..." -ForegroundColor Gray
    try {
        & pnputil.exe /delete-driver $oem /uninstall /force 2>&1
        Write-Host "    OK" -ForegroundColor Green
    } catch {
        Write-Host "    Failed: $_" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "Batch 1 complete. REBOOT NOW to verify GPU/Ethernet/Audio work." -ForegroundColor Yellow
Write-Host "After reboot, run this script again with -Batch 2" -ForegroundColor Yellow
Write-Host ""

# Ask for reboot
$reboot = Read-Host "Reboot now? (Y/N)"
if ($reboot -eq 'Y' -or $reboot -eq 'y') {
    Restart-Computer -Force
}
