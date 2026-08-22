# Batch 2: System & Misc drivers
# Run AFTER reboot from batch 1

$ErrorActionPreference = "Stop"

Write-Host "=== Batch 2: System & Misc Drivers ===" -ForegroundColor Cyan
Write-Host ""

$batch2 = @(
    "oem92.inf",  # armourycratecontrolinterface.inf - old ASUS Armoury
    "oem42.inf",  # asussci2.inf - old ASUS SCI2
    "oem58.inf",  # asussci2.inf - old ASUS SCI2
    "oem6.inf",   # prhiddrv.inf - old HID driver
    "oem82.inf",  # prhiddrv.inf - old HID driver (wait, this is the NEWER one)
    "oem63.inf",  # netrtwlane601.inf - old Realtek WiFi
    "oem80.inf"   # rt68cx21x64.inf - old Realtek Ethernet (already deleted in batch1)
)

# Actually, let me re-check which ones are safe to delete
# From the analysis:
# armourycratecontrolinterface.inf: DELETE oem92.inf (1.2.0.0), KEEP oem94.inf (1.2.0.2)
# asussci2.inf: DELETE oem42.inf (3.1.62.0) and oem58.inf (3.1.45.0), KEEP oem5.inf (3.1.64.0)
# prhiddrv.inf: DELETE oem6.inf (1.0.11.0), KEEP oem82.inf (1.0.16.0)
# netrtwlane601.inf: DELETE oem63.inf (6001.15.152.100), KEEP oem83.inf (6001.15.155.1)

$batch2 = @(
    "oem92.inf",  # armourycratecontrolinterface.inf
    "oem42.inf",  # asussci2.inf
    "oem58.inf",  # asussci2.inf
    "oem6.inf",   # prhiddrv.inf
    "oem63.inf"   # netrtwlane601.inf
)

foreach ($oem in $batch2) {
    Write-Host "  Deleting $oem..." -ForegroundColor Gray
    try {
        & pnputil.exe /delete-driver $oem /uninstall /force 2>&1
        Write-Host "    OK" -ForegroundColor Green
    } catch {
        Write-Host "    Failed: $_" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "Batch 2 complete. REBOOT to verify WiFi, ASUS services work." -ForegroundColor Yellow
