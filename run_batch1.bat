@echo off
echo Running Batch 1: GPU/Audio/Ethernet drivers...
powershell -ExecutionPolicy Bypass -Command "Start-Process powershell -Verb RunAs -ArgumentList '-ExecutionPolicy Bypass -File D:\release-foundation\cleanup_batch1.ps1' -Wait"
echo.
echo Done! Check the PowerShell window for results.
pause
