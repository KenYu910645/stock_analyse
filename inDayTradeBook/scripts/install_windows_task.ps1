param(
    [string]$TaskName = "InDayTradeBookRealtime",
    [string]$StartTime = "08:55",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Runner = Join-Path $PSScriptRoot "run_market_session.ps1"
$PowerShell = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"
$Argument = "-NoProfile -ExecutionPolicy Bypass -File `"$Runner`""

if ($DryRun) {
    Write-Output "DRY_RUN=1"
    Write-Output "task_name=$TaskName"
    Write-Output "runner=$Runner"
    Write-Output "start_time=$StartTime"
    Write-Output "days=Monday,Tuesday,Wednesday,Thursday,Friday"
    Write-Output "multiple_instances=IgnoreNew"
    exit 0
}

$Action = New-ScheduledTaskAction -Execute $PowerShell -Argument $Argument -WorkingDirectory $Root
$Trigger = New-ScheduledTaskTrigger `
    -Weekly `
    -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday `
    -At $StartTime
$Settings = New-ScheduledTaskSettingsSet `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 6) `
    -StartWhenAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Description "Record Fubon realtime trades and books from monitor_list.txt on weekdays." `
    -Force | Out-Null

Write-Output "registered_task=$TaskName"
