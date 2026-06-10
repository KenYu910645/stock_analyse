param(
    [string]$CredentialPath = "C:\CAFubon\credential.txt",
    [string]$StopAt = "13:30",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$LogDir = Join-Path $Root "log"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$OutLog = Join-Path $LogDir "realtime_collector_$Stamp.out.log"
$ErrLog = Join-Path $LogDir "realtime_collector_$Stamp.err.log"
$WrapperLog = Join-Path $LogDir "realtime_collector_$Stamp.wrapper.log"

function Write-WrapperLog {
    param([string]$Message)
    $Message | Out-File -FilePath $WrapperLog -Encoding utf8 -Append
}

function Add-DockerToPath {
    $dockerBin = "C:\Program Files\Docker\Docker\resources\bin"
    if (Test-Path $dockerBin) {
        $env:PATH = "$dockerBin;$env:PATH"
    }
}

function Invoke-Checked {
    param(
        [string]$FilePath,
        [string[]]$ArgumentList,
        [string]$Description
    )

    Write-WrapperLog "start_$Description=$(Get-Date -Format o)"
    & $FilePath @ArgumentList 2>&1 | Tee-Object -FilePath $WrapperLog -Append
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE"
    }
    Write-WrapperLog "done_$Description=$(Get-Date -Format o)"
}

function Start-DockerDesktop {
    $dockerDesktop = "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    if (Test-Path $dockerDesktop) {
        Write-WrapperLog "starting_docker_desktop=$(Get-Date -Format o)"
        Start-Process -FilePath $dockerDesktop -WindowStyle Hidden
    }
}

function Wait-DockerDaemon {
    param([int]$TimeoutSeconds = 300)

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        docker info *> $null
        if ($LASTEXITCODE -eq 0) {
            Write-WrapperLog "docker_ready=$(Get-Date -Format o)"
            return
        }

        Start-DockerDesktop
        Start-Sleep -Seconds 5
    }

    throw "Docker daemon did not become ready within $TimeoutSeconds seconds."
}

function Wait-TcpPort {
    param(
        [string]$HostName,
        [int]$Port,
        [int]$TimeoutSeconds = 120
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        $ok = (Test-NetConnection -ComputerName $HostName -Port $Port -WarningAction SilentlyContinue).TcpTestSucceeded
        if ($ok) {
            Write-WrapperLog "tcp_ready=$HostName`:$Port $(Get-Date -Format o)"
            return
        }
        Start-Sleep -Seconds 2
    }

    throw "$HostName`:$Port did not become ready within $TimeoutSeconds seconds."
}

function Load-Credentials {
    param([string]$Path)

    if (-not (Test-Path $Path)) {
        throw "Credential file not found: $Path"
    }

    Get-Content -Path $Path | ForEach-Object {
        $line = $_.Trim()
        if ($line -eq "" -or $line.StartsWith("#")) {
            return
        }

        $parts = $line -split "=", 2
        if ($parts.Count -ne 2) {
            return
        }

        $name = $parts[0].Trim()
        $value = $parts[1].Trim()
        if ($name -in @("FUBON_ID", "FUBON_PASSWORD", "FUBON_CERT_PATH")) {
            [Environment]::SetEnvironmentVariable($name, $value, "Process")
        }
    }

    [Environment]::SetEnvironmentVariable("FUBON_CERT_PASSWORD", $null, "Process")
    Remove-Item Env:FUBON_CERT_PASSWORD -ErrorAction SilentlyContinue
}

Set-Location -Path $Root
Add-DockerToPath

$stopParts = $StopAt.Split(":")
if ($stopParts.Count -ne 2) {
    throw "StopAt must be HH:mm, got: $StopAt"
}
$Target = (Get-Date).Date.AddHours([int]$stopParts[0]).AddMinutes([int]$stopParts[1])

$Symbols = python scripts/load_monitor_list.py --format csv
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($Symbols)) {
    throw "Failed to load monitor symbols."
}

$SymbolCount = ($Symbols -split ",").Count
$SubscriptionCount = $SymbolCount * 2

Write-WrapperLog "wrapper_start=$(Get-Date -Format o)"
Write-WrapperLog "root=$Root"
Write-WrapperLog "symbols=$SymbolCount"
Write-WrapperLog "subscriptions=$SubscriptionCount"
Write-WrapperLog "target_stop=$($Target.ToString('o'))"

if ($DryRun) {
    Write-Output "DRY_RUN=1"
    Write-Output "root=$Root"
    Write-Output "credential_path=$CredentialPath"
    Write-Output "symbols=$SymbolCount"
    Write-Output "subscriptions=$SubscriptionCount"
    Write-Output "now=$(Get-Date -Format o)"
    Write-Output "target_stop=$($Target.ToString('o'))"
    Write-Output "would_start=$(if ((Get-Date) -lt $Target) { 'true' } else { 'false' })"
    Write-Output "stdout=$OutLog"
    Write-Output "stderr=$ErrLog"
    Write-Output "wrapper_log=$WrapperLog"
    exit 0
}

if ((Get-Date) -ge $Target) {
    Write-WrapperLog "already_past_target=$(Get-Date -Format o)"
    exit 0
}

Load-Credentials -Path $CredentialPath
[Environment]::SetEnvironmentVariable("SYMBOLS", $Symbols, "Process")
[Environment]::SetEnvironmentVariable("CHANNELS", "trades,books", "Process")
[Environment]::SetEnvironmentVariable("PYTHONUNBUFFERED", "1", "Process")

Wait-DockerDaemon -TimeoutSeconds 300
Invoke-Checked -FilePath "docker" -ArgumentList @("compose", "up", "-d", "timescaledb") -Description "docker_compose_up"
Wait-TcpPort -HostName "localhost" -Port 5432 -TimeoutSeconds 180
Invoke-Checked -FilePath "python" -ArgumentList @("scripts/init_db.py") -Description "init_db"
Wait-TcpPort -HostName "localhost" -Port 5432 -TimeoutSeconds 60

$proc = Start-Process `
    -FilePath "python" `
    -ArgumentList @("-m", "realtime.main") `
    -WorkingDirectory $Root `
    -RedirectStandardOutput $OutLog `
    -RedirectStandardError $ErrLog `
    -PassThru

Write-WrapperLog "collector_pid=$($proc.Id)"
Write-WrapperLog "stdout=$OutLog"
Write-WrapperLog "stderr=$ErrLog"

while ((Get-Date) -lt $Target) {
    if ($proc.HasExited) {
        Write-WrapperLog "collector_exited_early=$(Get-Date -Format o); exit_code=$($proc.ExitCode)"
        exit $proc.ExitCode
    }
    Start-Sleep -Seconds 10
}

if (-not $proc.HasExited) {
    Write-WrapperLog "stopping_collector=$(Get-Date -Format o)"
    Stop-Process -Id $proc.Id -Force
}

Write-WrapperLog "wrapper_done=$(Get-Date -Format o)"
