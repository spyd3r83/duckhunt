# DuckHunt v2.0 - Windows Service Wrapper
# Manages keyboard monitor as Windows service

param(
    [ValidateSet('Install', 'Uninstall', 'Start', 'Stop', 'Status')]
    [string]$Action = 'Status',
    [string]$ServiceName = 'DuckHuntMonitor'
)

$script:ServiceDisplayName = "DuckHunt HID Injection Monitor"
$script:ServiceDescription = "Detects USB HID injection attacks using behavioral analysis"
$script:InstallPath = "C:\Program Files\DuckHunt"

function Test-Administrator {
    $currentPrincipal = New-Object Security.Principal.WindowsPrincipal(
        [Security.Principal.WindowsIdentity]::GetCurrent()
    )
    return $currentPrincipal.IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator
    )
}

function Install-DuckHuntService {
    if (-not (Test-Administrator)) {
        throw "Administrator privileges required"
    }

    Write-Host "Installing DuckHunt service..."

    # Create installation directory
    if (-not (Test-Path $script:InstallPath)) {
        New-Item -ItemType Directory -Path $script:InstallPath -Force | Out-Null
    }

    # Copy files
    $sourceDir = Split-Path -Parent $PSCommandPath
    Copy-Item -Path "$sourceDir\keyboard_monitor.ps1" -Destination "$script:InstallPath\" -Force
    Copy-Item -Path "$sourceDir\..\..\core\*.py" -Destination "$script:InstallPath\core\" -Force -Recurse
    Copy-Item -Path "$sourceDir\..\..\config\*" -Destination "$script:InstallPath\config\" -Force -Recurse

    # Create service using NSSM (Non-Sucking Service Manager) if available
    # Otherwise use New-Service
    $nssmPath = Get-Command nssm -ErrorAction SilentlyContinue

    if ($nssmPath) {
        # Use NSSM (recommended)
        & nssm install $ServiceName powershell.exe `
            "-ExecutionPolicy Bypass -File `"$script:InstallPath\keyboard_monitor.ps1`""
        & nssm set $ServiceName Description $script:ServiceDescription
        & nssm set $ServiceName Start SERVICE_AUTO_START

    } else {
        # Use New-Service (requires PowerShell wrapper)
        Write-Warning "NSSM not found. Using New-Service (requires wrapper script)"

        # Create wrapper script
        $wrapperScript = @"
`$ErrorActionPreference = 'Stop'
Set-Location '$script:InstallPath'
& powershell.exe -ExecutionPolicy Bypass -File '$script:InstallPath\keyboard_monitor.ps1'
"@
        $wrapperScript | Out-File -FilePath "$script:InstallPath\service_runner.ps1" -Encoding UTF8

        # Create service
        New-Service -Name $ServiceName `
            -DisplayName $script:ServiceDisplayName `
            -Description $script:ServiceDescription `
            -BinaryPathName "powershell.exe -ExecutionPolicy Bypass -File `"$script:InstallPath\service_runner.ps1`"" `
            -StartupType Automatic
    }

    Write-Host "Service installed successfully"
    Write-Host "Use 'sc start $ServiceName' or 'Start-Service $ServiceName' to start"
}

function Uninstall-DuckHuntService {
    if (-not (Test-Administrator)) {
        throw "Administrator privileges required"
    }

    Write-Host "Uninstalling DuckHunt service..."

    # Stop service if running
    $service = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
    if ($service -and $service.Status -eq 'Running') {
        Stop-Service -Name $ServiceName -Force
        Start-Sleep -Seconds 2
    }

    # Remove service
    $nssmPath = Get-Command nssm -ErrorAction SilentlyContinue
    if ($nssmPath) {
        & nssm remove $ServiceName confirm
    } else {
        sc.exe delete $ServiceName
    }

    Write-Host "Service uninstalled successfully"

    # Optionally remove installation directory
    $remove = Read-Host "Remove installation directory? (y/N)"
    if ($remove -eq 'y') {
        Remove-Item -Path $script:InstallPath -Recurse -Force
        Write-Host "Installation directory removed"
    }
}

function Start-DuckHuntService {
    if (-not (Test-Administrator)) {
        throw "Administrator privileges required"
    }

    Write-Host "Starting DuckHunt service..."
    Start-Service -Name $ServiceName
    Start-Sleep -Seconds 2

    $service = Get-Service -Name $ServiceName
    Write-Host "Service status: $($service.Status)"
}

function Stop-DuckHuntService {
    if (-not (Test-Administrator)) {
        throw "Administrator privileges required"
    }

    Write-Host "Stopping DuckHunt service..."
    Stop-Service -Name $ServiceName -Force
    Start-Sleep -Seconds 2

    $service = Get-Service -Name $ServiceName
    Write-Host "Service status: $($service.Status)"
}

function Get-DuckHuntServiceStatus {
    $service = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue

    if ($service) {
        Write-Host "Service: $($service.DisplayName)"
        Write-Host "Status: $($service.Status)"
        Write-Host "StartType: $($service.StartType)"
    } else {
        Write-Host "Service not installed"
    }
}

# Main execution
try {
    switch ($Action) {
        'Install' { Install-DuckHuntService }
        'Uninstall' { Uninstall-DuckHuntService }
        'Start' { Start-DuckHuntService }
        'Stop' { Stop-DuckHuntService }
        'Status' { Get-DuckHuntServiceStatus }
    }
} catch {
    Write-Error "Error: $_"
    exit 1
}
