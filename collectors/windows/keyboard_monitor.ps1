# DuckHunt v2.0 - Windows Keyboard Monitor
# PowerShell-based low-level keyboard hook
# Captures keystroke events and sends to analysis engine

param(
    [string]$ConfigPath = "..\..\config\duckhunt.v2.conf",
    [string]$PipeName = "duckhunt-events",
    [switch]$Debug
)

# Add necessary .NET types for keyboard hooking
Add-Type @"
using System;
using System.Runtime.InteropServices;
using System.Windows.Forms;
using System.Text;

public class KeyboardHook {
    private const int WH_KEYBOARD_LL = 13;
    private const int WM_KEYDOWN = 0x0100;
    private const int WM_SYSKEYDOWN = 0x0104;

    public delegate IntPtr LowLevelKeyboardProc(int nCode, IntPtr wParam, IntPtr lParam);

    [DllImport("user32.dll", CharSet = CharSet.Auto, SetLastError = true)]
    public static extern IntPtr SetWindowsHookEx(int idHook, LowLevelKeyboardProc lpfn, IntPtr hMod, uint dwThreadId);

    [DllImport("user32.dll", CharSet = CharSet.Auto, SetLastError = true)]
    public static extern bool UnhookWindowsHookEx(IntPtr hhk);

    [DllImport("user32.dll", CharSet = CharSet.Auto, SetLastError = true)]
    public static extern IntPtr CallNextHookEx(IntPtr hhk, int nCode, IntPtr wParam, IntPtr lParam);

    [DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
    public static extern IntPtr GetModuleHandle(string lpModuleName);

    [DllImport("user32.dll")]
    public static extern IntPtr GetForegroundWindow();

    [DllImport("user32.dll", CharSet = CharSet.Auto)]
    public static extern int GetWindowText(IntPtr hWnd, StringBuilder text, int count);

    [DllImport("user32.dll")]
    public static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint processId);

    [StructLayout(LayoutKind.Sequential)]
    public struct KBDLLHOOKSTRUCT {
        public uint vkCode;
        public uint scanCode;
        public uint flags;
        public uint time;
        public IntPtr dwExtraInfo;
    }

    // Flag constants
    public const uint LLKHF_INJECTED = 0x10;
    public const uint LLKHF_LOWER_IL_INJECTED = 0x02;
}
"@

# Global variables
$script:hookId = [IntPtr]::Zero
$script:lastKeystrokeTime = 0
$script:eventQueue = [System.Collections.Queue]::new()
$script:running = $true

# Key name mapping
$script:keyNames = @{
    0x08 = "BackSpace"
    0x09 = "Tab"
    0x0D = "Return"
    0x10 = "LShift"
    0x11 = "LCtrl"
    0x12 = "LAlt"
    0x14 = "CapsLock"
    0x1B = "Escape"
    0x20 = "Space"
    0x21 = "PageUp"
    0x22 = "PageDown"
    0x23 = "End"
    0x24 = "Home"
    0x25 = "Left"
    0x26 = "Up"
    0x27 = "Right"
    0x28 = "Down"
    0x2C = "PrintScreen"
    0x2D = "Insert"
    0x2E = "Delete"
    0x5B = "LWin"
    0x5C = "RWin"
    0x5D = "Apps"
}

function Get-KeyName {
    param([uint32]$vkCode)

    if ($script:keyNames.ContainsKey($vkCode)) {
        return $script:keyNames[$vkCode]
    }

    # Letters A-Z
    if ($vkCode -ge 0x41 -and $vkCode -le 0x5A) {
        return [char]$vkCode
    }

    # Numbers 0-9
    if ($vkCode -ge 0x30 -and $vkCode -le 0x39) {
        return [char]$vkCode
    }

    # Numpad 0-9
    if ($vkCode -ge 0x60 -and $vkCode -le 0x69) {
        return "Numpad$($vkCode - 0x60)"
    }

    return "Unknown_$vkCode"
}

function Get-ActiveWindow {
    $hwnd = [KeyboardHook]::GetForegroundWindow()
    $sb = New-Object System.Text.StringBuilder 256
    [KeyboardHook]::GetWindowText($hwnd, $sb, 256) | Out-Null

    $processId = 0
    [KeyboardHook]::GetWindowThreadProcessId($hwnd, [ref]$processId) | Out-Null

    $processName = ""
    try {
        $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
        $processName = $process.ProcessName
    } catch {}

    return @{
        Title = $sb.ToString()
        ProcessName = $processName
    }
}

function Send-EventToPipe {
    param([hashtable]$Event)

    try {
        $json = $Event | ConvertTo-Json -Compress

        # In production, this would send to named pipe
        # For now, write to event queue for Python to read
        $script:eventQueue.Enqueue($json)

        if ($Debug) {
            Write-Host "[EVENT] $json"
        }

    } catch {
        Write-Warning "Failed to send event: $_"
    }
}

# Keyboard hook callback
$script:hookCallback = {
    param($nCode, $wParam, $lParam)

    if ($nCode -ge 0 -and ($wParam -eq 0x0100 -or $wParam -eq 0x0104)) {
        try {
            # Parse keyboard event
            $kbd = [System.Runtime.InteropServices.Marshal]::PtrToStructure(
                $lParam,
                [KeyboardHook+KBDLLHOOKSTRUCT]
            )

            # Get current time
            $currentTime = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()

            # Calculate inter-keystroke interval
            $interKeyMs = 0
            if ($script:lastKeystrokeTime -gt 0) {
                $interKeyMs = $currentTime - $script:lastKeystrokeTime
            }
            $script:lastKeystrokeTime = $currentTime

            # Check if injected
            $isInjected = ($kbd.flags -band [KeyboardHook]::LLKHF_INJECTED) -ne 0

            # Get active window
            $window = Get-ActiveWindow

            # Get key name
            $keyName = Get-KeyName -vkCode $kbd.vkCode

            # Build event object
            $event = @{
                event_type = "keystroke"
                timestamp = $currentTime
                platform = "windows"
                key = $keyName
                key_code = $kbd.vkCode
                scan_code = $kbd.scanCode
                injected = $isInjected
                inter_event_ms = $interKeyMs
                window_name = $window.Title
                process_name = $window.ProcessName
                is_backspace = ($keyName -eq "BackSpace" -or $keyName -eq "Delete")
                modifiers = @()  # TODO: Track modifier state
            }

            # Send to analysis engine
            Send-EventToPipe -Event $event

        } catch {
            Write-Warning "Error in hook callback: $_"
        }
    }

    # Call next hook
    return [KeyboardHook]::CallNextHookEx([IntPtr]::Zero, $nCode, $wParam, $lParam)
}

function Start-KeyboardMonitoring {
    Write-Host "[DuckHunt] Starting keyboard monitoring..."
    Write-Host "[DuckHunt] Press Ctrl+C to stop"

    # Create script block delegate
    $hookProc = [KeyboardHook+LowLevelKeyboardProc]$script:hookCallback

    # Install keyboard hook
    $moduleHandle = [KeyboardHook]::GetModuleHandle("user32")
    $script:hookId = [KeyboardHook]::SetWindowsHookEx(
        13,  # WH_KEYBOARD_LL
        $hookProc,
        $moduleHandle,
        0
    )

    if ($script:hookId -eq [IntPtr]::Zero) {
        throw "Failed to install keyboard hook"
    }

    Write-Host "[DuckHunt] Keyboard hook installed successfully"

    # Message pump to keep hook alive
    try {
        # In production, this would run as Windows service
        # For testing, we'll use a simple loop
        while ($script:running) {
            Start-Sleep -Milliseconds 100

            # Process event queue
            while ($script:eventQueue.Count -gt 0) {
                $event = $script:eventQueue.Dequeue()
                # TODO: Send to Python analysis engine via IPC
                # For now, optionally write to file
                if ($Debug) {
                    Add-Content -Path "events.jsonl" -Value $event
                }
            }

            # Check for Windows messages
            [System.Windows.Forms.Application]::DoEvents()
        }
    } finally {
        # Cleanup
        if ($script:hookId -ne [IntPtr]::Zero) {
            [KeyboardHook]::UnhookWindowsHookEx($script:hookId) | Out-Null
            Write-Host "[DuckHunt] Keyboard hook removed"
        }
    }
}

function Stop-KeyboardMonitoring {
    Write-Host "[DuckHunt] Stopping keyboard monitoring..."
    $script:running = $false
}

# Handle Ctrl+C
$null = Register-EngineEvent -SourceIdentifier PowerShell.Exiting -Action {
    Stop-KeyboardMonitoring
}

# Main execution
try {
    # Check if running with elevated privileges
    $currentPrincipal = New-Object Security.Principal.WindowsPrincipal(
        [Security.Principal.WindowsIdentity]::GetCurrent()
    )
    $isAdmin = $currentPrincipal.IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator
    )

    if (-not $isAdmin) {
        Write-Warning "Not running with administrator privileges"
        Write-Warning "Some features may not work correctly"
    }

    # Start monitoring
    Start-KeyboardMonitoring

} catch {
    Write-Error "Fatal error: $_"
    Write-Error $_.ScriptStackTrace
    exit 1
}
