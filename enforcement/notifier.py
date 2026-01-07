"""
DuckHunt v2.0 - Cross-Platform Notification System
Sends user alerts for detected attacks
"""

import platform
import subprocess
from typing import Optional
from enum import Enum


class NotificationLevel(Enum):
    """Notification severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class Notifier:
    """Cross-platform notification sender"""

    def __init__(self, config: dict):
        """
        Initialize notifier

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.platform = platform.system().lower()
        self.enabled = config.get('notifications', {}).get('alert_on_detection', True)

    def send(self, title: str, message: str, level: NotificationLevel = NotificationLevel.WARNING):
        """
        Send notification

        Args:
            title: Notification title
            message: Notification message
            level: Severity level
        """
        if not self.enabled:
            return

        try:
            if self.platform == 'windows':
                self._notify_windows(title, message, level)
            elif self.platform == 'darwin':  # macOS
                self._notify_macos(title, message, level)
            elif self.platform == 'linux':
                self._notify_linux(title, message, level)
            else:
                print(f"[ALERT] {title}: {message}")

        except Exception as e:
            print(f"Failed to send notification: {e}")

    def _notify_windows(self, title: str, message: str, level: NotificationLevel):
        """Send Windows notification using PowerShell"""
        # Use Windows Toast notification
        ps_script = f"""
Add-Type -AssemblyName System.Windows.Forms

$notification = New-Object System.Windows.Forms.NotifyIcon
$notification.Icon = [System.Drawing.SystemIcons]::Warning
$notification.BalloonTipIcon = [System.Windows.Forms.ToolTipIcon]::Warning
$notification.BalloonTipTitle = "{title}"
$notification.BalloonTipText = "{message}"
$notification.Visible = $true
$notification.ShowBalloonTip(10000)

Start-Sleep -Seconds 10
$notification.Dispose()
"""
        try:
            subprocess.run(
                ['powershell', '-Command', ps_script],
                timeout=15,
                capture_output=True
            )
        except:
            # Fallback: message box
            subprocess.run(
                ['msg', '*', f'{title}\\n\\n{message}'],
                timeout=5,
                capture_output=True
            )

    def _notify_linux(self, title: str, message: str, level: NotificationLevel):
        """Send Linux notification using notify-send"""
        urgency_map = {
            NotificationLevel.INFO: 'low',
            NotificationLevel.WARNING: 'normal',
            NotificationLevel.CRITICAL: 'critical'
        }

        try:
            subprocess.run(
                ['notify-send', '-u', urgency_map[level], '-i', 'security-high', title, message],
                timeout=5,
                check=False
            )
        except FileNotFoundError:
            # notify-send not available, try zenity
            try:
                subprocess.run(
                    ['zenity', '--warning', '--title', title, '--text', message],
                    timeout=5,
                    check=False
                )
            except:
                print(f"[ALERT] {title}: {message}")

    def _notify_macos(self, title: str, message: str, level: NotificationLevel):
        """Send macOS notification using osascript"""
        script = f'display notification "{message}" with title "{title}" sound name "Funk"'

        try:
            subprocess.run(
                ['osascript', '-e', script],
                timeout=5,
                check=False
            )
        except:
            print(f"[ALERT] {title}: {message}")

    def send_attack_detected(self, confidence: float, attack_type: str):
        """Send attack detection notification"""
        title = "🦆 DuckHunt: Attack Detected"
        message = f"HID injection attack detected!\nType: {attack_type}\nConfidence: {confidence:.0%}"

        level = NotificationLevel.CRITICAL if confidence > 0.95 else NotificationLevel.WARNING

        self.send(title, message, level)

    def send_learning_complete(self, sample_count: int):
        """Send learning phase complete notification"""
        title = "🦆 DuckHunt: Learning Complete"
        message = f"Behavioral profile established with {sample_count:,} samples.\nEnforcement mode is now active."

        self.send(title, message, NotificationLevel.INFO)

    def send_false_positive_reminder(self):
        """Send reminder about false positive reporting"""
        title = "🦆 DuckHunt: False Positive?"
        message = "If this was not an attack, please report it so we can improve detection."

        self.send(title, message, NotificationLevel.INFO)


def test_notifier():
    """Test notifier"""
    config = {
        'notifications': {
            'alert_on_detection': True,
        }
    }

    notifier = Notifier(config)

    print("Testing notifications...")
    print(f"Platform: {notifier.platform}")

    # Test attack notification
    notifier.send_attack_detected(confidence=0.92, attack_type="Speed anomaly + PowerShell pattern")

    print("\nNotification test complete!")
    print("Check if you received a notification on your system")


if __name__ == '__main__':
    test_notifier()
