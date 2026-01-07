#!/usr/bin/env python3
"""
DuckHunt v2.0 - Linux evdev Keyboard Reader
Monitors keyboard input using evdev library
"""

import sys
import json
import time
from datetime import datetime
from pathlib import Path

try:
    import evdev
except ImportError:
    print("Error: evdev library not installed", file=sys.stderr)
    print("Install with: pip install evdev", file=sys.stderr)
    sys.exit(1)


class LinuxKeyboardMonitor:
    """Linux keyboard monitor using evdev"""

    def __init__(self, device_path=None, output_file=None):
        """
        Initialize keyboard monitor

        Args:
            device_path: Path to input device (e.g., /dev/input/event0)
            output_file: Path to output JSON Lines file
        """
        self.device_path = device_path or self._find_keyboard_device()
        self.output_file = output_file
        self.device = None
        self.last_event_time = 0
        self.running = True

    def _find_keyboard_device(self):
        """Find keyboard input device"""
        devices = [evdev.InputDevice(path) for path in evdev.list_devices()]

        for device in devices:
            # Look for device with EV_KEY capability
            if evdev.ecodes.EV_KEY in device.capabilities():
                # Check if it's a keyboard (has letter keys)
                keys = device.capabilities()[evdev.ecodes.EV_KEY]
                if evdev.ecodes.KEY_A in keys or evdev.ecodes.KEY_Q in keys:
                    print(f"Found keyboard: {device.name} at {device.path}")
                    return device.path

        raise Exception("No keyboard device found")

    def get_active_window(self):
        """Get active window title using xdotool"""
        try:
            import subprocess
            result = subprocess.run(
                ['xdotool', 'getactivewindow', 'getwindowname'],
                capture_output=True,
                text=True,
                timeout=0.1
            )
            return result.stdout.strip() if result.returncode == 0 else "Unknown"
        except:
            return "Unknown"

    def get_key_name(self, key_event):
        """Get human-readable key name"""
        try:
            key_code = key_event.code
            key_name = evdev.ecodes.KEY[key_code]

            # Remove KEY_ prefix
            if key_name.startswith('KEY_'):
                key_name = key_name[4:]

            # Special mappings
            mapping = {
                'LEFTSHIFT': 'LShift',
                'RIGHTSHIFT': 'RShift',
                'LEFTCTRL': 'LCtrl',
                'RIGHTCTRL': 'RCtrl',
                'LEFTALT': 'LAlt',
                'RIGHTALT': 'RAlt',
                'LEFTMETA': 'LWin',
                'RIGHTMETA': 'RWin',
                'ENTER': 'Return',
                'BACKSPACE': 'BackSpace',
                'ESC': 'Escape',
            }

            return mapping.get(key_name, key_name.capitalize())

        except:
            return f"Unknown_{key_event.code}"

    def is_backspace_key(self, key_name):
        """Check if key is a correction key"""
        return key_name in ['BackSpace', 'Delete']

    def monitor(self):
        """Start monitoring keyboard events"""
        print(f"[DuckHunt] Monitoring keyboard: {self.device_path}")
        print("[DuckHunt] Press Ctrl+C to stop")

        self.device = evdev.InputDevice(self.device_path)

        # Grab exclusive access (optional, prevents other apps from seeing keys)
        # self.device.grab()

        try:
            for event in self.device.read_loop():
                if not self.running:
                    break

                # Only process key down events
                if event.type == evdev.ecodes.EV_KEY:
                    key_event = evdev.categorize(event)

                    if key_event.keystate == evdev.KeyEvent.key_down:
                        self._process_keystroke(key_event, event)

        except KeyboardInterrupt:
            print("\n[DuckHunt] Stopping...")
        finally:
            # Release device
            # self.device.ungrab()
            self.device.close()

    def _process_keystroke(self, key_event, raw_event):
        """Process a keystroke event"""
        current_time = int(time.time() * 1000)  # milliseconds

        # Calculate inter-event time
        inter_event_ms = 0
        if self.last_event_time > 0:
            inter_event_ms = current_time - self.last_event_time
        self.last_event_time = current_time

        # Get key name
        key_name = self.get_key_name(key_event)

        # Get active window
        window_name = self.get_active_window()

        # Build event object
        event_data = {
            'event_type': 'keystroke',
            'timestamp': current_time,
            'platform': 'linux',
            'key': key_name,
            'key_code': key_event.scancode,
            'injected': False,  # evdev doesn't easily distinguish
            'inter_event_ms': inter_event_ms,
            'window_name': window_name,
            'is_backspace': self.is_backspace_key(key_name),
            'modifiers': []  # TODO: Track modifier state
        }

        # Output event
        self._output_event(event_data)

    def _output_event(self, event_data):
        """Output event as JSON"""
        json_str = json.dumps(event_data)

        # Print to stdout (can be piped to analysis engine)
        print(json_str, flush=True)

        # Optionally write to file
        if self.output_file:
            with open(self.output_file, 'a') as f:
                f.write(json_str + '\n')

    def stop(self):
        """Stop monitoring"""
        self.running = False


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='DuckHunt Linux Keyboard Monitor')
    parser.add_argument('--device', help='Input device path (e.g., /dev/input/event0)')
    parser.add_argument('--output', help='Output JSON Lines file')
    parser.add_argument('--list-devices', action='store_true', help='List available input devices')

    args = parser.parse_args()

    if args.list_devices:
        print("Available input devices:")
        devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
        for device in devices:
            caps = device.capabilities()
            has_keys = evdev.ecodes.EV_KEY in caps
            print(f"  {device.path}: {device.name} (keys: {has_keys})")
        sys.exit(0)

    try:
        monitor = LinuxKeyboardMonitor(
            device_path=args.device,
            output_file=args.output
        )
        monitor.monitor()

    except PermissionError:
        print("Error: Permission denied. Run with sudo:", file=sys.stderr)
        print("  sudo python3 evdev_reader.py", file=sys.stderr)
        sys.exit(1)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
