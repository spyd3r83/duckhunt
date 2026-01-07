#!/usr/bin/env python3
"""
DuckHunt v2.0 - macOS Keyboard Monitor
Monitors keyboard input using pynput (cross-platform alternative)
For production, would use native CGEvent/IOKit
"""

import sys
import json
import time
from datetime import datetime

try:
    from pynput import keyboard
except ImportError:
    print("Error: pynput library not installed", file=sys.stderr)
    print("Install with: pip install pynput", file=sys.stderr)
    sys.exit(1)


class MacOSKeyboardMonitor:
    """macOS keyboard monitor"""

    def __init__(self, output_file=None):
        self.output_file = output_file
        self.last_event_time = 0
        self.running = True

    def get_key_name(self, key):
        """Get human-readable key name"""
        try:
            if hasattr(key, 'char') and key.char:
                return key.char
            else:
                # Special keys
                key_str = str(key)
                return key_str.replace('Key.', '').capitalize()
        except:
            return 'Unknown'

    def on_press(self, key):
        """Handle key press event"""
        current_time = int(time.time() * 1000)

        inter_event_ms = 0
        if self.last_event_time > 0:
            inter_event_ms = current_time - self.last_event_time
        self.last_event_time = current_time

        key_name = self.get_key_name(key)

        event_data = {
            'event_type': 'keystroke',
            'timestamp': current_time,
            'platform': 'macos',
            'key': key_name,
            'inter_event_ms': inter_event_ms,
            'is_backspace': key_name in ['Backspace', 'Delete'],
            'modifiers': []
        }

        self._output_event(event_data)

    def _output_event(self, event_data):
        """Output event as JSON"""
        json_str = json.dumps(event_data)
        print(json_str, flush=True)

        if self.output_file:
            with open(self.output_file, 'a') as f:
                f.write(json_str + '\n')

    def monitor(self):
        """Start monitoring"""
        print("[DuckHunt] Monitoring keyboard on macOS")
        print("[DuckHunt] Press Ctrl+C to stop")

        with keyboard.Listener(on_press=self.on_press) as listener:
            listener.join()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='DuckHunt macOS Keyboard Monitor')
    parser.add_argument('--output', help='Output JSON Lines file')
    args = parser.parse_args()

    try:
        monitor = MacOSKeyboardMonitor(output_file=args.output)
        monitor.monitor()
    except KeyboardInterrupt:
        print("\n[DuckHunt] Stopping...")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
