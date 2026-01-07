#!/usr/bin/env python3
"""
DuckHunt v2.0 - Main Entry Point
Coordinates all components for HID injection detection
"""

import sys
import os
import argparse
import json
import configparser
from pathlib import Path
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.analyzer import BehavioralAnalyzer, KeystrokeEvent
from core.detector import PatternDetector, AttackClassifier
from core.profile_manager import ProfileManager
from core.privacy import PrivacyManager
from enforcement.policy_engine import PolicyEngine
from enforcement.notifier import Notifier, NotificationLevel


class DuckHunt:
    """Main DuckHunt application"""

    def __init__(self, config_path: str):
        """
        Initialize DuckHunt

        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()

        # Initialize components
        profile_path = self.config.get('advanced', {}).get('profile_path', './data/profile.json')
        self.profile_manager = ProfileManager(profile_path, self.config)
        self.profile = self.profile_manager.load_profile()

        self.analyzer = BehavioralAnalyzer(self.profile, self.config)
        self.detector = PatternDetector(self.config)
        self.policy_engine = PolicyEngine(self.config)
        self.privacy_manager = PrivacyManager(self.config)
        self.notifier = Notifier(self.config)

        # State
        self.running = False
        self.events_processed = 0
        self.attacks_detected = 0

    def _load_config(self) -> dict:
        """Load configuration from INI file"""
        if not self.config_path.exists():
            print(f"Warning: Config file not found: {self.config_path}")
            print("Using default configuration")
            return {}

        config = configparser.ConfigParser()
        config.read(self.config_path)

        # Convert to nested dict
        result = {}
        for section in config.sections():
            result[section] = {}
            for key, value in config.items(section):
                # Try to convert to appropriate type
                if value.lower() in ('true', 'yes', 'on'):
                    result[section][key] = True
                elif value.lower() in ('false', 'no', 'off'):
                    result[section][key] = False
                elif value.isdigit():
                    result[section][key] = int(value)
                elif value.replace('.', '', 1).isdigit():
                    result[section][key] = float(value)
                else:
                    result[section][key] = value

        return result

    def process_event(self, event_data: dict) -> dict:
        """
        Process a keystroke event

        Args:
            event_data: Event dictionary from collector

        Returns:
            Processing result
        """
        # Create KeystrokeEvent
        event = KeystrokeEvent(
            timestamp=event_data.get('timestamp', 0),
            key=event_data.get('key', ''),
            key_code=event_data.get('key_code', 0),
            inter_key_ms=event_data.get('inter_event_ms', 0),
            window_name=event_data.get('window_name', ''),
            window_category=self.privacy_manager.anonymize_window_name(
                event_data.get('window_name', '')
            ),
            injected=event_data.get('injected', False),
            modifiers=event_data.get('modifiers', []),
            is_backspace=event_data.get('is_backspace', False)
        )

        self.events_processed += 1

        # Run behavioral analysis
        analysis_result = self.analyzer.analyze_keystroke(event)

        # Check for pattern matches
        self.detector.add_keystroke(event.key, event.window_name)
        pattern_match = self.detector.check_pattern_match(
            list(self.detector.recent_keys),
            list(self.detector.recent_windows)
        )

        # Classify attack if anomaly detected
        classification = None
        if analysis_result.is_anomaly:
            classification = AttackClassifier.classify_attack(analysis_result, pattern_match)

        # Enforce policy
        enforcement_action = self.policy_engine.enforce(analysis_result, pattern_match)

        # Handle enforcement action
        if enforcement_action.send_alert and analysis_result.is_anomaly:
            self.notifier.send_attack_detected(
                confidence=analysis_result.confidence,
                attack_type=classification.get('primary_type', 'unknown') if classification else 'unknown'
            )
            self.attacks_detected += 1

        # Log if needed
        if enforcement_action.log_event:
            attack_data = {
                'attack_type': classification.get('primary_type') if classification else 'anomaly',
                'confidence': analysis_result.confidence,
                'characteristics': analysis_result.metrics,
                'action_taken': 'blocked' if not enforcement_action.allow_keystroke else 'logged',
                'content': event.key,  # Will be hashed by privacy manager
                'window_name': event.window_name
            }
            log_entry = self.privacy_manager.create_attack_log(attack_data)
            # TODO: Write to attack log file

        # Update profile (if in learning mode)
        if self.profile_manager.learning_mode and not analysis_result.is_anomaly:
            # TODO: Update profile with new data
            pass

        return {
            'allow': enforcement_action.allow_keystroke,
            'is_anomaly': analysis_result.is_anomaly,
            'confidence': analysis_result.confidence,
            'message': enforcement_action.message,
            'classification': classification
        }

    def run(self):
        """Run DuckHunt in interactive mode"""
        print("=" * 60)
        print("🦆 DuckHunt v2.0 - HID Injection Detection System")
        print("=" * 60)
        print()
        print(f"Configuration: {self.config_path}")
        print(f"Profile: {self.profile_manager.profile_path}")
        print(f"Learning phase: {self.profile_manager.get_learning_phase()}")
        print(f"Ready for enforcement: {self.profile_manager.is_ready_for_enforcement()}")
        print()

        # Check learning status
        if not self.profile_manager.is_ready_for_enforcement():
            print("⚠️  System is in learning mode")
            print(f"   Need {self.profile_manager.min_samples:,} samples")
            print(f"   Current: {self.profile.get('sample_count', 0):,}")
            print()

        print("Starting event processing...")
        print("Press Ctrl+C to stop")
        print()

        self.running = True

        try:
            # In production, this would read from collectors via IPC
            # For demo, we'll simulate some events
            self._demo_mode()

        except KeyboardInterrupt:
            print("\n\nStopping...")
        finally:
            self._shutdown()

    def _demo_mode(self):
        """Demo mode with simulated events"""
        print("[DEMO MODE] Simulating keystroke events...")
        print("In production, this would read from platform collectors\n")

        import random

        for i in range(100):
            if not self.running:
                break

            # Simulate normal typing
            event = {
                'timestamp': int(time.time() * 1000),
                'key': chr(97 + (i % 26)),
                'key_code': 65 + (i % 26),
                'inter_event_ms': 145 + random.gauss(0, 20),
                'window_name': 'Demo Application',
                'injected': False,
                'is_backspace': False,
                'modifiers': []
            }

            # Occasionally inject fast attack
            if i % 30 == 0:
                event['inter_event_ms'] = 15  # Very fast
                event['injected'] = True

            result = self.process_event(event)

            if result['is_anomaly']:
                print(f"  [ANOMALY] Confidence: {result['confidence']:.2%}, "
                      f"Action: {'BLOCKED' if not result['allow'] else 'ALLOWED'}")

            time.sleep(0.1)

        print(f"\nProcessed {self.events_processed} events")
        print(f"Detected {self.attacks_detected} attacks")

    def _shutdown(self):
        """Clean shutdown"""
        print("\nShutting down...")

        # Save profile
        self.profile_manager.save_profile()
        print("Profile saved")

        # Show statistics
        stats = self.analyzer.get_statistics()
        print(f"\nStatistics:")
        print(f"  Total keystrokes: {stats['total_keystrokes']}")
        print(f"  Total anomalies: {stats['total_anomalies']}")
        print(f"  Anomaly rate: {stats['anomaly_rate']:.2%}")

    def status(self):
        """Show system status"""
        print("=" * 60)
        print("🦆 DuckHunt v2.0 Status")
        print("=" * 60)
        print()

        # Profile status
        summary = self.profile_manager.get_profile_summary()
        print("Profile:")
        print(f"  Version: {summary['version']}")
        print(f"  Created: {summary['created_at']}")
        print(f"  Learning phase: {summary['learning_phase']}")
        print(f"  Sample count: {summary['sample_count']:,}")
        print(f"  Ready for enforcement: {summary['ready_for_enforcement']}")
        print(f"  Average speed (WPM): {summary['average_speed_wpm']:.1f}")
        print(f"  Error rate: {summary['error_rate']:.2%}")
        print(f"  Digraphs learned: {summary['digraph_count']}")
        print()

        # Privacy status
        privacy = self.privacy_manager.get_privacy_summary()
        print("Privacy:")
        print(f"  Retention days: {privacy['retention_days']}")
        print(f"  Anonymize logs: {privacy['anonymize_logs']}")
        print(f"  Raw keystroke storage: {privacy['raw_keystroke_storage']}")
        print()

        # Policy status
        policy_status = self.policy_engine.get_status()
        print("Enforcement:")
        print(f"  Policy: {policy_status['policy']}")
        print(f"  Currently blocked: {policy_status['is_blocked']}")
        print(f"  Consecutive anomalies: {policy_status['consecutive_anomalies']}")
        print()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='DuckHunt v2.0 - HID Injection Detection System',
        epilog='For more information, see README.v2.md'
    )

    parser.add_argument(
        '--config',
        default='config/duckhunt.v2.conf',
        help='Path to configuration file'
    )

    parser.add_argument(
        '--status',
        action='store_true',
        help='Show system status and exit'
    )

    parser.add_argument(
        '--learn',
        action='store_true',
        help='Run in learning mode (no enforcement)'
    )

    parser.add_argument(
        '--enforce',
        action='store_true',
        help='Enable enforcement mode'
    )

    args = parser.parse_args()

    try:
        app = DuckHunt(config_path=args.config)

        if args.status:
            app.status()
        else:
            app.run()

    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
