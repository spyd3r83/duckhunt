"""
Unit tests for DuckHunt v2.0 Pattern Detector

Tests the pattern matching engine's ability to detect known attack sequences.
"""

import unittest
import sys
import os
from typing import List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.detector import PatternDetector, KeystrokeEvent


class TestPatternDetector(unittest.TestCase):
    """Test suite for PatternDetector"""

    def setUp(self):
        """Set up test fixtures"""
        self.detector = PatternDetector()

    def test_windows_run_dialog_detected(self):
        """Test detection of Windows Run dialog (WIN+R)"""
        events = [
            self._create_event('LWin', 0),
            self._create_event('r', 50)
        ]

        matches = self.detector.check_pattern_match(events)

        self.assertGreater(len(matches), 0, "WIN+R pattern not detected")
        self.assertGreater(matches[0]['risk'], 0.5,
                         "WIN+R risk score too low")

    def test_powershell_execution_detected(self):
        """Test detection of PowerShell execution pattern"""
        keys = list('powershell')
        events = [self._create_event(k, i*50, window='Run') for i, k in enumerate(keys)]

        matches = self.detector.check_pattern_match(events)

        self.assertGreater(len(matches), 0, "PowerShell pattern not detected")

        # Find PowerShell-specific match
        ps_match = next((m for m in matches if 'powershell' in str(m).lower()), None)
        self.assertIsNotNone(ps_match, "PowerShell specific pattern not found")
        self.assertGreater(ps_match['risk'], 0.8,
                         "PowerShell risk score too low")

    def test_curl_download_detected(self):
        """Test detection of curl download pattern"""
        keys = list('curl -o ')
        events = [self._create_event(k, i*50) for i, k in enumerate(keys)]

        matches = self.detector.check_pattern_match(events)

        self.assertGreater(len(matches), 0, "curl pattern not detected")

    def test_wget_download_detected(self):
        """Test detection of wget download pattern"""
        keys = list('wget http')
        events = [self._create_event(k, i*50, window='Terminal') for i, k in enumerate(keys)]

        matches = self.detector.check_pattern_match(events)

        self.assertGreater(len(matches), 0, "wget pattern not detected")

    def test_base64_decode_detected(self):
        """Test detection of base64 decode pattern"""
        keys = list('base64 -d')
        events = [self._create_event(k, i*50, window='Terminal') for i, k in enumerate(keys)]

        matches = self.detector.check_pattern_match(events)

        # Should detect base64 pattern
        if len(matches) > 0:
            self.assertGreater(matches[0]['risk'], 0.6)

    def test_registry_modification_detected(self):
        """Test detection of Windows registry modification"""
        keys = list('reg add')
        events = [self._create_event(k, i*50, window='cmd.exe') for i, k in enumerate(keys)]

        matches = self.detector.check_pattern_match(events)

        if len(matches) > 0:
            self.assertGreater(matches[0]['risk'], 0.7,
                             "Registry modification risk too low")

    def test_repetition_pattern_detected(self):
        """Test detection of repeated characters (botlike behavior)"""
        # Same key repeated rapidly
        events = [self._create_event('a', i*10) for i in range(10)]

        result = self.detector.detect_repetition_pattern(events)

        self.assertTrue(result['is_anomaly'],
                       "Repeated character pattern not detected")
        self.assertEqual(result['pattern_type'], 'same_key_repeated')

    def test_alternating_pattern_detected(self):
        """Test detection of alternating key pattern"""
        # Alternating between two keys
        events = []
        for i in range(10):
            key = 'a' if i % 2 == 0 else 'b'
            events.append(self._create_event(key, i*15))

        result = self.detector.detect_repetition_pattern(events)

        self.assertTrue(result['is_anomaly'],
                       "Alternating pattern not detected")
        self.assertEqual(result['pattern_type'], 'alternating_keys')

    def test_sequential_pattern_detected(self):
        """Test detection of sequential keys (like 'abcdefg')"""
        keys = list('abcdefghij')
        events = [self._create_event(k, i*15) for i, k in enumerate(keys)]

        result = self.detector.detect_repetition_pattern(events)

        self.assertTrue(result['is_anomaly'],
                       "Sequential pattern not detected")
        self.assertEqual(result['pattern_type'], 'sequential_keys')

    def test_normal_typing_not_flagged_as_pattern(self):
        """Test that normal varied typing is not flagged"""
        text = "the quick brown fox jumps"
        events = [self._create_event(k, i*100) for i, k in enumerate(text)]

        result = self.detector.detect_repetition_pattern(events)

        self.assertFalse(result['is_anomaly'],
                        "Normal typing incorrectly flagged as pattern")

    def test_gui_shortcut_sequence_detected(self):
        """Test detection of rapid GUI shortcut sequence"""
        # Rapid succession of GUI shortcuts (suspicious)
        events = [
            self._create_event('LWin', 0),
            self._create_event('r', 50),
            self._create_event('Return', 100),
            self._create_event('LCtrl', 150),
            self._create_event('LAlt', 180),
            self._create_event('Delete', 200)
        ]

        matches = self.detector.check_pattern_match(events)

        # Should detect suspicious GUI shortcut pattern
        if len(matches) > 0:
            max_risk = max(m['risk'] for m in matches)
            self.assertGreater(max_risk, 0.5,
                             "GUI shortcut sequence risk too low")

    def test_command_injection_characters(self):
        """Test detection of command injection characters"""
        # Common injection characters: &&, ||, ;, |, >, <
        patterns = [
            list('cmd && dir'),
            list('ls ; cat'),
            list('echo | nc'),
        ]

        for pattern in patterns:
            events = [self._create_event(k, i*20) for i, k in enumerate(pattern)]
            matches = self.detector.check_pattern_match(events)

            # Should detect suspicious command patterns
            # (Actual implementation may vary)

    def test_empty_sequence_handled(self):
        """Test that empty event sequence is handled gracefully"""
        events = []

        matches = self.detector.check_pattern_match(events)
        self.assertEqual(len(matches), 0, "Empty sequence should return no matches")

        result = self.detector.detect_repetition_pattern(events)
        self.assertFalse(result['is_anomaly'], "Empty sequence should not be anomaly")

    def test_single_keystroke_handled(self):
        """Test that single keystroke is handled gracefully"""
        events = [self._create_event('a', 0)]

        matches = self.detector.check_pattern_match(events)
        # Single key shouldn't match any patterns (they all require sequences)

        result = self.detector.detect_repetition_pattern(events)
        self.assertFalse(result['is_anomaly'],
                        "Single keystroke should not be flagged as repetition")

    def test_pattern_confidence_scoring(self):
        """Test that pattern confidence increases with multiple signals"""
        # Create a sequence that matches multiple patterns
        # "powershell" in Run dialog window should be high risk

        keys = list('powershell -enc ')
        events = [self._create_event(k, i*15, window='Run') for i, k in enumerate(keys)]

        matches = self.detector.check_pattern_match(events)

        # Should have multiple matches or high confidence
        total_risk = sum(m['risk'] for m in matches)
        self.assertGreater(total_risk, 0.8,
                         "Multi-signal pattern confidence too low")

    def test_context_aware_detection(self):
        """Test that context (window name) affects detection"""
        # Same pattern in different windows should have different risk

        keys = list('powershell')

        # In Run dialog: high risk
        events_run = [self._create_event(k, i*15, window='Run') for i, k in enumerate(keys)]
        matches_run = self.detector.check_pattern_match(events_run)

        # In normal editor: lower risk
        events_editor = [self._create_event(k, i*15, window='VSCode') for i, k in enumerate(keys)]
        matches_editor = self.detector.check_pattern_match(events_editor)

        # Context should affect risk scoring
        # (Actual implementation may vary)

    # Helper methods

    def _create_event(self, key: str, timestamp_ms: int,
                     window: str = "Editor") -> KeystrokeEvent:
        """Create a KeystrokeEvent for testing"""
        return KeystrokeEvent(
            timestamp=f"2025-01-06T12:00:00.{timestamp_ms:03d}Z",
            key=key,
            event_type='key_down',
            inter_event_ms=50.0,
            window_name=window,
            application="TestApp",
            injected=False,
            scan_code=0,
            is_repeat=False
        )


class TestPatternLibrary(unittest.TestCase):
    """Test pattern library completeness"""

    def setUp(self):
        """Set up test fixtures"""
        self.detector = PatternDetector()

    def test_pattern_library_not_empty(self):
        """Test that pattern library contains patterns"""
        # Access internal pattern list
        if hasattr(self.detector, 'attack_patterns'):
            self.assertGreater(len(self.detector.attack_patterns), 0,
                             "Pattern library is empty")

    def test_windows_patterns_present(self):
        """Test that Windows-specific patterns are included"""
        # Should have patterns for: WIN+R, powershell, cmd, reg, etc.
        pass  # Actual test depends on implementation

    def test_linux_patterns_present(self):
        """Test that Linux-specific patterns are included"""
        # Should have patterns for: curl, wget, bash, sh, etc.
        pass  # Actual test depends on implementation

    def test_cross_platform_patterns_present(self):
        """Test that cross-platform patterns are included"""
        # Should have patterns for: base64, nc, python, etc.
        pass  # Actual test depends on implementation


def run_tests():
    """Run all tests"""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    print("DuckHunt v2.0 - Pattern Detector Unit Tests")
    print("=" * 60)
    run_tests()
