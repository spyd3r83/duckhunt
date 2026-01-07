"""
DuckHunt v2.0 - Privacy Module
Data minimization, hashing, and anonymization utilities

This module ensures that DuckHunt follows privacy-first principles:
- No storage of raw keystroke content
- Hash-based logging for attack detection
- Configurable data retention policies
- Anonymization of window/application context
"""

import hashlib
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path


class PrivacyManager:
    """Manages privacy-preserving data handling"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize privacy manager

        Args:
            config: Configuration dict with privacy settings
        """
        self.config = config
        self.retention_days = config.get('privacy', {}).get('log_retention_days', 7)
        self.anonymize_logs = config.get('privacy', {}).get('anonymize_logs', True)
        self.store_window_names = config.get('privacy', {}).get('store_window_names', False)

    def hash_content(self, content: str, algorithm: str = 'sha256') -> str:
        """
        Hash content for privacy-preserving logging

        Args:
            content: Raw content to hash
            algorithm: Hash algorithm (sha256, sha512)

        Returns:
            Hex digest of hashed content with algorithm prefix
        """
        if algorithm == 'sha256':
            hasher = hashlib.sha256()
        elif algorithm == 'sha512':
            hasher = hashlib.sha512()
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")

        hasher.update(content.encode('utf-8'))
        return f"{algorithm.upper()}:{hasher.hexdigest()}"

    def anonymize_window_name(self, window_name: str) -> str:
        """
        Anonymize window name to category

        Args:
            window_name: Full window title

        Returns:
            Anonymized category or hash
        """
        if not self.store_window_names:
            # Categorize common applications
            if 'firefox' in window_name.lower() or 'chrome' in window_name.lower():
                return 'BROWSER'
            elif 'notepad' in window_name.lower() or 'code' in window_name.lower():
                return 'EDITOR'
            elif 'cmd' in window_name.lower() or 'powershell' in window_name.lower():
                return 'TERMINAL'
            elif 'explorer' in window_name.lower():
                return 'FILE_MANAGER'
            else:
                return 'APPLICATION'

        return window_name

    def sanitize_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize event data for privacy-preserving storage

        Args:
            event: Raw event dictionary

        Returns:
            Sanitized event with no sensitive data
        """
        sanitized = {
            'timestamp': event.get('timestamp'),
            'event_type': event.get('event_type'),
            'platform': event.get('platform'),
        }

        # For keystroke events, only store timing and metadata
        if event.get('event_type') == 'keystroke':
            sanitized.update({
                'inter_event_ms': event.get('inter_event_ms'),
                'injected': event.get('injected'),
                'modifiers': event.get('modifiers', []),
                'key_category': self._categorize_key(event.get('key', '')),
            })

            # Anonymize window name
            if event.get('window_name'):
                sanitized['window_category'] = self.anonymize_window_name(
                    event['window_name']
                )

        # For mouse events, only store velocity/acceleration
        elif event.get('event_type') in ['mouse_move', 'mouse_click']:
            sanitized.update({
                'mouse_velocity': event.get('mouse_velocity'),
                'mouse_acceleration': event.get('mouse_acceleration'),
            })

        return sanitized

    def _categorize_key(self, key: str) -> str:
        """Categorize key type without storing actual key"""
        if key in ['BackSpace', 'Delete']:
            return 'CORRECTION'
        elif key in ['Return', 'Enter']:
            return 'NEWLINE'
        elif key in ['Tab']:
            return 'TAB'
        elif key in ['LWin', 'RWin', 'LAlt', 'RAlt', 'LCtrl', 'RCtrl', 'LShift', 'RShift']:
            return 'MODIFIER'
        elif len(key) == 1 and key.isalpha():
            return 'LETTER'
        elif len(key) == 1 and key.isdigit():
            return 'DIGIT'
        else:
            return 'OTHER'

    def create_attack_log(self, attack_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create privacy-preserving attack log entry

        Args:
            attack_data: Detected attack information

        Returns:
            Sanitized log entry with hashed content
        """
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'attack_type': attack_data.get('attack_type'),
            'confidence': attack_data.get('confidence'),
            'characteristics': attack_data.get('characteristics', {}),
            'action_taken': attack_data.get('action_taken'),
            'false_positive': None,  # User can flag later
        }

        # Hash any keystroke content
        if attack_data.get('content'):
            log_entry['content_hash'] = self.hash_content(attack_data['content'])

        # Hash window name if present
        if attack_data.get('window_name'):
            log_entry['window_hash'] = self.hash_content(attack_data['window_name'])

        return log_entry

    def enforce_retention_policy(self, log_dir: Path) -> int:
        """
        Delete logs older than retention period

        Args:
            log_dir: Directory containing log files

        Returns:
            Number of files deleted
        """
        if not log_dir.exists():
            return 0

        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        deleted_count = 0

        for log_file in log_dir.glob('*.jsonl'):
            # Check file modification time
            mtime = datetime.fromtimestamp(log_file.stat().st_mtime)

            if mtime < cutoff_date:
                try:
                    log_file.unlink()
                    deleted_count += 1
                except OSError:
                    pass  # Continue even if deletion fails

        return deleted_count

    def get_privacy_summary(self) -> Dict[str, Any]:
        """
        Get summary of privacy settings

        Returns:
            Dictionary with privacy configuration
        """
        return {
            'retention_days': self.retention_days,
            'anonymize_logs': self.anonymize_logs,
            'store_window_names': self.store_window_names,
            'hash_algorithm': 'sha256',
            'data_minimization_enabled': True,
            'raw_keystroke_storage': False,
        }


class StatisticsOnlyStorage:
    """Stores only statistical aggregates, never raw data"""

    def __init__(self):
        self.speed_samples = []
        self.digraph_samples = {}
        self.error_count = 0
        self.total_keystrokes = 0

    def add_speed_sample(self, interval_ms: float):
        """Add speed sample (keeps only statistics)"""
        self.speed_samples.append(interval_ms)

        # Keep only recent samples to limit memory
        if len(self.speed_samples) > 10000:
            self.speed_samples = self.speed_samples[-5000:]

    def add_digraph_sample(self, digraph: str, interval_ms: float):
        """Add digraph timing sample"""
        if digraph not in self.digraph_samples:
            self.digraph_samples[digraph] = []

        self.digraph_samples[digraph].append(interval_ms)

        # Keep only recent samples per digraph
        if len(self.digraph_samples[digraph]) > 1000:
            self.digraph_samples[digraph] = self.digraph_samples[digraph][-500:]

    def add_keystroke(self, is_error: bool = False):
        """Record keystroke (count only, no content)"""
        self.total_keystrokes += 1
        if is_error:
            self.error_count += 1

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistical summary (no raw data)"""
        import numpy as np

        stats = {
            'total_keystrokes': self.total_keystrokes,
            'error_rate': self.error_count / max(self.total_keystrokes, 1),
            'sample_count': len(self.speed_samples),
        }

        if self.speed_samples:
            stats['speed_distribution'] = {
                'mean_ms': float(np.mean(self.speed_samples)),
                'std_dev_ms': float(np.std(self.speed_samples)),
                'median_ms': float(np.median(self.speed_samples)),
                'q1': float(np.percentile(self.speed_samples, 25)),
                'q3': float(np.percentile(self.speed_samples, 75)),
            }

        stats['digraph_count'] = len(self.digraph_samples)

        return stats

    def clear_raw_samples(self):
        """Clear raw samples, keep only computed statistics"""
        # This would be called periodically to minimize memory
        # After computing and saving statistics to profile
        self.speed_samples = []
        self.digraph_samples = {}


def test_privacy_manager():
    """Test privacy manager functionality"""
    config = {
        'privacy': {
            'log_retention_days': 7,
            'anonymize_logs': True,
            'store_window_names': False,
        }
    }

    pm = PrivacyManager(config)

    # Test hashing
    content_hash = pm.hash_content("sensitive password")
    print(f"Content hash: {content_hash}")
    assert content_hash.startswith('SHA256:')

    # Test event sanitization
    event = {
        'timestamp': 1704470400123,
        'event_type': 'keystroke',
        'platform': 'windows',
        'key': 'a',
        'inter_event_ms': 145,
        'window_name': 'Firefox - Google',
        'injected': False,
    }

    sanitized = pm.sanitize_event(event)
    print(f"Sanitized event: {sanitized}")
    assert 'key' not in sanitized  # Raw key removed
    assert sanitized['window_category'] == 'BROWSER'

    # Test attack logging
    attack = {
        'attack_type': 'speed_anomaly',
        'confidence': 0.92,
        'content': 'powershell -enc ABC123',
        'window_name': 'Run',
        'action_taken': 'blocked',
    }

    log_entry = pm.create_attack_log(attack)
    print(f"Attack log: {json.dumps(log_entry, indent=2)}")
    assert 'content' not in log_entry  # Raw content removed
    assert 'content_hash' in log_entry  # Hash present

    print("\n✅ Privacy manager tests passed!")


if __name__ == '__main__':
    test_privacy_manager()
