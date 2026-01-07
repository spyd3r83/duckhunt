"""
DuckHunt v2.0 - Profile Manager
Handles user behavioral profile creation, loading, saving, and continuous learning

Profiles contain only statistical models of typing behavior:
- Speed distributions (mean, std, percentiles)
- Digraph timing patterns
- Error rate statistics
- Mouse movement characteristics (if enabled)

NO raw keystroke content is stored.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np


class ProfileManager:
    """Manages user behavioral profiles"""

    PROFILE_VERSION = "2.0"

    def __init__(self, profile_path: str, config: Dict[str, Any], baseline_path: str = None):
        """
        Initialize profile manager

        Args:
            profile_path: Path to user profile JSON file
            config: Configuration dictionary
            baseline_path: Path to universal baseline profile (optional)
        """
        self.profile_path = Path(profile_path)
        self.config = config
        self.profile = None
        self.learning_mode = config.get('learning', {}).get('enabled', True)
        self.continuous_learning = config.get('learning', {}).get('continuous', True)
        self.min_samples = config.get('learning', {}).get('min_samples', 10000)
        self.learning_rate = config.get('learning', {}).get('learning_rate', 0.05)

        # Baseline profile for immediate protection
        self.baseline_path = Path(baseline_path) if baseline_path else Path('config/baseline.profile.json')
        self.baseline_profile = None
        self.use_baseline = True  # Start with baseline until user profile trained

    def initialize_profile(self) -> Dict[str, Any]:
        """
        Create new empty profile

        Returns:
            New profile dictionary
        """
        return {
            'version': self.PROFILE_VERSION,
            'created_at': datetime.utcnow().isoformat() + 'Z',
            'last_updated': datetime.utcnow().isoformat() + 'Z',
            'learning_phase': 'initial',  # initial, continuous, stable
            'sample_count': 0,

            'speed_distribution': {
                'mean_ms': 0,
                'std_dev_ms': 0,
                'median_ms': 0,
                'q1': 0,
                'q3': 0,
                'min_ms': 0,
                'max_ms': 0,
            },

            'digraph_timings': {},

            'key_hold_duration': {},

            'typing_patterns': {
                'average_speed_wpm': 0,
                'burst_speed_wpm': 0,
                'pause_speed_wpm': 0,
                'error_rate': 0,
                'correction_pattern': {
                    'immediate_backspace_pct': 0,
                    'delayed_correction_pct': 0,
                },
            },

            'mouse_characteristics': {
                'enabled': self.config.get('privacy', {}).get('enable_mouse_tracking', False),
                'average_velocity_px_s': 0,
                'max_velocity_px_s': 0,
                'average_acceleration_px_s2': 0,
                'movement_smoothness': 0,
            },

            'temporal_patterns': {
                'active_hours': [],
                'typing_rhythm_variance': 0,
            },

            'metadata': {
                'platform': '',
                'keyboard_layout': 'en-US',
                'adaptive_learning_rate': self.learning_rate,
            },
        }

    def load_profile(self) -> Dict[str, Any]:
        """
        Load profile from disk

        Returns:
            Profile dictionary, or new profile if not found
        """
        if not self.profile_path.exists():
            self.profile = self.initialize_profile()
            return self.profile

        try:
            with open(self.profile_path, 'r') as f:
                self.profile = json.load(f)

            # Validate profile version
            if self.profile.get('version') != self.PROFILE_VERSION:
                print(f"Warning: Profile version mismatch. Expected {self.PROFILE_VERSION}, got {self.profile.get('version')}")
                # Could implement migration logic here

            return self.profile

        except json.JSONDecodeError as e:
            print(f"Error loading profile: {e}. Creating new profile.")
            self.profile = self.initialize_profile()
            return self.profile

    def save_profile(self):
        """Save profile to disk"""
        if self.profile is None:
            return

        self.profile['last_updated'] = datetime.utcnow().isoformat() + 'Z'

        # Ensure directory exists
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write: write to temp file, then rename
        temp_path = self.profile_path.with_suffix('.tmp')

        try:
            with open(temp_path, 'w') as f:
                json.dump(self.profile, f, indent=2)

            # Atomic rename
            temp_path.replace(self.profile_path)

        except Exception as e:
            print(f"Error saving profile: {e}")
            if temp_path.exists():
                temp_path.unlink()

    def update_speed_distribution(self, speed_samples: List[float]):
        """
        Update speed distribution statistics

        Args:
            speed_samples: List of inter-keystroke intervals in milliseconds
        """
        if not speed_samples:
            return

        if self.profile is None:
            self.load_profile()

        # Calculate statistics
        self.profile['speed_distribution'] = {
            'mean_ms': float(np.mean(speed_samples)),
            'std_dev_ms': float(np.std(speed_samples)),
            'median_ms': float(np.median(speed_samples)),
            'q1': float(np.percentile(speed_samples, 25)),
            'q3': float(np.percentile(speed_samples, 75)),
            'min_ms': float(np.min(speed_samples)),
            'max_ms': float(np.max(speed_samples)),
        }

        self.profile['sample_count'] = len(speed_samples)

    def update_digraph_timing(self, digraph: str, interval_ms: float):
        """
        Update or add digraph timing using exponential moving average

        Args:
            digraph: Two-character key pair (e.g., 'th', 'he')
            interval_ms: Time between the two keys
        """
        if self.profile is None:
            self.load_profile()

        if digraph not in self.profile['digraph_timings']:
            self.profile['digraph_timings'][digraph] = {
                'samples': 0,
                'mean_ms': interval_ms,
                'std_dev_ms': 0,
                'median_ms': interval_ms,
                'q1': interval_ms,
                'q3': interval_ms,
            }
        else:
            dg = self.profile['digraph_timings'][digraph]

            # Exponential moving average
            alpha = self.learning_rate
            old_mean = dg['mean_ms']
            new_mean = (1 - alpha) * old_mean + alpha * interval_ms

            # Update variance (simplified Welford's algorithm)
            if dg['samples'] > 0:
                delta = interval_ms - old_mean
                old_var = dg['std_dev_ms'] ** 2
                new_var = (1 - alpha) * old_var + alpha * (delta ** 2)
                dg['std_dev_ms'] = float(np.sqrt(max(new_var, 0)))

            dg['mean_ms'] = new_mean
            dg['samples'] += 1

    def update_error_rate(self, error_count: int, total_keystrokes: int):
        """
        Update typing error rate

        Args:
            error_count: Number of backspace/delete keys pressed
            total_keystrokes: Total keystrokes in period
        """
        if self.profile is None:
            self.load_profile()

        if total_keystrokes > 0:
            new_error_rate = error_count / total_keystrokes

            # Exponential moving average
            if self.profile['typing_patterns']['error_rate'] == 0:
                self.profile['typing_patterns']['error_rate'] = new_error_rate
            else:
                alpha = self.learning_rate
                old_rate = self.profile['typing_patterns']['error_rate']
                self.profile['typing_patterns']['error_rate'] = (1 - alpha) * old_rate + alpha * new_error_rate

    def update_typing_speed_wpm(self, speed_wpm: float, speed_type: str = 'average'):
        """
        Update typing speed in WPM

        Args:
            speed_wpm: Words per minute
            speed_type: 'average', 'burst', or 'pause'
        """
        if self.profile is None:
            self.load_profile()

        key_map = {
            'average': 'average_speed_wpm',
            'burst': 'burst_speed_wpm',
            'pause': 'pause_speed_wpm',
        }

        key = key_map.get(speed_type, 'average_speed_wpm')

        # Exponential moving average
        if self.profile['typing_patterns'][key] == 0:
            self.profile['typing_patterns'][key] = speed_wpm
        else:
            alpha = self.learning_rate
            old_speed = self.profile['typing_patterns'][key]
            self.profile['typing_patterns'][key] = (1 - alpha) * old_speed + alpha * speed_wpm

    def update_active_hours(self, hour: int):
        """
        Track active typing hours

        Args:
            hour: Hour of day (0-23)
        """
        if self.profile is None:
            self.load_profile()

        active_hours = set(self.profile['temporal_patterns']['active_hours'])
        active_hours.add(hour)
        self.profile['temporal_patterns']['active_hours'] = sorted(list(active_hours))

    def get_learning_phase(self) -> str:
        """
        Determine current learning phase

        Returns:
            'initial', 'continuous', or 'stable'
        """
        if self.profile is None:
            self.load_profile()

        sample_count = self.profile.get('sample_count', 0)

        if sample_count < self.min_samples:
            return 'initial'
        elif sample_count < self.min_samples * 3:
            return 'continuous'
        else:
            return 'stable'

    def is_ready_for_enforcement(self) -> bool:
        """
        Check if profile has enough samples for enforcement

        Returns:
            True if profile is ready, False otherwise
        """
        if self.profile is None:
            self.load_profile()

        return self.profile.get('sample_count', 0) >= self.min_samples

    def get_profile_summary(self) -> Dict[str, Any]:
        """
        Get human-readable profile summary

        Returns:
            Dictionary with profile statistics
        """
        if self.profile is None:
            self.load_profile()

        return {
            'version': self.profile.get('version'),
            'created_at': self.profile.get('created_at'),
            'learning_phase': self.get_learning_phase(),
            'sample_count': self.profile.get('sample_count', 0),
            'ready_for_enforcement': self.is_ready_for_enforcement(),
            'average_speed_wpm': self.profile['typing_patterns']['average_speed_wpm'],
            'error_rate': self.profile['typing_patterns']['error_rate'],
            'digraph_count': len(self.profile.get('digraph_timings', {})),
            'active_hours': len(self.profile['temporal_patterns']['active_hours']),
        }

    def backup_profile(self) -> Path:
        """
        Create backup of current profile

        Returns:
            Path to backup file
        """
        if not self.profile_path.exists():
            return None

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = self.profile_path.with_suffix(f'.{timestamp}.backup')

        try:
            import shutil
            shutil.copy2(self.profile_path, backup_path)
            return backup_path
        except Exception as e:
            print(f"Error creating backup: {e}")
            return None

    def reset_profile(self):
        """Reset profile to initial state"""
        self.profile = self.initialize_profile()
        self.save_profile()

    def load_baseline_profile(self) -> Dict[str, Any]:
        """
        Load universal baseline profile for immediate protection.

        Returns:
            Baseline profile dictionary, or None if not found
        """
        if not self.baseline_path.exists():
            print(f"Warning: Baseline profile not found at {self.baseline_path}")
            return None

        try:
            with open(self.baseline_path, 'r') as f:
                self.baseline_profile = json.load(f)

            print(f"Loaded baseline profile (sample_count: {self.baseline_profile.get('sample_count', 0)})")
            return self.baseline_profile

        except json.JSONDecodeError as e:
            print(f"Error loading baseline profile: {e}")
            return None

    def get_blended_profile(self) -> Dict[str, Any]:
        """
        Get blended profile (baseline + personalized).

        Uses weighted average based on sample count:
        - 0 samples: 100% baseline
        - 1000 samples: 50% baseline / 50% personal
        - 5000 samples: 20% baseline / 80% personal
        - 10000+ samples: 5% baseline / 95% personal

        Always keeps 5% baseline to prevent profile poisoning attacks.

        Returns:
            Blended profile dictionary
        """
        if self.profile is None:
            self.load_profile()

        if self.baseline_profile is None:
            self.load_baseline_profile()

        # If no baseline, return user profile
        if self.baseline_profile is None:
            return self.profile

        # Calculate baseline weight based on sample count
        sample_count = self.profile.get('sample_count', 0)
        baseline_weight = self._calculate_baseline_weight(sample_count)

        # Blend profiles
        blended = self._blend_profiles(self.baseline_profile, self.profile, baseline_weight)

        return blended

    def _calculate_baseline_weight(self, sample_count: int) -> float:
        """
        Calculate baseline weight based on sample count.

        Args:
            sample_count: Number of user samples

        Returns:
            Baseline weight (0.05 to 1.0)
        """
        if sample_count == 0:
            return 1.0
        elif sample_count < 1000:
            # Linear decrease from 1.0 to 0.5
            return 1.0 - (sample_count / 1000) * 0.5
        elif sample_count < 5000:
            # Linear decrease from 0.5 to 0.2
            return 0.5 - ((sample_count - 1000) / 4000) * 0.3
        elif sample_count < 10000:
            # Linear decrease from 0.2 to 0.05
            return 0.2 - ((sample_count - 5000) / 5000) * 0.15
        else:
            # Minimum 5% baseline (prevents profile poisoning)
            return 0.05

    def _blend_profiles(self, baseline: Dict, personal: Dict, baseline_weight: float) -> Dict:
        """
        Blend baseline and personal profiles using weighted average.

        Args:
            baseline: Baseline profile
            personal: Personal profile
            baseline_weight: Weight for baseline (0.0-1.0)

        Returns:
            Blended profile
        """
        personal_weight = 1.0 - baseline_weight

        blended = personal.copy()
        blended['baseline_weight'] = baseline_weight
        blended['is_blended'] = True

        # Blend speed distribution
        if 'speed_distribution' in baseline and 'speed_distribution' in personal:
            for key in ['mean_ms', 'std_dev_ms', 'median_ms', 'q1', 'q3']:
                if key in baseline['speed_distribution'] and key in personal['speed_distribution']:
                    baseline_val = baseline['speed_distribution'][key]
                    personal_val = personal['speed_distribution'][key]

                    # If personal value is still 0 (not trained), use baseline
                    if personal_val == 0:
                        blended['speed_distribution'][key] = baseline_val
                    else:
                        blended['speed_distribution'][key] = (
                            baseline_val * baseline_weight +
                            personal_val * personal_weight
                        )

        # Blend digraph timings
        if 'digraph_timings' in baseline:
            for digraph, stats in baseline['digraph_timings'].items():
                if digraph not in blended.get('digraph_timings', {}):
                    # Not in personal profile yet, use baseline
                    if 'digraph_timings' not in blended:
                        blended['digraph_timings'] = {}
                    blended['digraph_timings'][digraph] = stats.copy()
                else:
                    # Blend with personal
                    for key in ['mean_ms', 'std_dev_ms']:
                        if key in stats and key in blended['digraph_timings'][digraph]:
                            baseline_val = stats[key]
                            personal_val = blended['digraph_timings'][digraph][key]

                            blended['digraph_timings'][digraph][key] = (
                                baseline_val * baseline_weight +
                                personal_val * personal_weight
                            )

        # Blend typing patterns
        if 'typing_patterns' in baseline and 'typing_patterns' in personal:
            for key in ['average_speed_wpm', 'error_rate']:
                if key in baseline['typing_patterns'] and key in personal['typing_patterns']:
                    baseline_val = baseline['typing_patterns'][key]
                    personal_val = personal['typing_patterns'][key]

                    if personal_val == 0:
                        blended['typing_patterns'][key] = baseline_val
                    else:
                        blended['typing_patterns'][key] = (
                            baseline_val * baseline_weight +
                            personal_val * personal_weight
                        )

        return blended

    def get_effective_profile(self) -> Dict[str, Any]:
        """
        Get the effective profile to use for detection.

        Returns blended profile if sample count is low,
        otherwise returns user profile.

        Returns:
            Effective profile for detection
        """
        if self.profile is None:
            self.load_profile()

        sample_count = self.profile.get('sample_count', 0)

        # Use blended profile until sufficiently trained
        if sample_count < 10000:
            return self.get_blended_profile()
        else:
            return self.profile


def test_profile_manager():
    """Test profile manager functionality"""
    import tempfile

    # Create temporary profile file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_profile = f.name

    config = {
        'learning': {
            'enabled': True,
            'continuous': True,
            'min_samples': 100,
            'learning_rate': 0.05,
        },
        'privacy': {
            'enable_mouse_tracking': False,
        },
    }

    try:
        pm = ProfileManager(temp_profile, config)

        # Test profile initialization
        profile = pm.load_profile()
        print(f"New profile created: {profile['version']}")
        assert profile['version'] == '2.0'
        assert profile['sample_count'] == 0

        # Test speed distribution update
        speed_samples = [120, 145, 132, 158, 140, 125, 150, 135, 148, 142]
        pm.update_speed_distribution(speed_samples)
        print(f"Speed distribution: {pm.profile['speed_distribution']}")
        assert pm.profile['speed_distribution']['mean_ms'] > 0

        # Test digraph timing update
        pm.update_digraph_timing('th', 145.3)
        pm.update_digraph_timing('th', 142.1)
        pm.update_digraph_timing('he', 132.5)
        print(f"Digraph timings: {json.dumps(pm.profile['digraph_timings'], indent=2)}")
        assert 'th' in pm.profile['digraph_timings']
        assert pm.profile['digraph_timings']['th']['samples'] == 2

        # Test error rate update
        pm.update_error_rate(3, 100)
        print(f"Error rate: {pm.profile['typing_patterns']['error_rate']}")
        assert pm.profile['typing_patterns']['error_rate'] == 0.03

        # Test save/load
        pm.save_profile()
        pm2 = ProfileManager(temp_profile, config)
        pm2.load_profile()
        assert pm2.profile['version'] == '2.0'
        assert pm2.profile['sample_count'] == len(speed_samples)

        # Test learning phase
        phase = pm.get_learning_phase()
        print(f"Learning phase: {phase}")
        assert phase == 'initial'  # < 100 samples

        # Test profile summary
        summary = pm.get_profile_summary()
        print(f"Profile summary: {json.dumps(summary, indent=2)}")
        assert summary['ready_for_enforcement'] == False

        print("\n✅ Profile manager tests passed!")

    finally:
        # Cleanup
        if os.path.exists(temp_profile):
            os.unlink(temp_profile)


if __name__ == '__main__':
    test_profile_manager()
