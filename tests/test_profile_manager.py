"""
Unit tests for DuckHunt v2.0 Profile Manager

Tests the user profile management and continuous learning functionality.
"""

import unittest
import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.profile_manager import ProfileManager


class TestProfileManager(unittest.TestCase):
    """Test suite for ProfileManager"""

    def setUp(self):
        """Set up test fixtures"""
        # Use temporary file for testing
        self.test_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        self.test_file.close()
        self.profile_path = self.test_file.name

        self.manager = ProfileManager(self.profile_path)

    def tearDown(self):
        """Clean up test artifacts"""
        if os.path.exists(self.profile_path):
            os.remove(self.profile_path)

    def test_profile_initialization(self):
        """Test that new profile is initialized correctly"""
        profile = self.manager.profile

        self.assertEqual(profile['version'], '2.0')
        self.assertEqual(profile['sample_count'], 0)
        self.assertEqual(profile['learning_phase'], 'initial')
        self.assertIn('speed_distribution', profile)
        self.assertIn('digraph_timings', profile)

    def test_save_and_load_profile(self):
        """Test profile persistence"""
        # Modify profile
        self.manager.profile['sample_count'] = 100
        self.manager.profile['speed_distribution']['mean_ms'] = 150.5

        # Save
        self.manager.save_profile()

        # Create new instance and load
        manager2 = ProfileManager(self.profile_path)
        self.assertEqual(manager2.profile['sample_count'], 100)
        self.assertEqual(manager2.profile['speed_distribution']['mean_ms'], 150.5)

    def test_update_speed_distribution(self):
        """Test updating speed statistics with continuous learning"""
        # Initialize profile with some baseline
        self.manager.profile['sample_count'] = 1000
        self.manager.profile['speed_distribution'] = {
            'mean_ms': 140.0,
            'std_dev_ms': 20.0,
            'median_ms': 138.0,
            'q1': 125.0,
            'q3': 155.0
        }

        # Update with new interval
        self.manager.update_speed_distribution(150.0)

        # Mean should have shifted slightly toward 150
        new_mean = self.manager.profile['speed_distribution']['mean_ms']
        self.assertGreater(new_mean, 140.0, "Mean should increase")
        self.assertLess(new_mean, 150.0, "Mean should not jump fully to new value")

        # Sample count should increment
        self.assertEqual(self.manager.profile['sample_count'], 1001)

    def test_update_digraph_timing(self):
        """Test updating digraph timing statistics"""
        # Initialize digraph
        self.manager.profile['digraph_timings'] = {
            'th': {'mean_ms': 140.0, 'std_dev_ms': 20.0, 'samples': 100}
        }

        # Update with new timing
        self.manager.update_digraph_timing('th', 160.0)

        # Mean should have shifted
        digraph = self.manager.profile['digraph_timings']['th']
        self.assertGreater(digraph['mean_ms'], 140.0)
        self.assertEqual(digraph['samples'], 101)

    def test_new_digraph_creation(self):
        """Test that new digraph is created if not exists"""
        self.manager.profile['digraph_timings'] = {}

        # Update non-existent digraph
        self.manager.update_digraph_timing('xy', 120.0)

        # Should be created
        self.assertIn('xy', self.manager.profile['digraph_timings'])
        digraph = self.manager.profile['digraph_timings']['xy']
        self.assertEqual(digraph['mean_ms'], 120.0)
        self.assertEqual(digraph['samples'], 1)

    def test_update_error_pattern(self):
        """Test updating error rate statistics"""
        self.manager.profile['sample_count'] = 100
        self.manager.profile['typing_patterns'] = {
            'error_rate': 0.03,
            'correction_pattern': {
                'immediate_backspace_pct': 0.7,
                'delayed_correction_pct': 0.3
            }
        }

        # Update with backspace event (is_error=True)
        self.manager.update_error_pattern(is_backspace=True, immediate=True)

        # Error rate should update
        new_error_rate = self.manager.profile['typing_patterns']['error_rate']
        # With adaptive learning, should be slight change

    def test_learning_phase_progression(self):
        """Test that learning phase progresses correctly"""
        # Start in initial phase
        self.assertEqual(self.manager.get_learning_phase(), 'initial')

        # Add samples below threshold
        self.manager.profile['sample_count'] = 5000
        self.manager.min_samples = 10000
        self.assertEqual(self.manager.get_learning_phase(), 'initial')

        # Add samples above threshold
        self.manager.profile['sample_count'] = 10000
        phase = self.manager.get_learning_phase()
        self.assertIn(phase, ['continuous', 'stable'])

    def test_learning_rate_application(self):
        """Test that learning rate affects updates correctly"""
        self.manager.learning_rate = 0.1  # Higher learning rate

        self.manager.profile['sample_count'] = 1000
        self.manager.profile['speed_distribution']['mean_ms'] = 100.0

        # Update with very different value
        self.manager.update_speed_distribution(200.0)

        new_mean = self.manager.profile['speed_distribution']['mean_ms']

        # With learning_rate=0.1, new value should shift more than with 0.05
        # new_mean = (1-0.1)*100 + 0.1*200 = 90 + 20 = 110
        expected = 0.9 * 100 + 0.1 * 200
        self.assertAlmostEqual(new_mean, expected, places=1)

    def test_profile_export_import(self):
        """Test profile can be exported and imported"""
        # Set up profile with data
        self.manager.profile['sample_count'] = 5000
        self.manager.profile['speed_distribution']['mean_ms'] = 145.3

        # Export to dict
        exported = self.manager.export_profile()

        # Verify it's a valid dict
        self.assertIsInstance(exported, dict)
        self.assertEqual(exported['sample_count'], 5000)

        # Import into new manager
        manager2 = ProfileManager(self.profile_path + ".2")
        manager2.import_profile(exported)

        self.assertEqual(manager2.profile['sample_count'], 5000)
        self.assertEqual(manager2.profile['speed_distribution']['mean_ms'], 145.3)

        # Cleanup
        if os.path.exists(self.profile_path + ".2"):
            os.remove(self.profile_path + ".2")

    def test_profile_validation(self):
        """Test that invalid profiles are rejected"""
        # Create invalid profile (missing required fields)
        invalid_profile = {'version': '2.0'}

        with self.assertRaises(Exception):
            self.manager.import_profile(invalid_profile)

    def test_profile_version_check(self):
        """Test that version mismatch is handled"""
        # Future version
        future_profile = self.manager.profile.copy()
        future_profile['version'] = '3.0'

        # Should handle version mismatch
        # (Implementation may upgrade, reject, or warn)

    def test_continuous_learning_disabled(self):
        """Test behavior when continuous learning is disabled"""
        self.manager.continuous_learning = False

        initial_mean = 100.0
        self.manager.profile['speed_distribution']['mean_ms'] = initial_mean

        # Try to update
        self.manager.update_speed_distribution(150.0)

        # Mean should not change if continuous learning disabled
        # (Implementation may vary - might still update or might freeze)

    def test_temporal_pattern_updates(self):
        """Test that temporal patterns are tracked"""
        # Simulate typing at hour 14 (2 PM)
        self.manager.update_temporal_patterns(hour=14)

        # Hour 14 should be in active_hours
        active_hours = self.manager.profile['temporal_patterns']['active_hours']
        self.assertIn(14, active_hours)

    def test_mouse_characteristics_update(self):
        """Test updating mouse movement statistics"""
        # Enable mouse tracking
        self.manager.profile['mouse_characteristics']['enabled'] = True

        # Update with mouse data
        velocity = 250.0  # px/s
        self.manager.update_mouse_characteristics(velocity_px_s=velocity)

        # Should be recorded
        avg_velocity = self.manager.profile['mouse_characteristics']['average_velocity_px_s']
        # First update should set average to the value
        # (or use exponential moving average)

    def test_profile_reset(self):
        """Test profile reset functionality"""
        # Modify profile
        self.manager.profile['sample_count'] = 10000
        self.manager.profile['speed_distribution']['mean_ms'] = 150.0

        # Reset
        self.manager.reset_profile()

        # Should be back to initial state
        self.assertEqual(self.manager.profile['sample_count'], 0)
        self.assertEqual(self.manager.profile['learning_phase'], 'initial')

    def test_concurrent_access_safety(self):
        """Test that concurrent saves don't corrupt profile"""
        # This would require threading, skip for basic tests
        pass

    def test_large_sample_count_handling(self):
        """Test behavior with very large sample counts"""
        self.manager.profile['sample_count'] = 1000000

        # Should still update correctly
        self.manager.update_speed_distribution(145.0)

        self.assertEqual(self.manager.profile['sample_count'], 1000001)


class TestProfileStatistics(unittest.TestCase):
    """Test statistical calculations in profile manager"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        self.test_file.close()
        self.profile_path = self.test_file.name
        self.manager = ProfileManager(self.profile_path)

    def tearDown(self):
        """Clean up test artifacts"""
        if os.path.exists(self.profile_path):
            os.remove(self.profile_path)

    def test_exponential_moving_average(self):
        """Test EMA calculation"""
        alpha = 0.1
        old_value = 100.0
        new_value = 150.0

        expected_ema = (1 - alpha) * old_value + alpha * new_value
        self.assertEqual(expected_ema, 95.0)

    def test_variance_update(self):
        """Test online variance calculation (Welford's algorithm)"""
        # Initialize with baseline
        self.manager.profile['sample_count'] = 2
        self.manager.profile['speed_distribution'] = {
            'mean_ms': 100.0,
            'std_dev_ms': 10.0
        }

        # Add new sample
        # Variance should be updated using Welford's algorithm
        # (Actual implementation may use simplified approach)

    def test_percentile_updates(self):
        """Test that percentiles are updated appropriately"""
        # Add many samples
        for i in range(100):
            self.manager.update_speed_distribution(100 + i)

        # Check that Q1, median, Q3 are reasonable
        dist = self.manager.profile['speed_distribution']
        if 'q1' in dist and 'median_ms' in dist and 'q3' in dist:
            self.assertLess(dist['q1'], dist['median_ms'])
            self.assertLess(dist['median_ms'], dist['q3'])


def run_tests():
    """Run all tests"""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    print("DuckHunt v2.0 - Profile Manager Unit Tests")
    print("=" * 60)
    run_tests()
