"""
DuckHunt v2.0 - Test Runner

Runs all unit tests and integration tests for DuckHunt.
"""

import sys
import os
import unittest

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def run_all_tests():
    """Discover and run all tests"""
    print("=" * 70)
    print("DuckHunt v2.0 - Complete Test Suite")
    print("=" * 70)
    print()

    # Discover all test files
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(__file__)
    suite = loader.discover(start_dir, pattern='test_*.py')

    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print()
    print("=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print()

    if result.wasSuccessful():
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


def run_specific_module(module_name):
    """Run tests from a specific module"""
    print(f"Running tests from: {module_name}")
    print("=" * 70)

    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(module_name)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return 0 if result.wasSuccessful() else 1


def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        # Run specific test module
        module = sys.argv[1]
        return run_specific_module(module)
    else:
        # Run all tests
        return run_all_tests()


if __name__ == '__main__':
    sys.exit(main())
