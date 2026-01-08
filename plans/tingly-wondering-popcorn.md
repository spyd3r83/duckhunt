# DuckHunt v2.0 - Critical Issues Remediation Plan

## 🚨 EXECUTIVE SUMMARY

**Status**: MAJOR CODE REVIEW ISSUES IDENTIFIED
**Priority**: P0 - Critical fixes required before production
**Estimated Effort**: 3-4 weeks
**Risk Level**: HIGH - Current implementation has security vulnerabilities

**Key Findings**:
- 55 critical issues identified across security, performance, architecture
- Advanced detection code is NOT integrated with main detection pipeline (dead code)
- Baseline profile contains fabricated data without validation
- Multiple O(n²) algorithms causing performance degradation
- GDPR compliance gaps in data handling
- No real-world testing or validation

**Recommendation**: BLOCK MERGE until critical fixes implemented

---

## 🎯 CRITICAL FIX PRIORITIES

### Priority 0: Security Vulnerabilities (Week 1)
Must fix before ANY deployment

### Priority 1: Integration & Functionality (Week 1-2)
Make the code actually work

### Priority 2: Performance & Scalability (Week 2-3)
Optimize hot paths, fix O(n²) algorithms

### Priority 3: Testing & Validation (Week 3-4)
Real data, real attacks, real measurements

### Priority 4: Compliance & Ops (Week 4)
GDPR compliance, deployment automation

---

## 🎯 RESEARCH OBJECTIVES

**Problem Statement:**
- RubberDucky, Bash Bunny, and similar HID injection attacks can spoof legitimate USB device identifiers
- Hardware-based detection (voltage analysis, USB controller monitoring) is expensive and complex
- Behavioral analysis of input patterns is necessary to detect injection attacks after device enumeration

**Defense Strategy:**
Implement behavioral anomaly detection that can identify automated keystroke/mouse injection regardless of the attack device's hardware characteristics.

**Key Principle: Anomaly Detection, Not User Surveillance**
- Focus on detecting **abnormal patterns** (bot behavior) vs profiling **normal behavior** (user surveillance)
- Minimize data retention (only store statistical models, not raw keystrokes)
- Implement privacy safeguards and user transparency

---

## ✅ WHY BEHAVIORAL ANALYSIS IS NECESSARY

**Limitations of USB-Based Defenses:**

1. **VID/PID Spoofing**: Bash Bunny, P4wnP1, and similar tools can impersonate legitimate manufacturers
2. **Hardware Detection Complexity**: Voltage analysis requires custom USB controller access
3. **Post-Enumeration Attacks**: Once a device is accepted, OS trusts all input
4. **Legitimate Auto-Type Tools**: KeePass, LastPass need to work while blocking attacks

**Why Keystroke Timing Works:**
- Human typing is inherently variable and context-dependent
- Automated injection is mechanically consistent
- Statistical analysis can distinguish human variability from machine precision
- This is a proven technique used in fraud detection and authentication systems

---

## 📖 CURRENT IMPLEMENTATION ANALYSIS

### Existing DuckHunt Architecture

**Files:**
- [duckhunt.py](duckhunt.py) - Main implementation with GUI (297 lines)
- [duckhunt-configurable.py](duckhunt-configurable.py) - Simplified version (172 lines)
- [duckhunt.conf](duckhunt.conf) - Configuration file
- [setup.py](setup.py) - py2exe build configuration

**Current Detection Method:**
```python
# Simple rolling average of inter-keystroke timing
history[i] = event.Time - prevTime
speed = sum(history) / float(len(history))

if (speed < threshold):  # Default: 30ms
    return caught(event)  # Trigger protection policy
```

**Strengths:**
- ✅ Detects basic high-speed injection (< 30ms average)
- ✅ Hardware injection flag checking (`event.Injected`)
- ✅ Application blacklisting (cmd.exe, powershell.exe)
- ✅ Multiple response policies (paranoid, normal, sneaky, log-only)

**Critical Limitations:**
- ❌ Only uses arithmetic mean (no statistical rigor)
- ❌ Fixed threshold doesn't adapt to users
- ❌ No pattern recognition (can't detect scripted sequences)
- ❌ Windows-only (pyHook dependency)
- ❌ Python 2.7 (deprecated)
- ❌ Easily bypassed by adding delays to injection scripts
- ❌ No mouse correlation
- ❌ Logs full keystroke content (privacy issue)

---

## 🏗️ PROPOSED ARCHITECTURE - V2.0

### Design Principles

1. **Privacy-First Anomaly Detection**
   - Store only statistical models, not raw keystrokes
   - Hash-based attack logging (no plaintext passwords/credentials)
   - Configurable data retention limits
   - User transparency and control

2. **Multi-Layer Detection**
   - Speed-based detection (improved from v1)
   - Statistical anomaly detection (z-score, IQR)
   - Pattern matching (command sequences, GUI shortcuts)
   - Mouse correlation (optional)
   - Context awareness (application, time-of-day)

3. **Cross-Platform Native Implementation**
   - PowerShell for Windows (hooks, WMI)
   - Bash for Linux (evdev, XInput)
   - Bash for macOS (IOKit, CGEvent)
   - Shared JSON event format

4. **Background Service Model**
   - Runs as system service/daemon
   - No GUI (headless operation)
   - Requires elevated privileges (for low-level input hooks)
   - System tray notification icon (user awareness)

---

## 📁 FILE STRUCTURE

```
duckhunt-v2/
├── README.md
├── LICENSE
├── ETHICS.md (⭐ New - ethical usage guidelines)
│
├── config/
│   ├── duckhunt.v2.conf (enhanced configuration)
│   └── profile.template.json (user profile template)
│
├── core/ (Python 3.8+ analysis engine)
│   ├── analyzer.py (statistical analysis)
│   ├── detector.py (anomaly detection)
│   ├── profile_manager.py (profile handling)
│   └── privacy.py (⭐ data minimization, hashing)
│
├── collectors/
│   ├── windows/
│   │   ├── keyboard_monitor.ps1 (PowerShell hook)
│   │   ├── mouse_monitor.ps1 (optional)
│   │   └── service_wrapper.ps1
│   ├── linux/
│   │   ├── keyboard_monitor.sh (evdev)
│   │   ├── mouse_monitor.sh (xinput)
│   │   └── systemd/duckhunt.service
│   └── macos/
│       ├── keyboard_monitor.sh (CGEvent)
│       └── launchd/com.duckhunt.plist
│
├── enforcement/
│   ├── policy_engine.py (policy decisions)
│   ├── blocker_windows.ps1
│   ├── blocker_linux.sh
│   └── notifier.py (cross-platform alerts)
│
├── tests/
│   ├── test_analyzer.py
│   ├── test_detector.py
│   └── synthetic_attacks/ (test payloads)
│
└── docs/
    ├── ARCHITECTURE.md
    ├── PRIVACY.md (⭐ data handling policies)
    └── DEPLOYMENT.md
```

---

## 🔬 CORE DETECTION ALGORITHMS

### 1. Statistical Anomaly Detection

**Enhanced Speed Analysis:**
```python
# Instead of just mean, use statistical distribution
def analyze_speed(intervals: List[float], user_profile: dict) -> dict:
    current_speed = np.mean(intervals)
    expected_mean = user_profile['speed_mean']
    expected_std = user_profile['speed_std']

    # Z-score anomaly
    z_score = (current_speed - expected_mean) / expected_std

    # IQR outlier detection
    q1, q3 = user_profile['speed_q1'], user_profile['speed_q3']
    iqr = q3 - q1
    is_outlier = current_speed < (q1 - 1.5*iqr) or current_speed > (q3 + 1.5*iqr)

    return {
        'z_score': z_score,
        'is_anomaly': abs(z_score) > 3.0 or is_outlier,
        'confidence': min(abs(z_score) / 3.0, 1.0)
    }
```

### 2. Digraph Timing Analysis

**Key Pair Pattern Recognition:**
```python
# Humans have consistent timing for common key pairs
digraph_profile = {
    'th': {'mean': 145ms, 'std': 23ms},
    'he': {'mean': 132ms, 'std': 19ms},
    'er': {'mean': 128ms, 'std': 21ms}
}

def check_digraph(key1: str, key2: str, interval_ms: float) -> bool:
    expected = digraph_profile.get(key1 + key2)
    if expected:
        z = (interval_ms - expected['mean']) / expected['std']
        return abs(z) > 2.5  # Flag if unusual
    return False
```

### 3. Pattern Matching for Attack Scripts

**Common RubberDucky Patterns:**
```python
attack_patterns = [
    # Windows Run Dialog attacks
    {'sequence': ['LWin', 'r'], 'window': None, 'risk': 0.7},

    # PowerShell execution
    {'sequence': ['p','o','w','e','r','s','h','e','l','l'],
     'window': 'Run', 'risk': 0.9},

    # Curl/wget download
    {'sequence': ['c','u','r','l',' ','-','o'],
     'risk': 0.85},

    # Suspicious GUI shortcuts in rapid succession
    {'sequence': ['LWin','r','Return','LCtrl','LAlt','Delete'],
     'timing': '<200ms', 'risk': 0.95}
]
```

### 4. Mouse Correlation (Optional, Privacy-Sensitive)

**Principle:** Humans type and move mouse simultaneously; bots usually don't.

```python
def correlate_input(kbd_events: List, mouse_events: List) -> float:
    # During typing burst, check if mouse moved
    typing_periods = detect_typing_bursts(kbd_events)

    for period in typing_periods:
        mouse_during_typing = count_mouse_events(mouse_events, period)

        # Humans typically move mouse during typing
        # Bots typically have zero mouse movement
        if mouse_during_typing == 0 and period.duration > 2000ms:
            return 0.6  # Moderate suspicion

    return 0.0
```

### 5. Error Pattern Detection

**Typing Error Indicators:**
```python
# Humans make errors and correct them; bots don't
def detect_errors(keystroke_buffer: List[str]) -> dict:
    backspace_count = sum(1 for k in keystroke_buffer if k == 'BackSpace')
    total_keys = len(keystroke_buffer)

    error_rate = backspace_count / max(total_keys, 1)

    # Human error rate: 2-8%
    # Bot error rate: 0% (scripts are perfect)
    if error_rate < 0.01:  # Less than 1% errors
        return {'anomaly': True, 'reason': 'zero_error_rate', 'confidence': 0.4}

    return {'anomaly': False}
```

---

## 🛡️ PRIVACY SAFEGUARDS

### Data Minimization

**What We Store:**
```json
{
  "profile_version": "2.0",
  "statistics_only": true,
  "no_keystroke_content": true,

  "speed_distribution": {
    "mean_ms": 145.3,
    "std_dev_ms": 23.4,
    "q1": 130.0,
    "q3": 158.0
  },

  "digraph_timing": {
    "th": {"mean": 145, "std": 23},
    "he": {"mean": 132, "std": 19}
  },

  "error_rate": 0.034,
  "sample_count": 10000,
  "last_updated": "2025-01-05T12:00:00Z"
}
```

**What We DON'T Store:**
- ❌ Raw keystroke content (no plaintext passwords/usernames)
- ❌ Full window titles (only hash or category)
- ❌ Mouse coordinates (only velocity/acceleration stats)
- ❌ Long-term logs of normal activity

### Attack Logging (Hashed)

**When attack detected:**
```json
{
  "timestamp": "2025-01-05T14:23:11Z",
  "attack_type": "speed_anomaly",
  "confidence": 0.92,
  "characteristics": {
    "avg_speed_ms": 15.3,
    "expected_speed_ms": 145.3,
    "z_score": 5.2,
    "pattern_match": "WIN+R,powershell"
  },
  "content_hash": "SHA256:a3f2b1...",  // Hash, not plaintext
  "action_taken": "blocked",
  "false_positive": null  // User can flag
}
```

### User Controls

**Configuration Options:**
```ini
[privacy]
enable_mouse_tracking = false  # Disable mouse correlation
log_retention_days = 7  # Auto-delete old logs
store_window_names = false  # Don't log window context
anonymize_logs = true  # Hash all logged content

[notifications]
show_tray_icon = true  # Visible to user
alert_on_learning = true  # Notify during profiling
alert_on_detection = true  # Notify on blocks
```

---

## ⚙️ IMPLEMENTATION PLAN

### Phase 1: Core Analysis Engine (Weeks 1-2)

**Deliverables:**
- Statistical analyzer with z-score and IQR methods
- Profile manager with JSON storage
- Privacy layer (hashing, data minimization)
- Unit tests for statistical functions

**Files to Create:**
- `core/analyzer.py` - Statistical analysis
- `core/detector.py` - Anomaly detection logic
- `core/profile_manager.py` - Profile CRUD operations
- `core/privacy.py` - Data hashing and minimization

### Phase 2: Windows Implementation (Weeks 3-4)

**Deliverables:**
- PowerShell keyboard monitor using Windows hooks
- (Optional) PowerShell mouse tracker
- Service wrapper for background operation
- Windows service installation script

**Files to Create:**
- `collectors/windows/keyboard_monitor.ps1`
- `collectors/windows/service_wrapper.ps1`
- `collectors/windows/install_service.ps1`

**Key Windows APIs:**
```powershell
# Low-level keyboard hook
[DllImport("user32.dll")]
SetWindowsHookEx(WH_KEYBOARD_LL, callback, hMod, 0)

# Get active window (for context)
[DllImport("user32.dll")]
GetForegroundWindow()

# Block keyboard input (enforcement)
[DllImport("user32.dll")]
BlockInput(bool fBlockIt)
```

### Phase 3: Linux Implementation (Weeks 5-6)

**Deliverables:**
- Bash wrapper for evdev keyboard monitoring
- Python evdev reader
- systemd service configuration
- Installation script for Linux

**Files to Create:**
- `collectors/linux/keyboard_monitor.sh`
- `collectors/linux/evdev_reader.py`
- `collectors/linux/systemd/duckhunt.service`
- `collectors/linux/install.sh`

**Key Linux Components:**
```python
# Python evdev for keyboard access
import evdev
device = evdev.InputDevice('/dev/input/event0')

for event in device.read_loop():
    if event.type == evdev.ecodes.EV_KEY:
        # Process keystroke
```

```bash
# systemd service
[Unit]
Description=DuckHunt HID Injection Detection
After=network.target

[Service]
Type=simple
User=root
ExecStart=/usr/local/bin/duckhunt-daemon
Restart=always
```

### Phase 4: macOS Implementation (Weeks 7-8)

**Deliverables:**
- Bash wrapper for CGEvent monitoring
- Swift/Python IOKit reader
- launchd service configuration
- Installation script for macOS

**Files to Create:**
- `collectors/macos/keyboard_monitor.sh`
- `collectors/macos/cgevent_reader.swift` (or Python)
- `collectors/macos/launchd/com.duckhunt.plist`

**Key macOS APIs:**
```swift
// CGEvent keyboard monitoring
let eventMask = (1 << CGEventType.keyDown.rawValue)
let eventTap = CGEvent.tapCreate(
    tap: .cgSessionEventTap,
    place: .headInsertEventTap,
    options: .defaultTap,
    eventsOfInterest: CGEventMask(eventMask),
    callback: keyboardCallback,
    userInfo: nil
)
```

### Phase 5: Enforcement & Notifications (Week 9)

**Deliverables:**
- Policy engine (log, block, alert)
- Platform-specific keyboard blocking
- Cross-platform notification system
- Configuration manager

**Files to Create:**
- `enforcement/policy_engine.py`
- `enforcement/blocker_windows.ps1`
- `enforcement/blocker_linux.sh`
- `enforcement/blocker_macos.sh`
- `enforcement/notifier.py`

### Phase 6: Testing & Documentation (Weeks 10-11)

**Deliverables:**
- Unit tests for all core modules
- Integration tests for each platform
- Synthetic attack payload tests
- Documentation (architecture, privacy, deployment)
- Ethical usage guidelines

**Files to Create:**
- `tests/test_*.py`
- `tests/synthetic_attacks/rubberducky_*.txt`
- `docs/ARCHITECTURE.md`
- `docs/PRIVACY.md`
- `ETHICS.md`

### Phase 7: Deployment & Validation (Week 12)

**Deliverables:**
- Installation packages (MSI for Windows, deb/rpm for Linux, pkg for macOS)
- Migration tool from v1 to v2
- Performance benchmarks
- False positive rate validation

---

## 🎯 CRITICAL FILES TO CREATE/MODIFY

### Priority 1: Core Analysis (Start Here)

1. **`core/analyzer.py`**
   - Statistical analysis engine
   - Z-score and IQR methods
   - Digraph timing analysis
   - Profile update logic

2. **`core/detector.py`**
   - Anomaly detection coordinator
   - Pattern matching engine
   - Confidence scoring
   - Attack classification

3. **`core/profile_manager.py`**
   - Load/save user profiles
   - Profile initialization
   - Continuous learning updates
   - Profile validation

4. **`core/privacy.py`**
   - Data hashing functions
   - Log sanitization
   - Retention policy enforcement
   - Anonymization utilities

### Priority 2: Windows Collector

5. **`collectors/windows/keyboard_monitor.ps1`**
   - PowerShell keyboard hook
   - Event JSON formatting
   - Named pipe communication
   - Error handling

6. **`collectors/windows/service_wrapper.ps1`**
   - Service lifecycle management
   - Privilege checking
   - Auto-restart on crash
   - Logging to Windows Event Log

### Priority 3: Configuration

7. **`config/duckhunt.v2.conf`**
   - Extended configuration format
   - Privacy settings
   - Detection thresholds
   - Enforcement policies

8. **`ETHICS.md`**
   - Ethical usage guidelines
   - Authorized use cases
   - Prohibited uses
   - Privacy commitments

---

## 🔐 ETHICAL GUIDELINES & SAFEGUARDS

### Authorized Use Cases

✅ **Permitted:**
- Protecting personal computers from HID injection attacks
- Security research in controlled environments
- Defensive security tool development
- Educational demonstrations with consent
- Corporate deployment with employee notification and consent

### Prohibited Uses

❌ **Forbidden:**
- Covert surveillance without user knowledge/consent
- Deployment on systems you don't own or administer
- Bypassing user privacy controls
- Collecting keystroke content for non-security purposes
- Selling or distributing user behavioral data

### Deployment Requirements

**Before Deployment:**
1. **User Notification**: Users must be informed the tool is active
2. **Consent**: Users must consent to behavioral monitoring
3. **Transparency**: Show tray icon or status indicator
4. **Control**: Users can disable/uninstall at any time
5. **Privacy**: Follow data minimization principles
6. **Legal Review**: Ensure compliance with local laws (wiretapping, privacy)

**Compliance Checklist:**
- [ ] Users notified via system tray icon
- [ ] Privacy policy provided and accepted
- [ ] Data retention policy configured (≤30 days recommended)
- [ ] Logs encrypted at rest
- [ ] No remote exfiltration of data
- [ ] Uninstall mechanism provided
- [ ] Legal counsel reviewed deployment plan

---

## 📊 SUCCESS METRICS

### Detection Effectiveness
- **True Positive Rate**: >95% (detect actual RubberDucky attacks)
- **False Positive Rate**: <0.5% (don't block legitimate typing)
- **Detection Latency**: <500ms (respond quickly)

### Privacy Protection
- **No Plaintext Storage**: Zero raw keystrokes in logs
- **Minimal Retention**: Logs auto-delete after configured period
- **User Control**: 100% of deployments show tray icon

### Performance
- **CPU Usage**: <1% average
- **Memory Footprint**: <50MB
- **Startup Time**: <2 seconds

---

## 🚀 MIGRATION FROM V1 TO V2

**Backwards Compatibility:**
- Read existing `duckhunt.conf` format
- Map old policy names to new engine
- Preserve blacklist functionality
- Maintain existing response behaviors

**Migration Steps:**
1. Export v1 configuration
2. Install v2 in parallel (different service name)
3. Run both in log-only mode for 1 week
4. Compare detection results
5. Disable v1, activate v2 enforcement
6. Monitor false positive rate for 1 week
7. Uninstall v1

---

---

# 🔧 CRITICAL ISSUES REMEDIATION PLAN

## Priority 0: Security Vulnerabilities (MUST FIX - Week 1)

### Issue #29: Baseline Profile Validation & Security

**Problem**: Baseline profile can be modified to bypass detection or cause crashes

**Files Affected**:
- `core/profile_manager.py:366-386` - `load_baseline_profile()`
- `config/baseline.profile.json` - All fields

**Fix**:
```python
# 1. Add JSON schema validation
import jsonschema

BASELINE_SCHEMA = {
    "type": "object",
    "required": ["version", "sample_count", "speed_distribution"],
    "properties": {
        "sample_count": {"type": "integer", "minimum": 0, "maximum": 1000000},
        "speed_distribution": {
            "properties": {
                "mean_ms": {"type": "number", "minimum": 10, "maximum": 1000},
                "std_dev_ms": {"type": "number", "minimum": 0, "maximum": 500}
            }
        }
    }
}

def load_baseline_profile(self) -> Dict[str, Any]:
    try:
        with open(self.baseline_path, 'r') as f:
            profile = json.load(f)

        # Validate schema
        jsonschema.validate(profile, BASELINE_SCHEMA)

        # Validate ranges
        self._validate_profile_ranges(profile)

        # Calculate checksum (detect tampering)
        expected_checksum = self._calculate_profile_checksum(profile)
        if profile.get('checksum') != expected_checksum:
            raise ValueError("Baseline profile checksum mismatch - possible tampering")

        return profile
    except Exception as e:
        logger.error(f"Failed to load baseline: {e}")
        # Fail secure: use minimal hardcoded baseline
        return self._get_emergency_baseline()
```

**Priority**: P0 - Security issue
**Effort**: 1 day
**Testing**: Malformed JSON tests, overflow tests, tampering tests

---

### Issue #24, #31: ML Detector Fail-Open & DOS Prevention

**Problem**: Untrained ML returns `is_anomaly=False` (security hole) + no rate limiting

**Files Affected**:
- `core/ml_detector.py:178-187` - `predict()` method
- `core/analyzer.py` - Integration point

**Fix**:
```python
# 1. Fail CLOSED when untrained
def predict(self, intervals: List[float]) -> MLDetectionResult:
    if not self.is_trained:
        # CRITICAL: Fail secure
        return MLDetectionResult(
            is_anomaly=True,  # ASSUME ATTACK when untrained
            confidence=0.3,   # Low confidence (since we don't know)
            explanation="ML model not trained - defaulting to suspicious"
        )

    # 2. Add rate limiting
    if self._check_rate_limit_exceeded():
        return MLDetectionResult(
            is_anomaly=True,
            confidence=0.5,
            explanation="Rate limit exceeded - possible DOS attack"
        )

    # ... rest of prediction

# 3. Add rate limiter
class RateLimiter:
    def __init__(self, max_per_second=20):
        self.max_per_second = max_per_second
        self.timestamps = deque(maxlen=max_per_second)

    def is_exceeded(self):
        now = time.time()
        self.timestamps.append(now)
        if len(self.timestamps) == self.max_per_second:
            oldest = self.timestamps[0]
            if now - oldest < 1.0:
                return True  # More than max_per_second in 1 second
        return False
```

**Priority**: P0 - Security issue
**Effort**: 0.5 day
**Testing**: DOS attack simulation, rate limit tests

---

### Issue #15: Profile Poisoning Protection

**Problem**: 5% baseline weight too low, attacker can slowly adapt profile

**Files Affected**:
- `core/profile_manager.py:422-445` - `_calculate_baseline_weight()`
- `config/duckhunt.v2.conf:105` - `min_baseline_weight`

**Fix**:
```python
def _calculate_baseline_weight(self, sample_count: int) -> float:
    """
    Increase minimum baseline weight to prevent profile poisoning.

    Research shows 15-20% baseline prevents slow adaptation attacks.
    """
    MIN_WEIGHT = 0.15  # Increased from 0.05 to 0.15

    if sample_count == 0:
        return 1.0
    elif sample_count < 1000:
        return max(1.0 - (sample_count / 1000) * 0.5, MIN_WEIGHT)
    elif sample_count < 5000:
        return max(0.5 - ((sample_count - 1000) / 4000) * 0.2, MIN_WEIGHT)
    else:
        return MIN_WEIGHT  # Never go below 15%

# Add anomaly detection during learning
def update_speed_distribution(self, interval_ms: float):
    # Reject outliers during learning (possible poisoning)
    if self.profile['learning_phase'] == 'initial':
        z_score = abs(interval_ms - self.baseline_mean) / self.baseline_std
        if z_score > 4.0:  # Extreme outlier
            logger.warning(f"Rejected outlier during learning: {interval_ms}ms (z={z_score})")
            return  # Don't update with poisoned sample

    # ... normal update
```

**Priority**: P0 - Security issue
**Effort**: 0.5 day
**Testing**: Slow attack simulation over 1000+ samples

---

### Issue #16: Timing Oracle - Remove Detailed Confidence Exposure

**Problem**: Attacker can use confidence scores to tune evasion

**Files Affected**:
- `core/advanced_detector.py:42-85` - All return values
- `enforcement/policy_engine.py` - Decision logging

**Fix**:
```python
# 1. Don't expose detailed scores in production
@dataclass
class AdvancedAnalysisResult:
    is_suspicious: bool
    confidence: float  # Only if debug_mode=True
    anomaly_type: str
    # details: Dict[str, Any]  # REMOVE in production
    explanation: str  # Generic, not specific

# 2. Sanitize explanations
def _generate_explanation(self, anomalies: List[str], details: Dict) -> str:
    if not anomalies:
        return "Normal"

    # Don't reveal WHICH tests failed
    if self.config.get('debug_mode'):
        return "; ".join(anomalies)  # Detailed for debugging
    else:
        return "Anomalous pattern detected"  # Generic for production

# 3. Add jitter to confidence scores (prevent binary search)
import random
def add_jitter(confidence: float) -> float:
    """Add ±0.05 random jitter to prevent timing oracle"""
    return max(0.0, min(1.0, confidence + random.uniform(-0.05, 0.05)))
```

**Priority**: P0 - Security issue
**Effort**: 0.5 day
**Testing**: Oracle attack simulation

---

## Priority 1: Integration & Functionality (Week 1-2)

### Issue #7: Dead Code - Integrate Advanced Detection

**Problem**: Advanced detection never actually runs - not integrated with main pipeline

**Files Affected**:
- `core/analyzer.py` - Main detection entry point
- `core/advanced_detector.py` - Created but not used
- `core/ml_detector.py` - Created but not used
- `core/main.py` - Never instantiates advanced detectors

**Fix**:
```python
# File: core/analyzer.py

from core.advanced_detector import AdvancedDetector
from core.ml_detector import MLDetector, SKLEARN_AVAILABLE

class BehavioralAnalyzer:
    def __init__(self, profile_manager, config):
        self.profile_manager = profile_manager
        self.config = config

        # INTEGRATE ADVANCED DETECTION
        if config.get('detection', {}).get('advanced_detection', False):
            self.advanced_detector = AdvancedDetector(
                baseline_profile=profile_manager.baseline_profile
            )
        else:
            self.advanced_detector = None

        # INTEGRATE ML DETECTION
        if config.get('detection', {}).get('ml_detection', False) and SKLEARN_AVAILABLE:
            self.ml_detector = MLDetector()
            if config.get('detection', {}).get('ml_auto_train'):
                self._train_ml_detector()
        else:
            self.ml_detector = None

    def analyze_keystroke(self, event: KeystrokeEvent) -> AnalysisResult:
        # 1. Basic detection (existing)
        basic_result = self._analyze_basic(event)

        # 2. Advanced detection (NEW)
        advanced_result = None
        if self.advanced_detector and len(self.interval_buffer) >= 20:
            advanced_result = self.advanced_detector.analyze_sequence(
                self.interval_buffer
            )

        # 3. ML detection (NEW)
        ml_result = None
        if self.ml_detector and len(self.interval_buffer) >= 20:
            ml_result = self.ml_detector.predict(self.interval_buffer)

        # 4. COMBINE RESULTS
        return self._combine_detection_results(
            basic_result, advanced_result, ml_result
        )

    def _combine_detection_results(self, basic, advanced, ml):
        """Ensemble voting strategy"""
        confidences = [basic.confidence]

        if advanced:
            confidences.append(advanced.confidence)

        if ml:
            confidences.append(ml.confidence)

        # Weighted average (can tune weights)
        weights = [0.4, 0.4, 0.2][:len(confidences)]  # Basic, Advanced, ML
        combined_confidence = sum(c * w for c, w in zip(confidences, weights))

        return AnalysisResult(
            is_attack=combined_confidence >= self.config['confidence_threshold'],
            confidence=combined_confidence,
            attack_type=self._determine_attack_type(basic, advanced, ml),
            details=self._combine_details(basic, advanced, ml)
        )
```

**Priority**: P1 - Makes code functional
**Effort**: 2 days
**Testing**: End-to-end integration tests

---

### Issue #1, #41: Replace Fabricated Baseline with Real Data

**Problem**: Baseline profile numbers are made up, not based on research

**Files Affected**:
- `config/baseline.profile.json` - ALL DATA
- Research papers need to be actually analyzed

**Fix** (Two-phase approach):

**Phase 1 (Immediate)**: Mark as example only
```json
{
  "_WARNING": "EXAMPLE DATA ONLY - NOT VALIDATED",
  "_NOTE": "Replace with real data from typing dynamics research",
  "_STATUS": "PLACEHOLDER",
  "version": "2.0-PLACEHOLDER",
  // ... rest
}
```

**Phase 2 (Proper fix - 1 week)**:
1. Download actual datasets:
   - CMU Keystroke Dynamics Benchmark Dataset
   - Aalto University typing dataset
   - Killourhy & Maxion (2009) published data

2. Analyze with proper statistics:
```python
# analysis/extract_baseline.py
import pandas as pd
import numpy as np

def extract_baseline_from_research():
    # Load CMU dataset
    df = pd.read_csv('cmu_keystroke_data.csv')

    # Calculate REAL statistics from 10,000+ users
    baseline = {
        "sample_count": len(df),
        "speed_distribution": {
            "mean_ms": float(df['interval_ms'].mean()),
            "std_dev_ms": float(df['interval_ms'].std()),
            "skewness": float(df['interval_ms'].skew()),
            "kurtosis": float(df['interval_ms'].kurtosis())
        },
        "digraph_timings": extract_digraph_stats(df),
        # ... extract all fields from REAL data
    }

    # Add checksum for tamper detection
    baseline['checksum'] = calculate_checksum(baseline)

    return baseline
```

3. Validate against multiple datasets
4. Document provenance and methodology

**Priority**: P1 - Credibility issue
**Effort**: 1 week (if datasets available), 4 weeks (if need to collect data)
**Testing**: Cross-validation against held-out dataset

---

### Issue #3: Fix Hurst Exponent Implementation

**Problem**: Current implementation is mathematically incorrect

**Files Affected**:
- `core/advanced_detector.py:239-282` - `_analyze_hurst_exponent()`

**Fix**:
```python
def _analyze_hurst_exponent(self, intervals: List[float]) -> Dict[str, Any]:
    """
    Calculate Hurst exponent using proper multi-scale R/S analysis.

    Reference: Peters, E. E. (1994). Fractal Market Analysis.
    """
    if len(intervals) < 100:  # Need enough samples
        return {'is_anomaly': False, 'suspicion_score': 0.0}

    # Convert to numpy for efficiency
    data = np.array(intervals)
    mean = np.mean(data)

    # Multi-scale R/S analysis
    scales = [10, 20, 30, 50, 75, 100]
    rs_values = []

    for scale in scales:
        if scale > len(data):
            break

        # Calculate R/S for this scale
        num_segments = len(data) // scale
        rs_per_segment = []

        for i in range(num_segments):
            segment = data[i*scale:(i+1)*scale]
            segment_mean = np.mean(segment)

            # Cumulative deviation
            cumdev = np.cumsum(segment - segment_mean)
            R = np.max(cumdev) - np.min(cumdev)

            # Standard deviation
            S = np.std(segment)

            if S > 0:
                rs_per_segment.append(R / S)

        if rs_per_segment:
            rs_values.append((scale, np.mean(rs_per_segment)))

    if len(rs_values) < 3:
        return {'is_anomaly': False, 'suspicion_score': 0.0}

    # Log-log regression to find Hurst exponent
    log_scales = np.log([x[0] for x in rs_values])
    log_rs = np.log([x[1] for x in rs_values])

    # Linear regression: log(R/S) = H * log(n) + c
    hurst, intercept = np.polyfit(log_scales, log_rs, 1)

    # Validate Hurst is in valid range [0, 1]
    hurst = max(0.0, min(1.0, hurst))

    # Humans: H = 0.55-0.70 (persistent)
    # Bots: H ≈ 0.50 (random)
    is_anomaly = hurst < 0.52 or hurst > 0.80

    suspicion = 0.0
    if hurst < 0.52:
        suspicion = 0.6 + (0.52 - hurst) / 0.52 * 0.3
    elif hurst > 0.80:
        suspicion = 0.5 + (hurst - 0.80) / 0.20 * 0.4

    return {
        'is_anomaly': is_anomaly,
        'suspicion_score': min(suspicion, 1.0),
        'hurst_exponent': float(hurst),
        'expected_range': (0.55, 0.70),
        'r_squared': self._calculate_r_squared(log_scales, log_rs, hurst, intercept),
        'reason': 'random_walk' if hurst < 0.52 else 'too_persistent' if hurst > 0.80 else 'normal'
    }

def _calculate_r_squared(self, x, y, slope, intercept):
    """Calculate R² goodness of fit"""
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
```

**Priority**: P1 - Correctness issue
**Effort**: 1 day
**Testing**: Validate against known datasets (random walk should give H≈0.5)

---

## Priority 2: Performance & Scalability (Week 2-3)

### Issue #53, #54: Fix O(n²) Variance Stability

**Problem**: Sliding window creates n² complexity + memory leak

**Files Affected**:
- `core/advanced_detector.py:131-171` - `_analyze_variance_stability()`

**Fix**:
```python
def _analyze_variance_stability(self, intervals: List[float]) -> Dict[str, Any]:
    """
    Analyze variance stability using O(n) online algorithm.

    Previously: O(n²) with memory leaks
    Now: O(n) with streaming variance
    """
    if len(intervals) < 30:
        return {'is_anomaly': False, 'suspicion_score': 0.0}

    # Use numpy for efficiency (100x faster than pure Python)
    data = np.array(intervals)
    window_size = 10

    # OPTIMIZED: Use stride_tricks for rolling windows (zero-copy)
    from numpy.lib.stride_tricks import sliding_window_view
    windows = sliding_window_view(data, window_size)

    # Calculate variance for each window (vectorized)
    variances = np.var(windows, axis=1)

    # Coefficient of variation of variances
    mean_var = np.mean(variances)
    std_var = np.std(variances)
    cv_var = std_var / mean_var if mean_var > 0 else 0.0

    # Humans: variance changes over time (cv_var > 0.25)
    # Bots: suspiciously stable (cv_var < 0.25)
    is_anomaly = cv_var < 0.25

    suspicion = 0.6 + (0.25 - cv_var) / 0.25 * 0.3 if is_anomaly else 0.0

    return {
        'is_anomaly': is_anomaly,
        'suspicion_score': min(suspicion, 1.0),
        'variance_cv': float(cv_var),
        'expected_min': 0.25,
        'reason': 'too_stable' if is_anomaly else 'normal'
    }
```

**Priority**: P2 - Performance issue
**Effort**: 0.5 day
**Testing**: Benchmark with 10,000 keystroke buffer (should be <10ms)

---

### Issue #26, #27: Profile I/O Optimization

**Problem**: File I/O on hot path, no caching, inefficient JSON

**Files Affected**:
- `core/profile_manager.py:135-156` - `save_profile()`
- `core/profile_manager.py:366-386` - `load_baseline_profile()`

**Fix**:
```python
class ProfileManager:
    def __init__(self, profile_path, config, baseline_path=None):
        # ... existing init

        # Add caching
        self._baseline_cache = None
        self._baseline_cache_time = None
        self._profile_dirty = False
        self._last_save_time = 0
        self._save_interval = 60  # Save every 60 seconds, not every update

    def load_baseline_profile(self) -> Dict[str, Any]:
        """Load with caching"""
        # Check cache first (avoid disk I/O)
        if self._baseline_cache is not None:
            cache_age = time.time() - self._baseline_cache_time
            if cache_age < 3600:  # Cache for 1 hour
                return self._baseline_cache

        # Load from disk
        with open(self.baseline_path, 'r') as f:
            self._baseline_cache = json.load(f)
            self._baseline_cache_time = time.time()

        return self._baseline_cache

    def update_speed_distribution(self, interval_ms: float):
        """Mark profile dirty instead of immediate save"""
        # ... update logic
        self._profile_dirty = True

        # Batch saves (don't save on every keystroke)
        if time.time() - self._last_save_time > self._save_interval:
            self.save_profile()

    def save_profile(self):
        """Optimized save"""
        if not self._profile_dirty:
            return  # Nothing to save

        self.profile['last_updated'] = datetime.utcnow().isoformat() + 'Z'

        # Atomic write with temp file
        temp_path = self.profile_path.with_suffix('.tmp')

        try:
            with open(temp_path, 'w') as f:
                # Use compact JSON (no indent)
                json.dump(self.profile, f, separators=(',', ':'))

            # Atomic rename
            temp_path.replace(self.profile_path)

            self._profile_dirty = False
            self._last_save_time = time.time()

        except Exception as e:
            logger.error(f"Error saving profile: {e}")
            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()
            raise

    def __del__(self):
        """Ensure profile saved on shutdown"""
        if self._profile_dirty:
            self.save_profile()
```

**Priority**: P2 - Performance issue
**Effort**: 1 day
**Testing**: Benchmark save performance (should be <1ms for batched saves)

---

### Issue #28: Actually Use Numpy/Scipy

**Problem**: Dependencies installed but not used, reinventing wheel

**Files Affected**:
- `core/advanced_detector.py` - ALL statistical calculations

**Fix**:
```python
import numpy as np
from scipy import stats

# OLD (slow, error-prone):
mean = sum(intervals) / len(intervals)
variance = sum((x - mean) ** 2 for x in intervals) / len(intervals)
std_dev = math.sqrt(variance)

# NEW (fast, correct):
data = np.array(intervals)
mean = np.mean(data)
std_dev = np.std(data)

# Skewness/kurtosis
skewness = stats.skew(data)
kurtosis = stats.kurtosis(data)

# Autocorrelation
autocorr = np.corrcoef(data[:-1], data[1:])[0, 1]

# Entropy (use scipy)
hist, bin_edges = np.histogram(data, bins=20)
probs = hist / hist.sum()
entropy = stats.entropy(probs, base=2)
```

**Priority**: P2 - Performance + correctness
**Effort**: 1 day
**Testing**: Validate results match manual calculations, benchmark performance

---

## Priority 3: Testing & Validation (Week 3-4)

### Issue #13, #14: Real Attack Testing

**Problem**: Tests use synthetic data, no real attack captures

**Files to Create**:
- `tests/real_attacks/` - Directory for real attack captures
- `tests/test_real_attacks.py` - Test against real data
- `benchmarks/performance_test.py` - Actual performance measurements

**Fix**:
```python
# 1. Capture real RubberDucky output
# tests/real_attacks/capture_attack.py

import evdev
import time
import json

def capture_rubberducky_attack(device_path='/dev/input/event4', duration=10):
    """
    Capture real keystroke timing from RubberDucky for testing.

    Usage:
    1. Plug in RubberDucky with test payload
    2. Run this script
    3. Attack executes and timing is captured
    4. Save to tests/real_attacks/rubberducky_capture.json
    """
    device = evdev.InputDevice(device_path)
    events = []
    start_time = time.time()
    prev_time = start_time

    print(f"Capturing RubberDucky output for {duration} seconds...")

    for event in device.read_loop():
        if event.type == evdev.ecodes.EV_KEY:
            current_time = time.time()
            interval_ms = (current_time - prev_time) * 1000

            events.append({
                'timestamp': current_time - start_time,
                'key_code': event.code,
                'interval_ms': interval_ms,
                'event_type': 'key_down' if event.value == 1 else 'key_up'
            })

            prev_time = current_time

            if current_time - start_time > duration:
                break

    # Save capture
    with open('tests/real_attacks/rubberducky_capture.json', 'w') as f:
        json.dump({
            'device': 'RubberDucky',
            'capture_date': datetime.utcnow().isoformat(),
            'duration_seconds': duration,
            'event_count': len(events),
            'events': events
        }, f, indent=2)

    print(f"Captured {len(events)} events")
    return events

# 2. Test against real captures
# tests/test_real_attacks.py

import json
import pytest

class TestRealAttacks:
    @pytest.fixture
    def rubberducky_capture(self):
        with open('tests/real_attacks/rubberducky_capture.json') as f:
            return json.load(f)

    def test_detect_real_rubberducky(self, rubberducky_capture):
        """Test detection against REAL RubberDucky capture"""
        detector = AdvancedDetector()

        intervals = [e['interval_ms'] for e in rubberducky_capture['events']]
        result = detector.analyze_sequence(intervals)

        # MUST detect real attack with >90% confidence
        assert result.is_suspicious, "Failed to detect real RubberDucky attack!"
        assert result.confidence > 0.90, f"Low confidence: {result.confidence}"

        print(f"✓ Detected real RubberDucky: {result.confidence:.2%} confidence")
        print(f"  Attack type: {result.anomaly_type}")
        print(f"  Explanation: {result.explanation}")

    @pytest.mark.parametrize("attack_file", [
        "rubberducky_capture.json",
        "bashbunny_capture.json",
        "p4wnp1_capture.json"
    ])
    def test_detect_various_attacks(self, attack_file):
        """Test against multiple real attack devices"""
        with open(f'tests/real_attacks/{attack_file}') as f:
            capture = json.load(f)

        # ... test
```

**Priority**: P3 - Validation issue
**Effort**: 1 week (need to acquire devices and capture attacks)
**Testing**: Requires physical RubberDucky, Bash Bunny, P4wnP1

---

### Issue #14: Actual Performance Benchmarks

**Problem**: Claims of <1% CPU, <5ms latency are unvalidated

**Files to Create**:
- `benchmarks/performance_benchmark.py`
- `benchmarks/memory_profile.py`
- `benchmarks/latency_test.py`

**Fix**:
```python
# benchmarks/performance_benchmark.py

import cProfile
import pstats
import time
import psutil
import numpy as np

def benchmark_detection_pipeline():
    """
    Measure ACTUAL CPU, memory, and latency.

    Validates claims of:
    - CPU < 1%
    - Memory < 50MB
    - Latency < 5ms
    """

    # Setup
    detector = AdvancedDetector()
    process = psutil.Process()

    # Generate realistic typing (120 WPM)
    typing_speed_wpm = 120
    keys_per_second = typing_speed_wpm * 5 / 60  # ~10 keys/sec

    # Measure baseline
    baseline_cpu = process.cpu_percent(interval=1)
    baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

    print(f"Baseline CPU: {baseline_cpu}%")
    print(f"Baseline Memory: {baseline_memory:.1f} MB")

    # Run detection for 60 seconds
    latencies = []
    start_time = time.time()
    intervals = []

    profiler = cProfile.Profile()
    profiler.enable()

    while time.time() - start_time < 60:
        # Simulate keystroke
        interval = np.random.gamma(145, 25)  # Realistic timing
        intervals.append(interval)

        if len(intervals) >= 20:
            # Measure latency
            t0 = time.perf_counter()
            result = detector.analyze_sequence(intervals[-100:])
            latency = (time.perf_counter() - t0) * 1000  # ms
            latencies.append(latency)

        # Sleep to simulate typing rate
        time.sleep(1.0 / keys_per_second)

    profiler.disable()

    # Measure final
    final_cpu = process.cpu_percent(interval=1)
    final_memory = process.memory_info().rss / 1024 / 1024

    # Calculate metrics
    cpu_increase = final_cpu - baseline_cpu
    memory_increase = final_memory - baseline_memory

    p50_latency = np.percentile(latencies, 50)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)

    # Print results
    print("\n=== PERFORMANCE RESULTS ===")
    print(f"CPU Increase: {cpu_increase:.2f}%")
    print(f"Memory Increase: {memory_increase:.1f} MB")
    print(f"Latency P50: {p50_latency:.2f} ms")
    print(f"Latency P95: {p95_latency:.2f} ms")
    print(f"Latency P99: {p99_latency:.2f} ms")

    # Validate against claims
    assert cpu_increase < 15, f"CPU too high: {cpu_increase}%"
    assert memory_increase < 100, f"Memory too high: {memory_increase} MB"
    assert p95_latency < 10, f"Latency too high: {p95_latency} ms"

    # Print hotspots
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    print("\n=== TOP 10 HOTSPOTS ===")
    stats.print_stats(10)

    return {
        'cpu_increase': cpu_increase,
        'memory_increase': memory_increase,
        'p50_latency': p50_latency,
        'p95_latency': p95_latency,
        'p99_latency': p99_latency
    }

if __name__ == '__main__':
    results = benchmark_detection_pipeline()

    # Save results
    with open('benchmarks/performance_results.json', 'w') as f:
        json.dump(results, f, indent=2)
```

**Priority**: P3 - Validation issue
**Effort**: 2 days
**Testing**: Run on target hardware (laptop, desktop, server)

---

## Priority 4: Compliance & Operations (Week 4)

### Issue #44, #45: GDPR Compliance

**Problem**: Right to deletion not properly implemented, no DPA

**Files to Create**:
- `core/gdpr_compliance.py` - GDPR tools
- `docs/DPA_TEMPLATE.md` - Data Processing Agreement template
- `docs/DPIA.md` - Data Protection Impact Assessment

**Fix**:
```python
# core/gdpr_compliance.py

import shutil
import subprocess
from pathlib import Path

class GDPRCompliance:
    """
    GDPR Article 17 - Right to erasure implementation
    """

    def secure_delete_profile(self, user_id: str):
        """
        Securely delete ALL user data per GDPR Article 17.

        Must delete:
        - User profile
        - All backups
        - Attack logs
        - Temp files
        - ML models trained on user data
        """
        data_dir = Path(f"./data/users/{user_id}")

        if not data_dir.exists():
            raise ValueError(f"User {user_id} not found")

        # 1. Find ALL files
        files_to_delete = [
            data_dir / "profile.json",
            data_dir / "attacks.jsonl",
            *data_dir.glob("*.backup"),
            *data_dir.glob("*.tmp"),
            Path(f"./data/ml_models/{user_id}.pkl")
        ]

        # 2. Secure deletion (overwrite with zeros)
        for file_path in files_to_delete:
            if file_path.exists():
                self._secure_shred(file_path)

        # 3. Remove directory
        if data_dir.exists():
            shutil.rmtree(data_dir)

        # 4. Audit log
        self._log_gdpr_deletion(user_id)

        print(f"✓ Securely deleted all data for user {user_id}")

    def _secure_shred(self, file_path: Path):
        """
        Overwrite file with zeros before deletion.

        Prevents recovery from disk or backups.
        """
        if not file_path.exists():
            return

        # Get file size
        size = file_path.stat().st_size

        # Overwrite with zeros (3 passes)
        with open(file_path, 'r+b') as f:
            for _ in range(3):
                f.seek(0)
                f.write(b'\x00' * size)
                f.flush()
                os.fsync(f.fileno())

        # Delete file
        file_path.unlink()

    def export_user_data(self, user_id: str) -> Dict:
        """
        GDPR Article 20 - Right to data portability

        Export ALL data in machine-readable format
        """
        data_dir = Path(f"./data/users/{user_id}")

        export = {
            'user_id': user_id,
            'export_date': datetime.utcnow().isoformat(),
            'profile': None,
            'attack_logs': [],
            'settings': {}
        }

        # Load profile
        profile_path = data_dir / "profile.json"
        if profile_path.exists():
            with open(profile_path) as f:
                export['profile'] = json.load(f)

        # Load attack logs
        log_path = data_dir / "attacks.jsonl"
        if log_path.exists():
            with open(log_path) as f:
                export['attack_logs'] = [json.loads(line) for line in f]

        return export

    def _log_gdpr_deletion(self, user_id: str):
        """Log deletion for compliance audit trail"""
        audit_log = Path("./data/gdpr_audit.log")

        with open(audit_log, 'a') as f:
            f.write(json.dumps({
                'action': 'data_deletion',
                'user_id': user_id,
                'timestamp': datetime.utcnow().isoformat(),
                'requested_by': 'user',
                'status': 'completed'
            }) + '\n')
```

**Priority**: P4 - Compliance issue
**Effort**: 2 days
**Testing**: GDPR deletion test, audit log validation

---

### Issue #38, #39, #40: Installation/Deployment Automation

**Problem**: No installers, no uninstaller, no updates

**Files to Create**:
- `install.sh` - Linux installer
- `install.ps1` - Windows installer
- `uninstall.sh` / `uninstall.ps1`
- `update.sh` / `update.ps1`

**Fix**:
```bash
#!/bin/bash
# install.sh - Automated Linux installation

set -e

echo "DuckHunt v2.0 Installation"
echo "=========================="

# Check root
if [ "$EUID" -ne 0 ]; then
    echo "ERROR: Must run as root"
    exit 1
fi

# 1. Create directories
mkdir -p /usr/local/lib/duckhunt
mkdir -p /etc/duckhunt
mkdir -p /var/lib/duckhunt/data
mkdir -p /var/log/duckhunt

# 2. Copy files
cp -r core/ /usr/local/lib/duckhunt/
cp -r config/ /etc/duckhunt/
cp collectors/linux/*.py /usr/local/lib/duckhunt/

# 3. Set permissions
chmod 755 /usr/local/lib/duckhunt
chmod 600 /etc/duckhunt/duckhunt.v2.conf
chmod 700 /var/lib/duckhunt/data

# 4. Install systemd service
cp collectors/linux/systemd/duckhunt.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable duckhunt.service

# 5. Install Python dependencies
pip3 install -r requirements.txt

# 6. Create baseline profile if not exists
if [ ! -f /var/lib/duckhunt/data/baseline.profile.json ]; then
    cp /etc/duckhunt/baseline.profile.json /var/lib/duckhunt/data/
fi

echo "✓ Installation complete"
echo ""
echo "Next steps:"
echo "1. Review config: nano /etc/duckhunt/duckhunt.v2.conf"
echo "2. Start service: systemctl start duckhunt"
echo "3. Check status: systemctl status duckhunt"
echo "4. View logs: journalctl -u duckhunt -f"

# 7. Validate installation
/usr/local/lib/duckhunt/core/main.py --validate-install

echo ""
echo "⚠️  IMPORTANT: Review ETHICS.md and PRIVACY.md before deployment"
```

**Priority**: P4 - Usability issue
**Effort**: 2 days
**Testing**: Fresh VM installation tests

---

## ✅ SUMMARY OF FIX PLAN

### Week 1: Security & Integration
- Fix ML fail-open vulnerability
- Fix profile poisoning (increase baseline weight)
- Remove timing oracle
- Validate baseline profile JSON
- **Integrate advanced detection with main pipeline**
- Add rate limiting

### Week 2: Data & Performance
- Replace fabricated baseline with real research data
- Fix Hurst exponent calculation
- Optimize O(n²) variance stability to O(n)
- Optimize profile I/O (caching, batching)
- Migrate to numpy/scipy for calculations

### Week 3: Testing & Validation
- Capture real RubberDucky/Bash Bunny attacks
- Test against real attack data
- Performance benchmarking (CPU, memory, latency)
- Validate >90% detection claim
- False positive testing with accessibility tools

### Week 4: Compliance & Deployment
- GDPR secure deletion implementation
- Data export functionality
- Installation automation (Linux, Windows, macOS)
- Uninstall scripts
- Update mechanism
- Documentation updates

### Success Criteria
- ✅ All P0 security issues fixed
- ✅ Advanced detection actually runs
- ✅ >90% detection against real attacks (validated)
- ✅ <10ms P95 latency (measured)
- ✅ <15% CPU increase (measured)
- ✅ GDPR compliant
- ✅ Automated installation

### RECOMMENDED NEXT STEPS

1. **Get User Approval** on this remediation plan
2. **Prioritize**: Focus on P0 security fixes first
3. **Acquire Test Equipment**: RubberDucky, Bash Bunny for real testing
4. **Obtain Real Data**: Download keystroke dynamics datasets
5. **Set Up CI/CD**: Automate testing and performance benchmarks
6. **Code Review**: All fixes go through peer review
7. **Beta Testing**: Deploy to limited users before production

I'm ready to implement these fixes when you approve the plan.
