# DuckHunt v2.0 Enhancement Plan
## Advanced Behavioral HID Injection Detection System

**Context: Legitimate Security Research**
This plan addresses the real-world limitation that USB device whitelisting (VID/PID filtering) is defeated by sophisticated attacks like Bash Bunny that can spoof legitimate device identifiers. Behavioral analysis represents a valid defense-in-depth layer for detecting automated keystroke injection.

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

## ✅ NEXT STEPS

1. **Review this plan** - Ensure it aligns with your research objectives
2. **Clarify scope** - Which platforms are priority? (Windows first, then Linux/macOS?)
3. **Define timeline** - Is 12-week timeline acceptable?
4. **Approve privacy approach** - Confirm data minimization strategy
5. **Begin implementation** - Start with core analyzer module

I'm ready to proceed with implementation once you approve this plan.
