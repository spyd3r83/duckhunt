# DuckHunt v2.0 - Architecture Overview

## System Architecture

DuckHunt v2.0 is a modular, cross-platform HID injection detection system designed with privacy-first principles.

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Platform Collectors                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  PowerShell  │  │  Bash/evdev  │  │  Bash/pynput │     │
│  │  (Windows)   │  │  (Linux)     │  │  (macOS)     │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                          ↓ JSON Events
┌─────────────────────────────────────────────────────────────┐
│                    Core Analysis Engine                      │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Privacy Manager → Sanitize, Hash, Anonymize          │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Profile Manager → Load, Save, Update Profile         │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Behavioral Analyzer → Statistical Analysis           │  │
│  │   - Z-score, IQR, Digraph Timing, Error Patterns    │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Pattern Detector → Known Attack Patterns             │  │
│  │   - PowerShell, CMD, Curl, Netcat, etc.             │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                          ↓ Analysis Results
┌─────────────────────────────────────────────────────────────┐
│                    Enforcement Engine                        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Policy Engine → Decide Action                        │  │
│  │   - Log, Alert, Block, Adaptive                      │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Notifier → Send User Alerts                          │  │
│  │   - Windows Toast, Linux notify-send, macOS osascript│  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Platform Collectors

**Purpose:** Capture keystroke events at OS level

**Windows (PowerShell):**
- Uses `SetWindowsHookEx` for low-level keyboard hook
- Detects hardware vs software injection via flags
- Captures window context and process names
- Outputs JSON events

**Linux (Bash + Python evdev):**
- Uses `/dev/input/eventX` for keyboard events
- Requires root privileges for device access
- Uses xdotool for window name detection
- Outputs JSON events

**macOS (Bash + Python pynput):**
- Uses pynput library (production would use CGEvent/IOKit)
- Cross-platform compatible
- Outputs JSON events

**Event Format:**
```json
{
  "event_type": "keystroke",
  "timestamp": 1704470400123,
  "platform": "windows",
  "key": "a",
  "key_code": 65,
  "inter_event_ms": 145,
  "window_name": "Firefox",
  "injected": false,
  "is_backspace": false,
  "modifiers": []
}
```

### 2. Core Analysis Engine

#### Privacy Manager (`core/privacy.py`)
- **Data Minimization:** Only statistics stored, no raw content
- **Hashing:** SHA256 for attack logs
- **Anonymization:** Window names to categories
- **Retention:** Auto-delete old logs

#### Profile Manager (`core/profile_manager.py`)
- **Load/Save:** JSON profile storage
- **Continuous Learning:** Exponential moving average
- **Learning Phases:** Initial → Continuous → Stable
- **Validation:** Profile version checking

#### Behavioral Analyzer (`core/analyzer.py`)
- **Speed Analysis:** Z-score, IQR outlier detection
- **Digraph Timing:** Key-pair pattern analysis
- **Error Patterns:** Backspace rate tracking
- **Temporal Consistency:** Active hours checking
- **Confidence Scoring:** Multi-factor weighting

#### Pattern Detector (`core/detector.py`)
- **Attack Patterns:** 15+ known sequences
- **Repetition Detection:** Same key, alternating, sequential
- **Classification:** Attack type categorization

### 3. Enforcement Engine

#### Policy Engine (`enforcement/policy_engine.py`)
- **Policies:** Log, Alert, Sneaky, Normal, Paranoid, Adaptive
- **Decision Logic:** Confidence-based actions
- **State Management:** Block duration, consecutive anomalies

#### Notifier (`enforcement/notifier.py`)
- **Cross-Platform:** Windows, Linux, macOS
- **Severity Levels:** Info, Warning, Critical
- **Methods:** Toast, notify-send, osascript

## Data Flow

1. **Keystroke Capture:**
   - Platform collector hooks keyboard
   - Creates JSON event with timing, key, window
   - Sends to analysis engine

2. **Privacy Sanitization:**
   - Privacy manager removes sensitive data
   - Hashes content for attack logs
   - Anonymizes window names

3. **Behavioral Analysis:**
   - Calculate inter-keystroke intervals
   - Compare to user profile statistics
   - Compute z-scores and IQR bounds
   - Check digraph timing patterns
   - Analyze error rate

4. **Pattern Matching:**
   - Check for known attack sequences
   - Detect repetitive patterns
   - Classify attack type

5. **Confidence Scoring:**
   - Weight multiple signals
   - Combine speed + pattern + error metrics
   - Output 0.0-1.0 confidence

6. **Policy Enforcement:**
   - Decide action based on policy and confidence
   - Generate enforcement action
   - Send notifications if needed

7. **Learning Update:**
   - If not anomaly, update profile
   - Use exponential moving average
   - Save profile periodically

## Performance Characteristics

- **CPU Usage:** < 1% average (target)
- **Memory:** < 50MB (target)
- **Latency:** < 500ms detection (target)
- **False Positive Rate:** < 0.5% (target)

## Security Considerations

- **Elevated Privileges:** Required for low-level hooks
- **Input Blocking:** Can block keyboard (enforcement)
- **Data Storage:** Local only, no remote exfiltration
- **Encryption:** Logs hashed, not encrypted (optional)

## Extensibility

### Adding New Detection Methods

1. Add method to `BehavioralAnalyzer`
2. Update `_calculate_confidence` weighting
3. Add tests
4. Document in config

### Adding New Platforms

1. Create collector in `collectors/<platform>/`
2. Output standardized JSON events
3. Test with existing analysis engine
4. Update installation scripts

### Adding New Attack Patterns

1. Add pattern to `PatternDetector._load_attack_patterns()`
2. Define sequence, risk score, timing constraints
3. Test with synthetic payloads
4. Document in README

## Configuration

See `config/duckhunt.v2.conf` for all options.

Key settings:
- `policy`: Enforcement policy
- `confidence_threshold`: Block threshold (0.0-1.0)
- `learning_rate`: Adaptation speed
- `log_retention_days`: Auto-delete period

## Deployment Models

### Personal Use
- Install locally
- Run in learning mode for 2 weeks
- Enable enforcement
- Monitor false positives

### Corporate Deployment
- Centralized configuration
- User notification required
- Privacy policy compliance
- Regular review of alerts

## Future Enhancements

- Neural network-based detection
- Multi-user profile support
- Cloud sync (optional)
- Integration with EDR platforms
- Hardware USB monitoring
