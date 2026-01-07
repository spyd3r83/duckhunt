# DuckHunt v2.0 - Privacy Policy

## Privacy-First Design Philosophy

DuckHunt v2.0 is designed with **privacy by design** principles:

1. **Data Minimization:** Collect only what's necessary for security
2. **Statistical Storage:** Store aggregates, not raw data
3. **User Transparency:** Clear documentation of data handling
4. **User Control:** Users can view, delete, or disable at any time
5. **Local Processing:** All analysis happens locally, no cloud

---

## What Data We Collect

### During Normal Operation (Learning Mode)

**Keystroke Timing Statistics:**
- ✅ Inter-keystroke intervals (time between keys)
- ✅ Statistical distributions (mean, std dev, percentiles)
- ✅ Digraph timing (timing between specific key pairs)
- ✅ Error rate (percentage of backspace usage)

**We DO NOT collect:**
- ❌ Actual keys pressed (passwords, messages, etc.)
- ❌ Full keystroke sequences
- ❌ Clipboard content
- ❌ Screen captures

**Mouse Movement (Optional, Disabled by Default):**
- ✅ Velocity and acceleration statistics
- ✅ Click interval timing

**We DO NOT collect:**
- ❌ Mouse coordinates or trajectories
- ❌ Click locations
- ❌ What you clicked on

**Window Context (Anonymized):**
- ✅ Window categories (BROWSER, EDITOR, TERMINAL)
- ✅ Hash of window title (for correlation only)

**We DO NOT collect:**
- ❌ Full window titles (unless explicitly enabled)
- ❌ URLs or document names
- ❌ Application content

### During Attack Detection

**Attack Logs (Hashed):**
- ✅ Timestamp of detection
- ✅ Confidence score (0.0-1.0)
- ✅ Attack characteristics (speed, pattern type)
- ✅ SHA256 hash of suspected content
- ✅ Action taken (blocked, logged, alerted)

**We DO NOT store:**
- ❌ Plaintext attack content
- ❌ Full command sequences
- ❌ Passwords or credentials

---

## Data Storage

### User Profile (`data/profile.json`)

**Purpose:** Statistical model of your typing behavior

**Contents:**
```json
{
  "version": "2.0",
  "sample_count": 10000,
  "speed_distribution": {
    "mean_ms": 145.3,
    "std_dev_ms": 23.4
  },
  "digraph_timings": {
    "th": {"mean_ms": 145, "std_dev_ms": 23}
  },
  "error_rate": 0.034,
  "active_hours": [9, 10, 11, 14, 15, 16, 17]
}
```

**Retention:** Persistent (required for detection)
**Location:** Local filesystem only
**Encryption:** Optional (user-configurable)

### Attack Logs (`data/attacks.jsonl`)

**Purpose:** Record of detected attacks for analysis

**Contents:**
```json
{
  "timestamp": "2025-01-05T14:23:11Z",
  "attack_type": "speed_anomaly",
  "confidence": 0.92,
  "content_hash": "SHA256:a3f2b1...",
  "action_taken": "blocked"
}
```

**Retention:** 7 days (default, configurable)
**Location:** Local filesystem only
**Auto-Delete:** Yes, after retention period

### Configuration (`config/duckhunt.v2.conf`)

**Purpose:** User preferences and settings

**Sensitive Data:** May contain password (paranoid mode)
**Recommendation:** Protect with file permissions (chmod 600)

---

## Privacy Controls

### User Configuration Options

```ini
[privacy]
# Enable/disable mouse tracking (default: false)
enable_mouse_tracking = false

# Auto-delete logs after N days (default: 7)
log_retention_days = 7

# Store full window names or anonymize (default: false)
store_window_names = false

# Hash all logged content (default: true)
anonymize_logs = true

# Show system tray icon (transparency, default: true)
show_tray_icon = true
```

### User Rights

**Right to Access:**
```bash
# View your profile
python -m core.main --status

# View attack logs
cat data/attacks.jsonl
```

**Right to Delete:**
```bash
# Delete profile and start fresh
rm data/profile.json

# Delete attack logs
rm data/attacks.jsonl
```

**Right to Disable:**
```bash
# Stop service (Windows)
Stop-Service DuckHuntMonitor

# Stop service (Linux)
sudo systemctl stop duckhunt

# Uninstall completely
sudo bash collectors/linux/uninstall.sh
```

**Right to Export:**
```bash
# Export profile for backup
cp data/profile.json profile_backup_$(date +%Y%m%d).json
```

---

## Data Protection Measures

### Hashing

All potentially sensitive data is hashed using SHA256:

```python
def hash_content(content: str) -> str:
    """Hash content for privacy-preserving logging"""
    hasher = hashlib.sha256()
    hasher.update(content.encode('utf-8'))
    return f"SHA256:{hasher.hexdigest()}"
```

**What This Means:**
- Original content cannot be recovered from hash
- Hashes can be compared for pattern detection
- Privacy is preserved while maintaining security

### Anonymization

Window names are categorized instead of stored:

```python
# Input: "Firefox - Bank Account - Personal Finance"
# Output: "BROWSER"

# Input: "Visual Studio Code - passwords.txt"
# Output: "EDITOR"
```

**What This Means:**
- No record of what specific websites/documents you accessed
- Sufficient context for attack detection
- Privacy preserved

### Data Retention

Automatic deletion of old data:

```python
def enforce_retention_policy(log_dir: Path, retention_days: int):
    """Delete logs older than retention period"""
    cutoff_date = datetime.now() - timedelta(days=retention_days)
    for log_file in log_dir.glob('*.jsonl'):
        if log_file.stat().st_mtime < cutoff_date:
            log_file.unlink()
```

**What This Means:**
- Old attack logs automatically deleted
- Configurable retention period (default: 7 days)
- Minimizes data storage

---

## Compliance

### GDPR (General Data Protection Regulation)

**Lawful Basis:** Legitimate interest (security)

**Data Minimization:** ✅ Only statistical data stored
**Purpose Limitation:** ✅ Used only for security
**Storage Limitation:** ✅ Auto-delete after 7-30 days
**Privacy by Design:** ✅ Built-in privacy features
**Transparency:** ✅ Users informed of data processing
**Data Subject Rights:** ✅ Access, delete, export supported

**DPIA Required?** Recommended for corporate deployments

### CCPA (California Consumer Privacy Act)

**Notice:** ✅ Users informed of data collection
**Purpose:** ✅ Security purpose clearly stated
**No Sale:** ✅ Data not sold to third parties
**Access:** ✅ Users can request data access
**Deletion:** ✅ Users can request deletion

### Wiretapping Laws (US)

**Consent:** One-party consent (you consent to monitor your own system)
**Notice:** System tray icon provides notice
**Legitimate Purpose:** Security (protecting against attacks)
**Limited Scope:** Timing only, not content

**Note:** Check local laws. Some jurisdictions may have stricter requirements.

---

## Transparency Reports

DuckHunt does NOT:
- ❌ Send data to remote servers
- ❌ Share data with third parties
- ❌ Sell user data
- ❌ Track user behavior for non-security purposes
- ❌ Create unique user identifiers for tracking
- ❌ Correlate data across users

DuckHunt DOES:
- ✅ Store data locally only
- ✅ Use data only for attack detection
- ✅ Allow users to view/delete/export data
- ✅ Auto-delete old data
- ✅ Hash sensitive information
- ✅ Operate transparently (system tray icon)

---

## Privacy Audit

### Self-Audit Checklist

Organizations deploying DuckHunt should verify:

- [ ] Users have been notified
- [ ] Privacy policy has been provided
- [ ] System tray icon is enabled and visible
- [ ] Log retention period is configured
- [ ] Logs are stored securely (file permissions)
- [ ] No unauthorized access to logs
- [ ] Users can request data deletion
- [ ] Legal counsel has reviewed deployment

---

## Privacy Incidents

### Reporting

If you discover a privacy violation:

1. Document the issue
2. Email: privacy@example.com
3. Include: What data, when, how accessed
4. We will respond within 72 hours

### Our Commitments

- Acknowledge receipt within 24 hours
- Investigate within 72 hours
- Notify affected users if breach confirmed
- Implement fixes immediately
- Publish post-mortem (if significant)

---

## Questions or Concerns

For privacy questions:
- **Email:** privacy@example.com
- **Documentation:** See [ETHICS.md](../ETHICS.md)
- **Issues:** GitHub Issues for technical concerns

---

**Last Updated:** January 2025
**Version:** 2.0
