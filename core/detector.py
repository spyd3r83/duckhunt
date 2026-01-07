"""
DuckHunt v2.0 - Attack Detector
Pattern matching and attack classification engine

Detects common RubberDucky/HID injection attack patterns:
- GUI shortcut sequences (WIN+R, ALT+F4, etc.)
- Command execution patterns (powershell, cmd, bash, curl, wget)
- Repetitive keystroke patterns
- Rapid command sequences
"""

from typing import Dict, List, Tuple, Optional, Any
from collections import deque
from dataclasses import dataclass
import re


@dataclass
class AttackPattern:
    """Defines a known attack pattern"""
    name: str
    description: str
    sequence: List[str]  # Key sequence to match
    window_filter: Optional[str]  # Window name regex (None = any)
    risk_score: float  # 0.0 - 1.0
    timing_constraint: Optional[str]  # e.g., "<200ms" for rapid sequences


class PatternDetector:
    """Detects known HID injection attack patterns"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize pattern detector

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.pattern_detection_enabled = config.get('detection', {}).get('pattern_detection', True)

        # Circular buffer for recent keystrokes
        self.recent_keys = deque(maxlen=50)
        self.recent_windows = deque(maxlen=10)

        # Define attack patterns
        self.attack_patterns = self._load_attack_patterns()

    def _load_attack_patterns(self) -> List[AttackPattern]:
        """
        Load known attack patterns

        Returns:
            List of attack patterns
        """
        patterns = [
            # Windows GUI shortcuts
            AttackPattern(
                name="Windows Run Dialog",
                description="Opens Windows Run dialog (common RubberDucky entry point)",
                sequence=['LWin', 'r'],
                window_filter=None,
                risk_score=0.7,
                timing_constraint="<500ms"
            ),
            AttackPattern(
                name="Windows Task Manager",
                description="Opens Task Manager",
                sequence=['LCtrl', 'LShift', 'Escape'],
                window_filter=None,
                risk_score=0.5,
                timing_constraint=None
            ),
            AttackPattern(
                name="Windows Power Menu",
                description="Opens Power User menu",
                sequence=['LWin', 'x'],
                window_filter=None,
                risk_score=0.6,
                timing_constraint="<500ms"
            ),

            # PowerShell execution
            AttackPattern(
                name="PowerShell Launch",
                description="Typing 'powershell' command",
                sequence=['p','o','w','e','r','s','h','e','l','l'],
                window_filter=r"(Run|Command|cmd)",
                risk_score=0.9,
                timing_constraint="<2000ms"
            ),
            AttackPattern(
                name="PowerShell Encoded Command",
                description="PowerShell -enc (encoded command)",
                sequence=['p','o','w','e','r','s','h','e','l','l',' ','-','e','n','c'],
                window_filter=r"(Run|Command|cmd)",
                risk_score=0.95,
                timing_constraint="<3000ms"
            ),

            # Command execution
            AttackPattern(
                name="CMD Launch",
                description="Typing 'cmd' command",
                sequence=['c','m','d'],
                window_filter=r"Run",
                risk_score=0.85,
                timing_constraint="<1000ms"
            ),

            # Download commands
            AttackPattern(
                name="Curl Download",
                description="Using curl to download",
                sequence=['c','u','r','l',' ','-','o'],
                window_filter=r"(Terminal|PowerShell|cmd|bash)",
                risk_score=0.85,
                timing_constraint="<3000ms"
            ),
            AttackPattern(
                name="Wget Download",
                description="Using wget to download",
                sequence=['w','g','e','t',' '],
                window_filter=r"(Terminal|PowerShell|cmd|bash)",
                risk_score=0.85,
                timing_constraint="<2000ms"
            ),

            # Linux/Mac commands
            AttackPattern(
                name="Bash Launch",
                description="Typing 'bash' command",
                sequence=['b','a','s','h'],
                window_filter=r"(Terminal|Run)",
                risk_score=0.8,
                timing_constraint="<1000ms"
            ),
            AttackPattern(
                name="Netcat Reverse Shell",
                description="Netcat reverse shell command",
                sequence=['n','c',' ','-','e'],
                window_filter=r"(Terminal|bash)",
                risk_score=0.95,
                timing_constraint="<2000ms"
            ),

            # Base64 encoding (common obfuscation)
            AttackPattern(
                name="Base64 Decode Pipe",
                description="Base64 decode and execute",
                sequence=['b','a','s','e','6','4',' ','-','d'],
                window_filter=r"(Terminal|bash|PowerShell)",
                risk_score=0.90,
                timing_constraint="<2000ms"
            ),
        ]

        return patterns

    def check_pattern_match(self, recent_keys: List[str], recent_windows: List[str]) -> Optional[Dict[str, Any]]:
        """
        Check if recent keystrokes match known attack patterns

        Args:
            recent_keys: List of recent key presses
            recent_windows: List of recent window names

        Returns:
            Match information if pattern found, None otherwise
        """
        if not self.pattern_detection_enabled:
            return None

        current_window = recent_windows[-1] if recent_windows else ""

        for pattern in self.attack_patterns:
            # Check if key sequence matches
            if self._sequence_matches(recent_keys, pattern.sequence):
                # Check window filter
                if pattern.window_filter:
                    if not re.search(pattern.window_filter, current_window, re.IGNORECASE):
                        continue  # Window doesn't match

                # Pattern matched!
                return {
                    'pattern_matched': True,
                    'pattern_name': pattern.name,
                    'pattern_description': pattern.description,
                    'risk_score': pattern.risk_score,
                    'matched_sequence': ''.join(pattern.sequence),
                    'window': current_window
                }

        return None

    def _sequence_matches(self, haystack: List[str], needle: List[str]) -> bool:
        """
        Check if needle sequence appears in haystack

        Args:
            haystack: Full key sequence
            needle: Sequence to find

        Returns:
            True if needle is in haystack
        """
        if len(needle) > len(haystack):
            return False

        needle_lower = [k.lower() for k in needle]

        # Check all positions
        for i in range(len(haystack) - len(needle) + 1):
            window = [haystack[i + j].lower() for j in range(len(needle))]

            if window == needle_lower:
                return True

        return False

    def detect_repetitive_pattern(self, recent_keys: List[str]) -> Optional[Dict[str, Any]]:
        """
        Detect repetitive keystroke patterns (bots often repeat)

        Args:
            recent_keys: Recent key sequence

        Returns:
            Detection result if repetitive pattern found
        """
        if len(recent_keys) < 10:
            return None

        last_10 = recent_keys[-10:]

        # Pattern 1: Same key repeated
        if len(set(last_10)) == 1:
            return {
                'pattern_type': 'repeated_key',
                'description': f"Same key '{last_10[0]}' repeated 10 times",
                'risk_score': 0.7
            }

        # Pattern 2: Alternating pattern (e.g., ababababab)
        if len(last_10) >= 6:
            if (last_10[0] == last_10[2] == last_10[4] and
                last_10[1] == last_10[3] == last_10[5] and
                last_10[0] != last_10[1]):
                return {
                    'pattern_type': 'alternating_keys',
                    'description': f"Alternating pattern: {''.join(last_10[:6])}",
                    'risk_score': 0.75
                }

        # Pattern 3: Sequential pattern (e.g., abcdefgh)
        if self._is_sequential(last_10):
            return {
                'pattern_type': 'sequential_keys',
                'description': f"Sequential pattern: {''.join(last_10)}",
                'risk_score': 0.65
            }

        return None

    def _is_sequential(self, keys: List[str]) -> bool:
        """Check if keys are in sequential order"""
        if len(keys) < 4:
            return False

        try:
            # Check if alphabetical sequence
            codes = [ord(k.lower()) for k in keys if len(k) == 1]
            if len(codes) < 4:
                return False

            differences = [codes[i+1] - codes[i] for i in range(len(codes)-1)]
            return all(d == 1 for d in differences)

        except:
            return False

    def add_keystroke(self, key: str, window: str):
        """
        Add keystroke to detection buffer

        Args:
            key: Key pressed
            window: Active window name
        """
        self.recent_keys.append(key)
        self.recent_windows.append(window)


class AttackClassifier:
    """Classifies detected attacks into categories"""

    ATTACK_TYPES = {
        'speed_anomaly': 'Abnormally fast typing speed (likely injection)',
        'digraph_anomaly': 'Unusual key-pair timing patterns',
        'pattern_match': 'Matched known attack pattern',
        'zero_error_rate': 'Suspiciously perfect typing (no corrections)',
        'repetitive_pattern': 'Repetitive keystroke pattern',
        'injection_flag': 'Hardware injection flag detected',
        'temporal_anomaly': 'Typing at unusual time',
    }

    @staticmethod
    def classify_attack(analysis_result: Any, pattern_match: Optional[Dict]) -> Dict[str, Any]:
        """
        Classify attack based on analysis results

        Args:
            analysis_result: Result from BehavioralAnalyzer
            pattern_match: Result from PatternDetector

        Returns:
            Classification result
        """
        classification = {
            'attack_types': [],
            'primary_type': None,
            'severity': 'UNKNOWN',
            'description': '',
        }

        # Collect attack types from analysis
        if 'hardware_injection' in analysis_result.metrics:
            classification['attack_types'].append('injection_flag')

        if any('speed' in r.lower() for r in analysis_result.reasons):
            classification['attack_types'].append('speed_anomaly')

        if any('digraph' in r.lower() for r in analysis_result.reasons):
            classification['attack_types'].append('digraph_anomaly')

        if any('error' in r.lower() for r in analysis_result.reasons):
            classification['attack_types'].append('zero_error_rate')

        # Add pattern match if present
        if pattern_match and pattern_match.get('pattern_matched'):
            classification['attack_types'].append('pattern_match')
            classification['pattern_name'] = pattern_match.get('pattern_name')

        # Determine primary type (highest confidence)
        if classification['attack_types']:
            # Prioritize pattern match if present
            if 'pattern_match' in classification['attack_types']:
                classification['primary_type'] = 'pattern_match'
            else:
                classification['primary_type'] = classification['attack_types'][0]

        # Determine severity
        if analysis_result.confidence >= 0.95:
            classification['severity'] = 'CRITICAL'
        elif analysis_result.confidence >= 0.85:
            classification['severity'] = 'HIGH'
        elif analysis_result.confidence >= 0.70:
            classification['severity'] = 'MEDIUM'
        else:
            classification['severity'] = 'LOW'

        # Build description
        descriptions = []
        for attack_type in classification['attack_types']:
            if attack_type in AttackClassifier.ATTACK_TYPES:
                descriptions.append(AttackClassifier.ATTACK_TYPES[attack_type])

        classification['description'] = '; '.join(descriptions)

        return classification


def test_pattern_detector():
    """Test pattern detector"""
    config = {
        'detection': {
            'pattern_detection': True,
        }
    }

    detector = PatternDetector(config)

    # Test 1: PowerShell pattern
    print("Test 1: PowerShell execution pattern")
    keys = list('powershell')
    windows = ['Run'] * len(keys)

    for i, key in enumerate(keys):
        detector.add_keystroke(key, windows[i])

    match = detector.check_pattern_match(list(detector.recent_keys), list(detector.recent_windows))
    print(f"Pattern match: {match}")
    assert match is not None
    assert match['pattern_name'] == 'PowerShell Launch'
    assert match['risk_score'] >= 0.9

    # Test 2: Repetitive pattern
    print("\nTest 2: Repetitive key pattern")
    repetitive_keys = ['a'] * 10
    for key in repetitive_keys:
        detector.add_keystroke(key, 'Notepad')

    repetitive = detector.detect_repetitive_pattern(list(detector.recent_keys))
    print(f"Repetitive pattern: {repetitive}")
    assert repetitive is not None
    assert repetitive['pattern_type'] == 'repeated_key'

    # Test 3: Windows Run dialog
    print("\nTest 3: Windows Run dialog (WIN+R)")
    detector2 = PatternDetector(config)
    detector2.add_keystroke('LWin', 'Desktop')
    detector2.add_keystroke('r', 'Desktop')

    match2 = detector2.check_pattern_match(list(detector2.recent_keys), list(detector2.recent_windows))
    print(f"Pattern match: {match2}")
    assert match2 is not None
    assert match2['pattern_name'] == 'Windows Run Dialog'

    print("\n✅ Pattern detector tests passed!")


if __name__ == '__main__':
    test_pattern_detector()
