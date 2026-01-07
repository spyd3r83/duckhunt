"""
DuckHunt v2.0 - Policy Enforcement Engine
Decides what action to take when anomaly is detected
"""

from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass
import time


class EnforcementPolicy(Enum):
    """Available enforcement policies"""
    LOG_ONLY = "log"          # Only log, no action (safest)
    ALERT = "alert"            # Alert user but allow
    SNEAKY = "sneaky"          # Randomly drop keys to disrupt attack
    NORMAL = "normal"          # Temporarily block keyboard
    PARANOID = "paranoid"      # Block until password entered
    ADAPTIVE = "adaptive"      # Adjust response based on confidence


@dataclass
class EnforcementAction:
    """Action to take"""
    allow_keystroke: bool      # Allow or block this keystroke
    log_event: bool            # Log to attack log
    send_alert: bool           # Send user notification
    block_duration_ms: int     # How long to block (0 = no block)
    message: str               # Message to user
    require_password: bool     # Require password to unlock


class PolicyEngine:
    """Enforcement policy decision engine"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize policy engine

        Args:
            config: Configuration dictionary
        """
        self.config = config

        # Get policy from config
        policy_str = config.get('enforcement', {}).get('policy', 'adaptive')
        try:
            self.policy = EnforcementPolicy(policy_str)
        except ValueError:
            print(f"Warning: Unknown policy '{policy_str}', using ADAPTIVE")
            self.policy = EnforcementPolicy.ADAPTIVE

        # Configuration
        self.block_enabled = config.get('enforcement', {}).get('block_suspicious', True)
        self.alert_enabled = config.get('enforcement', {}).get('alert_on_detection', True)
        self.confidence_threshold = config.get('detection', {}).get('confidence_threshold', 0.85)

        # State
        self.blocked_until = 0
        self.consecutive_anomalies = 0
        self.sneaky_counter = 0

    def enforce(self, analysis_result: Any, pattern_match: Optional[Dict] = None) -> EnforcementAction:
        """
        Decide what action to take

        Args:
            analysis_result: Result from BehavioralAnalyzer
            pattern_match: Optional pattern match from PatternDetector

        Returns:
            EnforcementAction specifying what to do
        """

        # If not anomalous, allow
        if not analysis_result.is_anomaly:
            self.consecutive_anomalies = 0
            return EnforcementAction(
                allow_keystroke=True,
                log_event=False,
                send_alert=False,
                block_duration_ms=0,
                message="",
                require_password=False
            )

        # Anomaly detected - increment counter
        self.consecutive_anomalies += 1

        # Check if currently in blocked state
        if self.blocked_until > time.time() * 1000:
            remaining_ms = int(self.blocked_until - time.time() * 1000)
            return EnforcementAction(
                allow_keystroke=False,
                log_event=True,
                send_alert=False,
                block_duration_ms=remaining_ms,
                message=f"Keyboard blocked for {remaining_ms/1000:.1f}s",
                require_password=False
            )

        # Apply policy
        if self.policy == EnforcementPolicy.LOG_ONLY:
            return self._policy_log_only(analysis_result)

        elif self.policy == EnforcementPolicy.ALERT:
            return self._policy_alert(analysis_result)

        elif self.policy == EnforcementPolicy.SNEAKY:
            return self._policy_sneaky(analysis_result)

        elif self.policy == EnforcementPolicy.NORMAL:
            return self._policy_normal(analysis_result)

        elif self.policy == EnforcementPolicy.PARANOID:
            return self._policy_paranoid(analysis_result)

        elif self.policy == EnforcementPolicy.ADAPTIVE:
            return self._policy_adaptive(analysis_result, pattern_match)

        else:
            # Default: log only
            return self._policy_log_only(analysis_result)

    def _policy_log_only(self, result: Any) -> EnforcementAction:
        """Log anomaly but take no action"""
        return EnforcementAction(
            allow_keystroke=True,
            log_event=True,
            send_alert=False,
            block_duration_ms=0,
            message=f"Anomaly logged (confidence: {result.confidence:.2f})",
            require_password=False
        )

    def _policy_alert(self, result: Any) -> EnforcementAction:
        """Alert user but allow keystroke"""
        return EnforcementAction(
            allow_keystroke=True,
            log_event=True,
            send_alert=self.alert_enabled,
            block_duration_ms=0,
            message=f"Suspicious activity detected (confidence: {result.confidence:.2f})",
            require_password=False
        )

    def _policy_sneaky(self, result: Any) -> EnforcementAction:
        """Randomly drop keystrokes to disrupt attack"""
        self.sneaky_counter += 1

        # Drop every 7th keystroke
        drop_key = (self.sneaky_counter % 7) == 0

        return EnforcementAction(
            allow_keystroke=not drop_key,
            log_event=True,
            send_alert=False,
            block_duration_ms=0,
            message="Sneaky mode: occasionally dropping keys" if drop_key else "",
            require_password=False
        )

    def _policy_normal(self, result: Any) -> EnforcementAction:
        """Temporarily block keyboard"""
        block_duration = 30000  # 30 seconds

        # Set blocked until time
        self.blocked_until = time.time() * 1000 + block_duration

        return EnforcementAction(
            allow_keystroke=False,
            log_event=True,
            send_alert=self.alert_enabled,
            block_duration_ms=block_duration,
            message=f"Keyboard blocked for 30s (confidence: {result.confidence:.2f})",
            require_password=False
        )

    def _policy_paranoid(self, result: Any) -> EnforcementAction:
        """Block keyboard until password entered"""
        return EnforcementAction(
            allow_keystroke=False,
            log_event=True,
            send_alert=True,
            block_duration_ms=-1,  # Indefinite
            message=f"Keyboard locked - password required (confidence: {result.confidence:.2f})",
            require_password=True
        )

    def _policy_adaptive(self, result: Any, pattern_match: Optional[Dict]) -> EnforcementAction:
        """Adapt response based on confidence level"""

        confidence = result.confidence

        # Very high confidence (> 0.95) - Block immediately
        if confidence >= 0.95:
            block_duration = 60000  # 60 seconds
            self.blocked_until = time.time() * 1000 + block_duration

            return EnforcementAction(
                allow_keystroke=False,
                log_event=True,
                send_alert=True,
                block_duration_ms=block_duration,
                message=f"HIGH CONFIDENCE ATTACK: Keyboard blocked for 60s (confidence: {confidence:.2f})",
                require_password=False
            )

        # High confidence (0.85-0.95) - Alert and possibly block
        elif confidence >= self.confidence_threshold:
            # If pattern matched, block
            if pattern_match and pattern_match.get('pattern_matched'):
                block_duration = 30000
                self.blocked_until = time.time() * 1000 + block_duration

                return EnforcementAction(
                    allow_keystroke=False,
                    log_event=True,
                    send_alert=True,
                    block_duration_ms=block_duration,
                    message=f"Attack pattern detected: {pattern_match.get('pattern_name')} - Blocked for 30s",
                    require_password=False
                )
            else:
                # Just alert
                return EnforcementAction(
                    allow_keystroke=True,
                    log_event=True,
                    send_alert=True,
                    block_duration_ms=0,
                    message=f"Suspicious activity detected (confidence: {confidence:.2f})",
                    require_password=False
                )

        # Medium confidence (0.70-0.85) - Log and alert
        elif confidence >= 0.70:
            return EnforcementAction(
                allow_keystroke=True,
                log_event=True,
                send_alert=self.alert_enabled,
                block_duration_ms=0,
                message=f"Possible suspicious activity (confidence: {confidence:.2f})",
                require_password=False
            )

        # Low confidence (< 0.70) - Log only
        else:
            return EnforcementAction(
                allow_keystroke=True,
                log_event=True,
                send_alert=False,
                block_duration_ms=0,
                message=f"Minor anomaly logged (confidence: {confidence:.2f})",
                require_password=False
            )

    def reset_block(self):
        """Reset blocked state"""
        self.blocked_until = 0
        self.consecutive_anomalies = 0

    def get_status(self) -> Dict[str, Any]:
        """Get current enforcement status"""
        is_blocked = self.blocked_until > time.time() * 1000

        return {
            'policy': self.policy.value,
            'is_blocked': is_blocked,
            'blocked_until_ms': self.blocked_until if is_blocked else 0,
            'consecutive_anomalies': self.consecutive_anomalies,
        }


def test_policy_engine():
    """Test policy engine"""
    from dataclasses import dataclass

    @dataclass
    class MockResult:
        is_anomaly: bool
        confidence: float
        reasons: list
        metrics: dict

    config = {
        'enforcement': {
            'policy': 'adaptive',
            'block_suspicious': True,
            'alert_on_detection': True,
        },
        'detection': {
            'confidence_threshold': 0.85,
        },
    }

    engine = PolicyEngine(config)

    # Test 1: Normal typing (no anomaly)
    result = MockResult(is_anomaly=False, confidence=0.2, reasons=[], metrics={})
    action = engine.enforce(result)
    print(f"Normal typing: allow={action.allow_keystroke}, alert={action.send_alert}")
    assert action.allow_keystroke == True
    assert action.send_alert == False

    # Test 2: High confidence anomaly
    result = MockResult(is_anomaly=True, confidence=0.92, reasons=['speed'], metrics={})
    action = engine.enforce(result)
    print(f"High confidence: allow={action.allow_keystroke}, alert={action.send_alert}, msg={action.message}")
    assert action.allow_keystroke == False
    assert action.send_alert == True

    # Test 3: Medium confidence
    result = MockResult(is_anomaly=True, confidence=0.75, reasons=['speed'], metrics={})
    action = engine.enforce(result)
    print(f"Medium confidence: allow={action.allow_keystroke}, alert={action.send_alert}")

    print("\n✅ Policy engine tests passed!")


if __name__ == '__main__':
    test_policy_engine()
