"""
DuckHunt v2.0 - Statistical Analyzer
Behavioral analysis engine for detecting automated keystroke injection

Uses multiple statistical methods to distinguish human typing from bot injection:
- Z-score analysis for speed anomalies
- Interquartile Range (IQR) for outlier detection
- Digraph timing analysis
- Error pattern detection
- Temporal consistency checking
- Advanced detection (entropy, autocorrelation, Hurst exponent, etc.)
- ML-based anomaly detection (optional)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
from dataclasses import dataclass
from datetime import datetime

# Import advanced detection modules
from core.advanced_detector import AdvancedDetector
from core.ml_detector import MLDetector, SKLEARN_AVAILABLE


@dataclass
class KeystrokeEvent:
    """Represents a single keystroke event"""
    timestamp: int  # Unix timestamp in milliseconds
    key: str  # Key identifier
    key_code: int  # Numeric key code
    inter_key_ms: float  # Time since last keystroke
    window_name: str  # Active window
    window_category: str  # Anonymized window category
    injected: bool  # Hardware vs software injection flag
    modifiers: List[str]  # Active modifiers (Ctrl, Alt, etc.)
    is_backspace: bool  # Is this a correction key?


@dataclass
class AnalysisResult:
    """Result of behavioral analysis"""
    is_anomaly: bool  # Is this behavior anomalous?
    confidence: float  # Confidence score (0.0 - 1.0)
    reasons: List[str]  # List of anomaly reasons
    metrics: Dict[str, float]  # Detailed metrics
    recommendation: str  # Recommended action


class BehavioralAnalyzer:
    """Main behavioral analysis engine"""

    def __init__(self, profile: Dict[str, Any], config: Dict[str, Any]):
        """
        Initialize behavioral analyzer

        Args:
            profile: User behavioral profile
            config: Configuration dictionary
        """
        self.profile = profile
        self.config = config

        # Configuration
        self.confidence_threshold = config.get('detection', {}).get('confidence_threshold', 0.85)
        self.speed_threshold_ms = config.get('detection', {}).get('speed_threshold_ms', 30)
        self.enable_digraph = config.get('detection', {}).get('digraph_analysis', True)
        self.enable_mouse = config.get('detection', {}).get('mouse_correlation', False)

        # Circular buffers for analysis
        max_history = config.get('advanced', {}).get('max_history_size', 1000)
        self.history = deque(maxlen=max_history)
        self.digraph_buffer = deque(maxlen=2)

        # NEW: Interval buffer for advanced detection
        self.interval_buffer = deque(maxlen=100)

        # NEW: Initialize advanced detector
        if config.get('detection', {}).get('advanced_detection', False):
            self.advanced_detector = AdvancedDetector(
                baseline_profile=profile
            )
        else:
            self.advanced_detector = None

        # NEW: Initialize ML detector (optional)
        if config.get('detection', {}).get('ml_detection', False) and SKLEARN_AVAILABLE:
            self.ml_detector = MLDetector()
            # Note: ML model needs training data before it can be used
        else:
            self.ml_detector = None

        # Statistics
        self.total_keystrokes = 0
        self.total_anomalies = 0

    def analyze_keystroke(self, event: KeystrokeEvent) -> AnalysisResult:
        """
        Perform multi-dimensional analysis on keystroke event

        Args:
            event: Keystroke event to analyze

        Returns:
            Analysis result with anomaly detection
        """
        self.total_keystrokes += 1

        # Track intervals for advanced detection
        self.interval_buffer.append(event.inter_key_ms)

        # Run basic detection
        basic_result = self._run_basic_detection(event)

        # Run advanced detection if enabled and we have enough samples
        advanced_result = None
        if self.advanced_detector and len(self.interval_buffer) >= 20:
            try:
                advanced_result = self.advanced_detector.analyze_sequence(
                    list(self.interval_buffer)
                )
            except Exception as e:
                # Don't fail on advanced detection errors
                pass

        # Run ML detection if enabled and trained
        ml_result = None
        if self.ml_detector and len(self.interval_buffer) >= 20:
            try:
                if self.ml_detector.is_trained:
                    ml_result = self.ml_detector.predict(list(self.interval_buffer))
            except Exception as e:
                # Don't fail on ML detection errors
                pass

        # Combine all results using ensemble voting
        final_result = self._combine_detection_results(basic_result, advanced_result, ml_result)

        # Update statistics
        if final_result.is_anomaly:
            self.total_anomalies += 1

        # Add event to history
        self.history.append(event)
        if event.key and len(event.key) == 1:
            self.digraph_buffer.append(event.key.lower())

        return final_result

    def _run_basic_detection(self, event: KeystrokeEvent) -> Dict[str, Any]:
        """
        Run basic detection algorithms (original DuckHunt detection)

        Returns:
            Dictionary with detection results
        """
        anomalies = []
        metrics = {}

        # 1. Hardware injection flag check
        if event.injected:
            anomalies.append("Software injection detected (hardware flag)")
            metrics['hardware_injection'] = 1.0

        # 2. Speed-based analysis
        if len(self.history) >= 25:
            speed_result = self._analyze_speed(event)
            if speed_result['is_anomaly']:
                anomalies.extend(speed_result['reasons'])
                metrics.update(speed_result['metrics'])

        # 3. Digraph timing analysis
        if self.enable_digraph and len(self.digraph_buffer) >= 1:
            digraph_result = self._analyze_digraph(event)
            if digraph_result['is_anomaly']:
                anomalies.extend(digraph_result['reasons'])
                metrics.update(digraph_result['metrics'])

        # 4. Error pattern analysis
        error_result = self._analyze_error_patterns()
        if error_result['is_anomaly']:
            anomalies.extend(error_result['reasons'])
            metrics.update(error_result['metrics'])

        # 5. Temporal consistency
        temporal_result = self._analyze_temporal_consistency(event)
        if temporal_result['is_anomaly']:
            anomalies.extend(temporal_result['reasons'])
            metrics.update(temporal_result['metrics'])

        # Calculate basic confidence score
        confidence = self._calculate_confidence(metrics, anomalies)

        return {
            'is_anomaly': len(anomalies) > 0 and confidence >= self.confidence_threshold,
            'confidence': confidence,
            'reasons': anomalies,
            'metrics': metrics
        }

    def _combine_detection_results(self, basic: Dict, advanced, ml) -> AnalysisResult:
        """
        Combine results from basic, advanced, and ML detection using ensemble voting

        Args:
            basic: Basic detection result dict
            advanced: AdvancedAnalysisResult or None
            ml: MLDetectionResult or None

        Returns:
            Final AnalysisResult
        """
        # Start with basic results
        confidences = [basic['confidence']]
        all_reasons = list(basic['reasons'])
        all_metrics = dict(basic['metrics'])

        # Add advanced detection results
        if advanced and advanced.is_suspicious:
            confidences.append(advanced.confidence)
            all_reasons.append(f"Advanced: {advanced.anomaly_type}")
            all_metrics['advanced_confidence'] = advanced.confidence
            all_metrics['advanced_type'] = advanced.anomaly_type

        # Add ML detection results
        if ml and ml.is_anomaly:
            confidences.append(ml.confidence)
            all_reasons.append(f"ML: {ml.explanation}")
            all_metrics['ml_confidence'] = ml.confidence

        # Weighted ensemble voting
        # Basic: 40%, Advanced: 40%, ML: 20%
        weights = [0.4, 0.4, 0.2][:len(confidences)]
        combined_confidence = sum(c * w for c, w in zip(confidences, weights))

        # Determine if attack detected
        is_attack = combined_confidence >= self.confidence_threshold

        # Get recommendation based on final confidence
        recommendation = self._get_recommendation(combined_confidence, all_reasons)

        return AnalysisResult(
            is_anomaly=is_attack,
            confidence=combined_confidence,
            reasons=all_reasons,
            metrics=all_metrics,
            recommendation=recommendation
        )

    def _analyze_speed(self, event: KeystrokeEvent) -> Dict[str, Any]:
        """
        Analyze typing speed using multiple statistical methods

        Args:
            event: Current keystroke event

        Returns:
            Dictionary with analysis results
        """
        result = {'is_anomaly': False, 'reasons': [], 'metrics': {}}

        # Get recent speed samples
        recent_speeds = [e.inter_key_ms for e in list(self.history)[-25:]]
        current_avg_speed = np.mean(recent_speeds)

        # Expected values from profile
        expected_mean = self.profile.get('speed_distribution', {}).get('mean_ms', 150)
        expected_std = self.profile.get('speed_distribution', {}).get('std_dev_ms', 30)
        expected_q1 = self.profile.get('speed_distribution', {}).get('q1', 120)
        expected_q3 = self.profile.get('speed_distribution', {}).get('q3', 180)

        # Method 1: Z-score analysis
        if expected_std > 0:
            z_score = (current_avg_speed - expected_mean) / expected_std
            result['metrics']['speed_zscore'] = abs(z_score)

            if abs(z_score) > 3.0:
                result['is_anomaly'] = True
                result['reasons'].append(f"Speed z-score: {abs(z_score):.2f} (>3.0 std devs)")

        # Method 2: IQR outlier detection
        iqr = expected_q3 - expected_q1
        lower_bound = expected_q1 - 1.5 * iqr
        upper_bound = expected_q3 + 1.5 * iqr

        if current_avg_speed < lower_bound:
            result['is_anomaly'] = True
            result['reasons'].append(f"Speed {current_avg_speed:.1f}ms below IQR lower bound {lower_bound:.1f}ms")
            result['metrics']['speed_outlier_low'] = 1.0

        # Method 3: Legacy threshold check (backwards compatible)
        if current_avg_speed < self.speed_threshold_ms:
            result['is_anomaly'] = True
            result['reasons'].append(f"Speed {current_avg_speed:.1f}ms < threshold {self.speed_threshold_ms}ms")
            result['metrics']['speed_below_threshold'] = 1.0

        result['metrics']['current_avg_speed_ms'] = current_avg_speed
        result['metrics']['expected_speed_ms'] = expected_mean

        return result

    def _analyze_digraph(self, event: KeystrokeEvent) -> Dict[str, Any]:
        """
        Analyze digraph (key-pair) timing

        Args:
            event: Current keystroke event

        Returns:
            Dictionary with analysis results
        """
        result = {'is_anomaly': False, 'reasons': [], 'metrics': {}}

        # Get previous key
        prev_key = self.digraph_buffer[-1] if len(self.digraph_buffer) > 0 else None
        curr_key = event.key.lower() if event.key and len(event.key) == 1 else None

        if not prev_key or not curr_key:
            return result

        digraph = prev_key + curr_key

        # Check if we have statistics for this digraph
        digraph_stats = self.profile.get('digraph_timings', {}).get(digraph)

        if digraph_stats and digraph_stats.get('samples', 0) >= 10:
            expected_mean = digraph_stats.get('mean_ms', 0)
            expected_std = digraph_stats.get('std_dev_ms', 0)

            if expected_std > 0:
                z_score = (event.inter_key_ms - expected_mean) / expected_std
                result['metrics'][f'digraph_{digraph}_zscore'] = abs(z_score)

                if abs(z_score) > 2.5:
                    result['is_anomaly'] = True
                    result['reasons'].append(
                        f"Digraph '{digraph}' timing unusual: {event.inter_key_ms:.1f}ms "
                        f"(expected {expected_mean:.1f}ms ± {expected_std:.1f}ms, z={abs(z_score):.2f})"
                    )

        return result

    def _analyze_error_patterns(self) -> Dict[str, Any]:
        """
        Analyze typing error patterns

        Humans make errors (2-8% backspace rate), bots typically don't.

        Returns:
            Dictionary with analysis results
        """
        result = {'is_anomaly': False, 'reasons': [], 'metrics': {}}

        if len(self.history) < 50:
            return result  # Need enough samples

        # Count backspace/corrections in recent history
        recent_100 = list(self.history)[-100:]
        backspace_count = sum(1 for e in recent_100 if e.is_backspace)
        error_rate = backspace_count / len(recent_100)

        result['metrics']['error_rate'] = error_rate

        # Humans typically have 2-8% error rate
        # Bots typically have 0% error rate (scripts are perfect)
        if error_rate < 0.01 and len(recent_100) >= 100:
            result['is_anomaly'] = True
            result['reasons'].append(f"Suspiciously low error rate: {error_rate*100:.2f}% (<1%)")
            result['metrics']['zero_error_rate'] = 1.0

        return result

    def _analyze_temporal_consistency(self, event: KeystrokeEvent) -> Dict[str, Any]:
        """
        Check if typing occurs at unusual times

        Args:
            event: Current keystroke event

        Returns:
            Dictionary with analysis results
        """
        result = {'is_anomaly': False, 'reasons': [], 'metrics': {}}

        # Extract hour from timestamp
        hour = datetime.fromtimestamp(event.timestamp / 1000).hour

        # Check against learned active hours
        active_hours = self.profile.get('temporal_patterns', {}).get('active_hours', [])

        if active_hours and hour not in active_hours:
            # Typing at unusual hour (minor indicator)
            result['is_anomaly'] = True
            result['reasons'].append(f"Typing at unusual hour: {hour}:00")
            result['metrics']['unusual_hour'] = 0.3  # Lower weight

        return result

    def _calculate_confidence(self, metrics: Dict[str, float], anomalies: List[str]) -> float:
        """
        Calculate overall confidence score based on multiple signals

        Args:
            metrics: Dictionary of metric values
            anomalies: List of anomaly descriptions

        Returns:
            Confidence score between 0.0 (definitely human) and 1.0 (definitely bot)
        """
        score = 0.0

        # Weight different signals
        weights = {
            'hardware_injection': 0.15,
            'speed_zscore': 0.25,
            'speed_outlier_low': 0.20,
            'speed_below_threshold': 0.15,
            'zero_error_rate': 0.15,
            'unusual_hour': 0.05,
        }

        for metric_name, weight in weights.items():
            if metric_name in metrics:
                if metric_name == 'speed_zscore':
                    # Map z-score to 0-1 (z > 4 is very suspicious)
                    value = min(metrics[metric_name] / 4.0, 1.0)
                else:
                    value = metrics[metric_name]

                score += value * weight

        # Digraph anomalies (variable weight)
        digraph_metrics = [k for k in metrics.keys() if k.startswith('digraph_')]
        if digraph_metrics:
            avg_digraph_z = np.mean([metrics[k] for k in digraph_metrics])
            score += min(avg_digraph_z / 4.0, 1.0) * 0.20

        return min(score, 1.0)

    def _get_recommendation(self, confidence: float, anomalies: List[str]) -> str:
        """
        Get recommended action based on confidence

        Args:
            confidence: Confidence score
            anomalies: List of detected anomalies

        Returns:
            Recommendation string
        """
        if confidence < 0.5:
            return "ALLOW"
        elif confidence < 0.75:
            return "LOG"
        elif confidence < 0.90:
            return "ALERT"
        else:
            return "BLOCK"

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get analyzer statistics

        Returns:
            Dictionary with analysis statistics
        """
        return {
            'total_keystrokes': self.total_keystrokes,
            'total_anomalies': self.total_anomalies,
            'anomaly_rate': self.total_anomalies / max(self.total_keystrokes, 1),
            'history_size': len(self.history),
            'profile_sample_count': self.profile.get('sample_count', 0),
        }


def test_behavioral_analyzer():
    """Test behavioral analyzer"""
    # Mock profile
    profile = {
        'version': '2.0',
        'sample_count': 10000,
        'speed_distribution': {
            'mean_ms': 145.3,
            'std_dev_ms': 23.4,
            'q1': 130.0,
            'q3': 158.0,
        },
        'digraph_timings': {
            'th': {'mean_ms': 145, 'std_dev_ms': 23, 'samples': 100},
            'he': {'mean_ms': 132, 'std_dev_ms': 19, 'samples': 100},
        },
        'temporal_patterns': {
            'active_hours': [9, 10, 11, 14, 15, 16, 17],
        },
    }

    config = {
        'detection': {
            'confidence_threshold': 0.85,
            'speed_threshold_ms': 30,
            'digraph_analysis': True,
            'mouse_correlation': False,
        },
        'advanced': {
            'max_history_size': 1000,
        },
    }

    analyzer = BehavioralAnalyzer(profile, config)

    # Test 1: Normal human typing
    print("Test 1: Normal human typing")
    for i in range(30):
        event = KeystrokeEvent(
            timestamp=1704470400000 + i * 145,  # ~145ms intervals
            key=chr(97 + (i % 26)),  # a-z
            key_code=65 + (i % 26),
            inter_key_ms=145 + np.random.normal(0, 20),  # Normal variation
            window_name="Firefox",
            window_category="BROWSER",
            injected=False,
            modifiers=[],
            is_backspace=False
        )
        result = analyzer.analyze_keystroke(event)

    print(f"Normal typing - Anomaly: {result.is_anomaly}, Confidence: {result.confidence:.3f}")
    assert result.confidence < 0.5  # Should be low confidence

    # Test 2: Fast injection attack
    print("\nTest 2: Fast injection attack")
    for i in range(30):
        event = KeystrokeEvent(
            timestamp=1704470400000 + i * 15,  # 15ms intervals (very fast!)
            key=chr(97 + (i % 26)),
            key_code=65 + (i % 26),
            inter_key_ms=15,  # Consistent fast speed
            window_name="Command Prompt",
            window_category="TERMINAL",
            injected=True,  # Hardware injection flag
            modifiers=[],
            is_backspace=False
        )
        result = analyzer.analyze_keystroke(event)

    print(f"Fast injection - Anomaly: {result.is_anomaly}, Confidence: {result.confidence:.3f}")
    print(f"Reasons: {result.reasons}")
    assert result.is_anomaly == True
    assert result.confidence > 0.85

    # Test 3: Zero error rate (suspicious)
    print("\nTest 3: Zero error rate (perfect typing)")
    analyzer2 = BehavioralAnalyzer(profile, config)
    for i in range(150):
        event = KeystrokeEvent(
            timestamp=1704470400000 + i * 140,
            key=chr(97 + (i % 26)),
            key_code=65 + (i % 26),
            inter_key_ms=140,
            window_name="Notepad",
            window_category="EDITOR",
            injected=False,
            modifiers=[],
            is_backspace=False  # No corrections!
        )
        result = analyzer2.analyze_keystroke(event)

    print(f"Zero errors - Anomaly: {result.is_anomaly}, Confidence: {result.confidence:.3f}")
    print(f"Reasons: {result.reasons}")

    # Test statistics
    stats = analyzer.get_statistics()
    print(f"\nAnalyzer statistics: {stats}")

    print("\n✅ Behavioral analyzer tests passed!")


if __name__ == '__main__':
    test_behavioral_analyzer()
