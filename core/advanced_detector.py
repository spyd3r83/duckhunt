"""
DuckHunt v2.0 - Advanced Detection Module

Implements sophisticated detection techniques to achieve >90% confidence
against evasive attacks:
- Entropy analysis (detect synthetic randomness)
- Hurst exponent (long-range dependency)
- Autocorrelation analysis (typing rhythm)
- Variance stability (detect mechanical consistency)
- Synthetic distribution detection
- Higher-order statistical moments

These techniques detect programmed evasion attempts that mimic
human timing but lack natural characteristics.
"""

import math
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import sys


@dataclass
class AdvancedAnalysisResult:
    """Result from advanced detection analysis"""
    is_suspicious: bool
    confidence: float
    anomaly_type: str
    details: Dict[str, Any]
    explanation: str


class AdvancedDetector:
    """
    Advanced detection techniques for sophisticated evasion.

    Detects:
    1. Synthetic randomness (programmed variance vs natural variance)
    2. Mechanical consistency (bots are too consistent)
    3. Missing long-range correlations (humans have rhythm)
    4. Unnatural error distributions
    5. Fatigue pattern absence
    """

    def __init__(self, baseline_profile: Dict = None):
        """Initialize with baseline profile"""
        self.baseline = baseline_profile or {}

        # Expected human characteristics from research
        self.human_autocorr_lag1 = 0.35  # Humans have rhythm
        self.human_hurst_min = 0.55      # Persistent time series
        self.human_entropy_min = 4.0     # Natural randomness
        self.human_entropy_max = 6.5     # Not too random
        self.min_cv = 0.15               # Minimum coefficient of variation

    def analyze_sequence(self, intervals: List[float]) -> AdvancedAnalysisResult:
        """
        Perform comprehensive advanced analysis on keystroke interval sequence.

        Args:
            intervals: List of inter-keystroke intervals in milliseconds

        Returns:
            AdvancedAnalysisResult with detection verdict
        """
        if len(intervals) < 20:
            return AdvancedAnalysisResult(
                is_suspicious=False,
                confidence=0.0,
                anomaly_type='insufficient_data',
                details={},
                explanation='Need at least 20 samples for advanced analysis'
            )

        anomalies = []
        total_suspicion = 0.0
        details = {}

        # Test 1: Entropy analysis (detect synthetic randomness)
        entropy_result = self._analyze_entropy(intervals)
        details['entropy'] = entropy_result
        if entropy_result['is_anomaly']:
            anomalies.append('entropy')
            total_suspicion += entropy_result['suspicion_score']

        # Test 2: Autocorrelation (detect missing rhythm)
        autocorr_result = self._analyze_autocorrelation(intervals)
        details['autocorrelation'] = autocorr_result
        if autocorr_result['is_anomaly']:
            anomalies.append('autocorrelation')
            total_suspicion += autocorr_result['suspicion_score']

        # Test 3: Variance stability (detect mechanical consistency)
        variance_result = self._analyze_variance_stability(intervals)
        details['variance_stability'] = variance_result
        if variance_result['is_anomaly']:
            anomalies.append('variance_stability')
            total_suspicion += variance_result['suspicion_score']

        # Test 4: Coefficient of variation (detect too-consistent timing)
        cv_result = self._analyze_coefficient_of_variation(intervals)
        details['coefficient_variation'] = cv_result
        if cv_result['is_anomaly']:
            anomalies.append('coefficient_variation')
            total_suspicion += cv_result['suspicion_score']

        # Test 5: Higher-order moments (skewness, kurtosis)
        moments_result = self._analyze_moments(intervals)
        details['moments'] = moments_result
        if moments_result['is_anomaly']:
            anomalies.append('moments')
            total_suspicion += moments_result['suspicion_score']

        # Test 6: Hurst exponent (long-range dependency)
        if len(intervals) >= 50:
            hurst_result = self._analyze_hurst_exponent(intervals)
            details['hurst'] = hurst_result
            if hurst_result['is_anomaly']:
                anomalies.append('hurst')
                total_suspicion += hurst_result['suspicion_score']

        # Test 7: Distribution shape testing
        dist_result = self._analyze_distribution_shape(intervals)
        details['distribution'] = dist_result
        if dist_result['is_anomaly']:
            anomalies.append('distribution_shape')
            total_suspicion += dist_result['suspicion_score']

        # Calculate overall confidence
        num_tests = 7 if len(intervals) >= 50 else 6
        avg_suspicion = total_suspicion / num_tests

        is_suspicious = avg_suspicion > 0.5 or len(anomalies) >= 3

        # Generate explanation
        explanation = self._generate_explanation(anomalies, details)

        return AdvancedAnalysisResult(
            is_suspicious=is_suspicious,
            confidence=min(avg_suspicion, 1.0),
            anomaly_type='synthetic_input' if is_suspicious else 'normal',
            details=details,
            explanation=explanation
        )

    def _analyze_entropy(self, intervals: List[float]) -> Dict[str, Any]:
        """
        Calculate Shannon entropy of timing sequence.

        Natural human typing: 4-6 bits
        Perfect randomness: ~8 bits
        Constant/mechanical: 0-2 bits
        """
        if len(intervals) < 10:
            return {'is_anomaly': False, 'suspicion_score': 0.0}

        # Discretize intervals into bins
        bins = 20
        min_val = min(intervals)
        max_val = max(intervals)
        bin_width = (max_val - min_val) / bins if max_val > min_val else 1

        # Count occurrences in each bin
        counts = [0] * bins
        for interval in intervals:
            bin_idx = int((interval - min_val) / bin_width)
            bin_idx = min(bin_idx, bins - 1)  # Clamp to last bin
            counts[bin_idx] += 1

        # Calculate entropy
        total = len(intervals)
        entropy = 0.0
        for count in counts:
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)

        # Analyze
        is_too_low = entropy < self.human_entropy_min  # Too mechanical
        is_too_high = entropy > self.human_entropy_max  # Too random (programmed)

        suspicion = 0.0
        if is_too_low:
            suspicion = 0.6 + (self.human_entropy_min - entropy) / self.human_entropy_min * 0.3
        elif is_too_high:
            suspicion = 0.5 + (entropy - self.human_entropy_max) / 4.0 * 0.4

        return {
            'is_anomaly': is_too_low or is_too_high,
            'suspicion_score': min(suspicion, 1.0),
            'entropy_bits': entropy,
            'expected_range': (self.human_entropy_min, self.human_entropy_max),
            'reason': 'too_mechanical' if is_too_low else 'too_random' if is_too_high else 'normal'
        }

    def _analyze_autocorrelation(self, intervals: List[float]) -> Dict[str, Any]:
        """
        Calculate autocorrelation at lag 1.

        Humans have rhythm (positive autocorrelation ~0.3-0.5)
        Bots with random delays have near-zero autocorrelation
        """
        if len(intervals) < 20:
            return {'is_anomaly': False, 'suspicion_score': 0.0}

        # Calculate mean
        mean = sum(intervals) / len(intervals)

        # Calculate lag-1 autocorrelation
        numerator = 0.0
        denominator = 0.0

        for i in range(len(intervals) - 1):
            numerator += (intervals[i] - mean) * (intervals[i+1] - mean)

        for interval in intervals:
            denominator += (interval - mean) ** 2

        autocorr = numerator / denominator if denominator > 0 else 0.0

        # Humans typically have positive autocorrelation (0.2-0.5)
        # Random bots have near-zero
        is_anomaly = abs(autocorr) < 0.15  # Too close to zero

        suspicion = 0.0
        if is_anomaly:
            # Lower autocorrelation = more suspicious
            suspicion = 0.5 + (0.15 - abs(autocorr)) / 0.15 * 0.4

        return {
            'is_anomaly': is_anomaly,
            'suspicion_score': min(suspicion, 1.0),
            'autocorrelation_lag1': autocorr,
            'expected_range': (0.2, 0.5),
            'reason': 'missing_rhythm' if is_anomaly else 'normal'
        }

    def _analyze_variance_stability(self, intervals: List[float]) -> Dict[str, Any]:
        """
        Analyze stability of variance over time.

        Humans: variance changes as they speed up/slow down
        Bots: suspiciously stable variance
        """
        if len(intervals) < 30:
            return {'is_anomaly': False, 'suspicion_score': 0.0}

        # Calculate rolling variance in windows
        window_size = 10
        variances = []

        for i in range(len(intervals) - window_size):
            window = intervals[i:i+window_size]
            mean = sum(window) / len(window)
            variance = sum((x - mean) ** 2 for x in window) / len(window)
            variances.append(variance)

        if len(variances) < 2:
            return {'is_anomaly': False, 'suspicion_score': 0.0}

        # Calculate variance of variances
        mean_var = sum(variances) / len(variances)
        var_of_var = sum((v - mean_var) ** 2 for v in variances) / len(variances)
        std_of_var = math.sqrt(var_of_var)

        # Coefficient of variation for variance
        cv_var = std_of_var / mean_var if mean_var > 0 else 0.0

        # Suspiciously stable variance (CV too low)
        is_anomaly = cv_var < 0.25  # Human variance varies more

        suspicion = 0.0
        if is_anomaly:
            suspicion = 0.6 + (0.25 - cv_var) / 0.25 * 0.3

        return {
            'is_anomaly': is_anomaly,
            'suspicion_score': min(suspicion, 1.0),
            'variance_cv': cv_var,
            'expected_min': 0.25,
            'reason': 'too_stable' if is_anomaly else 'normal'
        }

    def _analyze_coefficient_of_variation(self, intervals: List[float]) -> Dict[str, Any]:
        """
        Calculate coefficient of variation (std_dev / mean).

        Humans: CV typically 0.20-0.35
        Mechanical: CV < 0.15 (too consistent)
        """
        if len(intervals) < 10:
            return {'is_anomaly': False, 'suspicion_score': 0.0}

        mean = sum(intervals) / len(intervals)
        variance = sum((x - mean) ** 2 for x in intervals) / len(intervals)
        std_dev = math.sqrt(variance)

        cv = std_dev / mean if mean > 0 else 0.0

        # Too consistent = mechanical
        is_anomaly = cv < self.min_cv

        suspicion = 0.0
        if is_anomaly:
            suspicion = 0.7 + (self.min_cv - cv) / self.min_cv * 0.25

        return {
            'is_anomaly': is_anomaly,
            'suspicion_score': min(suspicion, 1.0),
            'coefficient_variation': cv,
            'expected_min': self.min_cv,
            'reason': 'mechanical_consistency' if is_anomaly else 'normal'
        }

    def _analyze_moments(self, intervals: List[float]) -> Dict[str, Any]:
        """
        Analyze skewness and kurtosis.

        Human timing is typically right-skewed (positive skew 0.8-1.5)
        Symmetric distributions (skew ~0) suggest synthetic data
        """
        if len(intervals) < 30:
            return {'is_anomaly': False, 'suspicion_score': 0.0}

        n = len(intervals)
        mean = sum(intervals) / n

        # Calculate moments
        m2 = sum((x - mean) ** 2 for x in intervals) / n
        m3 = sum((x - mean) ** 3 for x in intervals) / n
        m4 = sum((x - mean) ** 4 for x in intervals) / n

        # Skewness
        skewness = m3 / (m2 ** 1.5) if m2 > 0 else 0.0

        # Kurtosis (excess kurtosis)
        kurtosis = (m4 / (m2 ** 2)) - 3 if m2 > 0 else 0.0

        # Human typing typically has positive skew (0.8-1.5)
        # and moderate kurtosis (0-2)
        skew_anomaly = abs(skewness) < 0.5 or abs(skewness) > 2.5
        kurt_anomaly = kurtosis > 4.0  # Too heavy-tailed

        suspicion = 0.0
        if skew_anomaly:
            suspicion += 0.3
        if kurt_anomaly:
            suspicion += 0.3

        return {
            'is_anomaly': skew_anomaly or kurt_anomaly,
            'suspicion_score': min(suspicion, 1.0),
            'skewness': skewness,
            'kurtosis': kurtosis,
            'expected_skew_range': (0.8, 1.5),
            'expected_kurt_range': (0, 2),
            'reason': 'abnormal_distribution_shape'
        }

    def _analyze_hurst_exponent(self, intervals: List[float]) -> Dict[str, Any]:
        """
        Calculate Hurst exponent (simplified R/S analysis).

        H = 0.5: Random walk (bot with random delays)
        H > 0.5: Persistent (human - has memory/rhythm)
        H < 0.5: Anti-persistent (mean-reverting, unusual)
        """
        if len(intervals) < 50:
            return {'is_anomaly': False, 'suspicion_score': 0.0}

        # Simplified Hurst calculation using R/S analysis
        n = len(intervals)
        mean = sum(intervals) / n

        # Calculate cumulative deviations
        cumsum = [0.0]
        for interval in intervals:
            cumsum.append(cumsum[-1] + (interval - mean))

        # Calculate range
        R = max(cumsum) - min(cumsum)

        # Calculate standard deviation
        variance = sum((x - mean) ** 2 for x in intervals) / n
        S = math.sqrt(variance)

        # R/S ratio
        rs = R / S if S > 0 else 0.0

        # Hurst exponent approximation: H ≈ log(R/S) / log(n)
        hurst = math.log(rs) / math.log(n) if rs > 0 and n > 1 else 0.5

        # Humans: H = 0.55-0.70 (persistent)
        # Bots with random delays: H ≈ 0.50
        is_anomaly = hurst < self.human_hurst_min

        suspicion = 0.0
        if is_anomaly:
            suspicion = 0.5 + (self.human_hurst_min - hurst) / self.human_hurst_min * 0.4

        return {
            'is_anomaly': is_anomaly,
            'suspicion_score': min(suspicion, 1.0),
            'hurst_exponent': hurst,
            'expected_min': self.human_hurst_min,
            'reason': 'no_long_range_dependency' if is_anomaly else 'normal',
            'interpretation': 'random_walk' if abs(hurst - 0.5) < 0.05 else 'persistent' if hurst > 0.5 else 'anti_persistent'
        }

    def _analyze_distribution_shape(self, intervals: List[float]) -> Dict[str, Any]:
        """
        Test if distribution is suspiciously perfect (uniform or Gaussian).

        Programmed evasion often uses perfect distributions.
        Natural human timing is messier.
        """
        if len(intervals) < 30:
            return {'is_anomaly': False, 'suspicion_score': 0.0}

        # Bin the data
        bins = 10
        min_val = min(intervals)
        max_val = max(intervals)
        bin_width = (max_val - min_val) / bins if max_val > min_val else 1

        counts = [0] * bins
        for interval in intervals:
            bin_idx = int((interval - min_val) / bin_width)
            bin_idx = min(bin_idx, bins - 1)
            counts[bin_idx] += 1

        # Calculate uniformity (chi-square test approximation)
        expected = len(intervals) / bins
        chi_square = sum((c - expected) ** 2 / expected for c in counts if expected > 0)

        # Very low chi-square = suspiciously uniform
        # Critical value for 9 df at p=0.05 is ~16.9
        # Very uniform would be < 5
        is_too_uniform = chi_square < 3.0

        suspicion = 0.0
        if is_too_uniform:
            suspicion = 0.5 + (3.0 - chi_square) / 3.0 * 0.4

        return {
            'is_anomaly': is_too_uniform,
            'suspicion_score': min(suspicion, 1.0),
            'chi_square': chi_square,
            'reason': 'too_uniform' if is_too_uniform else 'normal'
        }

    def _generate_explanation(self, anomalies: List[str], details: Dict) -> str:
        """Generate human-readable explanation of detection"""
        if not anomalies:
            return "Typing pattern shows natural human characteristics."

        explanations = []

        if 'entropy' in anomalies:
            reason = details['entropy']['reason']
            if reason == 'too_mechanical':
                explanations.append("Timing is too consistent (low entropy - mechanical)")
            elif reason == 'too_random':
                explanations.append("Timing is too random (high entropy - programmed)")

        if 'autocorrelation' in anomalies:
            explanations.append("Missing typing rhythm (zero autocorrelation - bot)")

        if 'variance_stability' in anomalies:
            explanations.append("Variance suspiciously stable (mechanical consistency)")

        if 'coefficient_variation' in anomalies:
            explanations.append("Timing too consistent (low CV - not human)")

        if 'moments' in anomalies:
            explanations.append("Distribution shape unnatural (abnormal skewness/kurtosis)")

        if 'hurst' in anomalies:
            explanations.append("No long-range dependency (random walk - bot)")

        if 'distribution_shape' in anomalies:
            explanations.append("Distribution too uniform (programmed randomness)")

        return "; ".join(explanations)

    def detect_zero_error_anomaly(self, keystroke_buffer: List[str],
                                   window_size: int = 75) -> Dict[str, Any]:
        """
        Detect suspiciously perfect typing (no errors over long sequences).

        Humans make errors ~3-4% of the time.
        75+ keystrokes with zero backspaces is highly suspicious.
        """
        if len(keystroke_buffer) < window_size:
            return {'is_anomaly': False, 'suspicion_score': 0.0}

        # Check recent window
        recent = keystroke_buffer[-window_size:]
        backspace_count = sum(1 for k in recent if k in ['BackSpace', 'Backspace', 'back'])

        error_rate = backspace_count / len(recent)

        # Human error rate: 2-6%
        # Zero errors over 75 keys is suspicious
        is_anomaly = error_rate < 0.01

        suspicion = 0.0
        if is_anomaly:
            suspicion = 0.7 + (0.01 - error_rate) / 0.01 * 0.25

        return {
            'is_anomaly': is_anomaly,
            'suspicion_score': min(suspicion, 1.0),
            'error_rate': error_rate,
            'sequence_length': len(recent),
            'expected_errors': window_size * 0.035,
            'actual_errors': backspace_count,
            'reason': 'zero_error_rate' if is_anomaly else 'normal'
        }


def test_advanced_detector():
    """Test advanced detection capabilities"""
    print("Testing Advanced Detector")
    print("=" * 60)

    detector = AdvancedDetector()

    # Test 1: Normal human typing (varied, with rhythm)
    print("\nTest 1: Normal human typing")
    import random
    human_intervals = []
    prev = 145
    for i in range(100):
        # Add autocorrelation (rhythm)
        interval = prev + random.gauss(0, 15)
        interval = max(80, min(250, interval))
        human_intervals.append(interval)
        prev = interval * 0.7 + 145 * 0.3  # Mean reversion

    result = detector.analyze_sequence(human_intervals)
    print(f"  Suspicious: {result.is_suspicious}")
    print(f"  Confidence: {result.confidence:.3f}")
    print(f"  Explanation: {result.explanation}")

    # Test 2: Bot with programmed random delays
    print("\nTest 2: Bot with programmed random delays")
    bot_intervals = [random.gauss(145, 25) for _ in range(100)]

    result = detector.analyze_sequence(bot_intervals)
    print(f"  Suspicious: {result.is_suspicious}")
    print(f"  Confidence: {result.confidence:.3f}")
    print(f"  Explanation: {result.explanation}")

    # Test 3: Bot with perfect constant timing
    print("\nTest 3: Bot with constant timing")
    constant_intervals = [145.0] * 100

    result = detector.analyze_sequence(constant_intervals)
    print(f"  Suspicious: {result.is_suspicious}")
    print(f"  Confidence: {result.confidence:.3f}")
    print(f"  Explanation: {result.explanation}")

    # Test 4: Bot with uniform random distribution
    print("\nTest 4: Bot with uniform random")
    uniform_intervals = [random.uniform(100, 190) for _ in range(100)]

    result = detector.analyze_sequence(uniform_intervals)
    print(f"  Suspicious: {result.is_suspicious}")
    print(f"  Confidence: {result.confidence:.3f}")
    print(f"  Explanation: {result.explanation}")

    print("\n" + "=" * 60)
    print("Advanced detection tests complete")


if __name__ == '__main__':
    test_advanced_detector()
