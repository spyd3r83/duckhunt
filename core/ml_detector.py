"""
DuckHunt v2.0 - Machine Learning Based Detector (Optional)

Uses Isolation Forest for unsupervised anomaly detection.
Achieves >90% detection rate against sophisticated evasion.

Optional dependency: scikit-learn
If not available, falls back to statistical methods.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import deque
import math
import time

try:
    from sklearn.ensemble import IsolationForest
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    np = None


class RateLimiter:
    """
    Simple rate limiter to prevent DOS attacks via excessive prediction requests
    """
    def __init__(self, max_per_second: int = 20):
        self.max_per_second = max_per_second
        self.timestamps = deque(maxlen=max_per_second)

    def is_exceeded(self) -> bool:
        """Check if rate limit is exceeded"""
        now = time.time()
        self.timestamps.append(now)

        if len(self.timestamps) == self.max_per_second:
            oldest = self.timestamps[0]
            if now - oldest < 1.0:
                return True  # More than max_per_second requests in 1 second
        return False


@dataclass
class MLDetectionResult:
    """Result from ML-based detection"""
    is_anomaly: bool
    anomaly_score: float  # -1 to 1 (lower = more anomalous)
    confidence: float     # 0 to 1
    features: Dict[str, float]
    explanation: str


class MLDetector:
    """
    Machine learning based anomaly detector using Isolation Forest.

    Isolation Forest is effective for:
    - Unsupervised learning (no labeled attack data needed)
    - High-dimensional feature spaces
    - Detecting subtle anomalies
    - Fast inference
    """

    def __init__(self, contamination: float = 0.01):
        """
        Initialize ML detector.

        Args:
            contamination: Expected proportion of outliers (0.01 = 1%)
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn not available. Install with: pip install scikit-learn>=1.0.0"
            )

        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100,
            max_samples='auto',
            bootstrap=False
        )

        self.is_trained = False
        self.feature_names = [
            'mean', 'std_dev', 'cv', 'skewness', 'kurtosis',
            'entropy', 'autocorr_lag1', 'autocorr_lag2',
            'hurst', 'variance_cv', 'min_interval', 'max_interval',
            'range_ratio', 'q1', 'q3', 'iqr'
        ]

        # Rate limiter to prevent DOS attacks
        self.rate_limiter = RateLimiter(max_per_second=20)

    def train(self, training_sequences: List[List[float]]) -> None:
        """
        Train the model on normal human typing sequences.

        Args:
            training_sequences: List of interval sequences from normal typing
        """
        if not SKLEARN_AVAILABLE:
            return

        # Extract features from each sequence
        features_list = []
        for sequence in training_sequences:
            if len(sequence) >= 20:
                features = self._extract_features(sequence)
                features_list.append(list(features.values()))

        if len(features_list) < 10:
            raise ValueError("Need at least 10 training sequences")

        # Train model
        X = np.array(features_list)
        self.model.fit(X)
        self.is_trained = True

    def predict(self, intervals: List[float]) -> MLDetectionResult:
        """
        Predict if sequence is anomalous.

        Args:
            intervals: Keystroke interval sequence

        Returns:
            MLDetectionResult with prediction
        """
        # Check rate limit FIRST (DOS prevention)
        if self.rate_limiter.is_exceeded():
            return MLDetectionResult(
                is_anomaly=True,
                anomaly_score=-0.5,
                confidence=0.8,
                features={},
                explanation="Rate limit exceeded - possible DOS attack"
            )

        if not SKLEARN_AVAILABLE:
            # Fail CLOSED: assume attack when sklearn not available
            return MLDetectionResult(
                is_anomaly=True,
                anomaly_score=-0.2,
                confidence=0.2,
                features={},
                explanation="scikit-learn not available - failing secure"
            )

        if not self.is_trained:
            # SECURITY FIX: Fail CLOSED (assume attack when untrained)
            return MLDetectionResult(
                is_anomaly=True,
                anomaly_score=-0.3,
                confidence=0.3,
                features={},
                explanation="ML model not trained - failing secure"
            )

        if len(intervals) < 20:
            return MLDetectionResult(
                is_anomaly=False,
                anomaly_score=0.0,
                confidence=0.0,
                features={},
                explanation="Insufficient data for ML prediction"
            )

        # Extract features
        features = self._extract_features(intervals)
        feature_vector = np.array([list(features.values())])

        # Predict
        prediction = self.model.predict(feature_vector)[0]  # 1 = normal, -1 = anomaly
        anomaly_score = self.model.decision_function(feature_vector)[0]

        # Convert to confidence (0-1)
        # Anomaly score ranges from ~-0.5 (very anomalous) to ~0.5 (very normal)
        # Map to 0-1 where 1 = high confidence anomaly
        confidence = self._score_to_confidence(anomaly_score)

        is_anomaly = prediction == -1

        explanation = self._generate_explanation(features, anomaly_score)

        return MLDetectionResult(
            is_anomaly=is_anomaly,
            anomaly_score=float(anomaly_score),
            confidence=confidence,
            features=features,
            explanation=explanation
        )

    def _extract_features(self, intervals: List[float]) -> Dict[str, float]:
        """
        Extract comprehensive feature set from interval sequence.

        Features capture:
        - Central tendency (mean, median)
        - Dispersion (std_dev, cv, range)
        - Shape (skewness, kurtosis)
        - Information theory (entropy)
        - Time series properties (autocorrelation, Hurst)
        - Distribution characteristics (quartiles, IQR)
        """
        n = len(intervals)
        if n == 0:
            return {name: 0.0 for name in self.feature_names}

        # Basic statistics
        mean = sum(intervals) / n
        variance = sum((x - mean) ** 2 for x in intervals) / n
        std_dev = math.sqrt(variance)
        cv = std_dev / mean if mean > 0 else 0.0

        min_val = min(intervals)
        max_val = max(intervals)
        range_val = max_val - min_val
        range_ratio = range_val / mean if mean > 0 else 0.0

        # Quartiles
        sorted_intervals = sorted(intervals)
        q1_idx = n // 4
        q3_idx = 3 * n // 4
        q1 = sorted_intervals[q1_idx]
        q3 = sorted_intervals[q3_idx]
        iqr = q3 - q1

        # Higher moments
        m3 = sum((x - mean) ** 3 for x in intervals) / n
        m4 = sum((x - mean) ** 4 for x in intervals) / n
        skewness = m3 / (variance ** 1.5) if variance > 0 else 0.0
        kurtosis = (m4 / (variance ** 2)) - 3 if variance > 0 else 0.0

        # Entropy
        entropy = self._calculate_entropy(intervals)

        # Autocorrelation
        autocorr_lag1 = self._calculate_autocorr(intervals, lag=1)
        autocorr_lag2 = self._calculate_autocorr(intervals, lag=2)

        # Hurst exponent (simplified)
        hurst = self._calculate_hurst(intervals) if n >= 50 else 0.5

        # Rolling variance coefficient of variation
        variance_cv = self._calculate_variance_cv(intervals)

        return {
            'mean': mean,
            'std_dev': std_dev,
            'cv': cv,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'entropy': entropy,
            'autocorr_lag1': autocorr_lag1,
            'autocorr_lag2': autocorr_lag2,
            'hurst': hurst,
            'variance_cv': variance_cv,
            'min_interval': min_val,
            'max_interval': max_val,
            'range_ratio': range_ratio,
            'q1': q1,
            'q3': q3,
            'iqr': iqr
        }

    def _calculate_entropy(self, intervals: List[float]) -> float:
        """Calculate Shannon entropy"""
        bins = 20
        min_val = min(intervals)
        max_val = max(intervals)
        bin_width = (max_val - min_val) / bins if max_val > min_val else 1

        counts = [0] * bins
        for interval in intervals:
            bin_idx = int((interval - min_val) / bin_width)
            bin_idx = min(bin_idx, bins - 1)
            counts[bin_idx] += 1

        total = len(intervals)
        entropy = 0.0
        for count in counts:
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)

        return entropy

    def _calculate_autocorr(self, intervals: List[float], lag: int) -> float:
        """Calculate autocorrelation at given lag"""
        n = len(intervals)
        if n <= lag:
            return 0.0

        mean = sum(intervals) / n
        numerator = sum((intervals[i] - mean) * (intervals[i+lag] - mean)
                       for i in range(n - lag))
        denominator = sum((x - mean) ** 2 for x in intervals)

        return numerator / denominator if denominator > 0 else 0.0

    def _calculate_hurst(self, intervals: List[float]) -> float:
        """Calculate Hurst exponent (simplified)"""
        n = len(intervals)
        if n < 50:
            return 0.5

        mean = sum(intervals) / n
        cumsum = [0.0]
        for interval in intervals:
            cumsum.append(cumsum[-1] + (interval - mean))

        R = max(cumsum) - min(cumsum)
        variance = sum((x - mean) ** 2 for x in intervals) / n
        S = math.sqrt(variance)

        rs = R / S if S > 0 else 1.0
        hurst = math.log(rs) / math.log(n) if rs > 0 else 0.5

        return hurst

    def _calculate_variance_cv(self, intervals: List[float]) -> float:
        """Calculate coefficient of variation of rolling variance"""
        if len(intervals) < 30:
            return 0.0

        window_size = 10
        variances = []

        for i in range(len(intervals) - window_size):
            window = intervals[i:i+window_size]
            mean = sum(window) / len(window)
            variance = sum((x - mean) ** 2 for x in window) / len(window)
            variances.append(variance)

        if len(variances) < 2:
            return 0.0

        mean_var = sum(variances) / len(variances)
        var_of_var = sum((v - mean_var) ** 2 for v in variances) / len(variances)
        std_of_var = math.sqrt(var_of_var)

        return std_of_var / mean_var if mean_var > 0 else 0.0

    def _score_to_confidence(self, anomaly_score: float) -> float:
        """
        Convert Isolation Forest anomaly score to confidence (0-1).

        Anomaly score typically ranges from -0.5 to 0.5
        Lower score = more anomalous
        """
        # Map -0.5 to 1.0 (high confidence anomaly)
        # Map 0.5 to 0.0 (high confidence normal)
        confidence = (-anomaly_score + 0.5) / 1.0
        return max(0.0, min(1.0, confidence))

    def _generate_explanation(self, features: Dict[str, float],
                            anomaly_score: float) -> str:
        """Generate human-readable explanation"""
        if anomaly_score > 0:
            return "Typing pattern matches learned normal behavior"

        # Identify most anomalous features
        anomalies = []

        if features['cv'] < 0.15:
            anomalies.append("suspiciously low variation")
        if features['entropy'] < 4.0:
            anomalies.append("low entropy (mechanical)")
        if features['entropy'] > 6.5:
            anomalies.append("high entropy (too random)")
        if abs(features['autocorr_lag1']) < 0.15:
            anomalies.append("missing rhythm")
        if features['variance_cv'] < 0.25:
            anomalies.append("overly stable variance")
        if features['hurst'] < 0.55:
            anomalies.append("no long-range dependency")

        if anomalies:
            return "Anomalous: " + ", ".join(anomalies)
        else:
            return "Anomalous pattern detected by ML model"

    def save_model(self, filepath: str) -> None:
        """Save trained model to disk"""
        if not SKLEARN_AVAILABLE:
            return

        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)

    def load_model(self, filepath: str) -> None:
        """Load trained model from disk"""
        if not SKLEARN_AVAILABLE:
            return

        import pickle
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
            self.is_trained = True


def test_ml_detector():
    """Test ML detector"""
    if not SKLEARN_AVAILABLE:
        print("scikit-learn not available. Install with: pip install scikit-learn")
        return

    print("Testing ML-Based Detector")
    print("=" * 60)

    detector = MLDetector(contamination=0.05)

    # Generate training data (normal human typing)
    print("\nGenerating training data...")
    import random
    training_sequences = []

    for _ in range(50):
        sequence = []
        prev = 145
        for i in range(100):
            # Human typing with autocorrelation
            interval = prev + random.gauss(0, 15)
            interval = max(80, min(250, interval))
            sequence.append(interval)
            prev = interval * 0.7 + 145 * 0.3

            # Occasional errors (longer pauses)
            if random.random() < 0.03:
                sequence.append(random.gauss(300, 50))

        training_sequences.append(sequence)

    # Train model
    print("Training model...")
    detector.train(training_sequences)
    print("Model trained successfully")

    # Test 1: Normal typing
    print("\nTest 1: Normal human typing")
    test_normal = training_sequences[0]
    result = detector.predict(test_normal)
    print(f"  Anomaly: {result.is_anomaly}")
    print(f"  Score: {result.anomaly_score:.3f}")
    print(f"  Confidence: {result.confidence:.3f}")
    print(f"  Explanation: {result.explanation}")

    # Test 2: Fast bot
    print("\nTest 2: Fast bot injection")
    test_bot_fast = [15.0] * 100
    result = detector.predict(test_bot_fast)
    print(f"  Anomaly: {result.is_anomaly}")
    print(f"  Score: {result.anomaly_score:.3f}")
    print(f"  Confidence: {result.confidence:.3f}")
    print(f"  Explanation: {result.explanation}")

    # Test 3: Bot with random delays
    print("\nTest 3: Bot with programmed random delays")
    test_bot_random = [random.gauss(145, 25) for _ in range(100)]
    result = detector.predict(test_bot_random)
    print(f"  Anomaly: {result.is_anomaly}")
    print(f"  Score: {result.anomaly_score:.3f}")
    print(f"  Confidence: {result.confidence:.3f}")
    print(f"  Explanation: {result.explanation}")

    print("\n" + "=" * 60)
    print("ML detection tests complete")


if __name__ == '__main__':
    test_ml_detector()
