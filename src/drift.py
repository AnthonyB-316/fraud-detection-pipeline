"""
Feature drift detection for fraud detection pipeline.
Monitors distribution shifts between training and production data.
"""

import logging
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class DriftResult:
    """Result of drift detection for a single feature."""

    feature: str
    drift_detected: bool
    drift_score: float  # 0-1, higher = more drift
    p_value: float
    test_statistic: float
    test_type: str
    reference_mean: float
    current_mean: float
    reference_std: float
    current_std: float


@dataclass
class DriftReport:
    """Complete drift report across all features."""

    timestamp: str
    total_features: int
    features_with_drift: int
    overall_drift_score: float
    drift_detected: bool
    feature_results: List[DriftResult]
    sample_size: int

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "total_features": self.total_features,
            "features_with_drift": self.features_with_drift,
            "overall_drift_score": self.overall_drift_score,
            "drift_detected": self.drift_detected,
            "sample_size": self.sample_size,
            "feature_results": [
                {
                    "feature": r.feature,
                    "drift_detected": r.drift_detected,
                    "drift_score": r.drift_score,
                    "p_value": r.p_value,
                    "test_type": r.test_type,
                    "reference_mean": r.reference_mean,
                    "current_mean": r.current_mean,
                }
                for r in self.feature_results
            ],
        }


class PopulationStabilityIndex:
    """
    Calculate Population Stability Index (PSI) for drift detection.
    PSI < 0.1: No significant drift
    PSI 0.1-0.25: Moderate drift
    PSI > 0.25: Significant drift
    """

    @staticmethod
    def calculate(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
        """Calculate PSI between reference and current distributions."""
        # Create bins based on reference distribution
        _, bin_edges = np.histogram(reference, bins=bins)

        # Calculate proportions in each bin
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        cur_counts, _ = np.histogram(current, bins=bin_edges)

        # Convert to proportions
        ref_props = ref_counts / len(reference)
        cur_props = cur_counts / len(current)

        # Avoid division by zero
        ref_props = np.clip(ref_props, 0.0001, None)
        cur_props = np.clip(cur_props, 0.0001, None)

        # Calculate PSI
        psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))

        return float(psi)


class DriftDetector:
    """
    Monitors feature drift between training data and production data.
    Uses multiple statistical tests for robust drift detection.
    """

    def __init__(
        self,
        reference_data: Optional[pd.DataFrame] = None,
        feature_columns: Optional[List[str]] = None,
        window_size: int = 1000,
        drift_threshold: float = 0.1,
        p_value_threshold: float = 0.05,
    ):
        """
        Initialize drift detector.

        Args:
            reference_data: Training data to use as reference distribution
            feature_columns: List of feature column names to monitor
            window_size: Number of recent samples to use for drift detection
            drift_threshold: PSI threshold for drift detection
            p_value_threshold: P-value threshold for statistical tests
        """
        self.reference_data = reference_data
        self.feature_columns = feature_columns or []
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.p_value_threshold = p_value_threshold

        # Store reference statistics
        self.reference_stats: Dict[str, dict] = {}

        # Rolling window of recent production data
        self.production_buffer: deque = deque(maxlen=window_size)

        if reference_data is not None:
            self._compute_reference_stats()

    def _compute_reference_stats(self):
        """Compute statistics for reference data."""
        if self.reference_data is None:
            return

        for col in self.feature_columns:
            if col in self.reference_data.columns:
                data = self.reference_data[col].dropna().values
                self.reference_stats[col] = {
                    "mean": float(np.mean(data)),
                    "std": float(np.std(data)),
                    "min": float(np.min(data)),
                    "max": float(np.max(data)),
                    "median": float(np.median(data)),
                    "q25": float(np.percentile(data, 25)),
                    "q75": float(np.percentile(data, 75)),
                    "data": data,  # Store for statistical tests
                }

    def set_reference_stats(self, stats: Dict[str, dict]):
        """Set reference statistics from saved training data."""
        self.reference_stats = stats
        self.feature_columns = list(stats.keys())

    def add_sample(self, sample: Dict[str, float]):
        """Add a new production sample to the buffer."""
        self.production_buffer.append(sample)

    def add_samples(self, samples: List[Dict[str, float]]):
        """Add multiple production samples."""
        for sample in samples:
            self.production_buffer.append(sample)

    def _ks_test(self, feature: str, current_data: np.ndarray) -> Tuple[float, float]:
        """Perform Kolmogorov-Smirnov test."""
        if feature not in self.reference_stats:
            return 0.0, 1.0

        ref_data = self.reference_stats[feature].get("data")
        if ref_data is None:
            return 0.0, 1.0

        statistic, p_value = stats.ks_2samp(ref_data, current_data)
        return float(statistic), float(p_value)

    def _detect_feature_drift(
        self, feature: str, current_data: np.ndarray
    ) -> DriftResult:
        """Detect drift for a single feature."""
        ref_stats = self.reference_stats.get(feature, {})
        ref_data = ref_stats.get("data", np.array([]))

        # Calculate current statistics
        current_mean = float(np.mean(current_data))
        current_std = float(np.std(current_data))

        # Perform KS test
        ks_statistic, p_value = self._ks_test(feature, current_data)

        # Calculate PSI
        if len(ref_data) > 0:
            psi = PopulationStabilityIndex.calculate(ref_data, current_data)
        else:
            psi = 0.0

        # Determine if drift is detected
        drift_detected = (psi > self.drift_threshold) or (
            p_value < self.p_value_threshold
        )

        # Normalize drift score to 0-1
        drift_score = min(psi / 0.25, 1.0)  # Normalize: 0.25 PSI = 1.0 drift score

        return DriftResult(
            feature=feature,
            drift_detected=drift_detected,
            drift_score=drift_score,
            p_value=p_value,
            test_statistic=ks_statistic,
            test_type="KS-test + PSI",
            reference_mean=ref_stats.get("mean", 0.0),
            current_mean=current_mean,
            reference_std=ref_stats.get("std", 0.0),
            current_std=current_std,
        )

    def detect_drift(self, min_samples: int = 100) -> Optional[DriftReport]:
        """
        Detect drift across all monitored features.

        Args:
            min_samples: Minimum samples required for drift detection

        Returns:
            DriftReport if enough samples, None otherwise
        """
        if len(self.production_buffer) < min_samples:
            logger.info(
                f"Not enough samples for drift detection: {len(self.production_buffer)}/{min_samples}"
            )
            return None

        # Convert buffer to DataFrame
        current_df = pd.DataFrame(list(self.production_buffer))

        feature_results = []
        features_with_drift = 0
        total_drift_score = 0.0

        for feature in self.feature_columns:
            if feature in current_df.columns and feature in self.reference_stats:
                current_data = current_df[feature].dropna().values

                if len(current_data) > 0:
                    result = self._detect_feature_drift(feature, current_data)
                    feature_results.append(result)

                    if result.drift_detected:
                        features_with_drift += 1
                    total_drift_score += result.drift_score

        if len(feature_results) == 0:
            return None

        overall_drift_score = total_drift_score / len(feature_results)
        drift_detected = features_with_drift > 0 or overall_drift_score > 0.5

        from datetime import datetime

        return DriftReport(
            timestamp=datetime.utcnow().isoformat(),
            total_features=len(feature_results),
            features_with_drift=features_with_drift,
            overall_drift_score=overall_drift_score,
            drift_detected=drift_detected,
            feature_results=feature_results,
            sample_size=len(self.production_buffer),
        )

    def get_feature_stats(self) -> Dict[str, dict]:
        """Get current production buffer statistics."""
        if len(self.production_buffer) == 0:
            return {}

        current_df = pd.DataFrame(list(self.production_buffer))
        stats = {}

        for feature in self.feature_columns:
            if feature in current_df.columns:
                data = current_df[feature].dropna()
                stats[feature] = {
                    "count": len(data),
                    "mean": float(data.mean()),
                    "std": float(data.std()),
                    "min": float(data.min()),
                    "max": float(data.max()),
                }

        return stats


# Global drift detector instance
_drift_detector: Optional[DriftDetector] = None


def get_drift_detector() -> DriftDetector:
    """Get or create global drift detector."""
    global _drift_detector
    if _drift_detector is None:
        _drift_detector = DriftDetector(window_size=1000)
    return _drift_detector


def initialize_drift_detector(
    reference_data: pd.DataFrame, feature_columns: List[str], **kwargs
) -> DriftDetector:
    """Initialize global drift detector with reference data."""
    global _drift_detector
    _drift_detector = DriftDetector(
        reference_data=reference_data, feature_columns=feature_columns, **kwargs
    )
    return _drift_detector
