# scientific_quality/monitoring/drift_detection.py
"""
Detect distribution drift in incoming KOs.
"""

import numpy as np
from typing import Dict, List
from scipy import stats


class DriftDetector:
    """
    Detect when incoming KO distribution differs from training data.
    """
    
    def __init__(
        self,
        reference_data: np.ndarray,
        method: str = "ks_test",
        p_threshold: float = 0.05,
    ):
        """
        Initialize drift detector.
        
        Args:
            reference_data: Training data distribution (n_samples, n_features)
            method: "ks_test" (Kolmogorov-Smirnov) or "wasserstein"
            p_threshold: P-value threshold for drift detection
        """
        self.reference_data = reference_data
        self.method = method
        self.p_threshold = p_threshold
        self.feature_names = None
    
    def detect_drift(
        self,
        new_data: np.ndarray,
        feature_names: List[str] = None,
    ) -> Dict[str, any]:
        """
        Detect drift between reference and new data.
        
        Args:
            new_data: New KO features to check
            feature_names: Names of features
        
        Returns:
            Dict with drift detection results
        """
        self.feature_names = feature_names
        
        n_features = self.reference_data.shape[1]
        drift_detected = False
        drifted_features = []
        p_values = []
        
        for i in range(n_features):
            ref_feature = self.reference_data[:, i]
            new_feature = new_data[:, i]
            
            if self.method == "ks_test":
                statistic, p_value = stats.ks_2samp(ref_feature, new_feature)
            else:
                # Wasserstein distance
                from scipy.stats import wasserstein_distance
                stat = wasserstein_distance(ref_feature, new_feature)
                # Convert to p-value approximation
                p_value = 1.0 if stat < 0.1 else 0.01
            
            p_values.append(p_value)
            
            if p_value < self.p_threshold:
                drift_detected = True
                feature_name = feature_names[i] if feature_names else f"feature_{i}"
                drifted_features.append({
                    "feature": feature_name,
                    "p_value": float(p_value),
                })
        
        # Overall drift metric
        fraction_drifted = len(drifted_features) / n_features
        
        result = {
            "drift_detected": drift_detected,
            "fraction_features_drifted": fraction_drifted,
            "drifted_features": drifted_features,
            "mean_p_value": float(np.mean(p_values)),
            "min_p_value": float(np.min(p_values)),
        }
        
        return result
    
    def needs_retraining(self, new_data: np.ndarray, threshold: float = 0.2) -> bool:
        """
        Check if model needs retraining based on drift severity.
        
        Args:
            new_data: New KO features
            threshold: Fraction of features that can drift before retraining
        
        Returns:
            True if retraining recommended
        """
        result = self.detect_drift(new_data)
        return result["fraction_features_drifted"] > threshold
