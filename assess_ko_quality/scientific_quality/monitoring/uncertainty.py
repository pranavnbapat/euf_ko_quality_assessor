# scientific_quality/monitoring/uncertainty.py
"""
Active learning via uncertainty sampling.
"""

import numpy as np
from typing import List, Dict, Any


class UncertaintySampler:
    """
    Identify KOs for active learning based on prediction uncertainty.
    """
    
    def __init__(self, threshold: float = 0.5):
        """
        Initialize sampler.
        
        Args:
            threshold: Uncertainty threshold for flagging
        """
        self.threshold = threshold
    
    def identify_uncertain_kos(
        self,
        kos: List[Dict],
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        top_k: int = 50,
    ) -> List[Dict]:
        """
        Identify KOs with high uncertainty for expert review.
        
        Args:
            kos: List of all KOs
            predictions: Predicted scores (n_kos, n_dims)
            uncertainties: Prediction uncertainties (n_kos, n_dims)
            top_k: Number of most uncertain KOs to return
        
        Returns:
            List of uncertain KOs with metadata
        """
        # Compute total uncertainty per KO
        total_uncertainty = np.sum(uncertainties, axis=1)
        
        # Find top K most uncertain
        top_indices = np.argsort(total_uncertainty)[-top_k:][::-1]
        
        uncertain_kos = []
        for idx in top_indices:
            uncertain_kos.append({
                "index": int(idx),
                "ko_id": kos[idx].get("_orig_id") or kos[idx].get("@id"),
                "title": kos[idx].get("title", "")[:100],
                "predictions": {f"dim_{i}": float(predictions[idx, i]) 
                               for i in range(predictions.shape[1])},
                "uncertainty": float(total_uncertainty[idx]),
                "reason": "high_prediction_uncertainty",
            })
        
        return uncertain_kos
    
    def identify_disagreement_cases(
        self,
        kos: List[Dict],
        predictions: np.ndarray,
        heuristic_scores: np.ndarray,
        threshold: float = 1.5,
    ) -> List[Dict]:
        """
        Identify KOs where ML model and heuristic disagree.
        
        These are interesting cases for expert review.
        """
        # Compare overall quality predictions
        ml_overall = predictions[:, -1]  # Assume last dim is overall
        heuristic_overall = heuristic_scores
        
        disagreement = np.abs(ml_overall - heuristic_overall)
        high_disagreement = disagreement > threshold
        
        indices = np.where(high_disagreement)[0]
        
        disagreement_cases = []
        for idx in indices[:50]:  # Top 50
            disagreement_cases.append({
                "index": int(idx),
                "ko_id": kos[idx].get("_orig_id") or kos[idx].get("@id"),
                "title": kos[idx].get("title", "")[:100],
                "ml_prediction": float(ml_overall[idx]),
                "heuristic_score": float(heuristic_overall[idx]),
                "disagreement": float(disagreement[idx]),
                "reason": "ml_heuristic_disagreement",
            })
        
        return disagreement_cases
