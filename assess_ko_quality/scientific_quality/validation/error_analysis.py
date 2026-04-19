# scientific_quality/validation/error_analysis.py
"""
Error analysis for model debugging.
"""

import numpy as np
from typing import Dict, List, Any


def analyze_errors(
    kos: List[Dict],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dimensions: List[str],
    top_n: int = 20,
) -> Dict[str, List[Dict]]:
    """
    Analyze the worst prediction errors.
    
    Args:
        kos: List of KO dictionaries
        y_true: True human judgments
        y_pred: Model predictions
        dimensions: List of dimension names
        top_n: Number of worst errors to return
    
    Returns:
        Dict mapping dimension -> list of error dicts
    """
    results = {}
    
    for i, dim in enumerate(dimensions):
        true = y_true[:, i]
        pred = y_pred[:, i]
        errors = np.abs(true - pred)
        
        # Find top N worst errors
        worst_indices = np.argsort(errors)[-top_n:][::-1]
        
        error_list = []
        for idx in worst_indices:
            error_list.append({
                "index": int(idx),
                "ko_id": kos[idx].get("_orig_id") or kos[idx].get("@id") or f"ko_{idx}",
                "title": kos[idx].get("title", "")[:100],
                "true_score": float(true[idx]),
                "predicted_score": float(pred[idx]),
                "error": float(errors[idx]),
            })
        
        results[dim] = error_list
    
    return results


def error_patterns(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dimensions: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Identify systematic error patterns.
    
    Returns:
        Dict with error pattern statistics
    """
    patterns = {}
    
    for i, dim in enumerate(dimensions):
        true = y_true[:, i]
        pred = y_pred[:, i]
        errors = pred - true  # Signed errors
        
        patterns[dim] = {
            "mean_error": float(np.mean(errors)),
            "std_error": float(np.std(errors)),
            "overestimation_rate": float(np.mean(errors > 0.5)),
            "underestimation_rate": float(np.mean(errors < -0.5)),
            "exact_match_rate": float(np.mean(np.abs(errors) < 0.5)),
        }
    
    return patterns
