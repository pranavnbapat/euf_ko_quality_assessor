# scientific_quality/explainability/shap_explainer.py
"""
SHAP-based explainability for quality predictions.
"""

import numpy as np
from typing import Dict, List, Any


class QualityExplainer:
    """
    Explain model predictions using SHAP values.
    """
    
    def __init__(self, model, feature_names: List[str]):
        """
        Initialize explainer.
        
        Args:
            model: Trained MultiTaskQualityModel
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names
        self.explainers = {}
    
    def fit(self, X_background: np.ndarray):
        """
        Fit SHAP explainers.
        
        Args:
            X_background: Background data for SHAP (subset of training data)
        """
        try:
            import shap
        except ImportError:
            raise ImportError("shap not installed. Run: pip install shap")
        
        print("Fitting SHAP explainers...")
        
        for dim, model in self.model.models.items():
            if hasattr(model, 'get_booster'):
                # XGBoost
                self.explainers[dim] = shap.TreeExplainer(model)
            else:
                # Fallback to KernelExplainer (slower)
                self.explainers[dim] = shap.KernelExplainer(model.predict, X_background)
        
        print("SHAP explainers ready.")
    
    def explain(
        self,
        X: np.ndarray,
        ko_id: str = None,
    ) -> Dict[str, Any]:
        """
        Explain predictions for a single KO.
        
        Args:
            X: Feature vector (1, n_features)
            ko_id: KO identifier
        
        Returns:
            Dict with SHAP values and interpretation
        """
        explanations = {}
        
        for dim, explainer in self.explainers.items():
            shap_values = explainer.shap_values(X)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # For multi-class
            
            # Get top features
            feature_importance = list(zip(self.feature_names, shap_values[0]))
            feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
            
            explanations[dim] = {
                "prediction": float(self.model.models[dim].predict(X)[0]),
                "base_value": float(explainer.expected_value if hasattr(explainer, 'expected_value') else 0),
                "top_positive": [(f, float(v)) for f, v in feature_importance if v > 0][:5],
                "top_negative": [(f, float(v)) for f, v in feature_importance if v < 0][:5],
            }
        
        return {
            "ko_id": ko_id,
            "explanations": explanations,
        }
    
    def explain_batch(
        self,
        X: np.ndarray,
        ko_ids: List[str] = None,
    ) -> List[Dict[str, Any]]:
        """Explain multiple KOs."""
        results = []
        for i in range(X.shape[0]):
            ko_id = ko_ids[i] if ko_ids else f"ko_{i}"
            results.append(self.explain(X[i:i+1], ko_id))
        return results
