# scientific_quality/models/multi_task_model.py
"""
Multi-task learning model for predicting multiple quality dimensions.
"""

import numpy as np
import pickle
from typing import Dict, List, Any, Optional
from pathlib import Path


class MultiTaskQualityModel:
    """
    Multi-task model that predicts all quality dimensions simultaneously.
    
    Uses XGBoost by default (best performance for tabular data).
    """
    
    def __init__(
        self,
        model_type: str = "xgboost",
        dimensions: List[str] = None,
        model_params: Optional[Dict] = None,
    ):
        """
        Initialize model.
        
        Args:
            model_type: "xgboost", "random_forest", or "lightgbm"
            dimensions: List of quality dimensions to predict
            model_params: Hyperparameters for base model
        """
        self.model_type = model_type
        self.dimensions = dimensions or ["structural", "semantic", "domain", "functional", "overall"]
        self.model_params = model_params or {}
        self.models = {}  # One model per dimension
        self.feature_names = None
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None):
        """
        Fit model on training data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target matrix (n_samples, n_dimensions)
            feature_names: List of feature names
        """
        self.feature_names = feature_names
        n_dims = y.shape[1]
        
        print(f"Training {self.model_type} models for {n_dims} dimensions...")
        
        for i, dim in enumerate(self.dimensions[:n_dims]):
            print(f"  Training {dim} model...")
            model = self._create_base_model()
            model.fit(X, y[:, i])
            self.models[dim] = model
        
        self.is_fitted = True
        print("Training complete.")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict quality scores.
        
        Args:
            X: Feature matrix (n_samples, n_features)
        
        Returns:
            Predictions (n_samples, n_dimensions)
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        predictions = np.zeros((X.shape[0], len(self.models)))
        
        for i, (dim, model) in enumerate(self.models.items()):
            predictions[:, i] = model.predict(X)
        
        # Clip to valid range (1-5)
        predictions = np.clip(predictions, 1, 5)
        
        return predictions
    
    def predict_with_uncertainty(self, X: np.ndarray, method: str = "ensemble") -> Dict[str, np.ndarray]:
        """
        Predict with uncertainty estimation.
        
        Args:
            X: Feature matrix
            method: "ensemble" or "bootstrap"
        
        Returns:
            Dict with 'mean', 'std', 'lower_ci', 'upper_ci'
        """
        if method == "ensemble" and self.model_type == "xgboost":
            # Use tree ensemble variance
            return self._predict_ensemble_variance(X)
        else:
            # Bootstrap resampling
            return self._predict_bootstrap(X)
    
    def _predict_ensemble_variance(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Use XGBoost's get_booster().predict(pred_leaf=True) for variance."""
        means = self.predict(X)
        
        # For each dimension, compute variance across trees
        stds = np.zeros_like(means)
        
        for i, (dim, model) in enumerate(self.models.items()):
            if hasattr(model, 'get_booster'):
                # Get predictions from individual trees
                booster = model.get_booster()
                leaf_preds = booster.predict(booster.DMatrix(X), pred_leaf=True)
                # Approximate std from tree variance
                stds[:, i] = np.std(leaf_preds, axis=1) * 0.1  # Scale factor
        
        return {
            "mean": means,
            "std": stds,
            "lower_ci": np.clip(means - 1.96 * stds, 1, 5),
            "upper_ci": np.clip(means + 1.96 * stds, 1, 5),
        }
    
    def _predict_bootstrap(self, X: np.ndarray, n_bootstrap: int = 50) -> Dict[str, np.ndarray]:
        """Bootstrap resampling for uncertainty."""
        # Not implemented for individual models
        # Would need to store bootstrap models
        means = self.predict(X)
        stds = np.ones_like(means) * 0.5  # Placeholder
        
        return {
            "mean": means,
            "std": stds,
            "lower_ci": np.clip(means - 1.96 * stds, 1, 5),
            "upper_ci": np.clip(means + 1.96 * stds, 1, 5),
        }
    
    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """Get feature importance for each dimension."""
        importance = {}
        
        for dim, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                imp = model.feature_importances_
                if self.feature_names:
                    importance[dim] = dict(zip(self.feature_names, imp))
                else:
                    importance[dim] = {f"feature_{i}": v for i, v in enumerate(imp)}
        
        return importance
    
    def _create_base_model(self):
        """Create base model instance."""
        if self.model_type == "xgboost":
            from xgboost import XGBRegressor
            params = {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
            }
            params.update(self.model_params)
            return XGBRegressor(**params)
        
        elif self.model_type == "random_forest":
            from sklearn.ensemble import RandomForestRegressor
            params = {
                "n_estimators": 200,
                "max_depth": 10,
                "random_state": 42,
            }
            params.update(self.model_params)
            return RandomForestRegressor(**params)
        
        elif self.model_type == "lightgbm":
            from lightgbm import LGBMRegressor
            params = {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.05,
                "random_state": 42,
            }
            params.update(self.model_params)
            return LGBMRegressor(**params)
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def save(self, path: str):
        """Save model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'model_type': self.model_type,
                'dimensions': self.dimensions,
                'models': self.models,
                'feature_names': self.feature_names,
                'is_fitted': self.is_fitted,
            }, f)
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str):
        """Load model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        instance = cls(
            model_type=data['model_type'],
            dimensions=data['dimensions'],
        )
        instance.models = data['models']
        instance.feature_names = data['feature_names']
        instance.is_fitted = data['is_fitted']
        
        print(f"Model loaded from {path}")
        return instance
