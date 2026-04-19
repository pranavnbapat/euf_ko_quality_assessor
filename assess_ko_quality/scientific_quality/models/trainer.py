# scientific_quality/models/trainer.py
"""
Model training pipeline with cross-validation and evaluation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr

from .multi_task_model import MultiTaskQualityModel


class ModelTrainer:
    """
    Training pipeline with proper validation and evaluation.
    """
    
    def __init__(
        self,
        model: MultiTaskQualityModel,
        cv_folds: int = 5,
        cv_strategy: str = "group_kfold",
    ):
        self.model = model
        self.cv_folds = cv_folds
        self.cv_strategy = cv_strategy
        self.results = {}
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray] = None,
        feature_names: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Train model with cross-validation.
        
        Args:
            X: Feature matrix
            y: Target matrix (human judgments)
            groups: Group labels for GroupKFold (e.g., institution)
            feature_names: Feature names
        
        Returns:
            Training results with CV metrics
        """
        print("=" * 60)
        print("TRAINING PIPELINE")
        print("=" * 60)
        
        # 1. Cross-validation
        cv_results = self._cross_validate(X, y, groups)
        
        # 2. Final model training on full data
        print("\nTraining final model on full dataset...")
        self.model.fit(X, y, feature_names)
        
        # 3. Feature importance
        importance = self.model.get_feature_importance()
        
        results = {
            "cv_results": cv_results,
            "feature_importance": importance,
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
        }
        
        self._print_results(results)
        
        return results
    
    def _cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Cross-validate model.
        
        Prevents data leakage by keeping groups (institutions) together.
        """
        print(f"\nCross-validation ({self.cv_folds} folds)...")
        
        if self.cv_strategy == "group_kfold" and groups is not None:
            cv = GroupKFold(n_splits=self.cv_folds)
            split_iter = cv.split(X, y, groups)
        else:
            from sklearn.model_selection import KFold
            cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
            split_iter = cv.split(X)
        
        # Store per-fold predictions
        all_predictions = []
        all_true = []
        
        for fold, (train_idx, val_idx) in enumerate(split_iter):
            print(f"  Fold {fold + 1}/{self.cv_folds}...")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model on this fold
            fold_model = MultiTaskQualityModel(
                model_type=self.model.model_type,
                dimensions=self.model.dimensions,
            )
            fold_model.fit(X_train, y_train)
            
            # Predict
            y_pred = fold_model.predict(X_val)
            
            all_predictions.append(y_pred)
            all_true.append(y_val)
        
        # Aggregate predictions
        predictions = np.vstack(all_predictions)
        true_values = np.vstack(all_true)
        
        # Compute metrics
        cv_results = self._compute_cv_metrics(true_values, predictions)
        
        return cv_results
    
    def _compute_cv_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """Compute cross-validation metrics."""
        results = {}
        
        for i, dim in enumerate(self.model.dimensions):
            true = y_true[:, i]
            pred = y_pred[:, i]
            
            results[dim] = {
                "pearson_r": pearsonr(true, pred)[0],
                "spearman_r": spearmanr(true, pred)[0],
                "mae": mean_absolute_error(true, pred),
                "rmse": np.sqrt(mean_squared_error(true, pred)),
                "mean_pred": np.mean(pred),
                "std_pred": np.std(pred),
            }
        
        return results
    
    def _print_results(self, results: Dict):
        """Print training results."""
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        
        print("\nCross-Validation Performance:")
        print("-" * 60)
        print(f"{'Dimension':<15} {'Pearson r':<12} {'MAE':<10} {'RMSE':<10}")
        print("-" * 60)
        
        for dim, metrics in results["cv_results"].items():
            print(f"{dim:<15} {metrics['pearson_r']:<12.3f} {metrics['mae']:<10.3f} {metrics['rmse']:<10.3f}")
        
        print("\nTop 10 Most Important Features (Overall Quality):")
        print("-" * 60)
        if "overall" in results["feature_importance"]:
            importance = results["feature_importance"]["overall"]
            sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
            for feat, imp in sorted_imp:
                print(f"  {feat}: {imp:.4f}")
    
    def evaluate_on_test_set(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate trained model on held-out test set.
        """
        print("\nEvaluating on test set...")
        
        y_pred = self.model.predict(X_test)
        
        results = {}
        for i, dim in enumerate(self.model.dimensions):
            true = y_test[:, i]
            pred = y_pred[:, i]
            
            results[dim] = {
                "pearson_r": pearsonr(true, pred)[0],
                "spearman_r": spearmanr(true, pred)[0],
                "mae": mean_absolute_error(true, pred),
                "rmse": np.sqrt(mean_squared_error(true, pred)),
            }
        
        print("\nTest Set Performance:")
        print("-" * 60)
        print(f"{'Dimension':<15} {'Pearson r':<12} {'MAE':<10} {'RMSE':<10}")
        print("-" * 60)
        for dim, metrics in results.items():
            print(f"{dim:<15} {metrics['pearson_r']:<12.3f} {metrics['mae']:<10.3f} {metrics['rmse']:<10.3f}")
        
        return results
