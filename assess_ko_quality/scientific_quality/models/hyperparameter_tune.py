# scientific_quality/models/hyperparameter_tune.py
"""
Hyperparameter tuning using Optuna.
"""

import numpy as np
import optuna
from typing import Dict, Any
from sklearn.model_selection import cross_val_score


class HyperparameterTuner:
    """Tune hyperparameters using Bayesian optimization."""
    
    def __init__(
        self,
        model_type: str = "xgboost",
        n_trials: int = 100,
        cv_folds: int = 3,
    ):
        self.model_type = model_type
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.best_params = None
    
    def tune(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_dim: int = 4,  # Tune for 'overall' quality by default
    ) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.
        
        Args:
            X: Feature matrix
            y: Target matrix
            target_dim: Which dimension to optimize for (default: overall)
        
        Returns:
            Best hyperparameters
        """
        print(f"Starting hyperparameter tuning ({self.n_trials} trials)...")
        
        def objective(trial):
            if self.model_type == "xgboost":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                    "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
                }
                from xgboost import XGBRegressor
                model = XGBRegressor(**params, random_state=42)
            
            elif self.model_type == "lightgbm":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                    "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
                    "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
                    "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
                }
                from lightgbm import LGBMRegressor
                model = LGBMRegressor(**params, random_state=42)
            
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            # Cross-validation
            scores = cross_val_score(
                model, X, y[:, target_dim],
                cv=self.cv_folds,
                scoring="neg_mean_absolute_error",
                n_jobs=-1,
            )
            
            return -scores.mean()  # Minimize MAE
        
        # Create study
        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(),
        )
        
        # Optimize
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        # Store best params
        self.best_params = study.best_params
        
        print(f"\nBest MAE: {study.best_value:.4f}")
        print("Best hyperparameters:")
        for k, v in self.best_params.items():
            print(f"  {k}: {v}")
        
        return self.best_params


def tune_hyperparameters(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "xgboost",
    n_trials: int = 100,
) -> Dict[str, Any]:
    """
    Convenience function for hyperparameter tuning.
    
    Usage:
        best_params = tune_hyperparameters(X_train, y_train)
        model = MultiTaskQualityModel(model_type="xgboost", model_params=best_params)
    """
    tuner = HyperparameterTuner(model_type=model_type, n_trials=n_trials)
    return tuner.tune(X, y)
