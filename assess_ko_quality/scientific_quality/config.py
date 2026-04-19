# scientific_quality/config.py
"""
Configuration for scientific quality assessment.
All hyperparameters and paths centralized.
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class ScientificQualityConfig:
    """Configuration for scientific KO quality assessment."""
    
    # ---------- Annotation Phase ----------
    min_inter_annotator_agreement: float = 0.7  # Cohen's Kappa threshold
    num_annotators: int = 3
    annotation_sample_size: int = 300
    
    # Quality dimensions to assess
    quality_dimensions: List[str] = field(
        default_factory=lambda: ["structural", "semantic", "domain", "functional", "overall"]
    )
    
    # ---------- Feature Engineering ----------
    # Embedding model
    embedding_model: str = "all-mpnet-base-v2"
    max_content_length_chars: int = 10000  # For embedding extraction
    
    # Readability metrics to compute
    readability_metrics: List[str] = field(
        default_factory=lambda: [
            "flesch_reading_ease",
            "flesch_kincaid_grade",
            "smog_index",
            "coleman_liau_index",
            "automated_readability_index",
            "dale_chall_readability_score",
        ]
    )
    
    # Information theory
    compute_entropy: bool = True
    compute_ttr: bool = True  # Type-token ratio
    compute_hapax_ratio: bool = True
    
    # ---------- Model Training ----------
    # Model architecture
    model_type: str = "xgboost"  # "xgboost", "random_forest", "lightgbm"
    
    # XGBoost hyperparameters (or use optuna to tune)
    xgb_n_estimators: int = 200
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.05
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    
    # Cross-validation
    cv_folds: int = 5
    cv_strategy: str = "group_kfold"  # Prevents leakage across institutions
    
    # Hyperparameter tuning
    use_optuna: bool = True
    optuna_trials: int = 100
    
    # ---------- Validation ----------
    min_correlation_threshold: float = 0.7  # Pearson r with human judgments
    calibration_bins: int = 5
    
    # ---------- Uncertainty ----------
    uncertainty_method: str = "ensemble"  # "ensemble", "quantile", "bootstrap"
    num_ensemble_members: int = 5
    confidence_level: float = 0.95  # For confidence intervals
    
    # ---------- Active Learning ----------
    uncertainty_sampling_threshold: float = 0.5
    max_active_learning_iterations: int = 10
    
    # ---------- Drift Detection ----------
    drift_detection_method: str = "ks_test"  # "ks_test", "tabular_drift"
    drift_p_value: float = 0.05
    
    # ---------- Paths ----------
    annotation_output_dir: str = "./annotations"
    model_output_dir: str = "./models"
    feature_cache_dir: str = "./feature_cache"
    shap_cache_dir: str = "./shap_cache"
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        for path in [
            self.annotation_output_dir,
            self.model_output_dir,
            self.feature_cache_dir,
            self.shap_cache_dir,
        ]:
            os.makedirs(path, exist_ok=True)
