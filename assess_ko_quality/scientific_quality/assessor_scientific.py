# scientific_quality/assessor_scientific.py
"""
Main entry point for scientific KO quality assessment.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional

from .config import ScientificQualityConfig
from .features.feature_extractor import FeatureExtractor, build_feature_matrix
from .models.multi_task_model import MultiTaskQualityModel
from .models.trainer import ModelTrainer
from .explainability.shap_explainer import QualityExplainer


class ScientificQualityAssessor:
    """
    Scientific, ML-based KO quality assessor.
    
    Usage:
        # Training
        assessor = ScientificQualityAssessor()
        assessor.train(kos_with_labels)
        assessor.save("path/to/model.pkl")
        
        # Inference
        assessor = ScientificQualityAssessor.load("path/to/model.pkl")
        result = assessor.assess_ko(ko_dict)
    """
    
    def __init__(self, config: Optional[ScientificQualityConfig] = None):
        """
        Initialize assessor.
        
        Args:
            config: Configuration object (creates default if None)
        """
        self.config = config or ScientificQualityConfig()
        self.feature_extractor = FeatureExtractor(
            embedding_model=self.config.embedding_model,
            use_embeddings=True,
        )
        self.model = None
        self.explainer = None
        self.is_trained = False
    
    def train(
        self,
        kos: List[Dict[str, Any]],
        labels: np.ndarray,
        groups: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Train the quality assessment model.
        
        Args:
            kos: List of KOs with features
            labels: Human judgments (n_kos, n_dimensions)
            groups: Group labels for GroupKFold (e.g., institution)
        
        Returns:
            Training results and metrics
        """
        print("=" * 70)
        print("SCIENTIFIC QUALITY ASSESSOR - TRAINING")
        print("=" * 70)
        
        # 1. Extract features
        print("\n[1/4] Extracting features...")
        feature_df = build_feature_matrix(kos, labels, self.feature_extractor)
        
        # Separate features and labels
        label_cols = [c for c in feature_df.columns if c.startswith("label_")]
        X = feature_df.drop(columns=label_cols + ["ko_id"]).values
        y = feature_df[label_cols].values
        feature_names = [c for c in feature_df.columns if c not in label_cols + ["ko_id"]]
        
        print(f"Feature matrix: {X.shape}")
        print(f"Labels: {y.shape}")
        
        # 2. Hyperparameter tuning (optional)
        if self.config.use_optuna:
            print("\n[2/4] Tuning hyperparameters...")
            from .models.hyperparameter_tune import tune_hyperparameters
            best_params = tune_hyperparameters(
                X, y,
                model_type=self.config.model_type,
                n_trials=self.config.optuna_trials,
            )
        else:
            best_params = {}
        
        # 3. Train model
        print("\n[3/4] Training model...")
        self.model = MultiTaskQualityModel(
            model_type=self.config.model_type,
            model_params=best_params,
        )
        
        trainer = ModelTrainer(self.model, cv_folds=self.config.cv_folds)
        results = trainer.train(X, y, groups, feature_names)
        
        # 4. Fit explainer
        print("\n[4/4] Fitting SHAP explainer...")
        self.explainer = QualityExplainer(self.model, feature_names)
        # Use subset for background
        bg_size = min(100, X.shape[0])
        self.explainer.fit(X[:bg_size])
        
        self.is_trained = True
        
        # Store feature names
        self.feature_names = feature_names
        
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        
        return results
    
    def assess_ko(
        self,
        ko: Dict[str, Any],
        explain: bool = True,
    ) -> Dict[str, Any]:
        """
        Assess quality of a single KO.
        
        Args:
            ko: KO dictionary
            explain: Whether to include SHAP explanations
        
        Returns:
            Quality assessment with predictions and diagnostics
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() or load() first.")
        
        # Extract features
        features = self.feature_extractor.extract(ko)
        X = np.array([[features[f] for f in self.feature_names]])
        
        # Predict
        predictions = self.model.predict(X)[0]
        
        # Predict with uncertainty
        uncertainty = self.model.predict_with_uncertainty(X)
        
        # Build result
        result = {
            "ko_id": ko.get("_orig_id") or ko.get("@id"),
            "title": ko.get("title", "")[:100],
        }
        
        # Add predictions for each dimension
        for i, dim in enumerate(self.model.dimensions):
            result[f"{dim}_quality"] = round(predictions[i], 2)
            result[f"{dim}_uncertainty"] = round(uncertainty["std"][0, i], 3)
            result[f"{dim}_ci_lower"] = round(uncertainty["lower_ci"][0, i], 2)
            result[f"{dim}_ci_upper"] = round(uncertainty["upper_ci"][0, i], 2)
        
        # Overall quality (weighted average)
        weights = {
            "structural": self.config.WEIGHTS["structural"],
            "semantic": self.config.WEIGHTS["semantic"],
            "functional": self.config.WEIGHTS["functional"],
            "domain": self.config.WEIGHTS["domain"],
        }
        
        overall = sum(
            predictions[i] * weights.get(dim, 25) / 100
            for i, dim in enumerate(self.model.dimensions[:-1])  # Exclude 'overall' itself
        )
        result["overall_quality"] = round(overall * 5, 2)  # Scale to 0-25
        result["confidence"] = "high" if np.mean(uncertainty["std"]) < 0.5 else "medium" if np.mean(uncertainty["std"]) < 1.0 else "low"
        
        # SHAP explanation
        if explain and self.explainer:
            explanation = self.explainer.explain(X, result["ko_id"])
            result["explanation"] = explanation["explanations"]
        
        return result
    
    def assess_batch(
        self,
        kos: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Assess multiple KOs efficiently.
        """
        return [self.assess_ko(ko, explain=False) for ko in kos]
    
    def save(self, path: str):
        """Save trained model."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = str(Path(path).with_suffix(".pkl"))
        self.model.save(model_path)
        
        # Save config and metadata
        import json
        metadata = {
            "config": self.config.__dict__,
            "feature_names": self.feature_names,
            "is_trained": self.is_trained,
        }
        meta_path = str(Path(path).with_suffix(".json"))
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str):
        """Load trained model."""
        import json
        
        # Load metadata
        meta_path = str(Path(path).with_suffix(".json"))
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        
        # Create instance
        config = ScientificQualityConfig(**metadata["config"])
        instance = cls(config)
        instance.feature_names = metadata["feature_names"]
        instance.is_trained = metadata["is_trained"]
        
        # Load model
        model_path = str(Path(path).with_suffix(".pkl"))
        instance.model = MultiTaskQualityModel.load(model_path)
        
        print(f"Model loaded from {path}")
        return instance
