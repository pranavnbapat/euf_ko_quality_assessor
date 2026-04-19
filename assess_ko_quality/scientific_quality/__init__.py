"""
Scientific KO Quality Assessment Framework

A machine learning-based approach to KO quality assessment with:
- Ground truth collection and inter-annotator agreement
- Rich feature engineering (embeddings, readability, information theory)
- Multi-task learning models
- Validation, calibration, and uncertainty quantification
- Explainability via SHAP values
- Active learning and drift detection

Usage:
    from scientific_quality.assessor_scientific import ScientificQualityAssessor
    
    assessor = ScientificQualityAssessor.load("path/to/model")
    result = assessor.assess_ko(ko_dict)
"""

__version__ = "1.0.0"

from .assessor_scientific import ScientificQualityAssessor
from .config import ScientificQualityConfig

__all__ = [
    "ScientificQualityAssessor",
    "ScientificQualityConfig",
]
