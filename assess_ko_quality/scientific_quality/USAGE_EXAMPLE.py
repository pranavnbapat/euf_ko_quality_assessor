#!/usr/bin/env python3
"""
Example usage of scientific KO quality assessment framework.

This script demonstrates the complete workflow:
1. Annotate a sample of KOs with human judgments
2. Extract features
3. Train ML model
4. Evaluate and interpret
5. Use for inference
"""

import numpy as np
from pathlib import Path

# Import scientific quality framework
from scientific_quality import ScientificQualityAssessor, ScientificQualityConfig
from scientific_quality.annotation import (
    QualityRubric,
    stratified_sample_for_annotation,
    compute_inter_annotator_agreement,
)
from scientific_quality.features import FeatureExtractor, build_feature_matrix


def example_workflow():
    """Complete example workflow."""
    
    # =========================================================================
    # PHASE 1: Ground Truth Collection
    # =========================================================================
    print("=" * 70)
    print("PHASE 1: Ground Truth Collection")
    print("=" * 70)
    
    # Export annotation guidelines for human experts
    QualityRubric.export_markdown("annotation_guidelines.md")
    print("Annotation guidelines exported to annotation_guidelines.md")
    
    # In practice, you would:
    # 1. Load all your KOs
    # all_kos = load_kos_from_jsonl("kos.jsonl")
    
    # 2. Select stratified sample for annotation
    # sample_kos = stratified_sample_for_annotation(all_kos, n_samples=300)
    
    # 3. Have experts annotate (using the rubric)
    # This would be done externally, resulting in a CSV like:
    # ko_id,structural,semantic,domain,functional,overall
    # ko_001,4,5,4,3,4
    # ...
    
    # For this example, we'll simulate annotations
    print("Simulating human annotations for demonstration...")
    n_kos = 300
    n_dims = 5
    # Simulate labels (normally these come from human experts)
    labels = np.random.randint(1, 6, size=(n_kos, n_dims))
    
    # Compute inter-annotator agreement (if multiple annotators)
    # annotations = {
    #     "annotator_1": [...],
    #     "annotator_2": [...],
    #     "annotator_3": [...],
    # }
    # agreement = compute_inter_annotator_agreement(annotations)
    # print(f"Inter-annotator agreement: {agreement['mean_kappa']:.3f}")
    
    # =========================================================================
    # PHASE 2: Feature Engineering
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: Feature Engineering")
    print("=" * 70)
    
    # Create feature extractor
    extractor = FeatureExtractor(
        embedding_model="all-mpnet-base-v2",
        use_embeddings=True,
        use_readability=True,
        use_information_theory=True,
    )
    
    # In practice:
    # feature_df = build_feature_matrix(kos, labels, extractor)
    # print(f"Extracted {feature_df.shape[1]} features")
    
    # =========================================================================
    # PHASE 3: Model Training
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 3: Model Training")
    print("=" * 70)
    
    # Create assessor with custom config
    config = ScientificQualityConfig(
        model_type="xgboost",
        use_optuna=True,
        optuna_trials=50,
        cv_folds=5,
    )
    
    assessor = ScientificQualityAssessor(config)
    
    # Train (in practice, use real kos and labels)
    # results = assessor.train(kos, labels)
    
    # Save trained model
    # assessor.save("models/quality_model_v1")
    
    # =========================================================================
    # PHASE 4: Inference
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 4: Inference")
    print("=" * 70)
    
    # Load trained model
    # assessor = ScientificQualityAssessor.load("models/quality_model_v1")
    
    # Assess a single KO
    example_ko = {
        "_orig_id": "ko_example_001",
        "title": "Sustainable Wheat Farming: Best Practices",
        "subtitle": "A comprehensive guide for small farmers",
        "description": "This KO covers sustainable wheat farming techniques...",
        "ko_content_flat": "Step 1: Soil preparation... Step 2: Seed selection...",
        "keywords": ["wheat", "sustainability", "farming", "agriculture"],
    }
    
    # result = assessor.assess_ko(example_ko)
    # print(f"Overall quality: {result['overall_quality']}")
    # print(f"Confidence: {result['confidence']}")
    
    print("\nExample workflow complete!")
    print("See USAGE_EXAMPLE.py for full code.")


if __name__ == "__main__":
    example_workflow()
