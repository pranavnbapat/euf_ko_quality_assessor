# scientific_quality/features/feature_extractor.py
"""
Main feature extraction pipeline combining all feature types.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from .embedding_features import EmbeddingFeatureExtractor
from .readability_features import extract_readability_features
from .information_theory import extract_information_features
from quality_text_utils import tokens, strip_stops


@dataclass
class FeatureExtractor:
    """
    Extract all features for KO quality assessment.
    """
    
    embedding_model: str = "all-mpnet-base-v2"
    use_embeddings: bool = True
    use_readability: bool = True
    use_information_theory: bool = True
    
    def __post_init__(self):
        """Initialize embedding extractor if needed."""
        self._emb_extractor = None
        if self.use_embeddings:
            self._emb_extractor = EmbeddingFeatureExtractor(self.embedding_model)
    
    def extract(self, ko: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract all features for a single KO.
        
        Returns flat dictionary of ~80-100 features.
        """
        features = {}
        
        # Get text fields
        title = ko.get("title", "")
        desc = ko.get("description", "")
        content = ko.get("ko_content_flat", "")
        keywords = ko.get("keywords", [])
        
        # 1. Basic structural features
        features["struct_title_length"] = len(title)
        features["struct_desc_length"] = len(desc)
        features["struct_content_length"] = len(content)
        features["struct_num_keywords"] = len(keywords)
        
        # 2. Token features
        content_tokens = tokens(content)
        content_tokens_nostop = strip_stops(content_tokens)
        
        features["struct_content_tokens"] = len(content_tokens)
        features["struct_content_unique_tokens"] = len(set(content_tokens))
        features["struct_content_tokens_nostop"] = len(content_tokens_nostop)
        
        # 3. Embedding features
        if self.use_embeddings and self._emb_extractor:
            emb_features = self._emb_extractor.extract_features(title, desc, content)
            features.update(emb_features)
        
        # 4. Readability features
        if self.use_readability:
            read_features = extract_readability_features(content)
            # Prefix to avoid collisions
            for k, v in read_features.items():
                features[f"read_{k}"] = v
        
        # 5. Information-theoretic features
        if self.use_information_theory and content_tokens_nostop:
            info_features = extract_information_features(content_tokens_nostop)
            for k, v in info_features.items():
                features[f"info_{k}"] = v
        
        return features
    
    def extract_batch(self, kos: List[Dict[str, Any]]) -> List[Dict[str, float]]:
        """Extract features for multiple KOs."""
        return [self.extract(ko) for ko in kos]


def build_feature_matrix(
    kos: List[Dict[str, Any]],
    labels: Optional[np.ndarray] = None,
    extractor: Optional[FeatureExtractor] = None,
) -> pd.DataFrame:
    """
    Build a feature matrix for ML training.
    
    Args:
        kos: List of KOs
        labels: Optional array of human judgments (n_kos, n_dimensions)
        extractor: Feature extractor (creates default if None)
    
    Returns:
        DataFrame with features and optionally labels
    """
    if extractor is None:
        extractor = FeatureExtractor()
    
    # Extract features for all KOs
    print(f"Extracting features for {len(kos)} KOs...")
    features_list = extractor.extract_batch(kos)
    
    # Convert to DataFrame
    df = pd.DataFrame(features_list)
    
    # Add labels if provided
    if labels is not None:
        dimensions = ["structural", "semantic", "domain", "functional", "overall"]
        for i, dim in enumerate(dimensions):
            if i < labels.shape[1]:
                df[f"label_{dim}"] = labels[:, i]
    
    # Add KO IDs for reference
    df["ko_id"] = [ko.get("_orig_id") or ko.get("@id") or f"ko_{i}" 
                   for i, ko in enumerate(kos)]
    
    print(f"Feature matrix shape: {df.shape}")
    print(f"Features: {list(df.columns)}")
    
    return df
