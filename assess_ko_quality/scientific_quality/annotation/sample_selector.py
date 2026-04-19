# scientific_quality/annotation/sample_selector.py
"""
Stratified sampling for annotation to ensure diverse, representative sample.
"""

import numpy as np
from typing import Dict, List, Any, Tuple


def stratified_sample_for_annotation(
    kos: List[Dict[str, Any]],
    n_samples: int = 300,
    stratify_by: List[str] = None,
    random_seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Select a stratified sample of KOs for annotation.
    
    Ensures coverage across:
    - Content length (short/medium/long)
    - Source institution (if available)
    - Keyword diversity
    - Heuristic quality score (if pre-computed)
    
    Args:
        kos: List of all KOs
        n_samples: Number of KOs to sample
        stratify_by: List of stratification criteria
        random_seed: For reproducibility
    
    Returns:
        Sampled KOs for annotation
    """
    if stratify_by is None:
        stratify_by = ["length", "heuristic_quality"]
    
    np.random.seed(random_seed)
    
    n_total = len(kos)
    if n_total <= n_samples:
        return kos
    
    # Compute stratification features
    features = _compute_stratification_features(kos)
    
    # Create strata
    strata = _assign_strata(features, stratify_by)
    
    # Sample from each stratum proportionally
    samples = []
    unique_strata = np.unique(strata)
    
    for stratum in unique_strata:
        stratum_indices = np.where(strata == stratum)[0]
        stratum_size = len(stratum_indices)
        
        # Proportional allocation
        target_size = int(np.ceil(n_samples * stratum_size / n_total))
        target_size = min(target_size, stratum_size)
        
        sampled = np.random.choice(stratum_indices, size=target_size, replace=False)
        samples.extend([kos[i] for i in sampled])
    
    # If we overshot due to rounding, randomly remove excess
    if len(samples) > n_samples:
        indices_to_keep = np.random.choice(len(samples), size=n_samples, replace=False)
        samples = [samples[i] for i in indices_to_keep]
    
    return samples


def _compute_stratification_features(kos: List[Dict]) -> Dict[str, np.ndarray]:
    """Extract features for stratification."""
    features = {
        "length": np.array([len(ko.get("ko_content_flat", "")) for ko in kos]),
        "num_keywords": np.array([len(ko.get("keywords", [])) for ko in kos]),
    }
    
    # Heuristic quality if available
    if "quality_score" in kos[0]:
        features["heuristic_quality"] = np.array([ko.get("quality_score", 0) for ko in kos])
    
    return features


def _assign_strata(features: Dict, stratify_by: List[str]) -> np.ndarray:
    """Assign each KO to a stratum based on features."""
    n = len(features["length"])
    strata = np.zeros(n, dtype=int)
    
    stratum_id = 0
    
    if "length" in stratify_by:
        # Bin by content length
        lengths = features["length"]
        short = lengths < 500
        medium = (lengths >= 500) & (lengths < 3000)
        long = lengths >= 3000
        
        strata[short] = stratum_id
        strata[medium] = stratum_id + 1
        strata[long] = stratum_id + 2
        stratum_id += 3
    
    if "heuristic_quality" in stratify_by and "heuristic_quality" in features:
        scores = features["heuristic_quality"]
        low = scores < 40
        mid = (scores >= 40) & (scores < 70)
        high = scores >= 70
        
        # Combine with existing strata
        new_strata = np.zeros(n, dtype=int)
        new_strata[low] = 0
        new_strata[mid] = 1
        new_strata[high] = 2
        
        strata = strata * 3 + new_strata
    
    return strata


def compute_coverage_metrics(
    sample: List[Dict],
    full_population: List[Dict]
) -> Dict[str, float]:
    """
    Compute how well the sample covers the population.
    
    Returns metrics like coverage of institutions, length ranges, etc.
    """
    metrics = {}
    
    # Length coverage
    sample_lengths = [len(ko.get("ko_content_flat", "")) for ko in sample]
    pop_lengths = [len(ko.get("ko_content_flat", "")) for ko in full_population]
    
    metrics["length_mean_diff"] = abs(np.mean(sample_lengths) - np.mean(pop_lengths))
    metrics["length_std_diff"] = abs(np.std(sample_lengths) - np.std(pop_lengths))
    
    # Keyword count coverage
    sample_kw = [len(ko.get("keywords", [])) for ko in sample]
    pop_kw = [len(ko.get("keywords", [])) for ko in full_population]
    
    metrics["keywords_mean_diff"] = abs(np.mean(sample_kw) - np.mean(pop_kw))
    
    return metrics
