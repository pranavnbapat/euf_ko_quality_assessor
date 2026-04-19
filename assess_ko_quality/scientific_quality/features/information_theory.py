# scientific_quality/features/information_theory.py
"""
Information-theoretic features for lexical richness.
"""

import numpy as np
from collections import Counter
from typing import Dict, List
import scipy.stats


def extract_information_features(tokens: List[str]) -> Dict[str, float]:
    """
    Extract information-theoretic features from tokenized text.
    
    These measure lexical diversity and information density.
    
    Args:
        tokens: List of tokens (already lowercased, no stopwords ideally)
    
    Returns:
        Dict with information-theoretic metrics
    """
    if not tokens:
        return {
            "shannon_entropy": 0.0,
            "type_token_ratio": 0.0,
            "hapax_legomena_ratio": 0.0,
            "dis_legomena_ratio": 0.0,
            "yules_k": 0.0,
            "simpsons_d": 0.0,
        }
    
    # Token frequency distribution
    token_counts = Counter(tokens)
    total_tokens = len(tokens)
    unique_tokens = len(token_counts)
    
    features = {}
    
    # 1. Shannon entropy (higher = more unpredictable = more diverse)
    probabilities = np.array(list(token_counts.values())) / total_tokens
    features["shannon_entropy"] = scipy.stats.entropy(probabilities)
    
    # 2. Type-Token Ratio (TTR)
    features["type_token_ratio"] = unique_tokens / total_tokens
    
    # 3. Hapax legomena (words appearing exactly once)
    hapax_count = sum(1 for count in token_counts.values() if count == 1)
    features["hapax_legomena_ratio"] = hapax_count / unique_tokens
    
    # 4. Dis legomena (words appearing exactly twice)
    dis_count = sum(1 for count in token_counts.values() if count == 2)
    features["dis_legomena_ratio"] = dis_count / unique_tokens
    
    # 5. Yule's K (vocabulary richness, lower = more diverse)
    features["yules_k"] = _compute_yules_k(token_counts, total_tokens)
    
    # 6. Simpson's D (probability two random tokens are same, lower = more diverse)
    features["simpsons_d"] = _compute_simpsons_d(token_counts, total_tokens)
    
    # 7. Honore's R (vocabulary richness accounting for text length)
    if hapax_count > 0:
        features["honores_r"] = 100 * np.log(total_tokens) / (1 - hapax_count / unique_tokens)
    else:
        features["honores_r"] = 0.0
    
    # 8. Guiraud's R (root TTR)
    features["guirauds_r"] = unique_tokens / np.sqrt(total_tokens)
    
    # 9. Herdan's C (log TTR)
    if total_tokens > 0:
        features["herdans_c"] = np.log(unique_tokens) / np.log(total_tokens)
    else:
        features["herdans_c"] = 0.0
    
    return features


def _compute_yules_k(token_counts: Counter, total_tokens: int) -> float:
    """
    Yule's K characteristic.
    Higher K = less diverse vocabulary.
    """
    if total_tokens == 0:
        return 0.0
    
    # Sum of i^2 * V(i) for i = 1, 2, ...
    # where V(i) is number of words occurring i times
    sum_i2_vi = sum(count ** 2 * sum(1 for c in token_counts.values() if c == count)
                    for count in set(token_counts.values()))
    
    k = 10**4 * (sum_i2_vi - total_tokens) / (total_tokens ** 2)
    return k


def _compute_simpsons_d(token_counts: Counter, total_tokens: int) -> float:
    """
    Simpson's Diversity Index.
    Probability that two randomly chosen tokens are the same.
    Lower D = more diverse.
    """
    if total_tokens <= 1:
        return 0.0
    
    d = sum(count * (count - 1) for count in token_counts.values())
    d = d / (total_tokens * (total_tokens - 1))
    return d
