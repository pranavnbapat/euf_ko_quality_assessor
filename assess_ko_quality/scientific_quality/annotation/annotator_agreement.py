# scientific_quality/annotation/annotator_agreement.py
"""
Inter-annotator agreement metrics and adjudication.
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict


def cohens_kappa(annotations1: List[float], annotations2: List[float]) -> float:
    """
    Compute Cohen's Kappa for two annotators.
    
    Args:
        annotations1: Scores from annotator 1
        annotations2: Scores from annotator 2
    
    Returns:
        Cohen's kappa statistic (-1 to 1, where 1 is perfect agreement)
    """
    if len(annotations1) != len(annotations2):
        raise ValueError("Annotation lists must have same length")
    
    n = len(annotations1)
    if n == 0:
        return 0.0
    
    # Convert to integers for categorical analysis
    a1 = np.array(annotations1, dtype=int)
    a2 = np.array(annotations2, dtype=int)
    
    # Observed agreement
    agreement = np.sum(a1 == a2) / n
    
    # Expected agreement by chance
    categories = np.unique(np.concatenate([a1, a2]))
    expected = 0.0
    
    for cat in categories:
        p1 = np.sum(a1 == cat) / n
        p2 = np.sum(a2 == cat) / n
        expected += p1 * p2
    
    # Cohen's kappa
    if expected == 1.0:  # Perfect agreement by chance definition
        return 1.0 if agreement == 1.0 else 0.0
    
    kappa = (agreement - expected) / (1 - expected)
    return kappa


def fleiss_kappa(annotations: List[List[float]]) -> float:
    """
    Compute Fleiss' Kappa for multiple annotators.
    
    Args:
        annotations: List of annotation lists, one per annotator
    
    Returns:
        Fleiss' kappa statistic
    """
    try:
        from statsmodels.stats.inter_rater import fleiss_kappa as fk
        
        # Convert to agreement table format
        n_items = len(annotations[0])
        n_annotators = len(annotations)
        
        # Build category count matrix
        categories = sorted(set([a for ann in annotations for a in ann]))
        n_categories = len(categories)
        
        table = np.zeros((n_items, n_categories))
        
        for i in range(n_items):
            for j, ann in enumerate(annotations):
                cat_idx = categories.index(ann[i])
                table[i, cat_idx] += 1
        
        return fk(table)
    except ImportError:
        print("statsmodels not installed, using approximate Fleiss' kappa")
        # Fallback: average pairwise Cohen's kappa
        kappas = []
        for i in range(len(annotations)):
            for j in range(i + 1, len(annotations)):
                kappas.append(cohens_kappa(annotations[i], annotations[j]))
        return np.mean(kappas)


def compute_inter_annotator_agreement(
    annotations: Dict[str, List[float]],
    method: str = "cohen"
) -> Dict[str, float]:
    """
    Compute inter-annotator agreement for multiple annotators.
    
    Args:
        annotations: Dict mapping annotator_id -> list of scores
        method: "cohen" (pairwise) or "fleiss" (multi-rater)
    
    Returns:
        Dict with agreement metrics
    """
    annotator_ids = list(annotations.keys())
    n_annotators = len(annotator_ids)
    
    if n_annotators < 2:
        return {"error": "Need at least 2 annotators"}
    
    results = {
        "num_annotators": n_annotators,
        "num_items": len(annotations[annotator_ids[0]]),
    }
    
    if method == "cohen" or n_annotators == 2:
        # Compute pairwise Cohen's kappa for all pairs
        kappas = []
        for i in range(n_annotators):
            for j in range(i + 1, n_annotators):
                id1, id2 = annotator_ids[i], annotator_ids[j]
                kappa = cohens_kappa(annotations[id1], annotations[id2])
                kappas.append(kappa)
                results[f"kappa_{id1}_vs_{id2}"] = round(kappa, 3)
        
        results["mean_kappa"] = round(np.mean(kappas), 3)
        results["min_kappa"] = round(np.min(kappas), 3)
        
        # Interpretation
        mean_kappa = results["mean_kappa"]
        if mean_kappa >= 0.8:
            results["agreement_level"] = "Almost perfect"
        elif mean_kappa >= 0.6:
            results["agreement_level"] = "Substantial"
        elif mean_kappa >= 0.4:
            results["agreement_level"] = "Moderate"
        elif mean_kappa >= 0.2:
            results["agreement_level"] = "Fair"
        else:
            results["agreement_level"] = "Slight/Poor"
    
    elif method == "fleiss":
        # Fleiss' kappa for multiple annotators
        ann_list = [annotations[aid] for aid in annotator_ids]
        kappa = fleiss_kappa(ann_list)
        results["fleiss_kappa"] = round(kappa, 3)
        results["mean_kappa"] = round(kappa, 3)
    
    return results


def adjudicate_disagreements(
    annotations: Dict[str, List[float]],
    threshold: float = 1.5
) -> Tuple[List[float], Dict[int, str]]:
    """
    Create consensus scores by adjudicating disagreements.
    
    Args:
        annotations: Dict mapping annotator_id -> list of scores
        threshold: If range of scores > threshold, flag for review
    
    Returns:
        Tuple of (consensus_scores, flags)
        flags is dict mapping item_index -> flag_reason
    """
    annotator_ids = list(annotations.keys())
    n_items = len(annotations[annotator_ids[0]])
    
    consensus_scores = []
    flags = {}
    
    for i in range(n_items):
        scores = [annotations[aid][i] for aid in annotator_ids]
        mean_score = np.mean(scores)
        score_range = np.max(scores) - np.min(scores)
        std_score = np.std(scores)
        
        # Round to nearest integer for final score
        consensus = int(round(mean_score))
        consensus_scores.append(consensus)
        
        # Flag if high disagreement
        if score_range > threshold:
            flags[i] = f"HIGH_DISAGREEMENT: range={score_range:.1f}, scores={scores}"
        elif std_score > 0.8:
            flags[i] = f"HIGH_VARIANCE: std={std_score:.2f}"
    
    return consensus_scores, flags


def identify_problematic_annotators(
    annotations: Dict[str, List[float]],
    min_kappa: float = 0.5
) -> List[str]:
    """
    Identify annotators who consistently disagree with others.
    
    Returns:
        List of annotator IDs with low agreement
    """
    annotator_ids = list(annotations.keys())
    problematic = []
    
    for aid in annotator_ids:
        # Compute average kappa with all other annotators
        kappas = []
        for other_id in annotator_ids:
            if other_id != aid:
                kappa = cohens_kappa(annotations[aid], annotations[other_id])
                kappas.append(kappa)
        
        mean_kappa = np.mean(kappas)
        if mean_kappa < min_kappa:
            problematic.append(f"{aid} (avg kappa={mean_kappa:.2f})")
    
    return problematic
