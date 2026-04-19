# scientific_quality/features/readability_features.py
"""
Readability metric extraction using textstat.
"""

import textstat
from typing import Dict


def extract_readability_features(text: str) -> Dict[str, float]:
    """
    Extract validated readability metrics.
    
    These are scientifically validated predictors of text difficulty.
    
    Returns:
        Dict with readability scores
    """
    if not text or len(text) < 100:
        # Return defaults for very short text
        return {
            "flesch_reading_ease": 0.0,
            "flesch_kincaid_grade": 0.0,
            "smog_index": 0.0,
            "coleman_liau_index": 0.0,
            "automated_readability_index": 0.0,
            "dale_chall_readability_score": 0.0,
            "difficult_words": 0,
            "sentence_count": 0,
            "word_count": 0,
        }
    
    features = {}
    
    # Core readability metrics
    features["flesch_reading_ease"] = textstat.flesch_reading_ease(text)
    features["flesch_kincaid_grade"] = textstat.flesch_kincaid_grade(text)
    features["smog_index"] = textstat.smog_index(text)
    features["coleman_liau_index"] = textstat.coleman_liau_index(text)
    features["automated_readability_index"] = textstat.automated_readability_index(text)
    features["dale_chall_readability_score"] = textstat.dale_chall_readability_score(text)
    
    # Additional metrics
    features["difficult_words"] = textstat.difficult_words(text)
    features["sentence_count"] = textstat.sentence_count(text)
    features["word_count"] = textstat.lexicon_count(text)
    
    # Syllable metrics
    features["syllable_count"] = textstat.syllable_count(text)
    features["avg_syllables_per_word"] = (
        features["syllable_count"] / max(1, features["word_count"])
    )
    
    # Sentence metrics
    features["avg_sentence_length"] = textstat.avg_sentence_length(text)
    
    # Reading time (in seconds, assuming 200 WPM)
    features["reading_time_seconds"] = features["word_count"] * 60 / 200
    
    # Grade level consensus (average of grade-based metrics)
    grade_metrics = [
        features["flesch_kincaid_grade"],
        features["smog_index"],
        features["coleman_liau_index"],
        features["automated_readability_index"],
    ]
    features["consensus_grade_level"] = sum(grade_metrics) / len(grade_metrics)
    
    return features
