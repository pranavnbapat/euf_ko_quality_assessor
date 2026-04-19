# scientific_quality/features/embedding_features.py
"""
Embedding-based semantic feature extraction.
"""

import numpy as np
from typing import Dict, List, Any, Optional


class EmbeddingFeatureExtractor:
    """
    Extract semantic features using sentence embeddings.
    """
    
    def __init__(self, model_name: str = "all-mpnet-base-v2", device: str = None):
        """
        Initialize embedding model.
        
        Args:
            model_name: Sentence-transformers model name
            device: 'cuda', 'cpu', or None for auto
        """
        self.model_name = model_name
        self._model = None
        self._device = device
        
    def _load_model(self):
        """Lazy load the embedding model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self._device)
    
    def extract_features(self, title: str, desc: str, content: str) -> Dict[str, float]:
        """
        Extract embedding-based features.
        
        Returns features like:
        - title_content_similarity
        - desc_content_similarity  
        - meta_content_similarity
        - content_self_similarity (paragraph coherence)
        - semantic_density
        """
        self._load_model()
        
        # Cap content for efficiency
        content_capped = content[:10000] if content else ""
        
        # Encode texts
        title_emb = self._model.encode(title, normalize_embeddings=True)
        desc_emb = self._model.encode(desc, normalize_embeddings=True)
        content_emb = self._model.encode(content_capped, normalize_embeddings=True)
        meta_emb = self._model.encode(f"{title} {desc}", normalize_embeddings=True)
        
        features = {}
        
        # 1. Cross-text similarities (semantic alignment)
        features["emb_title_content_sim"] = float(np.dot(title_emb, content_emb))
        features["emb_desc_content_sim"] = float(np.dot(desc_emb, content_emb))
        features["emb_meta_content_sim"] = float(np.dot(meta_emb, content_emb))
        
        # 2. Content self-similarity (paragraph coherence)
        features["emb_content_coherence"] = self._compute_coherence(content_capped)
        
        # 3. Semantic density (std of embedding components)
        features["emb_content_density"] = float(np.std(content_emb))
        features["emb_title_density"] = float(np.std(title_emb))
        
        # 4. Embedding norms (proxy for confidence/informativeness)
        features["emb_content_norm"] = float(np.linalg.norm(content_emb))
        features["emb_title_norm"] = float(np.linalg.norm(title_emb))
        
        return features
    
    def _compute_coherence(self, content: str, max_paragraphs: int = 10) -> float:
        """
        Compute coherence across paragraphs.
        
        High coherence = paragraphs are semantically related.
        Low coherence = disjointed content.
        """
        if not content:
            return 0.0
        
        # Split into paragraphs
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        if len(paragraphs) < 2:
            return 1.0  # Single paragraph, perfect coherence by definition
        
        # Sample paragraphs if too many
        if len(paragraphs) > max_paragraphs:
            indices = np.linspace(0, len(paragraphs) - 1, max_paragraphs, dtype=int)
            paragraphs = [paragraphs[i] for i in indices]
        
        # Encode paragraphs
        para_embs = self._model.encode(paragraphs, normalize_embeddings=True)
        
        # Compute pairwise similarities
        similarities = []
        for i in range(len(para_embs)):
            for j in range(i + 1, len(para_embs)):
                sim = np.dot(para_embs[i], para_embs[j])
                similarities.append(sim)
        
        return float(np.mean(similarities)) if similarities else 0.0


def extract_embedding_features_batch(
    kos: List[Dict[str, Any]],
    model_name: str = "all-mpnet-base-v2",
    batch_size: int = 32
) -> List[Dict[str, float]]:
    """
    Extract embedding features for multiple KOs efficiently.
    """
    extractor = EmbeddingFeatureExtractor(model_name)
    
    results = []
    for ko in kos:
        features = extractor.extract_features(
            title=ko.get("title", ""),
            desc=ko.get("description", ""),
            content=ko.get("ko_content_flat", "")
        )
        results.append(features)
    
    return results
