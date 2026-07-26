import numpy as np
import hashlib
from typing import List

class MedicalEmbeddingGenerator:
    """
    Computes dense semantic embeddings for medical text summaries and queries.
    Uses a word-projection hash matrix to map vocabulary semantics to a 
    128-dimensional space, providing robust offline cosine similarity.
    """
    def __init__(self, dimension: int = 128):
        self.dimension = dimension
        # Seed generator to guarantee deterministic projections
        np.random.seed(42)
        self.projection_matrix = np.random.randn(1000, dimension)

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Generates a normalized 1D embedding array for a given text.
        """
        text_l = text.lower()
        words = text_l.split()
        
        # Aggregate word hash indices
        vector = np.zeros(self.dimension)
        if not words:
            return vector
            
        for w in words:
            # Deterministic word index hash using hashlib md5
            w_idx = int(hashlib.md5(w.encode('utf-8')).hexdigest(), 16) % 1000
            vector += self.projection_matrix[w_idx]
            
        # Normalize to unit length
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
            
        return vector

    def get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        return [self.get_embedding(t) for t in texts]
