import numpy as np
from typing import List, Dict, Any, Tuple

from src.nlp.embeddings import MedicalEmbeddingGenerator

class LocalVectorDB:
    """
    A lightweight, in-memory Vector Database supporting cosine similarity search.
    """
    def __init__(self, embedder: MedicalEmbeddingGenerator):
        self.embedder = embedder
        self.documents: List[Dict[str, Any]] = []
        self.embeddings: List[np.ndarray] = []

    def add_document(self, text: str, metadata: Dict[str, Any] = {}):
        """
        Tokenizes, encodes, and indexes a text document.
        """
        embedding = self.embedder.get_embedding(text)
        self.documents.append({
            "text": text,
            "metadata": metadata
        })
        self.embeddings.append(embedding)

    def add_documents(self, documents: List[Tuple[str, Dict[str, Any]]]):
        for text, meta in documents:
            self.add_document(text, meta)

    def search(self, query: str, k: int = 2) -> List[Dict[str, Any]]:
        """
        Searches the index using cosine similarity and returns top k documents.
        """
        if not self.documents:
            return []
            
        q_emb = self.embedder.get_embedding(query)
        
        scores = []
        for idx, doc_emb in enumerate(self.embeddings):
            # Calculate cosine similarity (embeddings are already unit normalized)
            sim = float(np.dot(q_emb, doc_emb))
            scores.append((sim, self.documents[idx]))
            
        # Sort by similarity descending
        scores.sort(key=lambda x: x[0], reverse=True)
        
        # Format results with scores
        results = []
        for score, doc in scores[:k]:
            results.append({
                "text": doc["text"],
                "metadata": doc["metadata"],
                "similarity_score": score
            })
            
        return results
