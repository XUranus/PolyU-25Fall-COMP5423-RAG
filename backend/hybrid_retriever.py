#!/usr/bin/env python3
# hybrid_retriever.py
"""
Hybrid retrieval implementation combining sparse and dense methods.
"""

from typing import List, Tuple
import os
import logging
from retriever_base import BaseRetriever
from sparse_retriever import SparseRetriever
from dense_retriever import DenseRetriever

logger = logging.getLogger('RAG42')

class HybridRetriever(BaseRetriever):
    """
    Hybrid retrieval module combining sparse (BM25) and dense (BGE) methods.
    """
    def __init__(
        self,
        collection_path: str,
        sparse_model_name: str = "bm25s",
        dense_model_name: str = "BAAI/bge-small-en-v1.5",
        use_cache: bool = True,
        cache_dir: str = os.getenv('RAG42_CACHE_DIR', './cache'),
        alpha: float = 0.1  # Weight for sparse score in the hybrid fusion (1-alpha for dense)
    ):
        """
        Initializes the Hybrid Retriever.

        Args:
            collection_path: Path or identifier for the document collection.
            sparse_model_name: Name of the sparse model (currently only BM25 is implemented).
            dense_model_name: Name of the dense embedding model.
            use_cache: Whether to load pre-built indices if available.
            cache_dir: Directory to store cached indices.
            alpha: Weight for sparse score in the hybrid fusion (1-alpha for dense).
        """
        self.sparse_model_name = sparse_model_name
        self.dense_model_name = dense_model_name
        self.use_cache = use_cache
        self.alpha = alpha
        super().__init__(collection_path, cache_dir)
        
        # Initialize component retrievers
        self.sparse_retriever = SparseRetriever(
            collection_path=collection_path,
            sparse_model_name=sparse_model_name,
            use_cache=use_cache,
            cache_dir=cache_dir
        )
        self.dense_retriever = DenseRetriever(
            collection_path=collection_path,
            dense_model_name=dense_model_name,
            use_cache=use_cache,
            cache_dir=cache_dir
        )

    def retrieve(self, query: str, k: int = 20) -> List[Tuple[str, str, float]]:
        """
        Retrieves top-k documents using a hybrid approach (BM25 + BGE).

        Args:
            query: The query string.
            k: Number of documents to retrieve.

        Returns:
            List of tuples (doc_id, doc_text, score).
        """
        # Retrieve from both methods
        bm25_results = self.sparse_retriever.retrieve(query, k * 2)  # Retrieve more to merge
        bge_results = self.dense_retriever.retrieve(query, k * 2)

        # Extract scores and create score maps
        bm25_norm = self._normalize_scores([score for _, _, score in bm25_results])
        bge_norm = self._normalize_scores([score for _, _, score in bge_results])

        # Create score maps using doc_ids
        bm25_map = {doc_id: score for doc_id, _, score in zip([doc_id for doc_id, _, _ in bm25_results], bm25_norm)}
        bge_map = {doc_id: score for doc_id, _, score in zip([doc_id for doc_id, _, _ in bge_results], bge_norm)}

        # Fusion: Weighted sum (alpha for sparse, 1-alpha for dense)
        fused_scores = {}
        all_doc_ids = set(bm25_map.keys()) | set(bge_map.keys())
        for doc_id in all_doc_ids:
            fused_score = self.alpha * bm25_map.get(doc_id, 0.0) + (1 - self.alpha) * bge_map.get(doc_id, 0.0)
            fused_scores[doc_id] = fused_score

        # Sort and get top-k
        sorted_doc_ids = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        # Format results: (id, text, fused_score)
        results = [(doc_id, self.id_to_text[doc_id], score) for doc_id, score in sorted_doc_ids]
        
        logger.debug(f"Retrieved {len(results)} documents for query: {query}...")
        return results

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalizes a list of scores to [0, 1] using Min-Max scaling."""
        if not scores:
            return []
        min_score, max_score = min(scores), max(scores)
        if max_score == min_score:
            return [1.0] * len(scores)
        return [(s - min_score) / (max_score - min_score) for s in scores]