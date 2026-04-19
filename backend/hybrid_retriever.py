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
from reranker import CrossEncoderReranker

logger = logging.getLogger('RAG42')

class HybridRetriever(BaseRetriever):
    """
    Hybrid retrieval module combining sparse (BM25) and dense (BGE) methods.
    Uses Reciprocal Rank Fusion (RRF) for more robust score combination.
    """
    def __init__(
        self,
        collection_path: str,
        sparse_model_name: str = "bm25s",
        dense_model_name: str = "BAAI/bge-large-en-v1.5",
        use_cache: bool = True,
        cache_dir: str = os.getenv('RAG42_CACHE_DIR', './cache'),
        rrf_k: int = 60,  # RRF constant (standard value)
        use_reranker: bool = True,
        reranker_model: str = "BAAI/bge-reranker-v2-m3"
    ):
        """
        Initializes the Hybrid Retriever.

        Args:
            collection_path: Path or identifier for the document collection.
            sparse_model_name: Name of the sparse model (currently only BM25 is implemented).
            dense_model_name: Name of the dense embedding model.
            use_cache: Whether to load pre-built indices if available.
            cache_dir: Directory to store cached indices.
            rrf_k: RRF constant for reciprocal rank fusion (default 60).
            use_reranker: Whether to apply cross-encoder re-ranking after RRF fusion.
            reranker_model: Name of the cross-encoder model for re-ranking.
        """
        self.sparse_model_name = sparse_model_name
        self.dense_model_name = dense_model_name
        self.use_cache = use_cache
        self.rrf_k = rrf_k
        self.use_reranker = use_reranker
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

        # Initialize re-ranker (lazy, only if enabled)
        self.reranker = None
        self.reranker_model_name = reranker_model
        if self.use_reranker:
            self.reranker = CrossEncoderReranker(model_name=reranker_model)

    def retrieve(self, query: str, k: int = 20) -> List[Tuple[str, str, float]]:
        """
        Retrieves top-k documents using a hybrid approach (BM25 + BGE) with RRF.

        Args:
            query: The query string.
            k: Number of documents to retrieve.

        Returns:
            List of tuples (doc_id, doc_text, score).
        """
        # Retrieve from both methods
        bm25_results = self.sparse_retriever.retrieve(query, k * 3)  # Retrieve more candidates for RRF
        bge_results = self.dense_retriever.retrieve(query, k * 3)

        # Reciprocal Rank Fusion (RRF): score = sum(1 / (rrf_k + rank))
        fused_scores = {}

        for rank, (doc_id, _, _) in enumerate(bm25_results, start=1):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank)

        for rank, (doc_id, _, _) in enumerate(bge_results, start=1):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank)

        # Sort by fused RRF score and get candidates for re-ranking
        rrf_candidates = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

        if self.use_reranker and self.reranker:
            # Re-rank top candidates with cross-encoder
            candidate_docs = [(doc_id, self.id_to_text[doc_id], score) for doc_id, score in rrf_candidates[:k * 3]]
            results = self.reranker.rerank(query, candidate_docs, top_k=k)
        else:
            # Use RRF scores directly
            results = [(doc_id, self.id_to_text[doc_id], score) for doc_id, score in rrf_candidates[:k]]

        logger.debug(f"Retrieved {len(results)} documents for query: {query}...")
        return results
