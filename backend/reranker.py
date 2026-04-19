#!/usr/bin/env python3
# reranker.py

"""
Cross-encoder re-ranker for improving retrieval quality.
Uses a cross-encoder model to re-score retrieved documents.
"""

from typing import List, Tuple
import os
import logging
import numpy as np
from sentence_transformers import CrossEncoder

logger = logging.getLogger('RAG42')


class CrossEncoderReranker:
    """
    Re-ranks retrieved documents using a cross-encoder model.
    Cross-encoders jointly encode query-document pairs for more accurate
    relevance scoring than bi-encoder approaches.
    """
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        max_length: int = 512
    ):
        """
        Initializes the Cross-Encoder Re-ranker.

        Args:
            model_name: Name of the cross-encoder model.
            max_length: Maximum sequence length for the cross-encoder.
        """
        self.model_name = model_name
        self.max_length = max_length
        logger.info(f"Loading cross-encoder re-ranker: {model_name}")
        self.model = CrossEncoder(model_name, max_length=max_length)
        logger.info("Cross-encoder re-ranker loaded.")

    def rerank(
        self,
        query: str,
        documents: List[Tuple[str, str, float]],
        top_k: int = 10
    ) -> List[Tuple[str, str, float]]:
        """
        Re-ranks documents using the cross-encoder.

        Args:
            query: The query string.
            documents: List of (doc_id, doc_text, original_score) tuples.
            top_k: Number of documents to return after re-ranking.

        Returns:
            List of (doc_id, doc_text, reranker_score) tuples, sorted by score descending.
        """
        if not documents:
            return []

        # Build query-document pairs for the cross-encoder
        pairs = [(query, doc_text) for _, doc_text, _ in documents]

        # Score all pairs
        scores = self.model.predict(pairs, show_progress_bar=False)

        # Build results sorted by cross-encoder score
        results = []
        for i, (doc_id, doc_text, _) in enumerate(documents):
            results.append((doc_id, doc_text, float(scores[i])))

        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]
