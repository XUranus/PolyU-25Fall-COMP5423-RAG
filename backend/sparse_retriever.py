#!/usr/bin/env python3
# sparse_retriever.py

"""
Sparse retrieval implementation using BM25.
"""

from typing import List, Tuple
import bm25s
import os
import logging
import re
from retriever_base import BaseRetriever

logger = logging.getLogger('RAG42')


class SparseRetriever(BaseRetriever):
    """
    Sparse retrieval module using BM25.
    """
    def __init__(
        self,
        collection_path: str,
        sparse_model_name: str = "bm25s",
        use_cache: bool = True,
        cache_dir: str = os.getenv('RAG42_CACHE_DIR', './cache'),
        skip_load: bool = False
    ):
        """
        Initializes the Sparse Retriever.

        Args:
            collection_path: Path or identifier for the document collection.
            sparse_model_name: Name of the sparse model (currently only BM25 is implemented).
            use_cache: Whether to load a pre-built BM25 index if available.
            cache_dir: Directory to store cached indices.
            skip_load: If True, skip loading the collection (caller provides data).
        """
        self.sparse_model_name = sparse_model_name
        self.use_cache = use_cache
        super().__init__(collection_path, cache_dir, skip_load=skip_load)
        if not skip_load:
            self._build_index()


    def _build_index(self):          
        cache_path = os.path.join(self.cache_dir, f"bm25_index_{self.sparse_model_name.replace('/', '_')}.npz")
        if self.use_cache and os.path.exists(cache_path):
            logger.info(f"Loading cached BM25 index from {cache_path}...")
            self.bm25_retriever = bm25s.BM25.load(cache_path, load_corpus=False)
            logger.info("Cached BM25 index loaded.")
            return
        
        # No cache, Start to build index
        processed_docs = []
        for doc in self.doc_texts:
            # Basic cleaning that matches query preprocessing
            doc = re.sub(r'[^\w\s]', ' ', doc.lower())
            processed_docs.append(doc)

        logger.info("Building BM25 (sparse) index...")
        corpus_tokens = bm25s.tokenize(processed_docs, stopwords="en")
        
        # Tune BM25 hyperparameters - try different values
        self.bm25_retriever = bm25s.BM25(k1=1.5, b=0.75)  # Default is k1=1.5, b=0.75
        # Common good ranges: k1=1.2-2.0, b=0.5-0.8
        self.bm25_retriever.index(corpus_tokens)
        
        if self.use_cache:
            logger.info(f"Saving BM25 index to {cache_path}...")
            self.bm25_retriever.save(cache_path)
        logger.info("BM25 index built.")


    def retrieve(self, query: str, k: int = 20) -> List[Tuple[str, str, float]]:
        """
        Retrieves top-k documents using BM25.

        Args:
            query: The query string.
            k: Number of documents to retrieve.

        Returns:
            List of tuples (doc_id, doc_text, score).
        """
        # Preprocess query consistently with documents
        query = re.sub(r'[^\w\s]', ' ', query.lower())
        # Tokenize query - REMOVED 'stem=True'
        query_tokens = bm25s.tokenize([query], stopwords="en")
        scores, indices = self.bm25_retriever.retrieve(query_tokens, k=k)
        
        # Use raw scores directly - don't normalize per query
        raw_scores = [float(scores[0][i]) for i in range(len(indices[0]))]
        results = []
        for i in range(len(indices[0])):
            doc_idx = int(indices[0][i])
            doc_id = self.doc_ids[doc_idx]
            doc_text = self.doc_texts[doc_idx]
            results.append((doc_id, doc_text, raw_scores[i]))  # Keep raw scores
        return results
    
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Apply sigmoid normalization to preserve relative ordering"""
        scores_array = np.array(scores)
        # Sigmoid transformation preserves relative ordering better
        normalized = 1 / (1 + np.exp(-scores_array))
        return normalized.tolist()