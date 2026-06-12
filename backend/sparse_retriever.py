#!/usr/bin/env python3
# sparse_retriever.py

"""
Sparse retrieval implementation using BM25.
"""

from typing import List, Tuple
import bm25s
import os
import logging
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
        """Builds the BM25 index."""
        cache_path = os.path.join(self.cache_dir, f"bm25_index_{self.sparse_model_name.replace('/', '_')}.npz")
        if self.use_cache and os.path.exists(cache_path):
            logger.info(f"Loading cached BM25 index from {cache_path}...")
            # Load the index using bm25s
            self.bm25_retriever = bm25s.BM25.load(cache_path, load_corpus=False) # Don't load corpus again, we have self.doc_texts
            logger.info("Cached BM25 index loaded.")
            return

        logger.info("Building BM25 (sparse) index...")
        # Tokenize - REMOVED 'stem=True' as it's not a valid argument for bm25s.tokenize
        corpus_tokens = bm25s.tokenize(self.doc_texts, stopwords="en") # Removed stem=True
        # Build index
        self.bm25_retriever = bm25s.BM25()
        self.bm25_retriever.index(corpus_tokens)
        
        # Save the index if caching is enabled
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
        # Tokenize query - REMOVED 'stem=True'
        query_tokens = bm25s.tokenize([query], stopwords="en") # Removed stem=True
        scores, indices = self.bm25_retriever.retrieve(query_tokens, k=k)
        
        results = []
        for i in range(len(indices[0])):
            doc_idx = int(indices[0][i])
            doc_id = self.doc_ids[doc_idx]
            doc_text = self.doc_texts[doc_idx]
            score = float(scores[0][i])
            results.append((doc_id, doc_text, score))
        
        logger.debug(f"Retrieved {len(results)} documents for query: {query}...")
        return results