#!/usr/bin/env python3
# dense_retriever.py
"""
Dense retrieval implementation using sentence transformers and FAISS.
"""

from typing import List, Tuple
import faiss
from sentence_transformers import SentenceTransformer
import os
import logging
import numpy as np
from retriever_base import BaseRetriever

logger = logging.getLogger('RAG42')

class DenseRetriever(BaseRetriever):
    """
    Dense retrieval module using BGE/SentenceTransformer and FAISS.
    Supports configurable models: bge-small, bge-base, bge-large, gte, etc.
    """
    def __init__(
        self,
        collection_path: str,
        dense_model_name: str = "BAAI/bge-large-en-v1.5",
        use_cache: bool = True,
        cache_dir: str = os.getenv('RAG42_CACHE_DIR', './cache')
    ):
        """
        Initializes the Dense Retriever.

        Args:
            collection_path: Path or identifier for the document collection.
            dense_model_name: Name of the dense embedding model.
            use_cache: Whether to load a pre-built FAISS index if available.
            cache_dir: Directory to store cached indices.
        """
        self.dense_model_name = dense_model_name
        self.use_cache = use_cache
        # BGE models recommend prepending "Represent this sentence: " for queries
        self._query_prefix = "Represent this sentence: " if "bge" in dense_model_name.lower() else ""
        super().__init__(collection_path, cache_dir)
        self._build_index()


    def _build_index(self):
        """Builds the FAISS (dense) index."""
        cache_path = os.path.join(self.cache_dir, f"faiss_index_{self.dense_model_name.replace('/', '_')}.faiss")
        if self.use_cache and os.path.exists(cache_path):
            logger.info(f"Loading cached FAISS (dense) index from {cache_path}...")
            self.dense_model = SentenceTransformer(self.dense_model_name)
            self.dense_index = faiss.read_index(cache_path)
            logger.info("Cached FAISS dense index loaded.")
            return

        logger.info(f"Building FAISS (dense) index with {self.dense_model_name}...")
        self.dense_model = SentenceTransformer(self.dense_model_name)
        
        # Encode documents (batched for speed)
        logger.info("Encoding documents...")
        doc_embeddings = self.dense_model.encode(self.doc_texts, batch_size=64, show_progress_bar=True)
        
        # Build FAISS index (Inner Product for cosine similarity after normalization)
        dimension = doc_embeddings.shape[1]
        self.dense_index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(doc_embeddings) # Normalize embeddings for cosine similarity
        self.dense_index.add(doc_embeddings)

        # Save the index if caching is enabled
        if self.use_cache:
            logger.info(f"Saving FAISS index to {cache_path}...")
            faiss.write_index(self.dense_index, cache_path)
        
        logger.info("FAISS dense index built.")


    def retrieve(self, query: str, k: int = 20) -> List[Tuple[str, str, float]]:
        """
        Retrieves top-k documents using dense embeddings.

        Args:
            query: The query string.
            k: Number of documents to retrieve.

        Returns:
            List of tuples (doc_id, doc_text, score).
        """
        query_text = self._query_prefix + query
        query_emb = self.dense_model.encode([query_text])
        faiss.normalize_L2(query_emb)
        scores, indices = self.dense_index.search(query_emb, k=k)
        
        results = []
        for i in range(len(indices[0])):
            doc_idx = int(indices[0][i])
            doc_id = self.doc_ids[doc_idx]
            doc_text = self.doc_texts[doc_idx]
            score = float(scores[0][i])
            results.append((doc_id, doc_text, score))
        
        logger.debug(f"Retrieved {len(results)} documents for query: {query}...")
        return results