#!/usr/bin/env python3
# colbert_retriever.py

"""
Multi-vector retrieval implementation using ColBERT-style late interaction.
Uses the RAGKit-compatible approach with sentence-transformers ColBERT models.
"""

from typing import List, Tuple
import os
import logging
import numpy as np
from retriever_base import BaseRetriever

logger = logging.getLogger('RAG42')


class ColBERTRetriever(BaseRetriever):
    """
    Multi-vector retrieval using ColBERT-style late interaction models.
    Each document and query is represented as a set of token-level embeddings,
    and similarity is computed via MaxSim (sum of max similarities per query token).
    """
    def __init__(
        self,
        collection_path: str,
        model_name: str = "colbert-ir/colbertv2.0",
        use_cache: bool = True,
        cache_dir: str = os.getenv('RAG42_CACHE_DIR', './cache'),
        max_doc_length: int = 180,
        max_query_length: int = 32
    ):
        """
        Initializes the ColBERT Retriever.

        Args:
            collection_path: Path or identifier for the document collection.
            model_name: Name of the ColBERT model.
            use_cache: Whether to load pre-built index if available.
            cache_dir: Directory to store cached indices.
            max_doc_length: Maximum document token length for encoding.
            max_query_length: Maximum query token length for encoding.
        """
        self.model_name = model_name
        self.use_cache = use_cache
        self.max_doc_length = max_doc_length
        self.max_query_length = max_query_length
        super().__init__(collection_path, cache_dir)
        self._build_index()

    def _build_index(self):
        """Builds the ColBERT index by encoding all documents."""
        cache_path = os.path.join(self.cache_dir, f"colbert_index_{self.model_name.replace('/', '_')}.npy")
        cache_path_ids = os.path.join(self.cache_dir, f"colbert_index_{self.model_name.replace('/', '_')}_lengths.npy")

        if self.use_cache and os.path.exists(cache_path):
            logger.info(f"Loading cached ColBERT index from {cache_path}...")
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self.doc_embeddings = np.load(cache_path, allow_pickle=True)
            self.doc_lengths = np.load(cache_path_ids, allow_pickle=True)
            logger.info("Cached ColBERT index loaded.")
            return

        logger.info(f"Building ColBERT index with {self.model_name}...")
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(self.model_name)

        # Encode documents with token-level embeddings
        logger.info("Encoding documents with ColBERT token-level embeddings...")
        all_doc_embs = []
        self.doc_lengths = []

        batch_size = 32
        for i in range(0, len(self.doc_texts), batch_size):
            batch = self.doc_texts[i:i + batch_size]
            # Encode returns token-level embeddings for ColBERT models
            batch_embs = self.model.encode(
                batch,
                batch_size=batch_size,
                show_progress_bar=True
            )
            # If model returns 2D per doc, handle accordingly
            for emb in batch_embs:
                if isinstance(emb, np.ndarray):
                    if emb.ndim == 1:
                        # Reshape single-token output to 2D
                        emb = emb.reshape(1, -1)
                    all_doc_embs.append(emb)
                    self.doc_lengths.append(emb.shape[0])
                else:
                    all_doc_embs.append(np.array(emb))
                    self.doc_lengths.append(len(emb))

        self.doc_embeddings = all_doc_embs
        self.doc_lengths = np.array(self.doc_lengths)

        if self.use_cache:
            logger.info(f"Saving ColBERT index to {cache_path}...")
            np.save(cache_path, self.doc_embeddings, allow_pickle=True)
            np.save(cache_path_ids, self.doc_lengths, allow_pickle=True)

        logger.info("ColBERT index built.")

    def _maxsim_score(self, query_embs: np.ndarray, doc_embs: np.ndarray) -> float:
        """
        Compute ColBERT MaxSim score between query and document embeddings.
        Score = sum over query tokens of max similarity to any doc token.
        """
        # query_embs: (q_len, dim), doc_embs: (d_len, dim)
        # Compute pairwise cosine similarity
        query_norm = query_embs / (np.linalg.norm(query_embs, axis=1, keepdims=True) + 1e-10)
        doc_norm = doc_embs / (np.linalg.norm(doc_embs, axis=1, keepdims=True) + 1e-10)
        sim_matrix = np.dot(query_norm, doc_norm.T)  # (q_len, d_len)
        # MaxSim: for each query token, take max similarity to any doc token
        max_sims = np.max(sim_matrix, axis=1)  # (q_len,)
        return float(np.sum(max_sims))

    def retrieve(self, query: str, k: int = 20) -> List[Tuple[str, str, float]]:
        """
        Retrieves top-k documents using ColBERT MaxSim scoring.

        Args:
            query: The query string.
            k: Number of documents to retrieve.

        Returns:
            List of tuples (doc_id, doc_text, score).
        """
        # Encode query
        query_emb = self.model.encode([query])[0]
        if isinstance(query_emb, np.ndarray) and query_emb.ndim == 1:
            query_emb = query_emb.reshape(1, -1)

        # Score all documents
        scores = []
        for i, doc_emb in enumerate(self.doc_embeddings):
            score = self._maxsim_score(query_emb, doc_emb)
            scores.append((self.doc_ids[i], self.doc_texts[i], score))

        # Sort by score descending
        scores.sort(key=lambda x: x[2], reverse=True)
        return scores[:k]
