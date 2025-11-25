#!/usr/bin/env python3
# hybrid_retriever.py
"""
This class wraps your existing BM25 and BGE logic into a clean, reusable interface.
"""

from typing import List, Tuple, Dict
import bm25s
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import nltk
import logging
import json
import os

logger = logging.getLogger('RAG42')

class HybridRetriever:
    """
    A hybrid retrieval module combining BM25 (sparse) and BGE (dense) methods.
    Provides a simple API to retrieve documents given a query.
    """
    def __init__(
        self,
        collection_path: str, # Path to load collection data (or HuggingFace dataset name)
        sparse_model_name: str = "bm25s", # Not used directly, kept for potential future use
        dense_model_name: str = "BAAI/bge-small-en-v1.5",
        use_cache: bool = True, # If you save FAISS index / BM25 index
        cache_dir: str = "./cache" # Directory to save/load indices
    ):
        """
        Initializes the Hybrid Retriever.

        Args:
            collection_path (str): Path or identifier for the document collection.
            sparse_model_name (str): Name of the sparse model (currently only BM25 is implemented).
            dense_model_name (str): Name of the dense embedding model.
            use_cache (bool): Whether to load a pre-built FAISS/BM25 index if available.
            cache_dir (str): Directory to store cached indices.
        """
        self.sparse_model_name = sparse_model_name
        self.dense_model_name = dense_model_name
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True) # Ensure cache directory exists

        # Load collection
        self._load_collection(collection_path)

        # Initialize retrieval components
        self._build_sparse_index()
        self._build_dense_index()


    def _load_collection(self, path: str):
        """Loads the document collection from HuggingFace or a file."""
        from datasets import load_dataset
        logger.info(f"Loading collection from {path}...")
        hotpot_dataset = load_dataset(path)
        collection_dataset = hotpot_dataset["collection"]
        self.doc_texts = [ex["text"] for ex in collection_dataset]
        self.doc_ids = [ex["id"] for ex in collection_dataset]
        self.id_to_text = {ex["id"]: ex["text"] for ex in collection_dataset}
        logger.info(f"Loaded {len(self.doc_texts)} documents.")


    def _build_sparse_index(self):
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


    def _build_dense_index(self):
        """Builds the BGE (dense) index using FAISS."""
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


    def retrieve(self, query: str, k: int = 10, alpha: float = 0.5) -> List[Tuple[str, str, float]]:
        """
        Retrieves top-k documents using a hybrid approach (BM25 + BGE).

        Args:
            query (str): The query string.
            k (int): Number of documents to retrieve.
            alpha (float): Weight for BM25 score in the hybrid fusion (1-alpha for BGE).

        Returns:
            List[Tuple[str, str, float]]: List of tuples (doc_id, doc_text, score).
        """
        # Retrieve from both methods
        bm25_results = self._retrieve_sparse(query, k * 2) # Retrieve more to merge
        bge_results = self._retrieve_dense(query, k * 2)

        # Normalize scores
        bm25_norm = self._normalize_scores([score for _, score in bm25_results])
        bge_norm = self._normalize_scores([score for _, score in bge_results])

        # Create score maps
        # Use doc_ids from self.doc_ids based on the indices returned by retrieval methods
        bm25_map = {self.doc_ids[idx]: score for idx, score in zip([idx for idx, _ in bm25_results], bm25_norm)}
        bge_map = {self.doc_ids[idx]: score for idx, score in zip([idx for idx, _ in bge_results], bge_norm)}

        # Fusion: Weighted sum (can be changed to RRF if needed)
        fused_scores = {}
        all_doc_ids = set(bm25_map.keys()) | set(bge_map.keys())
        for doc_id in all_doc_ids:
            fused_score = alpha * bm25_map.get(doc_id, 0.0) + (1 - alpha) * bge_map.get(doc_id, 0.0)
            fused_scores[doc_id] = fused_score

        # Sort and get top-k
        sorted_doc_ids = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        # Format results: (id, text, fused_score)
        results = [(doc_id, self.id_to_text[doc_id], score) for doc_id, score in sorted_doc_ids]
        
        logger.debug(f"Retrieved {len(results)} documents for query: {query[:30]}...")
        return results


    def _retrieve_sparse(self, query: str, k: int) -> List[Tuple[int, float]]:
        """Internal method for sparse retrieval."""
        # Tokenize query - REMOVED 'stem=True'
        query_tokens = bm25s.tokenize([query], stopwords="en") # Removed stem=True
        scores, indices = self.bm25_retriever.retrieve(query_tokens, k=k)
        return [(int(indices[0][i]), float(scores[0][i])) for i in range(len(indices[0]))]


    def _retrieve_dense(self, query: str, k: int) -> List[Tuple[int, float]]:
        """Internal method for dense retrieval."""
        query_emb = self.dense_model.encode([query])
        faiss.normalize_L2(query_emb)
        scores, indices = self.dense_index.search(query_emb, k=k)
        return [(int(indices[0][i]), float(scores[0][i])) for i in range(len(indices[0]))]


    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalizes a list of scores to [0, 1] using Min-Max scaling."""
        if not scores:
            return []
        min_score, max_score = min(scores), max(scores)
        if max_score == min_score:
            return [1.0] * len(scores)
        return [(s - min_score) / (max_score - min_score) for s in scores]
