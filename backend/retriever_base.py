#!/usr/bin/env python3
# retriever_base.py

"""
Base class and factory for retrieval methods with concrete implementations for
sparse, dense, and hybrid retrieval.
"""

from typing import List, Tuple, Dict, Optional
import logging
import os
from abc import ABC, abstractmethod

logger = logging.getLogger('RAG42')

class BaseRetriever(ABC):
    """
    Abstract base class for all retrievers.
    Provides common functionality and factory method.
    """
    def __init__(
        self,
        collection_path: str,
        cache_dir: str = os.getenv('RAG42_CACHE_DIR', './cache')
    ):
        self.collection_path = collection_path
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load collection
        self._load_collection(collection_path)

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


    @abstractmethod
    def retrieve(self, query: str, k: int = 20) -> List[Tuple[str, str, float]]:
        """
        Abstract method to retrieve documents.
        
        Args:
            query: The query string
            k: Number of documents to retrieve
            
        Returns:
            List of tuples (doc_id, doc_text, score)
        """
        pass


    @classmethod
    def create_retriever(cls, retriever_type: str, **kwargs):
        """
        Factory method to create retriever instances.
        
        Args:
            retriever_type: One of 'sparse', 'static_embedding', 'dense', 'hybrid'
            **kwargs: Arguments for the specific retriever
            
        Returns:
            An instance of the specified retriever
        """
        if retriever_type.lower() == 'sparse':
            from sparse_retriever import SparseRetriever
            return SparseRetriever(**kwargs)
        elif retriever_type.lower() == 'static_embedding':
            from static_embedding_retriever import StaticEmbeddingRetriever
            return StaticEmbeddingRetriever(**kwargs)
        elif retriever_type.lower() == 'dense':
            from dense_retriever import DenseRetriever
            return DenseRetriever(**kwargs)
        elif retriever_type.lower() == 'hybrid':
            from hybrid_retriever import HybridRetriever
            return HybridRetriever(**kwargs)
        else:
            raise ValueError(f"Unknown retriever type: {retriever_type}")