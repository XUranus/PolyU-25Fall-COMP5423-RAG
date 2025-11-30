#!/usr/bin/env python3
# static_embedding_retriever.py
"""
Static embedding retrieval implementation using Word2Vec.
"""

from typing import List, Tuple
import os
import logging
import numpy as np
from retriever_base import BaseRetriever
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

logger = logging.getLogger('RAG42')

class StaticEmbeddingRetriever(BaseRetriever):
    """
    Static embedding retrieval module using Word2Vec.
    """
    def __init__(
        self,
        collection_path: str,
        embedding_dim: int = 100,
        window: int = 5,
        min_count: int = 1,
        workers: int = 4,
        use_cache: bool = True,
        cache_dir: str = os.getenv('RAG42_CACHE_DIR', './cache')
    ):
        """
        Initializes the Static Embedding Retriever.

        Args:
            collection_path: Path or identifier for the document collection.
            embedding_dim: Dimension of Word2Vec embeddings.
            window: Context window size for Word2Vec.
            min_count: Minimum word frequency for Word2Vec.
            workers: Number of threads for Word2Vec training.
            use_cache: Whether to load a pre-trained model if available.
            cache_dir: Directory to store cached models.
        """
        self.embedding_dim = embedding_dim
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.use_cache = use_cache
        super().__init__(collection_path, cache_dir)
        self._build_index()

    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenizes text using gensim's simple_preprocess."""
        return simple_preprocess(text, deacc=True)

    def _build_index(self):
        """Builds or loads the Word2Vec model."""
        cache_path = os.path.join(self.cache_dir, f"word2vec_model_{self.embedding_dim}d.model")
        if self.use_cache and os.path.exists(cache_path):
            logger.info(f"Loading cached Word2Vec model from {cache_path}...")
            self.word2vec_model = Word2Vec.load(cache_path)
            logger.info("Cached Word2Vec model loaded.")
        else:
            logger.info("Training Word2Vec model...")
            # Tokenize all documents
            tokenized_docs = [self._tokenize_text(text) for text in self.doc_texts]
            
            # Train Word2Vec model
            self.word2vec_model = Word2Vec(
                sentences=tokenized_docs,
                vector_size=self.embedding_dim,
                window=self.window,
                min_count=self.min_count,
                workers=self.workers
            )
            
            # Save the model if caching is enabled
            if self.use_cache:
                logger.info(f"Saving Word2Vec model to {cache_path}...")
                self.word2vec_model.save(cache_path)
            logger.info("Word2Vec model trained.")

        # Precompute document embeddings
        self._compute_document_embeddings()

    def _compute_document_embeddings(self):
        """Computes and stores document embeddings as average of word embeddings."""
        logger.info("Computing document embeddings...")
        self.doc_embeddings = []
        
        for text in self.doc_texts:
            tokens = self._tokenize_text(text)
            embeddings = []
            
            for token in tokens:
                if token in self.word2vec_model.wv:
                    embeddings.append(self.word2vec_model.wv[token])
            
            if embeddings:
                # Average the embeddings
                avg_embedding = np.mean(embeddings, axis=0)
            else:
                # If no tokens found, use zero vector
                avg_embedding = np.zeros(self.embedding_dim)
            
            self.doc_embeddings.append(avg_embedding)
        
        self.doc_embeddings = np.array(self.doc_embeddings)
        logger.info(f"Computed embeddings for {len(self.doc_embeddings)} documents.")

    def _get_query_embedding(self, query: str) -> np.ndarray:
        """Computes embedding for a query string."""
        tokens = self._tokenize_text(query)
        embeddings = []
        
        for token in tokens:
            if token in self.word2vec_model.wv:
                embeddings.append(self.word2vec_model.wv[token])
        
        if embeddings:
            # Average the embeddings
            avg_embedding = np.mean(embeddings, axis=0)
        else:
            # If no tokens found, use zero vector
            avg_embedding = np.zeros(self.embedding_dim)
        
        return avg_embedding

    def retrieve(self, query: str, k: int = 20) -> List[Tuple[str, str, float]]:
        """
        Retrieves top-k documents using Word2Vec embeddings.

        Args:
            query: The query string.
            k: Number of documents to retrieve.

        Returns:
            List of tuples (doc_id, doc_text, score).
        """
        query_embedding = self._get_query_embedding(query)
        
        # Compute cosine similarity between query and all document embeddings
        # Normalize embeddings for cosine similarity
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        doc_norms = self.doc_embeddings / (np.linalg.norm(self.doc_embeddings, axis=1, keepdims=True) + 1e-10)
        
        # Compute dot product (cosine similarity)
        similarities = np.dot(doc_norms, query_norm)
        
        # Get top-k most similar documents
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_indices:
            doc_id = self.doc_ids[idx]
            doc_text = self.doc_texts[idx]
            score = float(similarities[idx])
            results.append((doc_id, doc_text, score))
        
        logger.debug(f"Retrieved {len(results)} documents for query: {query}...")
        return results