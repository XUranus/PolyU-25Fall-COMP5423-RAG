#!/usr/bin/env python3
# instruction_retriever.py

"""
Dense retrieval with instruction prefix, using models like intfloat/e5-mistral-7b-instruct
or intfloat/multilingual-e5-large-instruct that require task-specific instruction prefixes.
"""

from typing import List, Tuple
import faiss
from sentence_transformers import SentenceTransformer
import os
import logging
import numpy as np
from retriever_base import BaseRetriever

logger = logging.getLogger('RAG42')


class InstructionRetriever(BaseRetriever):
    """
    Dense retrieval with instruction-based models (e.g., E5-instruct, GTE-instruct).
    These models require task-specific instruction prefixes for optimal performance.
    """
    def __init__(
        self,
        collection_path: str,
        dense_model_name: str = "intfloat/multilingual-e5-large-instruct",
        use_cache: bool = True,
        cache_dir: str = os.getenv('RAG42_CACHE_DIR', './cache'),
        query_instruction: str = "Given a question, retrieve relevant documents that answer the question."
    ):
        """
        Initializes the Instruction-based Dense Retriever.

        Args:
            collection_path: Path or identifier for the document collection.
            dense_model_name: Name of the instruction-based embedding model.
            use_cache: Whether to load a pre-built FAISS index if available.
            cache_dir: Directory to store cached indices.
            query_instruction: Instruction prefix for queries.
        """
        self.dense_model_name = dense_model_name
        self.use_cache = use_cache
        self.query_instruction = query_instruction
        super().__init__(collection_path, cache_dir)
        self._build_index()

    def _get_detailed_instruct(self, instruction: str, query: str) -> str:
        """Format query with instruction prefix as required by E5-instruct models."""
        return f"Instruct: {instruction}\nQuery: {query}"

    def _build_index(self):
        """Builds the FAISS index with instruction-based embeddings."""
        cache_path = os.path.join(self.cache_dir, f"faiss_index_{self.dense_model_name.replace('/', '_')}_instruct.faiss")
        if self.use_cache and os.path.exists(cache_path):
            logger.info(f"Loading cached FAISS (instruction) index from {cache_path}...")
            self.dense_model = SentenceTransformer(self.dense_model_name)
            self.dense_index = faiss.read_index(cache_path)
            logger.info("Cached FAISS instruction index loaded.")
            return

        logger.info(f"Building FAISS (instruction) index with {self.dense_model_name}...")
        self.dense_model = SentenceTransformer(self.dense_model_name)

        # Encode documents (documents don't need instruction prefix for E5)
        logger.info("Encoding documents (no instruction prefix)...")
        doc_embeddings = self.dense_model.encode(self.doc_texts, batch_size=32, show_progress_bar=True)

        # Build FAISS index (Inner Product for cosine similarity after normalization)
        dimension = doc_embeddings.shape[1]
        self.dense_index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(doc_embeddings)
        self.dense_index.add(doc_embeddings.astype(np.float32))

        # Save the index if caching is enabled
        if self.use_cache:
            logger.info(f"Saving FAISS instruction index to {cache_path}...")
            faiss.write_index(self.dense_index, cache_path)

        logger.info("FAISS instruction index built.")

    def retrieve(self, query: str, k: int = 20) -> List[Tuple[str, str, float]]:
        """
        Retrieves top-k documents using instruction-based dense embeddings.

        Args:
            query: The query string.
            k: Number of documents to retrieve.

        Returns:
            List of tuples (doc_id, doc_text, score).
        """
        # Format query with instruction prefix
        instruct_query = self._get_detailed_instruct(self.query_instruction, query)
        query_emb = self.dense_model.encode([instruct_query])
        faiss.normalize_L2(query_emb)
        scores, indices = self.dense_index.search(query_emb.astype(np.float32), k=k)

        results = []
        for i in range(len(indices[0])):
            doc_idx = int(indices[0][i])
            doc_id = self.doc_ids[doc_idx]
            doc_text = self.doc_texts[doc_idx]
            score = float(scores[0][i])
            results.append((doc_id, doc_text, score))

        logger.debug(f"Retrieved {len(results)} documents for query: {query}...")
        return results
