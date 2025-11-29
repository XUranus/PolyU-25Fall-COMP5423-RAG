#!/usr/bin/env python3
# hybrid_retriever_improved.py
"""
This class wraps BM25, Dense Retrieval (BGE/E5-Mistral), and ColBERT into a clean, reusable interface.
It uses Reciprocal Rank Fusion (RRF) for combining scores from different retrieval methods.
"""

from typing import List, Tuple, Dict, Optional
import bm25s
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import logging
import os
import json
import torch
from transformers import AutoModel, AutoTokenizer
import pickle # For saving/loading ColBERT doc token embeddings

logger = logging.getLogger('RAG42')

class HybridRetriever:
    """
    A hybrid retrieval module combining BM25 (sparse), Dense (BGE/E5-Mistral), and ColBERT (multi-vector) methods.
    Uses Reciprocal Rank Fusion (RRF) for combining scores.
    Provides a simple API to retrieve documents given a query.
    """
    def __init__(
        self,
        collection_path: str, # Path to load collection data (or HuggingFace dataset name)
        sparse_model_name: str = "bm25s", # Not used directly, kept for potential future use
        dense_model_name: str = "BAAI/bge-small-en-v1.5", # Can be BGE, E5-Mistral, or Qwen3-Embedding
        use_colbert: bool = False, # Whether to use ColBERT (GTE-ColBERT-v1) as the dense method
        colbert_model_name: str = "lightonai/GTE-ModernColBERT-v1", # Model name for ColBERT
        use_cache: bool = True, # If you save FAISS index / BM25 index / ColBERT embeddings
        cache_dir: str = os.getenv('RAG42_CACHE_DIR', './cache'), # Directory to save/load indices
        rrf_k: int = 60 # K parameter for RRF. Default is often 60.
    ):
        """
        Initializes the Hybrid Retriever.

        Args:
            collection_path (str): Path or identifier for the document collection.
            sparse_model_name (str): Name of the sparse model (currently only BM25 is implemented).
            dense_model_name (str): Name of the dense embedding model (BGE, E5-Mistral).
            use_colbert (bool): Whether to use ColBERT instead of standard dense retrieval.
            colbert_model_name (str): Name of the ColBERT model.
            use_cache (bool): Whether to load pre-built indices/embeddings if available.
            cache_dir (str): Directory to store cached indices/embeddings.
            rrf_k (int): K parameter for Reciprocal Rank Fusion.
        """
        self.sparse_model_name = sparse_model_name
        self.dense_model_name = dense_model_name
        self.use_colbert = use_colbert
        self.colbert_model_name = colbert_model_name
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.rrf_k = rrf_k
        os.makedirs(self.cache_dir, exist_ok=True)

        # Load collection
        self._load_collection(collection_path)

        # Initialize retrieval components
        self._build_sparse_index()
        if use_colbert:
            self._build_colbert_index()
        else:
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
            self.bm25_retriever = bm25s.BM25.load(cache_path, load_corpus=False)
            logger.info("Cached BM25 index loaded.")
            return

        logger.info("Building BM25 (sparse) index...")
        corpus_tokens = bm25s.tokenize(self.doc_texts, stopwords="en")
        self.bm25_retriever = bm25s.BM25()
        self.bm25_retriever.index(corpus_tokens)

        if self.use_cache:
             logger.info(f"Saving BM25 index to {cache_path}...")
             self.bm25_retriever.save(cache_path)
        logger.info("BM25 index built.")


    def _build_dense_index(self):
        """Builds the BGE/E5-Mistral (dense) index using FAISS."""
        cache_path = os.path.join(self.cache_dir, f"faiss_index_{self.dense_model_name.replace('/', '_')}.faiss")
        embeddings_cache_path = os.path.join(self.cache_dir, f"doc_embeddings_{self.dense_model_name.replace('/', '_')}.npy")
        if self.use_cache and os.path.exists(cache_path) and os.path.exists(embeddings_cache_path):
            logger.info(f"Loading cached FAISS (dense) index and embeddings from {cache_path} and {embeddings_cache_path}...")
            self.dense_model = SentenceTransformer(self.dense_model_name)
            self.dense_index = faiss.read_index(cache_path)
            # Load embeddings separately if needed later (e.g., for debugging)
            # self.doc_embeddings = np.load(embeddings_cache_path)
            logger.info("Cached FAISS dense index loaded.")
            return

        logger.info(f"Building FAISS (dense) index with {self.dense_model_name}...")
        self.dense_model = SentenceTransformer(self.dense_model_name)

        # Encode documents (batched for speed)
        logger.info("Encoding documents...")
        doc_embeddings = self.dense_model.encode(self.doc_texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True) # Normalize embeddings

        # Build FAISS index (Inner Product for cosine similarity after normalization)
        dimension = doc_embeddings.shape[1]
        self.dense_index = faiss.IndexFlatIP(dimension) # Cosine similarity with normalized vectors
        self.dense_index.add(doc_embeddings)

        # Save the index and embeddings if caching is enabled
        if self.use_cache:
            logger.info(f"Saving FAISS index to {cache_path} and embeddings to {embeddings_cache_path}...")
            faiss.write_index(self.dense_index, cache_path)
            np.save(embeddings_cache_path, doc_embeddings)
        logger.info("FAISS dense index built.")


    def _build_colbert_index(self):
        """Builds the ColBERT index (token embeddings) using FAISS."""
        cache_path = os.path.join(self.cache_dir, f"faiss_index_colbert_{self.colbert_model_name.replace('/', '_')}.faiss")
        embeddings_cache_path = os.path.join(self.cache_dir, f"doc_token_embeddings_colbert_{self.colbert_model_name.replace('/', '_')}.pkl")
        tokenizer_cache_path = os.path.join(self.cache_dir, f"tokenizer_colbert_{self.colbert_model_name.replace('/', '_')}.pkl")

        if self.use_cache and os.path.exists(embeddings_cache_path) and os.path.exists(tokenizer_cache_path):
            logger.info(f"Loading cached ColBERT token embeddings and tokenizer from {embeddings_cache_path} and {tokenizer_cache_path}...")
            with open(embeddings_cache_path, 'rb') as f:
                self.doc_token_embeddings = pickle.load(f)
            with open(tokenizer_cache_path, 'rb') as f:
                self.colbert_tokenizer = pickle.load(f)
            # Load a simple FAISS index for doc IDs if needed, or just use list index
            # For simplicity here, we'll just use the list index as the ID mapping.
            # A FAISS index could map doc_id to embedding_list_index if needed for large datasets.
            logger.info("Cached ColBERT token embeddings and tokenizer loaded.")
            return

        logger.info(f"Building ColBERT index with {self.colbert_model_name}...")
        # Load model and tokenizer
        self.colbert_model = AutoModel.from_pretrained(self.colbert_model_name, trust_remote_code=True)
        self.colbert_tokenizer = AutoTokenizer.from_pretrained(self.colbert_model_name, trust_remote_code=True)

        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.colbert_model.to(device)
        self.colbert_model.eval()

        logger.info("Encoding documents into token embeddings for ColBERT...")
        doc_token_embeddings_list = []
        batch_size = 16 # Smaller batch size due to potential memory usage of token embeddings
        for i in range(0, len(self.doc_texts), batch_size):
            batch_texts = self.doc_texts[i:i+batch_size]
            # Tokenize and encode
            inputs = self.colbert_tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                # Get token embeddings (last_hidden_state)
                outputs = self.colbert_model(**inputs)
                # For GTE-ColBERT-v1, the output is typically the token embeddings
                # Shape: (batch_size, seq_len, hidden_dim)
                batch_embeddings = outputs.last_hidden_state.cpu().numpy()

            doc_token_embeddings_list.extend(batch_embeddings)

            if i % (batch_size * 10) == 0: # Log progress every 10 batches
                logger.info(f"Encoded {i+batch_size} / {len(self.doc_texts)} documents.")

        self.doc_token_embeddings = doc_token_embeddings_list # List of numpy arrays (seq_len, hidden_dim)

        # Save the token embeddings and tokenizer if caching is enabled
        if self.use_cache:
            logger.info(f"Saving ColBERT token embeddings to {embeddings_cache_path} and tokenizer to {tokenizer_cache_path}...")
            with open(embeddings_cache_path, 'wb') as f:
                pickle.dump(self.doc_token_embeddings, f)
            with open(tokenizer_cache_path, 'wb') as f:
                pickle.dump(self.colbert_tokenizer, f)
        logger.info("ColBERT index (token embeddings) built.")


    def retrieve(self, query: str, k: int = 10, alpha_bm25: float = 0.2, alpha_dense_or_colbert: float = 0.8) -> List[Tuple[str, str, float]]:
        """
        Retrieves top-k documents using a hybrid approach (BM25 + Dense/ColBERT).
        Uses Reciprocal Rank Fusion (RRF) for combining scores.

        Args:
            query (str): The query string.
            k (int): Number of documents to retrieve.
            alpha_bm25 (float): Weight for BM25 score in the initial fusion (if not using RRF).
                               If using RRF, this parameter is ignored for the final score calculation.
                               It's kept for potential fallback or internal weighting within methods.
            alpha_dense_or_colbert (float): Weight for Dense/ColBERT score in the initial fusion (if not using RRF).
                                            If using RRF, this parameter is ignored for the final score calculation.

        Returns:
            List[Tuple[str, str, float]]: List of tuples (doc_id, doc_text, rrf_score).
        """
        # Retrieve from both methods
        bm25_results = self._retrieve_sparse(query, k * 2) # Retrieve more to merge
        if self.use_colbert:
            colbert_results = self._retrieve_colbert(query, k * 2)
            dense_or_colbert_results = colbert_results
        else:
            dense_results = self._retrieve_dense(query, k * 2)
            dense_or_colbert_results = dense_results

        # Get doc_ids and scores from each method
        bm25_ids = [self.doc_ids[idx] for idx, _ in bm25_results]
        bm25_scores = [score for _, score in bm25_results]

        dense_or_colbert_ids = [self.doc_ids[idx] for idx, _ in dense_or_colbert_results]
        dense_or_colbert_scores = [score for _, score in dense_or_colbert_results]

        # Create RRF scores
        fused_scores_rrf = self._reciprocal_rank_fusion(
            [bm25_ids, dense_or_colbert_ids],
            [bm25_scores, dense_or_colbert_scores],
            k=self.rrf_k
        )

        # Sort and get top-k based on RRF score
        sorted_doc_ids = sorted(fused_scores_rrf.items(), key=lambda x: x[1], reverse=True)[:k]

        # Format results: (id, text, rrf_score)
        results = [(doc_id, self.id_to_text[doc_id], score) for doc_id, score in sorted_doc_ids]

        logger.debug(f"Retrieved {len(results)} documents for query: {query} using RRF.")
        return results


    def _retrieve_sparse(self, query: str, k: int) -> List[Tuple[int, float]]:
        """Internal method for sparse retrieval."""
        query_tokens = bm25s.tokenize([query], stopwords="en")
        scores, indices = self.bm25_retriever.retrieve(query_tokens, k=k)
        return [(int(indices[0][i]), float(scores[0][i])) for i in range(len(indices[0]))]


    def _retrieve_dense(self, query: str, k: int) -> List[Tuple[int, float]]:
        """Internal method for dense retrieval (BGE/E5-Mistral)."""
        # Apply instruction if needed (e.g., for E5-Mistral or Qwen3-Embedding)
        # Example for E5-Mistral: query_for_embedding = f"Represent this sentence for searching relevant passages: {query}"
        # Example for Qwen3-Embedding: Check official documentation for correct format.
        query_for_embedding = query # Default for BGE
        if "e5-mistral" in self.dense_model_name.lower():
            query_for_embedding = f"Represent this sentence for searching relevant passages: {query}"
        elif "qwen3-embedding" in self.dense_model_name.lower():
            # Placeholder - Check Qwen3-Embedding docs for correct format
            # query_for_embedding = f"Qwen3-Embedding instruction format: {query}" # Replace with actual format
            logger.warning("Qwen3-Embedding format not implemented yet. Using raw query.")
            query_for_embedding = query # Replace with correct format

        query_emb = self.dense_model.encode([query_for_embedding], normalize_embeddings=True) # Normalize query
        scores, indices = self.dense_index.search(query_emb, k=k)
        return [(int(indices[0][i]), float(scores[0][i])) for i in range(len(indices[0]))]


    def _retrieve_colbert(self, query: str, k: int) -> List[Tuple[int, float]]:
        """Internal method for ColBERT retrieval."""
        # Apply instruction if needed (e.g., for Qwen3-Embedding used as ColBERT)
        # Placeholder - Check if GTE-ColBERT-v1 needs specific query formatting
        query_for_embedding = query

        # Tokenize query
        query_inputs = self.colbert_tokenizer(query_for_embedding, return_tensors="pt", max_length=512, truncation=True, padding=True)
        # Move to device if using GPU
        device = next(self.colbert_model.parameters()).device
        query_inputs = {k: v.to(device) for k, v in query_inputs.items()}

        with torch.no_grad():
            query_outputs = self.colbert_model(**query_inputs)
            # Shape: (1, seq_len_q, hidden_dim)
            query_token_embs = query_outputs.last_hidden_state[0] # (seq_len_q, hidden_dim)

        # Compute similarities: Query tokens vs Document tokens
        scores_per_doc = []
        for doc_idx, doc_token_emb_array in enumerate(self.doc_token_embeddings):
            # doc_token_emb_array: (seq_len_d, hidden_dim)
            doc_token_embs = torch.tensor(doc_token_emb_array, dtype=torch.float32).to(device) # (seq_len_d, hidden_dim)

            # Compute similarity matrix: (seq_len_q, hidden_dim) @ (hidden_dim, seq_len_d) -> (seq_len_q, seq_len_d)
            sim_matrix = torch.matmul(query_token_embs, doc_token_embs.transpose(0, 1)) # (seq_len_q, seq_len_d)

            # For each query token, find the max similarity over document tokens
            max_sim_per_query_token, _ = torch.max(sim_matrix, dim=1) # (seq_len_q,)

            # Sum the max similarities across query tokens (ColBERT max-similarity aggregation)
            colbert_score = torch.sum(max_sim_per_query_token).item()
            scores_per_doc.append(colbert_score)

        # Get top-k document indices based on ColBERT scores
        top_k_indices = np.argsort(scores_per_doc)[::-1][:k]
        top_k_scores = [scores_per_doc[i] for i in top_k_indices]

        return [(int(idx), float(score)) for idx, score in zip(top_k_indices, top_k_scores)]


    def _reciprocal_rank_fusion(self, lists_of_ids: List[List[str]], lists_of_scores: List[List[float]], k: int = 60) -> Dict[str, float]:
        """
        Calculates RRF scores by combining multiple ranked lists.

        Args:
            lists_of_ids: A list of lists, where each inner list contains doc_ids from a retrieval method.
            lists_of_scores: A list of lists, where each inner list contains scores corresponding to the ids in lists_of_ids.
            k: The k parameter for RRF. Default is 60.

        Returns:
            A dictionary mapping doc_id to its fused RRF score.
        """
        fused_scores = {}
        for ids, scores in zip(lists_of_ids, lists_of_scores):
            # Create a map of doc_id to its raw score for this list
            score_map = {doc_id: score for doc_id, score in zip(ids, scores)}
            # Get the ranked order based on raw scores (descending)
            ranked_ids = sorted(score_map.keys(), key=lambda x: score_map[x], reverse=True)

            for rank, doc_id in enumerate(ranked_ids, start=1):
                # Calculate RRF contribution: 1 / (k + rank)
                rrf_contribution = 1.0 / (k + rank)
                fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + rrf_contribution

        return fused_scores


    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalizes a list of scores to [0, 1] using Min-Max scaling."""
        if not scores:
            return []
        min_score, max_score = min(scores), max(scores)
        if max_score == min_score:
            return [1.0] * len(scores)
        return [(s - min_score) / (max_score - min_score) for s in scores]
