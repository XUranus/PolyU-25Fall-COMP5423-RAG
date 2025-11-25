#!/usr/bin/env python3
# rag_pipeline.py

"""
This is the main orchestrator. It takes a query, runs retrieval, generation, and formats the response for your database.
"""


from typing import List, Dict, Any, Tuple, Optional
from hybrid_retriever import HybridRetriever # Import our retriever class
from qwen_generator import QwenGenerator # Import our generator class
import logging

logger = logging.getLogger(__name__)

class RAGPipeline:
    """
    The main RAG Pipeline orchestrator.
    Integrates retrieval and generation modules to produce a final answer,
    along with supporting information like retrieved documents and thinking process.
    Designed to support single-turn and future multi-turn interactions.
    """
    def __init__(self, retriever: HybridRetriever, generator: QwenGenerator):
        """
        Initializes the RAG Pipeline.

        Args:
            retriever (HybridRetriever): The retrieval module instance.
            generator (QwenGenerator): The generation module instance.
        """
        self.retriever = retriever
        self.generator = generator

    def run(
        self,
        query: str,
        session_history: Optional[List[Dict[str, str]]] = None,
        top_k: int = 10,
        max_doc_chars: int = 2000
    ) -> Dict[str, Any]:
        """
        Executes the full RAG pipeline for a single query.

        Args:
            query (str): The user's query.
            session_history (Optional[List[Dict[str, str]]]): For future multi-turn support.
            top_k (int): Number of documents to retrieve.
            max_doc_chars (int): Maximum characters per retrieved document snippet.

        Returns:
            Dict[str, Any]: A dictionary containing the answer, retrieved docs, and thinking process.
        """
        # --- Step 1: Query Handling (Future Multi-turn) ---
        # If session_history is provided, implement query reformulation here.
        # For now, just use the query as is.
        processed_query = query
        if session_history:
            logger.info("Multi-turn support not yet implemented in this basic version.")
            # TODO: Implement query reformulation logic here
            # processed_query = self._reformulate_query(query, session_history)

        # --- Step 2: Retrieve Documents ---
        logger.info(f"Retrieving top {top_k} documents for query: {query[:50]}...")
        retrieved_docs_with_scores = self.retriever.retrieve(processed_query, k=top_k)
        # Extract just the document texts for generation
        retrieved_docs_texts = [doc_text for doc_id, doc_text, score in retrieved_docs_with_scores]

        # --- Step 3: Generate Answer ---
        logger.info("Generating answer...")
        generation_result = self.generator.generate(processed_query, retrieved_docs_texts, max_doc_chars)

        # --- Step 4: Format Output for Database/Response ---
        response_data = {
            "answer": generation_result["answer"],
            "retrieved_docs": retrieved_docs_with_scores, # [(id, text, score), ...]
            "thinking_process": generation_result.get("reasoning", "") # If generator provides reasoning (e.g., from Feature B)
        }

        logger.info("RAG pipeline completed successfully.")
        return response_data

    # --- Placeholder for Future Multi-turn Feature ---
    def _reformulate_query(self, query: str, history: List[Dict[str, str]]) -> str:
        """
        (Placeholder) Reformulates a follow-up query based on conversation history.
        This is where Feature A (Multi-Turn Search) logic would go.
        """
        # Example: "What about his wife?" -> "Where was Barack Obama's wife born?"
        # Requires an LLM or coreference resolution logic.
        # This is complex and often requires another LLM call.
        # For now, just return the original query.
        logger.warning("Query reformulation is a placeholder and not implemented.")
        return query
