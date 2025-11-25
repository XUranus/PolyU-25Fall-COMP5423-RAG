#!/usr/bin/env python3
# rag_pipeline.py

"""
This is the main orchestrator. It takes a query, runs retrieval, generation, and formats the response for your database.
"""


from typing import List, Dict, Any, Tuple, Optional
import logging

from agentic_workflow import AgenticWorkflow
from hybrid_retriever import HybridRetriever
from qwen_generator import QwenGenerator


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
        self.agentic_workflow = AgenticWorkflow(retriever, generator)

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
            logger.debug("Multi-turn support not yet implemented in this basic version.")
            # TODO: Implement query reformulation logic here
            # processed_query = self._reformulate_query(query, session_history)

        # --- Step 2: Retrieve Documents ---
        #logger.debug(f"Retrieving top {top_k} documents for query: {query[:50]}...")
        #retrieved_docs_with_scores = self.retriever.retrieve(processed_query, k = top_k)
        # Extract just the document texts for generation
        #retrieved_docs_texts = [doc_text for doc_id, doc_text, score in retrieved_docs_with_scores]

        # --- Step 3: Generate Answer ---
        thinking_process = []
        logger.debug("Generating answer With Agentic Workflow...")
        #generation_result = self.generator.generate_from_docs(processed_query, retrieved_docs_texts, max_doc_chars) # deprecated
        final_answer, intermediate_steps = self.agentic_workflow.run(processed_query)

        for step in intermediate_steps:
            thinking_process_item = {}
            step_no = step['step']
            step_description = step['description']
            thinking_process_item['step'] = step_no
            thinking_process_item['description'] = f"[{step_no}] {step_description}"
            # TODO:: add more details based on step type
            # if step['type'] in ['multi_hop_retrieval', 'multi_hop_sub_retrieval', 'single_hop_retrieval']:
            #     retrieved_docs = step.get('retrieved_docs', [])
            #     thinking_process_item['retrieved_docs'] = retrieved_docs
            if 'retrieved_docs' in step:
                thinking_process_item['retrieved_docs'] = [{'id' : id, 'text': text, 'score': score} for id, text, score in step['retrieved_docs']]
            thinking_process.append(thinking_process_item)
                
        # --- Step 4: Format Output for Database/Response ---
        response_data = {
            "answer": final_answer,
            "thinking_process":thinking_process
        }

        logger.debug("RAG pipeline completed successfully.")
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
