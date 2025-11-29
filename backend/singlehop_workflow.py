#!/usr/bin/env python3
# singlehop_workflow.py

"""
This module implements an singe-hop workflow.
It's only used as a baseline method for test
"""

from typing import List, Dict, Any, Tuple, Optional
from hybrid_retriever import HybridRetriever # Import our retriever class
import logging
import re

logger = logging.getLogger('RAG42')


# --- SingleHop Workflow Core Algorithm ---

class SingleHopWorkflow:
    def __init__(self,
                 retriever: HybridRetriever,
                 generator):
        self.retriever = retriever
        self.generator = generator


    def answer_from_docs(self, query: str, retrieved_docs: List[str], max_doc_chars: int = 2000) -> str:
        """
        Builds the prompt string from the documents for the LLM.
        Generates an answer based on the query and retrieved documents.

        Args:
            query (str): The user's query.
            retrieved_docs (List[str]): List of retrieved document texts.
            max_doc_chars (int): Max characters per doc snippet in prompt.

        Returns:
            str : the answer
        """
        evidence_snippets = "\n".join(
            [f"[{i+1}] {doc[:max_doc_chars]}" for i, doc in enumerate(retrieved_docs)]
        )
        
        prompt = (
            "Question:\n"
            f"{query}\n\n"
            f"nContext: {evidence_snippets}\n\n"
        )

        logger.debug(f"Generated prompt\n: {prompt}")
        response = self.generator.generate(prompt)
        if response.startswith("Answer:"):
            response = response[len("Answer:"):]
        return response
    

    def run(self, question: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Main function to execute the single-hop workflow.

        Args:
            question (str): The input question.

        Returns:
            Tuple[str, List[Dict[str, Any]]]: A tuple containing the final answer (str)
            and a list of intermediate steps (List[Dict]) for the UI.
        """
        steps_log = []

        # If not multi-hop, run standard single-turn RAG
        retrieved_docs = self.retriever.retrieve(question, k = 10)
        steps_log.append({
            "step": 1,
            "description": "Standard RAG Retrieval (Single-Hop)",
            "type" : "single_hop_retrieval",
            "query": question,
            "retrieved_docs": retrieved_docs
        })

        final_answer = self.answer_from_docs(question, [doc_text for (_, doc_text, _) in retrieved_docs])
        steps_log.append({
            "step": 2,
            "description": "Standard RAG Generation (Single-Hop)",
            "type" : "single_hop_generation",
            "query": question,
            "result": final_answer
        })

        return final_answer, steps_log
