#!/usr/bin/env python3
# singlehop_workflow.py

"""
This module implements an singe-hop workflow.
It's only used as a baseline method for test
"""

from typing import List, Dict, Any, Tuple, Optional
from retriever_base import BaseRetriever
import logging
import re

logger = logging.getLogger('RAG42')


# --- SingleHop Workflow Core Algorithm ---

class SingleHopWorkflow:
    def __init__(self,
                 retriever: BaseRetriever,
                 generator):
        self.retriever = retriever
        self.generator = generator


    def answer_from_docs(self, query: str, retrieved_docs: List[str], max_total_chars: int = 8000) -> str:
        """
        Builds the prompt string from the documents for the LLM.
        Generates an answer based on the query and retrieved documents.
        Truncates evidence to fit within max_total_chars (approximate token budget).

        Args:
            query (str): The user's query.
            retrieved_docs (List[str]): List of retrieved document texts.
            max_total_chars (int): Max total characters for all evidence snippets combined.

        Returns:
            str : the answer
        """
        snippets = []
        remaining = max_total_chars
        for i, doc in enumerate(retrieved_docs):
            if remaining <= 0:
                break
            snippet = doc[:remaining]
            snippets.append(f"[{i+1}] {snippet}")
            remaining -= len(snippet)

        evidence_snippets = "\n".join(snippets)

        prompt = (
            "Answer the question using ONLY the information in the evidence below. "
            "Your answer must be a short phrase, a single entity name, or 'yes'/'no'. "
            "Do NOT write a full sentence. Do NOT add explanations.\n\n"

            "Evidence:\n"
            f"{evidence_snippets}\n\n"

            f"Question: {query}\n\n"

            "Answer:"
        )

        logger.debug(f"Generated prompt\n: {prompt}")
        response = self.generator.generate(prompt)
        return self._post_process_answer(response)

    def _post_process_answer(self, answer: str) -> str:
        """
        Post-processes the LLM output to extract a clean answer.
        """
        answer = answer.strip()

        prefixes = [
            "Final Answer:", "Final answer:", "The answer is:", "The final answer is:",
            "Answer:", "A:", "Based on the evidence,", "Based on the information provided,",
        ]
        for prefix in prefixes:
            if answer.startswith(prefix):
                answer = answer[len(prefix):].strip()

        lines = [l.strip() for l in answer.split('\n') if l.strip()]
        if lines:
            answer = lines[0]

        if answer.endswith('.'):
            answer = answer[:-1].strip()

        return answer
    

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
