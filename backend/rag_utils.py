#!/usr/bin/env python3
# rag_utils.py

"""
Shared utility functions for RAG workflows.
"""

from typing import List, Optional
import logging

logger = logging.getLogger('RAG42')


def post_process_answer(answer: str) -> str:
    """
    Post-processes the LLM output to extract a clean answer.
    Strips common prefixes, verbose formulations, and normalizes
    to the short entity-style format expected by HotpotQA.

    Args:
        answer: Raw LLM output.

    Returns:
        Cleaned answer string.
    """
    answer = answer.strip()

    # Remove common prefixes
    prefixes = [
        "Final Answer:", "Final answer:", "The answer is:", "The final answer is:",
        "Answer:", "A:", "Based on the evidence,", "Based on the information provided,",
    ]
    for prefix in prefixes:
        if answer.startswith(prefix):
            answer = answer[len(prefix):].strip()

    # If the answer is multi-line, take the first non-empty line
    lines = [line.strip() for line in answer.split('\n') if line.strip()]
    if lines:
        answer = lines[0]

    # Remove trailing period if present (HotpotQA answers typically don't end with period)
    if answer.endswith('.'):
        answer = answer[:-1].strip()

    return answer


def build_evidence_snippets(retrieved_docs: List[str], max_total_chars: int = 8000) -> str:
    """
    Builds numbered evidence snippets from retrieved documents,
    truncating to fit within a character budget.

    Args:
        retrieved_docs: List of document text strings.
        max_total_chars: Maximum total characters for all evidence combined.

    Returns:
        Formatted evidence string with numbered snippets.
    """
    snippets = []
    remaining = max_total_chars
    for i, doc in enumerate(retrieved_docs):
        if remaining <= 0:
            break
        snippet = doc[:remaining]
        snippets.append(f"[{i + 1}] {snippet}")
        remaining -= len(snippet)
    return "\n".join(snippets)
