#!/usr/bin/env python3
# openai_generator.py
"""
This class wraps the LLM generation logic via OpenAI styped API calls
You need to set environment variables RAG42_OPENAI_API_KEY and RAG42_OPENAI_API_URL before using this class.
"""

import logging
from typing import List, Dict 
from openai import OpenAI
import os

logger = logging.getLogger('RAG42')


class OpenAIGenerator:
    """
    Generator module using OpenAI API as a provider instead of using local hugging face API to improve peformance.
    """
    def __init__(self, model_name: str = "qwen-turbo"):
        """
        Initializes the OpenAI Generator.

        Args:
            model_name (str): The name of the model to use.
        """
        self.model_name = model_name
        logger.info(f"init OpenAI API, model: {model_name}")
        
        self.api_key = os.getenv("RAG42_OPENAI_API_KEY")
        self.api_url = os.getenv("RAG42_OPENAI_API_URL")
        assert self.api_key is not None, "RAG42_OPENAI_API_KEY environment variable not set"
        assert self.api_url is not None, "RAG42_OPENAI_API_URL environment variable not set"

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_url
        )
        logger.info(f"{self.model_name} model loaded successfully.")


    def generate_from_docs(self, query: str, retrieved_docs: List[str], max_doc_chars: int = 2000) -> str:
        """
        Generates an answer based on the query and retrieved documents.

        Args:
            query (str): The user's query.
            retrieved_docs (List[str]): List of retrieved document texts.
            max_doc_chars (int): Max characters per doc snippet in prompt.

        Returns:
            str : the answer
        """
        prompt = self._build_prompt(query, retrieved_docs, max_doc_chars)
        logger.debug(f"Generated prompt\n: {prompt}")
        response = self.generate(prompt)

        logger.info("Generation completed.")
        return response
    

    def generate(self, prompt: str) -> str:
        """
        Generates an answer based on the query only.

        Args:
            prompt (str): The final prompt
        Returns:
            str: The answer
        """
        logger.debug(f'OpenAI API generate: {prompt}')    
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
                # stream=False is the default
            )
            # Extract the full reply
            reply = response.choices[0].message.content
            return reply
        except Exception as e:
            logger.error(f"Error using OpenAI API to generate: {e}")
            return f'Failed to retrieve chat history {e}'


    def _build_prompt(self, query: str, retrieved_docs: List[str], max_doc_chars: int) -> str:
        """
        Builds the prompt string for the LLM.
        """
        evidence_snippets = "\n".join(
            [f"[{i+1}] {doc[:max_doc_chars]}" for i, doc in enumerate(retrieved_docs)]
        )
        
        prompt = (
            "You are a helpful assistant. Answer the question based only on the provided evidence.\n\n"
            "Evidence:\n"
            f"{evidence_snippets}\n\n"
            f"Question: {query}\n\n"
            "Answer concisely and factually. If the evidence does not contain the answer, say 'I don't know.'\n"
            "Answer:"
        )
        return prompt
