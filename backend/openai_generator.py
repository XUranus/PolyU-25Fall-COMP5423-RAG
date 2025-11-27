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
import traceback

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
            logger.error(traceback.format_exc())
            return f'Failed to retrieve chat history {e}'


