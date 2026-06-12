#!/usr/bin/env python3
# generator_base.py

"""
Abstract base class for LLM generators.
All generator implementations must conform to this interface.
"""

from abc import ABC, abstractmethod
import logging

logger = logging.getLogger('RAG42')


class BaseGenerator(ABC):
    """
    Abstract base class for all generators.
    Provides a common interface for text generation.
    """

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generates a response based on the given prompt.

        Args:
            prompt: The input prompt string.

        Returns:
            The generated response string.
        """
        pass
