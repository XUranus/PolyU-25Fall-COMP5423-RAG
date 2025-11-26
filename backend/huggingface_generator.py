#!/usr/bin/env python3
# huggingface_generator.py
"""
This class wraps the local Qwen generation logic, making it easy to switch models or add features like reasoning.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging
from typing import List, Dict 

logger = logging.getLogger('RAG42')


class HuggingfaceGenerator:
    """
    Generator module using models like Qwen2.5 Instruct models from Huggingface.
    Handles prompt formatting and answer generation.
    Can be extended for Feature B (Agentic Workflow).
    """
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        """
        Initializes the Qwen Generator.

        Args:
            model_name (str): The name of the Qwen model to use.
        """
        self.model_name = model_name
        logger.info(f"Loading Qwen model: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        self.model.eval()
        logger.info("Qwen model loaded successfully.")


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
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=1024, # Adjust as needed
            do_sample=False, # Deterministic for consistency
            pad_token_id=self.tokenizer.eos_token_id
        )

        # Decode only the newly generated part
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        logger.info("Generation completed.")
        # For now, return just the answer. If Feature B is implemented,
        # this could parse out reasoning steps from the response.
        return response


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
