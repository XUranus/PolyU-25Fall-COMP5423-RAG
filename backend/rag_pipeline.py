#!/usr/bin/env python3
# rag_pipeline.py

"""
This is the main orchestrator. It takes a query, runs retrieval, generation, and formats the response for your database.
"""


from typing import List, Dict, Any, Tuple, Optional
import logging

from agentic_workflow import AgenticWorkflow
from hybrid_retriever import HybridRetriever
from huggingface_generator import HuggingfaceGenerator
from openai_generator import OpenAIGenerator

logger = logging.getLogger('RAG42')

class RAGPipeline:
    """
    The main RAG Pipeline orchestrator.
    Integrates retrieval and generation modules to produce a final answer,
    along with supporting information like retrieved documents and thinking process.
    Designed to support single-turn and future multi-turn interactions.
    """
    def __init__(self, retriever: HybridRetriever):
        """
        Initializes the RAG Pipeline.

        Args:
            retriever (HybridRetriever): The retrieval module instance.
            generator (QwenGenerator): The generation module instance.
        """
        self.retriever = retriever
        self.generator_map = {} # generator map


    def init_generator(self, model_name : str):
        """
        Initialize a model and add it into the generator map
        
        Args:
            model_name (str) : The huggingface model or the openai API model
        """
        if model_name in self.generator_map:
            logger.debug(f"{model_name} already been loaded, skip")
            return self.generator_map[model_name]
        logger.info("start init new generator: {model_name}")
        if model_name == "Qwen/Qwen2.5-0.5B-Instruct":
            self.generator_map[model_name] = HuggingfaceGenerator(model_name=model_name)
        else:
            self.generator_map[model_name] = OpenAIGenerator(model_name=model_name)
        logger.info("new generator: {model_name} loaded.")
        return self.generator_map[model_name]


    def run(
        self,
        query: str,
        session_history: Optional[List[Dict[str, str]]] = None,
        top_k: int = 10,
        max_doc_chars: int = 2000,
        model_name : str = "Qwen/Qwen2.5-0.5B-Instruct"
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
        # check generator
        self.init_generator(model_name)
        generator = self.generator_map[model_name]

        thinking_process = []

        # --- Step 1: Query Handling (Future Multi-turn) ---
        need_reformulate = False
        if session_history:
            if len(session_history) < 2:
                logger.debug("Session history too short; using original query.")
            else:
                logger.debug("Reformulating query based on session history...")
                need_reformulate = True
        else:
            logger.debug("No session history provided; using original query.")

        # --- Step 2: Generate Answer Via Agentic Workflow ---
        logger.debug("Generating answer with Agentic Workflow...")
        agentic_workflow = AgenticWorkflow(self.retriever, generator, need_reformulate, session_history)
        final_answer, intermediate_steps = agentic_workflow.run(query)

        for step in intermediate_steps:
            thinking_process_item = {}
            step_no = step['step']
            step_description = step['description']
            step_type = step['type']
            thinking_process_item['type'] = step['type']
            thinking_process_item['step'] = step_no
            thinking_process_item['description'] = f"[{step_no}] {step_description}"
            if step_type == 'multi_hop_sub_generation':
                thinking_process_item['result'] = step['result']
            if 'retrieved_docs' in step:
                thinking_process_item['retrieved_docs'] = [{'id' : id, 'text': text, 'score': score} for id, text, score in step['retrieved_docs']]
            thinking_process.append(thinking_process_item)
                
        # --- Step 3: Format Output for Database/Response ---
        response_data = {
            "answer": final_answer,
            "thinking_process":thinking_process
        }

        logger.debug("RAG pipeline completed successfully.")
        return response_data

