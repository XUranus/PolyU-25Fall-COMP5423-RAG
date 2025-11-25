#!/usr/bin/env python3
# agentic_workflow.py

"""
This module implements an agentic workflow that integrates retrieval and generation
to provide answers along with reasoning steps. It builds upon the RAG pipeline to
support multi-turn interactions and detailed thought processes.
"""

from typing import List, Dict, Any, Tuple, Optional
from hybrid_retriever import HybridRetriever # Import our retriever class
from qwen_generator import QwenGenerator # Import our generator class
import logging
import re

logger = logging.getLogger('RAG42')


# --- Agentic Workflow Core Algorithm ---

class AgenticWorkflow:
    def __init__(self, retriever: HybridRetriever, generator: QwenGenerator):
        self.retriever = retriever
        self.generator = generator


    def identify_multi_hop_pattern(self, question: str) -> Optional[str]:
        """
        Heuristically identifies if a question might be multi-hop based on keywords/phrases.
        This is a simple rule-based check. More sophisticated methods could use NER/RE.
        """
        question_lower = question.lower()
        # Common patterns for multi-hop questions in datasets like HotpotQA
        multi_hop_indicators = [
            r'\bwhen.*won\b', # "when he won", "when she was born"
            r'\bwhere.*was.*born\b',
            r'\bwho.*won.*in\b',
            r'\bsecond.*finisher.*who.*drive.*for\b', #
            r'\bfirst.*won.*then.*work.*for\b',
            # TODO:: Add more patterns as needed
        ]
        for pattern in multi_hop_indicators:
            if re.search(pattern, question_lower):
                return pattern
        return None


    def decompose_query(self, question: str) -> List[str]:
        """
        Decomposes a multi-hop question into sub-questions using the LLM.
        """
        decomposition_prompt = \
            f"""
            You are given a complex question that requires multiple steps to answer.
            Please break it down into simpler, sequential sub-questions that can be answered independently.
            The sub-questions should be self-contained and lead logically to the final answer.

            Question: {question}

            Please list the sub-questions, one per line, starting with "Sub-question 1:", "Sub-question 2:", etc.
            """
        decomposition_response = self.generator.generate(decomposition_prompt)

        # Simple parsing of the LLM's decomposition output
        sub_questions = []
        lines = decomposition_response.split('\n')
        for line in lines:
            # Match lines like "Sub-question 1: Who is X?" or "Sub-question 2: Where is Y?"
            match = re.match(r'^Sub-question\s+\d+:\s*(.+)', line.strip(), re.IGNORECASE)
            if match:
                sub_q = match.group(1).strip()
                if sub_q:
                    sub_questions.append(sub_q)
        return sub_questions


    def synthesize_answer(self, question: str, sub_answers: List[str]) -> str:
        """
        Synthesizes the final answer from the original question and the answers to sub-questions.
        """
        context_for_synthesis = "\n".join([f"Sub-answer {i+1}: {ans}" for i, ans in enumerate(sub_answers)])
        synthesis_prompt = \
            f"""
            You are given a complex question and the answers to its constituent sub-questions.
            Use this information to provide a concise and accurate final answer to the original question.

            Original Question: {question}
            Sub-answers:
            {context_for_synthesis}

            Final Answer:
            """
        final_answer = self.generator.generate(synthesis_prompt)
        # Post-process to remove potential prefixes like "The final answer is..." ::TODO
        # A simple heuristic might be to take the last non-empty line.
        lines = final_answer.strip().split('\n')
        for line in reversed(lines):
            if line.strip():
                return line.strip()
        return final_answer.strip() # Fallback if no lines match heuristic


    def run(self, question: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Main function to execute the agentic workflow.

        Args:
            question (str): The input question.
            retriever (Retriever): Instance of retrieval module.
            generator (Generator): Instance of generation module.

        Returns:
            Tuple[str, List[Dict[str, Any]]]: A tuple containing the final answer (str)
            and a list of intermediate steps (List[Dict]) for the UI.
        """
        steps_log = []
        sub_questions = []
        original_question = question
        step_cnt = 1

        # 1. Identify if the question is multi-hop
        multi_hop_pattern = self.identify_multi_hop_pattern(question)
        is_multi_hop = multi_hop_pattern is not None

        steps_log.append({
            "step": step_cnt,
            "description": f"Initial Analysis: Question is {'multi-hop' if is_multi_hop else 'single-hop'}. Pattern detected: {multi_hop_pattern or 'None'}.",
            "type" : "identify",
            "query": question,
            "result": None
        })
        step_cnt += 1

        # 2. Try decompose the multi-hop question
        if is_multi_hop:
            sub_questions = self.decompose_query(question)
            steps_log.append({
                "step": step_cnt,
                "description": f"Query decomposition: The question was broken down into {len(sub_questions)} sub-questions.",
                "type" : "decomposition",
                "sub_questions": sub_questions,
                "result": None
            })
            step_cnt += 1
            if len(sub_questions) < 2:
                # If decomposition failed, fallback to single-hop
                is_multi_hop = False
                steps_log.append({
                    "step": step_cnt,
                    "description": "Decomposition yielded less than 2 sub-questions. Falling back to single-hop processing.",
                    "type" : "fallback",
                    "result": None
                })
                step_cnt += 1

        if is_multi_hop:
            # 3. Process each sub-question
            sub_answers = []
            for i, sub_q in enumerate(sub_questions):
                # Retrieve relevant documents for the sub-question
                retrieved_docs = self.retriever.retrieve(sub_q, k = 10) # Adjust top_k as needed
                steps_log.append({
                    "step": f"2.{step_cnt}",
                    "description": f"Retrieval for Sub-question {i+1} : {sub_q}",
                    "type" : "multi_hop_sub_retrieval",
                    "query": sub_q,
                    "retrieved_docs": retrieved_docs
                })
                step_cnt += 1

                # Generate an answer for the sub-question using retrieved docs
                sub_answer = self.generator.generate_from_docs(sub_q, [doc_text for (_, doc_text, _) in retrieved_docs])
                sub_answers.append(sub_answer)

                steps_log.append({
                    "step": f"3.{step_cnt}",
                    "description": f"Generated Answer for Sub-question {i+1} : {sub_q}",
                    "type" : "multi_hop_sub_generation",
                    "query": sub_q,
                    "result": sub_answer
                })
                step_cnt += 1

            # 4. Synthesize the final answer from sub-answers
            final_answer = self.synthesize_answer(original_question, sub_answers)
            steps_log.append({
                "step": step_cnt,
                "description": "Synthesis: Final answer generated from sub-answers.",
                "type" : "multi_hop_synthesize_answer",
                "result": final_answer
            })
            step_cnt += 1

        else:
            # If not multi-hop, run standard single-turn RAG
            retrieved_docs = self.retriever.retrieve(question, k = 10)
            steps_log.append({
                "step": step_cnt,
                "description": "Standard RAG Retrieval (Single-Hop)",
                "type" : "single_hop_retrieval",
                "query": question,
                "retrieved_docs": retrieved_docs
            })
            step_cnt += 1

            final_answer = self.generator.generate_from_docs(question, [doc_text for (_, doc_text, _) in retrieved_docs])
            steps_log.append({
                "step": step_cnt,
                "description": "Standard RAG Generation (Single-Hop)",
                "type" : "single_hop_generation",
                "query": question,
                "result": final_answer
            })

        return final_answer, steps_log
