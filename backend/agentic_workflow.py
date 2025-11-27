#!/usr/bin/env python3
# agentic_workflow.py

"""
This module implements an agentic workflow that integrates retrieval and generation
to provide answers along with reasoning steps. It builds upon the RAG pipeline to
support multi-turn interactions and detailed thought processes.
"""

from typing import List, Dict, Any, Tuple, Optional
from hybrid_retriever import HybridRetriever # Import our retriever class
import logging
import re

logger = logging.getLogger('RAG42')


# --- Agentic Workflow Core Algorithm ---

class AgenticWorkflow:
    def __init__(self,
                 retriever: HybridRetriever,
                 generator,
                 need_reformulate : bool = False,
                 session_history : List = []):
        self.retriever = retriever
        self.generator = generator
        self.need_reformulate = need_reformulate
        self.session_history = session_history
        self.MAX_DECOMPOSITION_STEPS = 10

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
            "Answer the question using ONLY the information in the evidence below. "
            "If the evidence does not contain enough information to answer the question, respond with exactly: 'I don't know.'\n\n"
            
            "Evidence:\n"
            f"{evidence_snippets}\n\n"
            
            f"Question: {query}\n\n"
            
            "Answer (be concise and factual):"
        )
        logger.debug(f"Generated prompt\n: {prompt}")
        response = self.generator.generate(prompt)
        return response
    

    def reformulate_query(self, query: str, history: List[Dict[str, str]]) -> str:
        """
        Reformulates a follow-up query into a standalone query using conversation history.
        Uses the generator LLM to perform coreference resolution and context expansion.

        Example:
            History: [{"sender": "user", "content": "Where was Barack Obama born?"},
                    {"sender": "bot", "content": "Barack Obama was born in Honolulu, Hawaii."}]
            Current query: "What about his wife?"
            Output: "Where was Barack Obama's wife born?"

        Args:
            generator: The selected generator
            query (str): The current user query.
            history (List[Dict[str, str]]): Prior conversation turns.

        Returns:
            str: A contextually reformulated standalone query.
        """
        try:
            # Build conversation transcript
            conversation_lines = []
            for turn in history:
                role = "User" if turn["sender"] == "user" else "Assistant"
                conversation_lines.append(f"{role}: {turn['content']}")
            conversation_text = "\n".join(conversation_lines)
            
            # Construct prompt for query reformulation
            reformulation_prompt = (
                "You are a precise query rewriting assistant. Your ONLY task is to rewrite the final user query as a standalone question by resolving coreferences (e.g., 'he', 'his', 'it') using the conversation history. "
                "Do NOT change the user's intent, add information, or answer the question. Return ONLY the rewritten question—no prefix, no explanation.\n\n"
                
                "Conversation History:\n"
                f"{conversation_text}\n\n"
                
                "Final User Query:\n"
                f"{query}\n\n"
                
                "Standalone Question:"
            )
            logger.debug(f"Reformulation prompt:\n {reformulation_prompt}")
            # Use the generator to reformulate
            reformulated = self.generator.generate(reformulation_prompt).strip()

            # Fallback if LLM returns empty or malformed output
            if not reformulated or len(reformulated) < 3:
                logger.warning("Query reformulation returned empty or short output. Falling back to original query.")
                return query
            
            logger.debug(f"Reformulated query: {reformulated}")
            return reformulated

        except Exception as e:
            logger.error(f"Error during query reformulation: {e}. Falling back to original query.")
            return query



    def identify_multi_hop_pattern(self, question: str) -> Optional[str]:
        """
        Heuristically identifies if a question might be multi-hop based on keywords/phrases.
        This is a simple rule-based check. More sophisticated methods could use NER/RE.
        """
        question_lower = question.lower().strip()

        # Enhanced multi-hop indicators with more comprehensive patterns
        multi_hop_indicators = [
            # Temporal relationships
            r'\bwhen.*(?:won|received|got|achieved|was awarded)\b',
            r'\bwhen.*was.*born\b',
            r'\b(?:before|after|during|following).*when\b',
            # Causal relationships
            r'\b(?:what|who|which).*led to\b',
            r'\b(?:what|who|which).*caused\b',
            r'\b(?:result of|consequence of|due to)\b',
            # Comparative/relational
            r'\b(?:who|what|which).*(?:after|before|instead of|rather than)\b',
            r'\b(?:first.*then|initially.*subsequently|earlier.*later)\b',
            # Entity relationship chains
            r'\b(?:who|what).*works for.*who\b',
            r'\b(?:where|which).*located in.*where\b',
            r'\b(?:what|which).*created by.*who\b',
            # Multi-entity references
            r'\b(?:both.*and|either.*or|neither.*nor).*\b(?:who|what|where)\b',
        ]
        
        for pattern in multi_hop_indicators:
            if re.search(pattern, question_lower):
                return pattern
        return None


    def decompose_query(self, question: str) -> List[str]:
        """
        Decomposes a multi-hop question into sub-questions using the LLM.
        """
        few_shot = (
            "Example:\n"
            "Complex Question: Who was the director of the movie that won Best Picture at the 2020 Oscars?\n"
            "Sub-questions:\n"
            "1. Which movie won Best Picture at the 2020 Oscars?\n"
            "2. Who directed [answer from 1]?\n\n"
        )

        decomposition_prompt = (
            "You are an expert at breaking down complex questions. Decompose the following question into a sequence of simple, answerable sub-questions. "
            "Each sub-question must build logically on the previous one and use concrete entities. Do NOT answer—just list sub-questions.\n\n"
            
            f"{few_shot}"
            f"Complex Question: {question}\n"
            "Sub-questions:\n"
            "1."
        )
        
        logger.debug(f'decompose_query prompt :\n {decomposition_prompt}')
        decomposition_response = self.generator.generate(decomposition_prompt)
        logger.debug(f'decomposition_response: \n {decomposition_response}')

        # Enhanced parsing with multiple formats
        sub_questions = []
        lines = decomposition_response.split('\n')
        
        for line in lines:
            line = line.strip()
            # Match numbered lists (1., 2., etc.)
            numbered_match = re.match(r'^\d+\.\s*(.+)', line)
            if numbered_match:
                sub_q = numbered_match.group(1).strip()
                if sub_q and not sub_q.lower().startswith('provide up to') and not sub_q.startswith('<sub_questions>'):
                    sub_questions.append(sub_q)
            # Alternative format: "Sub-question 1:", "Sub-question 2:", etc.
            elif ':' in line and ('sub-question' in line.lower() or 'sub question' in line.lower()):
                match = re.match(r'^.*?:\s*(.+)', line.strip(), re.IGNORECASE)
                if match:
                    sub_q = match.group(1).strip()
                    if sub_q:
                        sub_questions.append(sub_q)

        # Remove empty strings and limit to max steps
        sub_questions = [sq for sq in sub_questions if sq.strip()][:self.MAX_DECOMPOSITION_STEPS]
        return sub_questions



    def synthesize_answer(self, question: str, sub_answers: List[str]) -> str:
        """
        Synthesizes the final answer from the original question and the answers to sub-questions.
        """
        context_for_synthesis = "\n".join([f"Sub-answer {i+1}: {ans}" for i, ans in enumerate(sub_answers)])
        synthesis_prompt = (
            "You are given a complex question and the verified answers to its sub-questions. "
            "Combine ONLY the provided sub-answers to form a final, concise answer to the original question. "
            "Do NOT use external knowledge or speculate. If sub-answers are insufficient, say 'I don't know.'\n\n"
            
            f"Original Question: {question}\n\n"
            "Sub-answers:\n"
            f"{context_for_synthesis}\n\n"
            
            "Final Answer:"
        )
        final_answer = self.generator.generate(synthesis_prompt)
        # A simple heuristic might be to take the last non-empty line.
        lines = final_answer.strip().split('\n')
        for line in reversed(lines):
            if line.strip():
                return line.strip()
        final_answer = final_answer.strip()
        # Post-process to remove potential prefixes like "The final answer is..."
        if final_answer.startswith("Final Answer:"):
            final_answer = final_answer[len("Final Answer:"):]
        return final_answer


    def run(self, question: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Main function to execute the agentic workflow.

        Args:
            question (str): The input question.

        Returns:
            Tuple[str, List[Dict[str, Any]]]: A tuple containing the final answer (str)
            and a list of intermediate steps (List[Dict]) for the UI.
        """
        steps_log = []
        sub_questions = []
        original_question = question
        step_cnt = 1

        if self.need_reformulate:
            new_query = self.reformulate_query(question, self.session_history)
            if new_query != question:
                steps_log.append({
                    "step": step_cnt,
                    "type" : "query_reformulation",
                    "description": f"Reformulated query to: *'{new_query}'*"
                })
                step_cnt += 1
                question = new_query

        # 1. Identify if the question is multi-hop
        multi_hop_pattern = self.identify_multi_hop_pattern(question)
        is_multi_hop = multi_hop_pattern is not None
        logger.debug(f'question = {question}, is_multi_hop = {is_multi_hop}')

        steps_log.append({
            "step": step_cnt,
            "description": f"Initial Analysis: Question is {'multi-hop' if is_multi_hop else 'single-hop'}. Pattern detected: `{multi_hop_pattern or 'None'}`.",
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
                    "description": "Decomposition yielded less than 2 sub-questions, falling back to single-hop processing.",
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
                    "step": step_cnt,
                    "description": f"Retrieval for Sub-question {i+1} : *{sub_q}*",
                    "type" : "multi_hop_sub_retrieval",
                    "query": sub_q,
                    "retrieved_docs": retrieved_docs
                })
                step_cnt += 1

                # Generate an answer for the sub-question using retrieved docs
                sub_answer = self.answer_from_docs(sub_q, [doc_text for (_, doc_text, _) in retrieved_docs])
                sub_answers.append(sub_answer)

                steps_log.append({
                    "step": step_cnt,
                    "description": f"Generated Answer for Sub-question {i+1}",
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

            final_answer = self.answer_from_docs(question, [doc_text for (_, doc_text, _) in retrieved_docs])
            steps_log.append({
                "step": step_cnt,
                "description": "Standard RAG Generation (Single-Hop)",
                "type" : "single_hop_generation",
                "query": question,
                "result": final_answer
            })

        return final_answer, steps_log
