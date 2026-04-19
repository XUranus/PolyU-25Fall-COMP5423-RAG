#!/usr/bin/env python3
# agentic_workflow.py

"""
This module implements an agentic workflow that integrates retrieval and generation
to provide answers along with reasoning steps. It builds upon the RAG pipeline to
support multi-turn interactions and detailed thought processes.
"""

from typing import List, Dict, Any, Tuple, Optional
from retriever_base import BaseRetriever
import logging
import re

logger = logging.getLogger('RAG42')


# --- Agentic Workflow Core Algorithm ---

class AgenticWorkflow:
    def __init__(self,
                 retriever: BaseRetriever,
                 generator,
                 need_reformulate : bool = False,
                 session_history : List = []):
        self.retriever = retriever
        self.generator = generator
        self.need_reformulate = need_reformulate
        self.session_history = session_history
        self.MAX_DECOMPOSITION_STEPS = 10

    def answer_from_docs(self, query: str, retrieved_docs: List[str], max_total_chars: int = 8000, prior_answers: List[str] = None) -> str:
        """
        Builds the prompt string from the documents for the LLM.
        Generates an answer based on the query and retrieved documents.
        Truncates evidence to fit within max_total_chars (approximate token budget).

        Args:
            query (str): The user's query.
            retrieved_docs (List[str]): List of retrieved document texts.
            max_total_chars (int): Max total characters for all evidence snippets combined.
            prior_answers (List[str]): Answers from previous sub-questions for chain reasoning.

        Returns:
            str : the answer
        """
        # Build evidence snippets with total character budget
        # Approximate: 1 token ~= 4 chars for English text
        # max_total_chars=8000 ~= 2000 tokens for evidence
        snippets = []
        remaining = max_total_chars
        for i, doc in enumerate(retrieved_docs):
            if remaining <= 0:
                break
            snippet = doc[:remaining]
            snippets.append(f"[{i+1}] {snippet}")
            remaining -= len(snippet)

        evidence_snippets = "\n".join(snippets)

        prior_context = ""
        if prior_answers:
            prior_context = "Previous sub-answers:\n" + "\n".join(
                [f"- {ans}" for ans in prior_answers]
            ) + "\n\nUse these answers if they help answer the current question.\n\n"

        prompt = (
            "Answer the question using ONLY the information in the evidence below. "
            "Your answer must be a short phrase, a single entity name, or 'yes'/'no'. "
            "Do NOT write a full sentence. Do NOT add explanations.\n\n"

            f"{prior_context}"
            "Evidence:\n"
            f"{evidence_snippets}\n\n"

            f"Question: {query}\n\n"

            "Answer:"
        )
        logger.debug(f"Generated prompt\n: {prompt}")
        response = self.generator.generate(prompt)
        return self._post_process_answer(response)
    

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
        Uses the LLM to determine if a question is multi-hop.
        Falls back to heuristic regex if LLM is unavailable.
        For HotpotQA specifically, most questions are multi-hop, so this
        errs on the side of decomposition.
        """
        try:
            classification_prompt = (
                "Determine whether the following question requires reasoning across multiple pieces of information (multi-hop) "
                "or can be answered with a single fact (single-hop).\n\n"
                "A multi-hop question typically:\n"
                "- Asks about two or more entities and their relationship\n"
                "- Requires finding one entity to answer about another\n"
                "- Contains chains like 'Who is the director of the movie that won...'\n\n"
                f"Question: {question}\n\n"
                "Respond with ONLY 'multi-hop' or 'single-hop'."
            )
            response = self.generator.generate(classification_prompt).strip().lower()
            if 'multi' in response:
                return "llm_classified_multi_hop"
            elif 'single' in response:
                return None
            # If ambiguous response, default to multi-hop for HotpotQA
            return "llm_classified_multi_hop"
        except Exception as e:
            logger.warning(f"LLM multi-hop classification failed: {e}. Falling back to heuristic.")
            return self._heuristic_multi_hop_check(question)

    def _heuristic_multi_hop_check(self, question: str) -> Optional[str]:
        """
        Fallback heuristic check for multi-hop questions using regex patterns.
        """
        question_lower = question.lower().strip()

        multi_hop_indicators = [
            r'\bwhen.*(?:won|received|got|achieved|was awarded)\b',
            r'\bwhen.*was.*born\b',
            r'\b(?:before|after|during|following).*when\b',
            r'\b(?:what|who|which).*led to\b',
            r'\b(?:what|who|which).*caused\b',
            r'\b(?:result of|consequence of|due to)\b',
            r'\b(?:who|what|which).*(?:after|before|instead of|rather than)\b',
            r'\b(?:first.*then|initially.*subsequently|earlier.*later)\b',
            r'\b(?:who|what).*works for.*who\b',
            r'\b(?:where|which).*located in.*where\b',
            r'\b(?:what|which).*created by.*who\b',
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
            "You are given a complex question and the answers to its sub-questions. "
            "Combine the sub-answers to form a final, concise answer to the original question. "
            "Your answer must be a short phrase, a single entity name, or 'yes'/'no'. "
            "Do NOT write a full sentence. Do NOT add explanations.\n\n"

            f"Original Question: {question}\n\n"
            "Sub-answers:\n"
            f"{context_for_synthesis}\n\n"

            "Final Answer:"
        )
        final_answer = self.generator.generate(synthesis_prompt)
        return self._post_process_answer(final_answer)

    def verify_answer(self, question: str, answer: str, evidence: str) -> str:
        """
        Verifies the generated answer against the evidence using the LLM.
        If the answer is contradicted or unsupported, returns an empty string
        to signal that regeneration should be attempted.

        Args:
            question: The original question.
            answer: The generated answer to verify.
            evidence: The evidence text used to generate the answer.

        Returns:
            The verified answer, or empty string if verification fails.
        """
        verify_prompt = (
            "Verify whether the following answer is directly supported by the evidence.\n\n"
            f"Question: {question}\n"
            f"Answer: {answer}\n\n"
            f"Evidence:\n{evidence[:3000]}\n\n"
            "Is the answer supported by the evidence? Respond with ONLY 'yes' or 'no'."
        )
        try:
            response = self.generator.generate(verify_prompt).strip().lower()
            if 'yes' in response:
                return answer
            else:
                logger.debug(f"Answer verification failed for: {answer}")
                return ""
        except Exception as e:
            logger.warning(f"Answer verification error: {e}. Keeping original answer.")
            return answer

    def answer_with_verification(self, query: str, retrieved_docs: List[str], max_retries: int = 1, prior_answers: List[str] = None) -> str:
        """
        Generates an answer with optional verification and retry.
        If verification fails, regenerates the answer up to max_retries times.

        Args:
            query: The question to answer.
            retrieved_docs: List of retrieved document texts.
            max_retries: Number of retry attempts if verification fails.
            prior_answers: Answers from previous sub-questions for chain reasoning.

        Returns:
            The best answer found.
        """
        evidence = "\n".join([doc[:2000] for doc in retrieved_docs])
        answer = self.answer_from_docs(query, retrieved_docs, prior_answers=prior_answers)

        for _ in range(max_retries):
            verified = self.verify_answer(query, answer, evidence)
            if verified:
                return verified
            # Regenerate
            answer = self.answer_from_docs(query, retrieved_docs, prior_answers=prior_answers)

        return answer

    def _post_process_answer(self, answer: str) -> str:
        """
        Post-processes the LLM output to extract a clean answer.
        Strips common prefixes, verbose formulations, and normalizes
        to the short entity-style format expected by HotpotQA.
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
        lines = [l.strip() for l in answer.split('\n') if l.strip()]
        if lines:
            answer = lines[0]

        # Remove trailing period if present (HotpotQA answers typically don't end with period)
        if answer.endswith('.'):
            answer = answer[:-1].strip()

        return answer


    def run(self, question: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Main function to execute the agentic workflow.

        Always attempts decomposition since HotpotQA is a multi-hop dataset.
        Falls back to single-hop if decomposition yields <2 sub-questions.

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

        # 1. Always attempt decomposition (HotpotQA is multi-hop by design)
        sub_questions = self.decompose_query(question)
        is_multi_hop = len(sub_questions) >= 2

        steps_log.append({
            "step": step_cnt,
            "description": f"Query decomposition: The question was broken down into {len(sub_questions)} sub-questions. {'Proceeding with multi-hop.' if is_multi_hop else 'Falling back to single-hop.'}",
            "type" : "decomposition",
            "sub_questions": sub_questions,
            "result": None
        })
        step_cnt += 1

        if is_multi_hop:
            # 2. Retrieve for the original question AND sub-questions
            # Some supporting docs may only be found via the original full question
            original_retrieved = self.retriever.retrieve(question, k = 10)
            steps_log.append({
                "step": step_cnt,
                "description": "Retrieval for original question (multi-hop context)",
                "type" : "multi_hop_original_retrieval",
                "query": question,
                "retrieved_docs": original_retrieved
            })
            step_cnt += 1

            # 3. Process each sub-question with chain reasoning
            sub_answers = []
            for i, sub_q in enumerate(sub_questions):
                # Retrieve relevant documents for the sub-question
                retrieved_docs = self.retriever.retrieve(sub_q, k = 10)
                steps_log.append({
                    "step": step_cnt,
                    "description": f"Retrieval for Sub-question {i+1} : *{sub_q}*",
                    "type" : "multi_hop_sub_retrieval",
                    "query": sub_q,
                    "retrieved_docs": retrieved_docs
                })
                step_cnt += 1

                # Generate an answer for the sub-question using retrieved docs
                # Pass previous sub-answers as context for chain reasoning
                doc_texts = [doc_text for (_, doc_text, _) in retrieved_docs]
                sub_answer = self.answer_from_docs(sub_q, doc_texts, prior_answers=sub_answers[:i])
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

        else:
            # If not multi-hop, run standard single-turn RAG with verification
            retrieved_docs = self.retriever.retrieve(question, k = 10)
            steps_log.append({
                "step": step_cnt,
                "description": "Standard RAG Retrieval (Single-Hop)",
                "type" : "single_hop_retrieval",
                "query": question,
                "retrieved_docs": retrieved_docs
            })

            final_answer = self.answer_with_verification(question, [doc_text for (_, doc_text, _) in retrieved_docs])
            steps_log.append({
                "step": step_cnt,
                "description": "Standard RAG Generation (Single-Hop, with verification)",
                "type" : "single_hop_generation",
                "query": question,
                "result": final_answer
            })

        return final_answer, steps_log
