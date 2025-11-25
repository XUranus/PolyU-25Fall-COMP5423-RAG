from typing import List, Dict, Optional, Tuple, Any
import logging
import sys

from hybrid_retriever import HybridRetriever
from qwen_generator import QwenGenerator
from rag_pipeline import RAGPipeline




my_retriever = HybridRetriever(collection_path="izhx/COMP5423-25Fall-HQ-small")
my_generator = QwenGenerator(model_name="Qwen/Qwen2.5-0.5B-Instruct")
rag = RAGPipeline(retriever=my_retriever, generator=my_generator)


print("Running Agentic Workflow for complex multi-hop question...")
query = "The second place finisher of the 2011 Gran Premio Santander d'Italia drove for who when he won the 2009 FIA Formula One World Championship?"
response = rag.run(query)
final_ans = response['answer']
thinking_process = response['thinking_process']

print("\n--- Final Answer ---")
print(final_ans)
print(thinking_process)


# Example single-hop question
print("\n\nRunning Agentic Workflow for Simple Question...")
query = "Who won the 2009 FIA Formula One World Championship?"
response = rag.run(query)
final_ans = response['answer']
thinking_process = response['thinking_process']

print("\n--- Final Answer ---")
print(final_ans)
print(thinking_process)