import time

now = time.time()

from rag_pipeline import RAGPipeline
from hybrid_retriever import HybridRetriever
from qwen_generator import QwenGenerator


print(f'import take {time.time() - now}')
now = time.time()

user_message = "Are Geoff Masters and Jimmy Connors both former tennis players?"
# --- Initialize RAG Components ---
# Load collection from HuggingFace (or path to local data)
retriever = HybridRetriever(collection_path="izhx/COMP5423-25Fall-HQ-small")
generator = QwenGenerator(model_name="Qwen/Qwen2.5-0.5B-Instruct")
rag_pipeline = RAGPipeline(retriever=retriever, generator=generator)


print(f'init take {time.time() - now}')
now = time.time()


# --- Run RAG Pipeline ---
# Note: For now, session_history is passed as None.
# You can retrieve it from the DB later if implementing multi-turn.
rag_result = rag_pipeline.run(query=user_message, session_history=None)


print(f'query take {time.time() - now}')
now = time.time()

bot_response = rag_result["answer"]
retrieved_docs = rag_result["retrieved_docs"] # [(id, text, score), ...]
thinking_process = rag_result["thinking_process"]


print("\n\n")
print(bot_response)

print("\n\n")
print(retrieved_docs)

print("\n\n")
print(thinking_process)
