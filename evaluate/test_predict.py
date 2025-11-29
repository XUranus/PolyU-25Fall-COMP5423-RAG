from typing import List, Dict, Optional, Tuple, Any
import json
import time
import argparse
from tqdm import tqdm
import os
import traceback
import sys
import logging
from concurrent.futures import ThreadPoolExecutor


APPNAME = "RAG42"

def setup_logger(name, log_file, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Avoid adding handlers multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] [%(message)s]')
    file_handler.setFormatter(file_format)

    logger.addHandler(file_handler)
    return logger

RAG42_STORAGE_DIR = os.getenv('RAG42_STORAGE_DIR', '.')
LOGGER_PATH = os.path.join(RAG42_STORAGE_DIR, 'rag.log')
logger = setup_logger(APPNAME, LOGGER_PATH, level=logging.INFO)

sys.path.append(os.path.join(os.path.dirname(__file__), '..', "backend"))
from rag_pipeline import RAGPipeline
from hybrid_retriever import HybridRetriever

MODEL_NAME = "qwen2.5-7b-instruct"

my_retriever = HybridRetriever(
    collection_path = "izhx/COMP5423-25Fall-HQ-small",
    sparse_model_name = "bm25s",
    dense_model_name = "BAAI/bge-small-en-v1.5", # "Qwen/Qwen3-Embedding-4B",
    use_cache = True,
    cache_dir = os.getenv('RAG42_CACHE_DIR', './cache'),
)

def read_jsonl(file_path):
    data = list()
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    print(f'[{time.asctime()}] Read {len(data)} from {file_path}')
    return data


def main():
    parser = argparse.ArgumentParser(description='Evaluate retrieval results.')
    parser.add_argument('--dataset', '-d', type=str, required=True, help='Path to the jsonl file test predict on.')
    args = parser.parse_args()
    dataset_path = args.dataset
    print(f'using dataset: {dataset_path}')
    dataset = read_jsonl(dataset_path)

    RAG42_CACHE_DIR = os.getenv('RAG42_CACHE_DIR')
    print(f'RAG42_CACHE_DIR = {RAG42_CACHE_DIR}')
    rag = RAGPipeline(retriever=my_retriever, enable_agentic_workflow = True)

    def task(data):
        try:
            id = data['id']
            question = data['text']
            response = rag.run(query = question, model_name = MODEL_NAME)
            find_docs = []
            answer = response["answer"]
            thinking_process = response["thinking_process"]
            for step in thinking_process:
                if "retrieved_docs" in step:
                    find_docs.extend(step["retrieved_docs"])
            
            seen_ids = set()
            retrieved_docs = []
            for doc in find_docs:
                if doc['id'] not in seen_ids:
                    seen_ids.add(doc['id'])
                    retrieved_docs.append([doc['id'], doc['score']])

            predict = {
                'id' : id,
                'question' : question,
                'answer' : answer,
                'retrieved_docs' : retrieved_docs
            }
            return json.dumps(predict)
        except Exception as e:
            traceback.print_exc()
            print(f"{e}\n\n")

    now = time.time()
    with ThreadPoolExecutor(max_workers = 20) as executor:
        results = list(executor.map(task, dataset))
        with open("./test_prediction.jsonl", "a") as f:
            for result in results:
                f.write(result + "\n")
    print(f'time elasped {time.time() - now}')


if __name__ == '__main__':
    print(f"RAG42_CACHE_DIR = {os.getenv('RAG42_CACHE_DIR')}")
    print(f"RAG42_OPENAI_API_KEY = {os.getenv('RAG42_OPENAI_API_KEY')}")
    print(f"RAG42_OPENAI_API_URL = {os.getenv('RAG42_OPENAI_API_URL')}")
    main()

    # rag = RAGPipeline(retriever=my_retriever, enable_agentic_workflow=False)
    # response = rag.run(query = "When was the British financial analyst to whom Sophie Winkleman was a wife born?", model_name = "qwen-turbo")
    # print(response)



# python test_predict.py --validation

