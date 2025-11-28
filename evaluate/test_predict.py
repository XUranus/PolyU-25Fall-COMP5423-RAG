from typing import List, Dict, Optional, Tuple, Any
import json
import time
import argparse
from tqdm import tqdm
import os
import traceback
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.append(os.path.join(os.path.dirname(__file__), '..', "backend"))

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

    from rag_pipeline import RAGPipeline
    from hybrid_retriever import HybridRetriever
    my_retriever = HybridRetriever(collection_path="izhx/COMP5423-25Fall-HQ-small")
    rag = RAGPipeline(retriever=my_retriever)

    predict_list = []

    def task(data):
        try:
            id = data['id']
            question = data['text']
            response = rag.run(query = question, model_name = "qwen-turbo")
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
            print(retrieved_docs)
            print("\n\n")

    with ThreadPoolExecutor(max_workers = 10) as executor:
        results = list(executor.map(task, dataset))
        with open("./test_prediction.jsonl", "a") as f:
            for result in results:
                f.write(result + "\n")


if __name__ == '__main__':
    main()

# python test_predict.py --validation

