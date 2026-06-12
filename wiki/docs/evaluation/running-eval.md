---
title: Running Evaluations
sidebar_position: 4
description: Step-by-step guide to running the RAG42 evaluation pipeline.
---

# Running Evaluations

This guide walks you through the three-step evaluation pipeline: generate predictions, evaluate retrieval, and evaluate QA.

## Prerequisites

Before running evaluations, make sure you have:

- A test dataset in JSONL format (e.g., `hotpotqa_dev.jsonl`)
- Required Python packages installed: `pytrec_eval`, `pandas`, `tqdm`
- Environment variables set:
  - `RAG42_CACHE_DIR` -- path to cached BM25 indices and models
  - `RAG42_OPENAI_API_KEY` -- API key (if using OpenAI-compatible generators)
  - `RAG42_OPENAI_API_URL` -- API endpoint URL

```bash
pip install pytrec_eval pandas tqdm
```

## Step 1: Generate Predictions

Run `test_predict.py` to execute the RAG pipeline on every question in the test dataset. This produces a JSONL file with predictions.

```bash
cd evaluate

python test_predict.py \
    --dataset ../data/hotpotqa_dev.jsonl \
    --generator qwen2.5-7b-instruct \
    --retriever hybrid
```

### Arguments

| Argument | Description | Options |
|----------|-------------|---------|
| `--dataset`, `-d` | Path to the test JSONL file | Any `.jsonl` path |
| `--generator`, `-g` | LLM to use for answer generation | `qwen2.5-0.5b-instruct`, `qwen2.5-1.5b-instruct`, `qwen2.5-3b-instruct`, `qwen2.5-7b-instruct` |
| `--retriever`, `-r` | Retriever type | `sparse`, `static_embedding`, `dense`, `instruction`, `colbert`, `hybrid` |

### What Happens

1. The script loads the dataset and initializes the RAG pipeline.
2. It uses a `ThreadPoolExecutor` with 4 workers to run predictions concurrently.
3. For each question, it collects retrieved documents from all reasoning steps (including sub-questions in agentic mode), keeping the highest score per document.
4. Results are written to `./test_prediction.jsonl`.

:::warning
The generator is initialized once before the thread pool starts to avoid thread-safety issues. Do not modify this behavior.
:::

### Output Format

Each line in `test_prediction.jsonl` is a JSON object:

```json
{
  "id": "hotpot_train_00001",
  "question": "When was the British financial analyst born?",
  "answer": "1952",
  "retrieved_docs": [
    ["doc_id_1", 0.892],
    ["doc_id_2", 0.756],
    ["doc_id_3", 0.621]
  ]
}
```

## Step 2: Evaluate Retrieval

Run `eval_retrieval.py` to compute retrieval metrics (nDCG, MAP, Recall@k, Precision@k).

```bash
python eval_retrieval.py \
    --gold ../data/hotpotqa_dev.jsonl \
    --pred ./test_prediction.jsonl
```

### Arguments

| Argument | Description |
|----------|-------------|
| `--gold`, `-g` | Path to the gold standard JSONL file (must contain `supporting_ids`) |
| `--pred`, `-p` | Path to the prediction JSONL file from Step 1 |

### Example Output

```json
{
  "map_at_2": 0.312,
  "map_at_5": 0.387,
  "map_at_10": 0.412,
  "ndcg_at_2": 0.358,
  "ndcg_at_5": 0.431,
  "ndcg_at_10": 0.467,
  "recall_at_2": 0.298,
  "recall_at_5": 0.512,
  "recall_at_10": 0.689,
  "precision_at_2": 0.341,
  "precision_at_5": 0.218,
  "precision_at_10": 0.142
}
```

## Step 3: Evaluate QA

Run `eval_hotpotqa.py` to compute answer metrics (EM, F1) and joint metrics.

```bash
python eval_hotpotqa.py \
    --gold ../data/hotpotqa_dev.jsonl \
    --pred ./test_prediction.jsonl \
    --topk 5
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--gold`, `-g` | Path to the gold standard JSONL file | (required) |
| `--pred`, `-p` | Path to the prediction JSONL file from Step 1 | (required) |
| `--topk`, `-k` | Number of top retrieved docs to use for supporting facts evaluation | `5` |

### Example Output

```json
{
  "em": 0.412,
  "f1": 0.538,
  "prec": 0.591,
  "recall": 0.502,
  "sp_em": 0.287,
  "sp_f1": 0.456,
  "sp_prec": 0.512,
  "sp_recall": 0.418,
  "joint_em": 0.156,
  "joint_f1": 0.312,
  "joint_prec": 0.367,
  "joint_recall": 0.278
}
```

## Diagnosing Issues

Use this table to interpret mismatched scores and identify where to focus your improvements:

| Retrieval | QA | Likely Problem | Where to Look |
|-----------|-----|---------------|---------------|
| High nDCG, High F1 | High EM | Everything works well | Tune for efficiency |
| High nDCG, Low F1 | Low EM | **Poor generation** -- retriever finds docs but LLM ignores or misinterprets them | Generator prompt, model choice, context window |
| Low nDCG, High F1 | High EM | Lucky guesses -- answer happens to be simple | Test on harder questions |
| Low nDCG, Low F1 | Low EM | **Poor retrieval** -- wrong documents sent to generator | Retriever type, embedding model, BM25 index |
| High SP Precision | Low SP Recall | Retriever is conservative -- returns few but correct docs | Increase top-k, lower score threshold |
| Low SP Precision | High SP Recall | Retriever returns many docs including noise | Improve re-ranking, reduce top-k |
| High Answer F1 | Low SP F1 | Answer is right but for the wrong reasons | Review supporting fact alignment |
| High Joint F1 | Low Joint EM | Close but not exact answers | Focus on answer normalization and precision |

## Tips for Improving Scores

### Retrieval

- **Try hybrid retrieval** -- combining BM25 (sparse) with dense embeddings often outperforms either alone.
- **Increase top-k** -- retrieving more documents gives the generator more context, at the cost of more noise.
- **Use a reranker** -- a cross-encoder reranker can significantly improve the precision of top results.

### Generation

- **Use a larger model** -- `qwen2.5-7b-instruct` typically outperforms `qwen2.5-0.5b-instruct` on complex questions.
- **Improve prompts** -- clearer instructions help the generator focus on relevant context.
- **Enable agentic workflow** -- multi-step reasoning with sub-questions can decompose complex questions.

### General

- **Cache your BM25 index** -- building the index from scratch takes 3+ hours. The `build.sh` script downloads a pre-built cache.
- **Run on a subset first** -- test your pipeline on 50-100 questions before running the full dataset.
- **Compare against baselines** -- track your scores over time to measure improvement.
