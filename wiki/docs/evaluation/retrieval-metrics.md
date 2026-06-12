---
title: Retrieval Metrics
sidebar_position: 2
description: Understanding nDCG, MAP, Recall@k, and Precision@k for measuring retrieval quality.
---

# Retrieval Metrics

These metrics answer one question: **did the retriever find the right documents?** RAG42 uses the `pytrec_eval` library to compute all four metrics efficiently.

## nDCG@k (Normalized Discounted Cumulative Gain)

nDCG measures how well the retriever **ranks** relevant documents. A relevant document at position 1 is worth more than the same document at position 10.

### Step 1: DCG (Discounted Cumulative Gain)

For each position `i` in the ranked results, the relevance score is discounted by a logarithmic factor:

`DCG@k = SUM_i rel_i / log2(i + 1)`

| Position (i) | Discount factor: 1 / log2(i+1) |
|:---:|:---:|
| 1 | 1.000 |
| 2 | 0.631 |
| 3 | 0.500 |
| 4 | 0.431 |
| 5 | 0.387 |
| 10 | 0.290 |

### Step 2: IDCG (Ideal DCG)

IDCG is the DCG you would get if all relevant documents were ranked at the top. This is the best possible ranking.

### Step 3: Normalize

`nDCG@k = DCG@k / IDCG@k`

The result is always between 0.0 and 1.0.

### Worked Example

Suppose there are 3 relevant documents (A, B, C) and the retriever returns them at positions 1, 4, and 8 (with irrelevant documents in between):

```
DCG@10  = 1/1.000 + 1/1.322 + 0/1.585 + 1/1.807 + ... + 1/2.807 + ...
        = 1.000 + 0.631 + 0 + 0.431 + 0 + 0 + 0 + 0.290 + 0 + 0
        = 2.352

IDCG@10 = 1/1.000 + 1/1.322 + 1/1.585 + 0 + ...    (all 3 at top)
        = 1.000 + 0.631 + 0.500
        = 2.131
```

Wait -- DCG > IDCG? That cannot happen. Let me recalculate with proper log values:

```
Position 1:  1 / log2(2)  = 1 / 1.000 = 1.000
Position 4:  1 / log2(5)  = 1 / 2.322 = 0.431
Position 8:  1 / log2(9)  = 1 / 3.170 = 0.315

DCG@10  = 1.000 + 0.431 + 0.315 = 1.746
IDCG@10 = 1.000 + 0.631 + 0.500 = 2.131
nDCG@10 = 1.746 / 2.131 = 0.819
```

An nDCG of 0.819 is quite good -- the relevant documents are mostly near the top.

## MAP (Mean Average Precision)

MAP evaluates the **complete ranking** of all relevant documents, not just the top-k.

### Precision@k

Precision at each position tells you what fraction of documents seen so far are relevant:

`Precision@k = (relevant docs in top k) / k`

### Average Precision (AP)

Average Precision averages the Precision@k values at every position where a relevant document appears:

`AP = (1/R) * SUM_k Precision@k * rel_k`

where `R` is the total number of relevant documents and `rel_k` is 1 if the document at position k is relevant, 0 otherwise.

### Worked Example

Suppose there are 3 relevant documents (A, B, C) out of 10 retrieved. They appear at positions 1, 3, and 7:

| Position | Doc | Relevant? | Precision@k |
|:---:|:---:|:---:|:---:|
| 1 | A | Yes | 1/1 = 1.000 |
| 2 | -- | No | -- |
| 3 | B | Yes | 2/3 = 0.667 |
| 4-6 | -- | No | -- |
| 7 | C | Yes | 3/7 = 0.429 |

```
AP = (1.000 + 0.667 + 0.429) / 3 = 0.699
```

MAP is the AP averaged over all queries in the dataset.

## Recall@k

Recall@k measures what **fraction of all relevant documents** appear in the top-k results:

`Recall@k = (relevant docs in top k) / (total relevant docs)`

If there are 4 relevant documents and k=10 retrieves 3 of them, Recall@10 = 0.75.

:::info
Recall@k is especially important in RAG because if the retriever misses a supporting document, the generator has no chance of producing a correct answer.
:::

## Precision@k

Precision@k measures what **fraction of the top-k retrieved documents** are actually relevant:

`Precision@k = (relevant docs in top k) / k`

If k=10 and 6 of the retrieved documents are relevant, Precision@10 = 0.60.

## Implementation in RAG42

The retrieval evaluation script (`evaluate/eval_retrieval.py`) uses the `pytrec_eval` library for efficient computation:

```python title="evaluate/eval_retrieval.py"
import pytrec_eval
import pandas as pd

def compute_metrics(qrels, results, k_values=[2, 5, 10]):
    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])

    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels, {map_string, ndcg_string, recall_string, precision_string}
    )
    scores_by_query = evaluator.evaluate(results)
    scores = pd.DataFrame.from_dict(scores_by_query.values()).mean()

    metrics = dict()
    for prefix in ('map_cut', 'ndcg_cut', 'recall', 'P'):
        name = 'precision' if prefix == 'P' else prefix.split('_')[0]
        for k in k_values:
            metrics[f'{name}_at_{k}'] = scores[f'{prefix}_{k}']
    return metrics
```

### How qrels and results are built

The gold file provides ground-truth supporting document IDs. The prediction file provides retrieved documents with scores:

```python
# Ground truth: each query maps to its relevant document IDs (relevance = 1)
qrels = {i['id']: {d: 1 for d in i['supporting_ids']} for i in gold_data}

# Predictions: each query maps to (doc_id, score) pairs
results = {i['id']: {d: s for d, s in i['retrieved_docs']} for i in pred_data}
```

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

:::note
`pytrec_eval` is a Python binding for the trec_eval tool used in TREC competitions. Install it with `pip install pytrec_eval`.
:::
