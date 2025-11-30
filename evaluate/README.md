# COMP5423 25Fall

A sampled subset, called HQ-small, for develop and evaluation, which is released at https://huggingface.co/datasets/izhx/COMP5423-25Fall-HQ-small.


#### Generation and Evaluation on `validation.jsonl`
1. Specify the `validation.jsonl` path
```bash
export RAG_VALIDATION_SET_PATH="./validation.jsonl"
```
1. Generate `test_prediction.jsonl` to evaluate.
```bash
rm -rf ./test_prediction.jsonl
python test_predict.py -d $RAG_VALIDATION_SET_PATH -g qwen2.5-7b-instruct -r hybrid # generate test_prediction.json
python eval_retrieval.py --gold $RAG_VALIDATION_SET_PATH --pred test_prediction.jsonl
python eval_hotpotqa.py --gold $RAG_VALIDATION_SET_PATH --pred test_prediction.jsonl
```

This step utilizes Aliyun Qwen API to do fast generation, you need to have the following env properly set.

```bash
RAG42_CACHE_DIR=/home/xuranus/workspace/PolyUAIBD25Fall/COMP5423/lab/RAG/PolyU-25Fall-COMP5423-RAG/backend/cache
RAG42_OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
RAG42_OPENAI_API_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

#### Result
Using model `qwen2.5-7b-instruct`

1. Sparse Retrieval (**BM25s**)
```bash
```

2. Dense Retrieval (**BAAI/bge-small-en-v1.5**)
```bash
{
  "map_at_2": 0.5385,
  "map_at_5": 0.6542833333333333,
  "map_at_10": 0.6714460317460317,
  "ndcg_at_2": 0.6088105533796958,
  "ndcg_at_5": 0.7373518860505552,
  "ndcg_at_10": 0.7640084897735142,
  "recall_at_2": 0.572,
  "recall_at_5": 0.7993333333333333,
  "recall_at_10": 0.866,
  "precision_at_2": 0.572,
  "precision_at_5": 0.31973333333333337,
  "precision_at_10": 0.17320000000000002
}
{
  "em": 0.33466666666666667,
  "f1": 0.4700240510637789,
  "prec": 0.4612523861763853,
  "recall": 0.5703745791245788,
  "sp_em": 0.0,
  "sp_f1": 0.4565714285714254,
  "sp_prec": 0.3195999999999921,
  "sp_recall": 0.799,
  "joint_em": 0.0,
  "joint_f1": 0.2286205844338048,
  "joint_prec": 0.15981681677792353,
  "joint_recall": 0.49603922558922536
}
```

3. Hybrid (weighted **BM25s** + **BAAI/bge-small-en-v1.5**)
```bash
{
  "map_at_2": 0.5383333333333333,
  "map_at_5": 0.6538722222222221,
  "map_at_10": 0.6703305555555555,
  "ndcg_at_2": 0.6085088275323213,
  "ndcg_at_5": 0.7368920283946678,
  "ndcg_at_10": 0.7625980595311833,
  "recall_at_2": 0.572,
  "recall_at_5": 0.799,
  "recall_at_10": 0.8633333333333333,
  "precision_at_2": 0.572,
  "precision_at_5": 0.3196,
  "precision_at_10": 0.17266666666666666
}
{
  "em": 0.3333333333333333,
  "f1": 0.4663219990563274,
  "prec": 0.4584881085471521,
  "recall": 0.5612420394420391,
  "sp_em": 0.0,
  "sp_f1": 0.45695238095237783,
  "sp_prec": 0.31986666666665875,
  "sp_recall": 0.7996666666666666,
  "joint_em": 0.0,
  "joint_f1": 0.2281344848166965,
  "joint_prec": 0.1596235705101507,
  "joint_recall": 0.4900559283309282
}
```