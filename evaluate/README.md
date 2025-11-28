# COMP5423 25Fall

A sampled subset, called HQ-small, for develop and evaluation, which is released at https://huggingface.co/datasets/izhx/COMP5423-25Fall-HQ-small.


#### Generation and Evaluation on `validation.jsonl`
1. Specify the `validation.jsonl` path
```bash
export RAG_VALIDATION_SET_PATH="../../COMP5423-25Fall-HQ-small/validation.jsonl"
```
1. Generate `test_prediction.jsonl` to evaluate.
```bash
python test_predict.py -d $RAG_VALIDATION_SET_PATH$
python eval_retrieval.py --gold $RAG_VALIDATION_SET_PATH --pred test_prediction.jsonl
1python eval_hotpotqa.py --gold $RAG_VALIDATION_SET_PATH --pred test_prediction.jsonl
```

This step utilizes Aliyun Qwen API to do fast generation, you need to have the following env properly set.

```bash
RAG42_CACHE_DIR=/home/xuranus/workspace/PolyUAIBD25Fall/COMP5423/lab/RAG/PolyU-25Fall-COMP5423-RAG/backend/cache
RAG42_OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
RAG42_OPENAI_API_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```