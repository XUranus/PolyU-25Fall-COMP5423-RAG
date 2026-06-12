---
title: QA Metrics
sidebar_position: 3
description: Understanding Exact Match, F1 Score, Supporting Facts, and Joint metrics for answer quality.
---

# QA Metrics

These metrics answer: **did the system give the right answer, and did it cite the right evidence?** RAG42 implements the standard HotpotQA evaluation protocol from `evaluate/eval_hotpotqa.py`.

## Exact Match (EM)

Exact Match checks whether the predicted answer matches the gold answer **after normalization**. It is strict -- the answer is either exactly right (1) or wrong (0).

### Normalization Steps

Before comparison, both answers go through four normalization steps in this order:

| Step | What it does | Example |
|------|-------------|---------|
| 1. Lowercase | Convert to lowercase | `"New York"` -> `"new york"` |
| 2. Remove punctuation | Strip all punctuation characters | `"new york!"` -> `"new york"` |
| 3. Remove articles | Remove `a`, `an`, `the` | `"the new york"` -> `" new york"` |
| 4. Fix whitespace | Collapse multiple spaces to one | `" new york"` -> `"new york"` |

```python title="evaluate/eval_hotpotqa.py"
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))
```

### Examples

| Prediction | Gold | After normalization | EM |
|-----------|------|-------------------|:---:|
| `"New York"` | `"new york"` | `"new york"` vs `"new york"` | 1 |
| `"the capital"` | `"capital"` | `"capital"` vs `"capital"` | 1 |
| `"Yes"` | `"No"` | `"yes"` vs `"no"` | 0 |
| `"San Francisco!"` | `"San Francisco"` | `"san francisco"` vs `"san francisco"` | 1 |

## F1 Score

F1 measures **token overlap** between prediction and gold answer. It is more forgiving than EM -- partial credit is given for partially correct answers.

### How it works

1. Normalize both answers (same steps as EM).
2. Split into tokens (words).
3. Count overlapping tokens using `Counter` intersection.
4. Compute precision, recall, and F1.

```python
precision = num_overlapping_tokens / num_prediction_tokens
recall    = num_overlapping_tokens / num_gold_tokens
f1        = 2 * precision * recall / (precision + recall)
```

### Worked Example

```
Prediction: "the capital of France is Paris"
Gold:       "Paris"

Normalized prediction tokens: ["capital", "france", "paris"]  (3 tokens)
Normalized gold tokens:       ["paris"]                        (1 token)

Overlapping tokens: ["paris"]  (1 token)

Precision = 1/3 = 0.333
Recall    = 1/1 = 1.000
F1        = 2 * 0.333 * 1.000 / (0.333 + 1.000) = 0.500
```

### Special Cases for Yes/No Answers

For `yes`, `no`, and `noanswer` responses, the F1 function returns zero unless the prediction and gold match exactly:

```python
if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
    return (0, 0, 0)  # (f1, precision, recall)
```

This prevents inflated scores from partial token overlap on binary answers.

## Supporting Facts Evaluation

In HotpotQA, each answer is supported by specific Wikipedia paragraphs (the "supporting facts"). This metric evaluates whether the retriever found those exact paragraphs.

### Set Operations

The predicted supporting facts are compared against gold supporting facts using set operations:

| Metric | Definition |
|--------|-----------|
| **TP** (True Positive) | Predicted facts that are also in gold |
| **FP** (False Positive) | Predicted facts that are NOT in gold |
| **FN** (False Negative) | Gold facts that were NOT predicted |

```python title="evaluate/eval_hotpotqa.py"
def update_sp(metrics, prediction, gold):
    cur_sp_pred = set(map(tuple, prediction))
    gold_sp_pred = set(map(tuple, gold))
    tp, fp, fn = 0, 0, 0
    for e in cur_sp_pred:
        if e in gold_sp_pred:
            tp += 1
        else:
            fp += 1
    for e in gold_sp_pred:
        if e not in cur_sp_pred:
            fn += 1
```

From these, the standard metrics are computed:

```
SP Precision = TP / (TP + FP)
SP Recall    = TP / (TP + FN)
SP F1        = 2 * SP_Precision * SP_Recall / (SP_Precision + SP_Recall)
SP EM        = 1 if FP + FN == 0 (perfect set match), else 0
```

### Worked Example

```
Gold supporting facts:      {"doc_A", "doc_B", "doc_C"}
Predicted supporting facts: {"doc_A", "doc_B", "doc_D"}

TP = 2  (doc_A, doc_B)
FP = 1  (doc_D)
FN = 1  (doc_C)

SP Precision = 2/3 = 0.667
SP Recall    = 2/3 = 0.667
SP F1        = 0.667
SP EM        = 0   (not a perfect set match)
```

## Joint Metrics

Joint metrics combine answer quality and retrieval quality. They penalize systems that get one right but not the other.

### Joint EM

```
Joint EM = Answer EM * Supporting Facts EM
```

Both must be exactly correct for Joint EM to be 1.

### Joint F1

```
Joint Precision = Answer Precision * SP Precision
Joint Recall    = Answer Recall * SP Recall
Joint F1        = 2 * Joint_Precision * Joint_Recall / (Joint_Precision + Joint_Recall)
```

### Why Joint Metrics Matter

| Scenario | Answer EM | SP EM | Joint EM |
|----------|:---------:|:-----:|:--------:|
| Right answer, right evidence | 1 | 1 | **1** |
| Right answer, wrong evidence | 1 | 0 | **0** |
| Wrong answer, right evidence | 0 | 1 | **0** |
| Wrong answer, wrong evidence | 0 | 0 | **0** |

A system that guesses correctly without finding the right documents scores 0 on Joint EM.

## Full Evaluation Code

The main evaluation loop processes every question and accumulates metrics:

```python title="evaluate/eval_hotpotqa.py"
def eval(predictions, gold_data, topk=5):
    metrics = {
        'em': 0, 'f1': 0, 'prec': 0, 'recall': 0,
        'sp_em': 0, 'sp_f1': 0, 'sp_prec': 0, 'sp_recall': 0,
        'joint_em': 0, 'joint_f1': 0, 'joint_prec': 0, 'joint_recall': 0
    }
    for dp in gold_data:
        pred = predictions[dp['id']]
        em, prec, recall = update_answer(metrics, pred['answer'], dp['answer'])
        pred_sp = sorted(pred['retrieved_docs'], key=lambda x: x[1], reverse=True)[:topk]
        pred_sp = [x[0] for x in pred_sp]
        sp_em, sp_prec, sp_recall = update_sp(metrics, pred_sp, dp['supporting_ids'])

        # Joint metrics
        joint_prec = prec * sp_prec
        joint_recall = recall * sp_recall
        joint_f1 = 2 * joint_prec * joint_recall / (joint_prec + joint_recall) if joint_prec + joint_recall > 0 else 0
        joint_em = em * sp_em
        metrics['joint_em'] += joint_em
        metrics['joint_f1'] += joint_f1

    N = len(gold_data)
    for k in metrics.keys():
        metrics[k] /= N  # Average over all questions
    return metrics
```

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
