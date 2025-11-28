# https://raw.githubusercontent.com/hotpotqa/hotpot/master/hotpot_evaluate_v1.py

import argparse
import json
import re
import string
import time
from collections import Counter


def read_jsonl(file_path):
    data = list()
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    print(f'[{time.asctime()}] Read {len(data)} from {file_path}')
    return data


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


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def update_answer(metrics, prediction, gold):
    em = exact_match_score(prediction, gold)
    f1, prec, recall = f1_score(prediction, gold)
    metrics['em'] += float(em)
    metrics['f1'] += f1
    metrics['prec'] += prec
    metrics['recall'] += recall
    return em, prec, recall


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
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if fp + fn == 0 else 0.0
    metrics['sp_em'] += em
    metrics['sp_f1'] += f1
    metrics['sp_prec'] += prec
    metrics['sp_recall'] += recall
    return em, prec, recall


def eval(predictions, gold_data, topk=5):
    # with open(prediction_file) as f:
    #     prediction = json.load(f)
    # with open(gold_file) as f:
    #     gold = json.load(f)

    metrics = {
        'em': 0, 'f1': 0, 'prec': 0, 'recall': 0,
        'sp_em': 0, 'sp_f1': 0, 'sp_prec': 0, 'sp_recall': 0,
        'joint_em': 0, 'joint_f1': 0, 'joint_prec': 0, 'joint_recall': 0
    }
    for dp in gold_data:
        pred = predictions[dp['id']]
        can_eval_joint = True
        if pred.get('answer', None):
            em, prec, recall = update_answer(metrics, pred['answer'], dp['answer'])
        else:
            print('missing answer {}'.format(dp['id']))
            can_eval_joint = False
        if pred.get('retrieved_docs', None):
            # Keep top k docs
            pred_sp = sorted(pred['retrieved_docs'], key=lambda x: x[1], reverse=True)[:topk]
            pred_sp = [x[0] for x in pred_sp]
            sp_em, sp_prec, sp_recall = update_sp(metrics, pred_sp, dp['supporting_ids'])
        else:
            print('missing sp fact {}'.format(dp['id']))
            can_eval_joint = False

        if can_eval_joint:
            joint_prec = prec * sp_prec
            joint_recall = recall * sp_recall
            if joint_prec + joint_recall > 0:
                joint_f1 = 2 * joint_prec * joint_recall / (joint_prec + joint_recall)
            else:
                joint_f1 = 0.
            joint_em = em * sp_em

            metrics['joint_em'] += joint_em
            metrics['joint_f1'] += joint_f1
            metrics['joint_prec'] += joint_prec
            metrics['joint_recall'] += joint_recall

    N = len(gold_data)
    for k in metrics.keys():
        metrics[k] /= N

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate Hotpot QA results.')
    parser.add_argument('--gold', '-g', type=str, required=True, help='Path to the gold file.')
    parser.add_argument('--pred', '-p', type=str, required=True, help='Path to the predicted file.')
    parser.add_argument('--topk', '-k', type=int, default=5, help='Top k docs in eval.')
    args = parser.parse_args()
    gold_data = read_jsonl(args.gold)
    pred_data = read_jsonl(args.pred)
    assert len(gold_data) == len(pred_data), "Gold and pred files must have the same number of entries."

    predictions = {i['id']: i for i in pred_data}
    metrics = eval(predictions, gold_data, args.topk)
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == '__main__':
    main()
