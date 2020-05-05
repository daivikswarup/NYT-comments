import os
import re
from tabulate import tabulate
from sklearn.metrics import cohen_kappa_score
import numpy as np


gt_dir = 'selected/gt'
manual_dir = 'selected/manual'
P = 5

def rand_ndcg(scores):
    scores = np.array(scores)/np.max(scores)
    rels = np.exp2(scores) -1
    sum_weights = np.sum(1/np.log2(np.arange(min(P,len(scores)))+2))
    rdcg = np.mean(rels) * sum_weights
    idcg = dcg(scores, np.argsort(scores)[::-1])
    return rdcg/idcg

def dcg(scores, ranks):
    scores = np.array(scores)/np.max(scores)
    dcg = 0
    for i in range(min(len(scores), P)):
        s = scores[ranks[i]]
        dcg += (np.exp2(s)-1)/np.log2(i+2)
    return dcg

def ndcg(gt_score, preds):
    gt_score = np.array(gt_score) / np.max(gt_score)
    best_order = np.argsort(gt_score)[::-1]
    predicted_order = np.argsort(preds)[::-1]
    return dcg(gt_score,predicted_order)/dcg(gt_score, best_order)



def get_score(fname):
    scores = []
    with open(fname, 'r') as f:
        for line in f:
            scores.append(int(line.split('\t')[1]))
    return scores

def get_comments(fname):
    scores = []
    with open(fname, 'r') as f:
        for line in f:
            scores.append(line.split('\t')[0])
    return scores

def get_target_predictions(gt, pred):
    targets = []
    predictions = []
    def get_label(a,b):
        if a>b:
            return 1
        elif a<b:
            return -1
        else:
            return 0

    for i in range(len(gt)):
        for j in range(i+1, len(gt)):
            if gt[i] == gt[j] or pred[i] == pred[j]:
                continue # ignore ties
            targets.append(get_label(gt[i], gt[j]))
            predictions.append(get_label(pred[i], pred[j]))
    return targets, predictions




all_targets = []
all_preds = []

per_file_evals = []

for fil in os.listdir(gt_dir):
    f_gt = os.path.join(gt_dir, fil)
    f_manual = os.path.join(manual_dir, fil)

    gt_scores = get_score(f_gt)
    manual_scores = get_score(f_manual)
    print(gt_scores)
    print(manual_scores)
    if max(gt_scores) == 0:
        continue
    combined_table = list(zip(get_comments(f_gt), gt_scores, manual_scores))
    header = ['Comment', 'Ground Truth', 'Manually assigned score(0-4)']
    table = re.sub(' +', ' ',tabulate(combined_table, header,
                                      tablefmt='latex'))
    print(table)
    flat_targets, flat_predictions = get_target_predictions(gt_scores,
                                                            manual_scores)
    all_targets.extend(flat_targets)
    all_preds.extend(flat_predictions)
    file_kappa = cohen_kappa_score(flat_targets, flat_predictions)
    file_ndcg = ndcg(gt_scores, manual_scores)
    rnd_ndcg = rand_ndcg(gt_scores)
    per_file_evals.append((file_kappa, file_ndcg, rnd_ndcg))

print(len(all_targets))
print(cohen_kappa_score(all_targets, all_preds))

print(per_file_evals)
print('NDCG = %f'%np.mean([x[1] for x in per_file_evals]))
print('Random NDCG = %f'%np.mean([x[2] for x in per_file_evals]))


