import os
import re
from tabulate import tabulate
from sklearn.metrics import cohen_kappa_score
import numpy as np


gt_file = 'gt.txt'
annotated = 'to_annotate.txt'

with open(gt_file, 'r') as f:
    gt_labels = [int(x) for x in f]

with open(annotated, 'r') as f:
    annotations = [int(x.split('\t')[1]) for x in f]

print(cohen_kappa_score(gt_labels, annotations))

