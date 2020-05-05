# takes train_80_20
# randomly selects N rows
# writes to_annotate.txt, gt.txt
import sys
import os
import numpy as np

inp = sys.argv[1]
N = int(sys.argv[2])

data = []
with open(inp, 'r') as f:
    lines = f.read().splitlines()

for l in lines:
    comment, clas = l.split('\t')
    clas = int(clas)
    data.append((comment, clas))

selected_data = np.random.choice(len(data), N, replace=False)
data_sample = [data[i] for i in selected_data]

with open('to_annotate.txt', 'w') as f:
    f.write('\n'.join([x[0] for x in data_sample]))

with open('gt.txt', 'w') as f:
    f.write('\n'.join([str(x[1]) for x in data_sample]))

