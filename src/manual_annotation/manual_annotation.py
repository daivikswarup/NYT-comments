import sys
import os
import numpy as np

N = 5

def read_file(data):
    comments = []
    for l in data:
        c, s = l.split('\t')
        s = int(s)
        comments.append((c,s))
    # Shuffle
    perm = np.random.permutation(len(comments))
    shuffled_comments = [comments[i] for i in perm]
    return shuffled_comments


fmap = {}
for fil in os.listdir('../data/val/'):
    fname = os.path.join('../data/val/', fil)
    with open(fname, 'r') as f:
        data = f.read().splitlines()
        fdata = read_file(data)
        if len(fdata) < 3 or len(fdata) > 5:
            continue
        if max([x[1] for x in fdata]) == 0:
            continue
        fmap[fil] = fdata 

files = list(fmap.keys())
selected = [files[i] for i in np.random.choice(len(files), N)]

for fil in selected:
    with open('selected/gt/%s'%fil, 'w') as f:
        f.write('\n'.join(['%s\t%d'%(c, s) for c,s in fmap[fil]]))
    with open('selected/manual/%s'%fil, 'w') as f:
        f.write('\n'.join(['%s\t'%c for c,s in fmap[fil]]))

