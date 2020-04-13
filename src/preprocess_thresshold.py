import os
import sys
import tqdm
import numpy as np

inputdir = sys.argv[1]
outfile = sys.argv[2]

comments = []
labels = []
for fil in tqdm.tqdm(os.listdir(inputdir)):
    fname = os.path.join(inputdir, fil)
    fcomments = []
    fscores = []
    with open(fname, 'r') as f:
        for line in f.read().splitlines():
            print(line)
            if not line.strip():
                continue
            comment, score = line.split('\t')
            score = int(score)
            fscores.append(score)
            fcomments.append(comment)
        if len(fscores) == 1 or max(fscores) == 0:
            continue
        median = np.median(fscores)
        label = [ int(x>=median) for x in fscores]
        comments.extend(fcomments)
        labels.append(label)

with open(outfile, 'w') as f:
    for comment, label in zip(comments, labels):
        outfile.write('%s\t%d\n'%(comment, label))


        
            


