import sys
import os

inp = sys.argv[1]
outdir = sys.argv[2]


with open(inp, 'r') as f:
    for i, line in enumerate(f):
        comment, score = line.split('\t')
        score = int(score)
        with open(os.path.join(outdir, '%d.txt'%i), 'w') as f:
            f.write(comment)

