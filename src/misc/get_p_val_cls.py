import pickle
import sys
import numpy as np
import tqdm

def resample(arr):
    n = len(arr)
    return arr[np.random.choice(n,n),:]

b = 10000
f1 = sys.argv[1]
f2 = sys.argv[2]
gtf = sys.argv[3]

with open(f1, 'rb') as f:
    d1 = np.array(pickle.load(f))


with open(f2, 'rb') as f:
    d2 = np.array(pickle.load(f))
with open(gtf, 'rb') as f:
    gt = np.array(pickle.load(f))

acc1 = d1 == gt
acc2 = d2 == gt
   
cat = np.stack([acc1, acc2], axis=1)

count_null = 0
for i in tqdm.trange(b):
    rscat = resample(cat)
    if np.mean(rscat[:,0]) < np.mean(rscat[:,1]):
        count_null += 1

print("p = %f"%(count_null/b))
