import pickle
import sys
import numpy as np
import tqdm

def resample(lis):
    n = len(lis)
    return [lis[np.random.randint(n)] for i in range(n)]

b = 10000
f1 = sys.argv[1]
f2 = sys.argv[2]

with open(f1, 'rb') as f:
    d1 = pickle.load(f)


with open(f2, 'rb') as f:
    d2 = pickle.load(f)
    
count_null = 0
for i in tqdm.trange(b):
    lis = resample(list(zip(d1,d2)))
    s1, s2 = zip(*lis)
    if np.mean(s1) < np.mean(s2):
        count_null += 1

print("p = %f"%(count_null/b))
