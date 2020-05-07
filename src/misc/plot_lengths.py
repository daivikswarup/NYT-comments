import matplotlib.pyplot as plt
import sys
import numpy as np


fname = sys.argv[1]


with open(sys.argv[1], 'r') as f:
    data = f.readlines()

comms_pos = [len(x.split('\t')[0].split()) for x in data if int(x.split('\t')[1])==1]
comms_neg = [len(x.split('\t')[0].split()) for x in data if \
           int(x.split('\t')[1])==0]

print(np.mean(comms_pos))
print(np.mean(comms_neg))


# comms_pos = [len(x)/max(1,len(x.split('\t')[0].split())) for x in data if int(x.split('\t')[1])==1]
# comms_neg = [len(x)/max(1,len(x.split('\t')[0].split())) for x in data if \
#            int(x.split('\t')[1])==0]




plt.hist(comms_pos, bins=100, alpha=0.5, label='Constructive', density=True)
plt.hist(comms_neg, bins=100, alpha=0.5, label='Not Constructive', density=True)
plt.legend()
plt.title('Lengths of constructive and non constructive comments')
plt.xlabel('Number of Words')
plt.ylabel('Fraction of Comments')
plt.show()

