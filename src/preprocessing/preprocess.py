import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import tqdm
import re
from collections import defaultdict
np.random.seed(0)

def clean(comment):
    html_pattern = re.compile('<.*?>')
    text = re.sub('[\t\n\r]', ' ',re.sub(html_pattern, ' ', comment))
    asci = re.sub(r'[^\x00-\x7F]+',' ', text) 
    return asci

def read_data(datadir= '../data'):
    all_data = defaultdict(list)
    for fil in tqdm.tqdm(os.listdir(datadir)):
        fname = os.path.join(datadir, fil)
        if not fil.startswith('Comments'):
            continue
        data = pd.read_csv(fname).to_dict('records')
        lis = []
        for row in tqdm.tqdm(data):
            if int(row['depth']) != 1:
                continue
            all_data[row['articleID']].append((clean(row['commentBody']),
                                               row['recommendations']))
    sorted_data = {}
    for key in all_data:
        comments = all_data[key]
        _,reccommends = zip(*comments)
        if max(reccommends) == 0:
            # remove articles with insufficient data
            continue
        argsort = np.argsort(reccommends)
        sorted_comments = [comments[i] for i in argsort[::-1]]
        sorted_data[key] = sorted_comments
    return sorted_data

def writedir(split, data, keys, datadir='../data'):
    folder = os.path.join(datadir, split)
    for key in tqdm.tqdm(keys):
        fname = os.path.join(folder, key + '.txt')
        with open(fname, 'w') as f:
            for comment, score in data[key]:
                f.write('%s\t%d\n'%(comment, score))



data = read_data()

articles = list(data.keys())
np.random.shuffle(articles)

train_split = 0.6
val_split = 0.2
test_split = 0.2

train_len = int(train_split * len(articles))
val_len = int(val_split * len(articles))
test_len = len(articles) - train_len - val_len

training_data = articles[:train_len]
val_data = articles[train_len:train_len+val_len]
test_data = articles[-test_len:]

writedir('train', data, training_data)
writedir('val', data, val_data)
writedir('test', data, test_data)





# comments, reccomendations = zip(*data)
# nonzero = [x for x in reccomendations if x !=0]
# print(len(nonzero)/len(data))
# # print(np.median(reccomendations))
# plt.hist(nonzero, bins=10)
# plt.show()



