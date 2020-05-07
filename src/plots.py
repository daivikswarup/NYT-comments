import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
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
        for row in tqdm.tqdm(data):
            all_data[row['articleID']].append((clean(row['commentBody']),
                                               row['recommendations']))
    data = []
    for key in all_data:
        comments = all_data[key]
        _,reccommends = zip(*comments)
        #mx = np.max(reccommends)
        mx = 1
        argsort = np.argsort(reccommends)
        sorted_comments = [(comments[i][0], comments[i][1]/mx) for i in argsort[::-1]]
        data.extend(sorted_comments)
    return data

def writedir(split, data, keys, datadir='../data'):
    folder = os.path.join(datadir, split)
    for key in tqdm.tqdm(keys):
        fname = os.path.join(folder, key + '.txt')
        with open(fname, 'w') as f:
            for comment, score in data[key]:
                f.write('%s\t%d\n'%(comment, score))



data = read_data()



comments, reccomendations = zip(*data)
print('fsfafd')
print('affdsf')
print(len([x for x in reccomendations if x > 0]) / len(reccomendations))
cutoff = np.percentile(reccomendations, 90)
print(cutoff)
recommendations = [x for x in reccomendations if x >0 and x < cutoff]
plt.hist(reccomendations, bins=100)
plt.xlabel('number of recommendations')
plt.ylabel('number of comments')
plt.show()

wlen = [len(re.split('\s+', x)) for x in tqdm.tqdm(comments)] 
#slen = [len(sent_tokenize(x)) for x in tqdm.tqdm(comments)]

#print(np.mean(slen))
#print(np.median(slen))

plt.hist(wlen, bins=100)
plt.xlabel('number of words')
plt.ylabel('number of comments')
plt.show()
plt.hist(slen, bins=100)

plt.xlabel('number of sentences')
plt.ylabel('number of comments')
plt.show()


