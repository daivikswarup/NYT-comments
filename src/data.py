"""
Preprocessing modules for various components
Can be cleaned up
"""


import numpy as np
import os
import tqdm
import torch
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.tokenize import word_tokenize
from collections import Counter, defaultdict
from settings import *
import tqdm
from scipy.special import logsumexp
from transformers import *
import numpy as np


def logsigmoid(x):
    mx = np.maximum(-x, 0)
    return -(mx + np.log(np.exp(-mx) + np.exp(-x-mx))) 

def sigmoid(x):
    return 1/(1+np.exp(-x))


class BertVectorizer:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.model.eval()
        self.model.cuda()
    
    def fit(self, x):
        pass

    def transform(self, comments):
        encoded = [torch.tensor([self.tokenizer.encode(x)], device='cuda') for \
                   x in tqdm.tqdm(comments, desc='encoding')]
        vectors = []
        for enc in tqdm.tqdm(encoded):
            output = self.model(torch.tensor(enc,
                                             device=DEVICE))
            vectors.append(output[0][0,0,:].cpu().detach().numpy())
            del output
        return np.stack(vectors)




class Vocab:
    def __init__(self, comments, vocab_size):
        wordcounts = Counter()
        for comment in tqdm.tqdm(comments):
            for word in comment:
                wordcounts[word] += 1
        most_common = wordcounts.most_common(vocab_size)
        vocab = [x[0] for x in most_common] + [UNK_TOKEN]
        self.unk_id = vocab_size
        self.pad_id = vocab_size + 1
        self.vocab_size = vocab_size
        vocab_dict = defaultdict(lambda:self.unk_id)
        for i, word in enumerate(vocab):
            vocab_dict[word] = i
        self.vocab = vocab
        self.vocab_dict = vocab_dict

    def encode(self, comments):
        """Takes list of strings, returns list of np arrays, lengths"""
        encoded_comments = [np.asarray([self.vocab_dict[w] for w in \
                                             comment]) for comment in \
                                 tqdm.tqdm(comments)]
        lengths = np.array([len(com) for com in
                                 tqdm.tqdm(encoded_comments)])
        return encoded_comments, lengths


    def get_glove(self, path = 'glove.6B.200d.txt'):
        vectors = np.random.randn(self.vocab_size +2, 200)/20
        with open(path, 'r') as f:
            for l in f:
                spl = l.split()
                word = spl[0]
                if word in self.vocab_dict:
                    vectors[self.vocab_dict[word],:] = np.array([float(val) for \
                                                                 val in spl[1:]])
        return torch.tensor(vectors, device=DEVICE)


    
        



class dataset:
    def __init__(self, fil, vectorizer=None):
        self.comments = []
        self.labels = []
        with open(fil, 'r') as f:
            for line in f.readlines():
                comment, label = line.split('\t')
                self.comments.append(comment)
                self.labels.append(int(label))
        if vectorizer is None:
            # vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=200000)
            # vectorizer = CountVectorizer(ngram_range = (1,2),\
            #                             max_features=20000)
            vectorizer = BertVectorizer()
            vectorizer.fit(self.comments)
        self.vectorizer = vectorizer
        self.vectors = self.vectorizer.transform(self.comments)
        # lengths = np.array([len(x.split()) for x in self.comments])
        # self.vectors = np.concatenate([self.vectors, np.expand_dims(lengths, 1)], axis=1)
        #self.vectors = np.expand_dims(np.asarray([len(x.split()) for x in
        #                           self.comments]),1)
        self.labels = np.asarray(self.labels)
        
    
    def get_xy(self):
        return self.vectors, self.labels


class dataset_lstm:
    # for tokenizing and batching
 
    def __init__(self, fil,vocab=None, vocab_size = 20000):
        self.comments = []
        self.labels = []
        with open(fil, 'r') as f:
            for line in tqdm.tqdm(f.read().splitlines()):
                comment, label = line.split('\t')
                tokens = self.get_tokens(comment)
                if len(tokens) == 0:
                    continue
                self.comments.append(tokens)
                self.labels.append(int(label))
        self.labels = np.asarray(self.labels)

        if vocab is None:
            self.vocab = Vocab(self.comments, vocab_size)
        else:
            self.vocab = vocab
        self.encoded_comments, self.lengths = self.vocab.encode(self.comments)


    def get_tokens(self, comment):
        tokens = word_tokenize(comment)
        tokens = [word.lower() for word in tokens]
        return tokens

    def get_batches(self, batch_size = 32, permute=True):
        if permute:
            perm = np.random.permutation(len(self.comments))
        else:
            perm = np.arange(len(self.comments))

        for i in tqdm.trange(0, len(self.comments), batch_size):
            x = [torch.tensor(self.encoded_comments[j], dtype=torch.long,
                            device=DEVICE) for j in perm[i:i+batch_size]]
            lens = torch.tensor(self.lengths[perm[i:i+batch_size]],
                                dtype=torch.long, device=DEVICE)
            labels = torch.tensor(self.labels[perm[i:i+batch_size]],
                                  dtype=torch.float, device=DEVICE)
            yield x, lens, labels

           
class dataset_bert:
    # for tokenizing and batching
 
    def __init__(self, fil,vocab=None, vocab_size = 20000):
        self.comments = []
        self.labels = []
        self.tokenizer = BertModel.from_pretrained('bert-base-uncased')
        with open(fil, 'r') as f:
            for line in tqdm.tqdm(f.read().splitlines()):
                comment, label = line.split('\t')
                self.comments.append(comment)
                self.labels.append(int(label))
        self.labels = np.asarray(self.labels)
        self.encoded_comments = [self.tokenizer.encode(comment) for comment in
                                 comments]
        



    def get_batches(self, batch_size = 32, permute=True):
        if permute:
            perm = np.random.permutation(len(self.comments))
        else:
            perm = np.arange(len(self.comments))

        for i in tqdm.trange(0, len(self.comments), batch_size):
            # comments = [self.comments[i]
            x = [torch.tensor(self.encoded_comments[j], dtype=torch.long,
                            device=DEVICE) for j in perm[i:i+batch_size]]
            lens = torch.tensor(self.lengths[perm[i:i+batch_size]],
                                dtype=torch.long, device=DEVICE)
            labels = torch.tensor(self.labels[perm[i:i+batch_size]],
                                  dtype=torch.float, device=DEVICE)
            yield x, lens, labels



class dataset_ranking:
    def __init__(self, directory, vectorizer=None):
        fmap = defaultdict(list)
        self.all_comments = []
        print('reading data')
        for fil in os.listdir(directory):
            fname = os.path.join(directory, fil)
            with open(fname, 'r') as f:
                for line in f:
                    comment, score = line.split('\t')
                    score = int(score)
                    fmap[fil].append((comment, score))
                    self.all_comments.append(comment)
        print('fitting vectorizer')

        if vectorizer is None:
            vectorizer = TfidfVectorizer(min_df=20)
            # vectorizer = CountVectorizer(ngram_range=(1,2))
            vectorizer.fit(self.all_comments)
        self.num_features = len(vectorizer.get_feature_names())
        self.vectorizer = vectorizer
        self.vectorized_fmap = defaultdict(list)
        for fil, data in tqdm.tqdm(fmap.items(), desc='Transforming'):
            comments = [x[0] for x in data]
            max_score = np.max([x[1] for x in data])
            if max_score == 0:
                continue
            vecs = self.vectorizer.transform(comments)
            self.vectorized_fmap[fil] = [(comment, vector, score, score/max_score) for
                            ((comment, score),vector) in zip(data, vecs)]
        # self.all_ids = []
        # for fil, data in tqdm.tqdm(self.vectorized_fmap.items()):
        #     n = len(data)
        #     for i in range(n):
        #         for j in range(i, n):
        #             if data[i][2] > data[j][2]:
        #                 self.all_ids.append((fil, i, j))
        self.files = list(self.vectorized_fmap.keys())
        self.num_samples = [(len(self.vectorized_fmap[f]) *
                            (len(self.vectorized_fmap[f])-1))/2 for f in self.files]
        self.num_samples = [len(comments) for f, comments in
                           self.vectorized_fmap.items()]
        self.epoch_size = np.sum([len(comments) for f, comments in \
                                  self.vectorized_fmap.items()])
        print("Total_traiing examples = %d"%np.sum(self.num_samples))
        self.num_samples/=np.sum(self.num_samples)
        
        
    
    def get_batches(self, bsize = 64):
        # permutation = np.random.permutation(len(self.all_ids))
        # each comment gets seen once on average
        for i in tqdm.trange(int(self.epoch_size/bsize), desc='Training batches'):
            vec1 =[]
            vec2 = []
            targets = []
            while len(vec1) < bsize:
                s = np.random.choice(len(self.files), p=self.num_samples)
                selected_file = self.vectorized_fmap[self.files[s]]
                a,b = np.random.choice(len(selected_file), 2, replace=False)
                if selected_file[a][2] > selected_file[b][2]+DELTA:
                    vec1.append(selected_file[a][1].todense())
                    vec2.append(selected_file[b][1].todense())
                    targets.append(1.0)
                elif selected_file[b][2] > selected_file[a][2]+DELTA:
                    targets.append(0.0)
                    vec1.append(selected_file[a][1].todense())
                    vec2.append(selected_file[b][1].todense())
                # targets.append(sigmoid(selected_file[a][3] - \
                #                        selected_file[b][3]))
            vec1 = np.stack(vec1)
            vec2 = np.stack(vec2)
            targets = np.stack(targets)
            yield vec1, vec2, targets

                
            # for j in permutation[i:i+bsize]:
            #     f, a, b = self.all_ids[j]
            #     vec1.append(self.vectorized_fmap[f][a][1].todense())
            #     vec2.append(self.vectorized_fmap[f][b][1].todense())
            # vec1 = np.stack(vec1)
            # vec2 = np.stack(vec2)
            # yield vec1, vec2

    def eval_batches(self):
        for fil in tqdm.tqdm(self.vectorized_fmap):
            comments, vectors, unnormalized, normalized_scores = zip(*self.vectorized_fmap[fil])
            vectors = [x.todense() for x in vectors]
            yield np.stack(vectors), np.stack(normalized_scores)




class dataset_ranking_lstm:
    def __init__(self, directory, vocab=None, vocab_size=20000):
        fmap = defaultdict(list)
        self.all_comments = []
        print('reading data')
        for fil in os.listdir(directory):
            fname = os.path.join(directory, fil)
            with open(fname, 'r') as f:
                for line in f:
                    comment, score = line.split('\t')
                    score = int(score)
                    fmap[fil].append((comment, score))
                    self.all_comments.append(comment)
        if vocab is None:
            self.vocab = Vocab(self.all_comments, vocab_size)
        else:
            self.vocab = vocab
        self.vectorized_fmap = defaultdict(list)
        for fil, data in tqdm.tqdm(fmap.items(), desc='Transforming'):
            comments = [x[0] for x in data]
            max_score = np.max([x[1] for x in data])
            if max_score == 0:
                continue
            vecs, lens = self.vocab.encode(comments)
            self.vectorized_fmap[fil] = [(comment, vector, score, score/max_score) for
                            ((comment, score),vector) in zip(data, vecs)]
        # self.all_ids = []
        # for fil, data in tqdm.tqdm(self.vectorized_fmap.items()):
        #     n = len(data)
        #     for i in range(n):
        #         for j in range(i, n):
        #             if data[i][2] > data[j][2]:
        #                 self.all_ids.append((fil, i, j))
        self.files = list(self.vectorized_fmap.keys())
        # self.num_samples = [(len(self.vectorized_fmap[f]) *
        #                     (len(self.vectorized_fmap[f])-1))/2 for f in self.files]
        self.num_samples = [len(comments) for f, comments in
                           self.vectorized_fmap.items()]
        self.epoch_size = np.sum([len(comments) for f, comments in \
                                  self.vectorized_fmap.items()])
        print("Total_traiing examples = %d"%np.sum(self.num_samples))
        self.num_samples/=np.sum(self.num_samples)
        
        
    
    def get_batches(self, bsize = 64):
        # permutation = np.random.permutation(len(self.all_ids))
        # each comment gets seen once on average
        for i in tqdm.trange(int(self.epoch_size/bsize), desc='Training batches'):
            vec1 =[]
            vec2 = []
            targets = []
            while len(vec1) < bsize:
                s = np.random.choice(len(self.files), p=self.num_samples)
                selected_file = self.vectorized_fmap[self.files[s]]
                a,b = np.random.choice(len(selected_file), 2, replace=False)
                vec1.append(torch.tensor(selected_file[a][1], device=DEVICE,\
                                        dtype=torch.long))
                vec2.append(torch.tensor(selected_file[b][1], device=DEVICE,\
                                        dtype=torch.long))
                if selected_file[a][3] > selected_file[b][3]+DELTA:
                    targets.append(1.0)
                    vec1.append(torch.tensor(selected_file[a][1], device=DEVICE,\
                                        dtype=torch.long))
                    vec2.append(torch.tensor(selected_file[b][1], device=DEVICE,\
                                        dtype=torch.long))
                elif selected_file[b][3] > selected_file[a][3]+DELTA:
                    targets.append(0.0)
                    vec1.append(torch.tensor(selected_file[a][1], device=DEVICE,\
                                        dtype=torch.long))
                    vec2.append(torch.tensor(selected_file[b][1], device=DEVICE,\
                                        dtype=torch.long))
            l1 = torch.tensor([len(x) for x in vec1], device=DEVICE,\
                                  dtype=torch.long)
            l2 = torch.tensor([len(x) for x in vec2], device=DEVICE,\
                                  dtype=torch.long)
            targets = torch.tensor(np.stack(targets), device=DEVICE, \
                                   dtype=torch.float)
            yield vec1, l1, vec2,l2, targets

                
            # for j in permutation[i:i+bsize]:
            #     f, a, b = self.all_ids[j]
            #     vec1.append(self.vectorized_fmap[f][a][1].todense())
            #     vec2.append(self.vectorized_fmap[f][b][1].todense())
            # vec1 = np.stack(vec1)
            # vec2 = np.stack(vec2)
            # yield vec1, vec2

    def eval_batches(self):
        for fil in tqdm.tqdm(self.vectorized_fmap):
            comments, vectors, unnormalized, normalized_scores = zip(*self.vectorized_fmap[fil])
            vectors = [torch.tensor(x, dtype= torch.long, device=DEVICE) for x \
                              in vectors]
            lengths = torch.tensor([len(x) for x in vectors], device=DEVICE,
                                   dtype= torch.long)
            yield vectors, lengths, np.stack(normalized_scores)



