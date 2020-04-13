import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.tokenize import word_tokenize
from collections import Counter, defaultdict
from settings import *

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
            vectorizer = TfidfVectorizer(ngram_range=(1,2))
            # vectorizer = CountVectorizer(ngram_range=(1,2))
            vectorizer.fit(self.comments)
        self.vectorizer = vectorizer
        self.vectors = self.vectorizer.transform(self.comments)
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
            for line in f.readlines():
                comment, label = line.split('\t')
                self.comments.append(self.get_tokens(comment))
                self.labels.append(int(label))
        self.labels = np.asarray(self.labels)
        if vocab is None:
            wordcounts = Counter()
            for comment in self.comments:
                wordcounts += Counter(comment)
            most_common = workcounts.most_common(vocab_size)
            vocab = [x[0] for x in most_common] + [UNK_TOKEN]
        self.unk_id = len(vocab)
        self.pad_id = len(vocab) + 1
        vocab_dict = defaultdict(self.unk_id)
        for i, word in enumerate(vocab):
            vocab_dict[word] = i
        self.vocab = vocab
        self.vocab_dict = vocab_dict
        self.encoded_comments = [np.asarray([self.vocab_dict[w] for w in \
                                             comment]) for comment in \
                                 self.comments]
        self.lengths = np.array([len(com) for com in self.encoded_comments])


    def get_tokens(self, comment):
        tokens = word_tokenize(comment)
        tokens = [word.lower() for word in tokens]
        return tokens

    def get_batches(self, batch_size = 32, permute=True):
        if permute:
            perm = np.random.permutation(len(self.comments))
        else:
            perm = np.arange(len(self.comments))

        for i in range(0, len(self.comments), batch_size):
            x = [torch.tensor(self.encoded_comments[j], dtype=torch.int32,
                            device=DEVICE) for j in perm[i:i+batch_size]]
            lens = torch.tensor(self.lengths[perm[i:i+batch_size]],
                                dtype=torch.int32, device=DEVICE)
            labels = torch.tensor(self.labels[perm[i:i+batch_size]],
                                  dtype=torch.int32, device=DEVICE)
            yield x, lens, labels

            
