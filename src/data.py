import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer



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
            #vectorizer = TfidfVectorizer(ngram_range=(1,2))
            vectorizer = CountVectorizer(ngram_range=(1,2))
            vectorizer.fit(self.comments)
        self.vectorizer = vectorizer
        self.vectors = self.vectorizer.transform(self.comments)
        self.labels = np.asarray(self.labels)
        
    
    def get_xy(self):
        return self.vectors, self.labels



