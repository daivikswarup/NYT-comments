# simple baseline
from data import dataset
import numpy as np
from sklearn.linear_model import LogisticRegression
import sys
from sklearn import preprocessing



traindata = dataset('train_80_20.txt')
vectorizer = traindata.vectorizer
valdata = dataset('val_80_20.txt', vectorizer)
testdata = dataset('test_80_20.txt', vectorizer)

trainx, trainy = traindata.get_xy()
valx, valy = valdata.get_xy()


print('training')
clf = LogisticRegression(random_state=0, verbose=1,max_iter=1000).fit(trainx, trainy)

print(clf.score(valx, valy))



