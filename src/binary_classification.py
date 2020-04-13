# simple baseline
from data import dataset
import numpy as np
from sklearn.linear_model import LogisticRegression

traindata = dataset('train.txt')
vectorizer = traindata.vectorizer
valdata = dataset('val.txt', vectorizer)
testdata = dataset('test.txt', vectorizer)

trainx, trainy = traindata.get_xy()
valx, valy = valdata.get_xy()

print('training')
clf = LogisticRegression(random_state=0, verbose=1).fit(trainx, trainy)

print(clf.score(valx, valy))



