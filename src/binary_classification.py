# simple baseline
from data import dataset
import numpy as np
from sklearn.linear_model import LogisticRegression

traindata = dataset('train_90_10.txt')
vectorizer = traindata.vectorizer
valdata = dataset('val_90_10.txt', vectorizer)
testdata = dataset('test_90_10.txt', vectorizer)

trainx, trainy = traindata.get_xy()
valx, valy = valdata.get_xy()

print('training')
clf = LogisticRegression(random_state=0, verbose=1).fit(trainx, trainy)

print(clf.score(valx, valy))



